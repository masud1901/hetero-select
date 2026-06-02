from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .client import fedprox_train
from .compression import (
    adaptive_ratios,
    compute_hess_diag,
    cosine_theta,
    get_layer_ranges,
    markov_sample_layers,
    resolve_beta,
    topk_compress,
    uniform_ratios,
)
from .data import make_loaders, partition_data
from .models import build_model, n_params, size_mb
from .scoring import score_clients, softmax_select
from .server import calibrate_bn, evaluate, score_weighted_aggregate
from .simulator import sim_round_time, sim_traffic_mb
from .utils import get_device


def run_experiment(
    cfg: Dict[str, Any],
    fl: Dict[str, Any],
    train_ds,
    test_loader,
    rng: np.random.RandomState,
    *,
    device: Optional[torch.device] = None,
    log_every: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    if device is None:
        device = get_device()

    dataset = cfg["dataset"]
    variant = cfg.get("variant", "adaptive")
    mu = cfg.get("mu", fl["mu"])
    seed = cfg["seed"]

    if verbose:
        bar = "━" * 68
        print(f"\n{bar}")
        print(
            f"  {dataset.upper()}  ψ={cfg['psi']}  seed={seed}  "
            f"variant={variant}  μ={mu}"
        )
        print(bar)

    cidx = partition_data(train_ds, cfg, fl["num_clients"], seed)
    loaders = make_loaders(train_ds, cidx, fl["batch_size"])
    model = build_model(dataset, device)
    N = n_params(model)
    R_mb = size_mb(model)
    R_bits = N * 32

    layer_ranges = get_layer_ranges(model)

    bw_mean = (fl["bw_min_mbps"] + fl["bw_max_mbps"]) / 2.0 * 1e6
    tc_mean = (fl["comp_min_s"]  + fl["comp_max_s"])  / 2.0
    T_budget = fl["theta_total"] * R_bits / bw_mean + fl["local_steps"] * tc_mean

    rng_c = np.random.RandomState(seed + 9999)
    c_time = rng_c.uniform(
        fl["comp_min_s"], fl["comp_max_s"], fl["num_clients"]
    )

    ebufs: Dict[int, torch.Tensor] = {
        k: torch.zeros(N, device=device) for k in range(fl["num_clients"])
    }
    markov_masks: Dict[int, Optional[torch.Tensor]] = {
        k: None for k in range(fl["num_clients"])
    }
    sel_counts: Dict[int, int] = {}
    last_selected: Dict[int, int] = {}
    client_grads_prev: Dict[int, torch.Tensor] = {}
    server_grad_avg: Optional[torch.Tensor] = None
    momentum_buf = torch.zeros(N, device=device)

    target = fl["target_acc"][dataset]
    logs: List[Dict[str, Any]] = []
    cum_t = cum_mb = 0.0
    peak = 0.0
    rounds_to_target: Optional[int] = None
    time_to_target: Optional[float] = None
    traffic_to_target: Optional[float] = None
    last10: List[float] = []
    t0 = time.time()

    for rnd in range(1, fl["num_rounds"] + 1):
        scores, raw_losses, score_parts = score_clients(
            model, loaders, fl["eval_batches"], device,
            server_grad_avg, client_grads_prev,
            sel_counts, last_selected, rnd, fl,
        )

        tau = fl["tau_0"] * (1.0 - 0.5 * min(rnd / fl["num_rounds"], 1.0))
        sel = softmax_select(scores, fl["clients_per_round"], tau, rng)
        for k in sel:
            sel_counts[k] = sel_counts.get(k, 0) + 1
            last_selected[k] = rnd

        theta_t = cosine_theta(rnd, fl)
        beta_t = resolve_beta(theta_t, fl)
        lr_base = fl["local_lr"] * (1.0 - 0.5 * min(rnd / fl["num_rounds"], 1.0))

        if variant == "stress":
            norm = raw_losses / (raw_losses.max() + 1e-8)
            inv = 1.0 - norm
            bw_mbps = (
                fl["bw_min_mbps"]
                + (fl["bw_max_mbps"] - fl["bw_min_mbps"]) * inv
            )
        else:
            bw_mbps = rng.uniform(
                fl["bw_min_mbps"], fl["bw_max_mbps"], fl["num_clients"]
            )
        bw_bps = bw_mbps * 1e6

        if variant == "uniform":
            theta_k = uniform_ratios(fl["clients_per_round"], theta_t)
        else:
            theta_k = adaptive_ratios(
                scores, sel, theta_t, bw_bps, R_bits, T_budget,
            )

        compressed: List[torch.Tensor] = []
        scores_log: List[float] = []
        theta_k_log: List[float] = []
        round_grads: Dict[int, torch.Tensor] = {}

        for i, k in enumerate(sel):
            if fl.get("uniform_lr"):
                lr_k = lr_base
            else:
                lr_k = min(
                    lr_base * (1.0 + float(scores[k])), fl["lr_scale_cap"]
                )
            delta, local_model = fedprox_train(
                model, loaders[k], fl["local_steps"],
                lr_k, mu, device, fl["grad_clip"],
            )

            if rnd > fl["warmup_rounds"]:
                sel_layers = markov_sample_layers(
                    markov_masks[k], layer_ranges,
                    fl["newton_Q"], fl["newton_lambda"], rng,
                )
                g_clones = [p.data.clone() for p in model.parameters()]
                h_diag = compute_hess_diag(
                    local_model, loaders[k], mu, g_clones, device,
                )
            else:
                sel_layers = []
                h_diag = None

            comp, ebufs[k], new_mask = topk_compress(
                delta, float(theta_k[i]), ebufs[k], beta_t,
                layer_ranges, sel_layers, h_diag,
            )
            markov_masks[k] = new_mask

            compressed.append(comp)
            round_grads[k] = comp.cpu()
            scores_log.append(float(scores[k]))
            theta_k_log.append(float(theta_k[i]))

        agg_scores = (
            [1.0] * len(scores_log) if fl.get("uniform_aggregation")
            else scores_log
        )
        momentum_buf = score_weighted_aggregate(
            model, compressed, agg_scores, momentum_buf, fl["beta_s"],
        )

        client_grads_prev.update(round_grads)
        s_arr = np.array(scores_log, dtype=np.float32)
        s_arr = s_arr / (s_arr.sum() + 1e-8)
        with torch.no_grad():
            delta_scored_cpu = torch.zeros(N)
            for w, comp in zip(s_arr, compressed):
                delta_scored_cpu.add_(comp.cpu(), alpha=float(w))
        server_grad_avg = (
            0.9 * server_grad_avg + 0.1 * delta_scored_cpu
            if server_grad_avg is not None
            else delta_scored_cpu
        )

        if dataset in ("cifar100", "tinyimagenet"):
            calibrate_bn(
                model, [loaders[k] for k in sel[:3]],
                fl["bn_calib_batches"], device,
            )

        acc = evaluate(model, test_loader, device)

        r_time = sim_round_time(
            theta_k, bw_bps[sel], R_bits, fl["local_steps"], c_time[sel],
        )
        r_traffic = sim_traffic_mb(theta_k, R_mb)
        cum_t += r_time
        cum_mb += r_traffic

        peak = max(peak, acc)
        if rounds_to_target is None and acc >= target:
            rounds_to_target = rnd
            time_to_target = cum_t
            traffic_to_target = cum_mb
        last10.append(acc)
        if len(last10) > 10:
            last10.pop(0)

        logs.append(dict(
            round          = rnd,
            test_acc       = round(float(acc), 6),
            tau            = round(float(tau), 4),
            lr_base        = round(float(lr_base), 6),
            theta_t        = round(float(theta_t), 5),
            beta_t         = round(float(beta_t), 5),
            selected       = sel.tolist(),
            scores_sel     = [round(s, 5) for s in scores_log],
            theta_k        = [round(t, 5) for t in theta_k_log],
            bw_mbps_sel    = [round(float(bw_mbps[k]), 3) for k in sel],
            losses_mean    = round(float(raw_losses.mean()), 5),
            losses_std     = round(float(raw_losses.std()),  5),
            score_V_mean   = round(float(score_parts["V"].mean()),  5),
            score_D_mean   = round(float(score_parts["D"].mean()),  5),
            score_F_mean   = round(float(score_parts["F"].mean()),  5),
            score_St_mean  = round(float(score_parts["St"].mean()), 5),
            sim_time_s     = round(r_time, 3),
            sim_traffic_mb = round(r_traffic, 5),
            cum_time_s     = round(cum_t, 3),
            cum_traffic_mb = round(cum_mb, 4),
            wall_s         = round(time.time() - t0, 1),
        ))

        if verbose and (rnd % log_every == 0 or rnd == 1):
            w_str = " [WARMUP]" if rnd <= fl["warmup_rounds"] else ""
            print(
                f"  [{rnd:3d}] acc={acc:.4f}  θ={theta_t:.3f}"
                f"  β={beta_t:.3f}  Σmb={cum_mb:7.1f}"
                f"  Σt={cum_t:7.0f}s  wall={time.time() - t0:5.0f}s{w_str}"
            )

    final = logs[-1]["test_acc"]
    s_drop = peak - final
    m_last = float(np.mean(last10))

    summary: Dict[str, Any] = dict(
        peak_acc             = round(peak,   6),
        final_acc            = round(final,  6),
        mean_last10_acc      = round(m_last, 6),
        stability_drop       = round(s_drop, 6),
        rounds_to_target     = rounds_to_target,
        time_to_target_s     = (
            round(time_to_target, 2) if time_to_target is not None else None
        ),
        traffic_to_target_mb = (
            round(traffic_to_target, 3) if traffic_to_target is not None else None
        ),
        total_sim_time_s     = round(cum_t, 2),
        total_traffic_mb     = round(cum_mb, 3),
        target_hit           = rounds_to_target is not None,
        mu                   = mu,
        model_params         = N,
        model_mb             = round(R_mb, 2),
    )

    if verbose:
        print(
            f"\n  RESULT  peak={peak:.4f}  final={final:.4f}"
            f"  drop={s_drop:.4f}  target={'✓' if rounds_to_target else '✗'}"
        )
        if rounds_to_target is not None:
            assert time_to_target is not None and traffic_to_target is not None
            print(
                f"          rounds={rounds_to_target}  "
                f"time={time_to_target:.0f}s  traffic={traffic_to_target:.1f}MB"
            )

    return dict(
        config=cfg,
        fl_config=fl,
        summary=summary,
        rounds=logs,
    )
