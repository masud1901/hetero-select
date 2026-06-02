from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_theta(rnd: int, fl: dict) -> float:
    if rnd <= fl["warmup_rounds"]:
        return 1.0
    t_eff = rnd - fl["warmup_rounds"]
    T_eff = max(fl["num_rounds"] - fl["warmup_rounds"], 1)
    theta = fl["theta_total"] * (
        1.0 + fl["alpha_cos"] * math.cos(math.pi * (t_eff - 1) / max(T_eff - 1, 1))
    )
    return max(float(theta), fl["theta_floor"])


def adaptive_beta(theta_t: float, beta_min: float = 0.85, beta_max: float = 0.97) -> float:
    return beta_min + (beta_max - beta_min) * (1.0 - theta_t)


def resolve_beta(theta_t: float, fl: dict) -> float:
    """Adaptive β by default; if fl['static_beta'] is set, return it verbatim."""
    sb = fl.get("static_beta")
    if sb is not None:
        return float(sb)
    return adaptive_beta(theta_t)


def adaptive_ratios(
    scores: np.ndarray,
    selected: Sequence[int],
    theta_t: float,
    bw_bps: np.ndarray,
    model_bits: int,
    T_budget: float,
) -> np.ndarray:
    s = scores[list(selected)].copy()
    s_mean = s.mean()
    if s_mean < 1e-8:
        s = np.ones(len(selected))
        s_mean = 1.0
    theta_star = (s / s_mean) * theta_t
    theta_ceil = np.clip(bw_bps[list(selected)] * T_budget / model_bits, 0.0, 1.0)
    return np.clip(np.minimum(theta_star, theta_ceil), 0.005, 1.0)


def uniform_ratios(m: int, theta_t: float) -> np.ndarray:
    return np.full(m, theta_t)


def get_layer_ranges(model: nn.Module) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    ptr = 0
    for module in model.modules():
        if list(module.children()):
            continue
        n = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )
        if n > 0:
            ranges.append((ptr, ptr + n))
            ptr += n
    return ranges


def markov_sample_layers(
    prev_mask: Optional[torch.Tensor],
    layer_ranges: Sequence[Tuple[int, int]],
    Q: int,
    lam: float,
    rng: np.random.RandomState,
) -> List[int]:
    L = len(layer_ranges)
    if prev_mask is None:
        importance = np.ones(L) / L
    else:
        importance = np.array([
            prev_mask[s:e].float().mean().item() for s, e in layer_ranges
        ])
        total = importance.sum()
        importance = importance / total if total > 1e-8 else np.ones(L) / L
    p = (1.0 - lam) * importance + lam / L
    p /= p.sum()
    return rng.choice(L, size=min(Q, L), replace=False, p=p).tolist()


def compute_hess_diag(
    local_model: nn.Module,
    loader,
    mu: float,
    g_clones: Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    local_model.train()
    local_model.zero_grad()
    params = [p for p in local_model.parameters() if p.requires_grad]

    out = local_model(x)
    ce = F.cross_entropy(out, y)
    prox = sum(
        ((p - gp.detach().to(device)) ** 2).sum()
        for p, gp in zip(local_model.parameters(), g_clones)
    )
    loss = ce + (mu / 2.0) * prox

    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    grads = [
        g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)
    ]

    z_parts = [
        torch.randint(0, 2, p.shape, device=device).float() * 2 - 1 for p in params
    ]
    z_flat = torch.cat([z.flatten() for z in z_parts])
    gz = (torch.cat([g.flatten() for g in grads]) * z_flat).sum()

    hvps = torch.autograd.grad(gz, params, allow_unused=True)
    hvp_flat = torch.cat([
        h.detach().flatten()
        if h is not None else torch.zeros(p.numel(), device=device)
        for h, p in zip(hvps, params)
    ])
    local_model.zero_grad()
    return (z_flat.detach() * hvp_flat).cpu()


def topk_compress(
    delta: torch.Tensor,
    ratio: float,
    ebuf: torch.Tensor,
    beta: float,
    layer_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    sel_layers: Optional[Sequence[int]] = None,
    h_diag: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v = delta + ebuf
    k = max(1, int(ratio * v.numel()))

    scoring = v.pow(2).clone()
    if h_diag is not None and sel_layers and layer_ranges is not None:
        h_full = h_diag.to(v.device)
        for l_idx in sel_layers:
            if l_idx >= len(layer_ranges):
                continue
            s, e = layer_ranges[l_idx]
            h_l = h_full[s:e].abs().clamp(min=1e-6)
            scoring[s:e] = v[s:e].pow(2) / h_l

    idx = scoring.topk(k).indices
    comp = torch.zeros_like(v)
    comp[idx] = v[idx]
    mask = torch.zeros(v.numel(), dtype=torch.bool, device=v.device)
    mask[idx] = True

    new_ebuf = beta * (v - comp)
    return comp, new_ebuf, mask.cpu()
