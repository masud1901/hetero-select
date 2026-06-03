from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def compute_loss_scores(
    model: nn.Module,
    loaders,
    eval_batches: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.train(False)
    losses = np.zeros(len(loaders))
    for k, ld in enumerate(loaders):
        total, nb = 0.0, 0
        for i, (x, y) in enumerate(ld):
            if i >= eval_batches:
                break
            total += F.cross_entropy(model(x.to(device)), y.to(device)).item()
            nb += 1
        losses[k] = total / max(nb, 1)
    mn, mx = float(losses.min()), float(losses.max())
    if mx - mn > 1e-8:
        scores = (losses - mn) / (mx - mn + 1e-8)
    else:
        scores = np.ones(len(losses))
    model.train(True)
    return scores, losses


def compute_diversity_scores(
    num_clients: int,
    server_grad_avg: Optional[torch.Tensor],
    client_grads_prev: Dict[int, Optional[torch.Tensor]],
) -> np.ndarray:
    if server_grad_avg is None or len(client_grads_prev) == 0:
        return np.full(num_clients, 0.5)

    out = np.full(num_clients, 0.5)
    avg_norm = server_grad_avg.norm().item()
    if avg_norm < 1e-8:
        return out

    for k, g_k in client_grads_prev.items():
        if g_k is None:
            continue
        gk_norm = g_k.norm().item()
        if gk_norm < 1e-8:
            continue
        cos = (g_k * server_grad_avg).sum().item() / (gk_norm * avg_norm + 1e-8)
        out[k] = float(np.clip(1.0 - cos, 0.0, 1.0))
    return out


def compute_fairness_scores(
    num_clients: int,
    sel_counts: Dict[int, int],
    rnd: int,
) -> np.ndarray:
    counts = np.array([sel_counts.get(k, 0) for k in range(num_clients)], dtype=float)
    mean_c = counts.mean() if counts.mean() > 0 else 1.0
    return np.clip(1.0 - counts / mean_c, -1.0, 1.0)


def compute_staleness_scores(
    num_clients: int,
    last_selected: Dict[int, int],
    rnd: int,
    gamma: float,
) -> np.ndarray:
    raw = np.zeros(num_clients)
    for k in range(num_clients):
        t_since = rnd - last_selected.get(k, 0)
        raw[k] = gamma * math.log(1.0 + t_since)
    mn, mx = float(raw.min()), float(raw.max())
    if mx - mn > 1e-8:
        raw = (raw - mn) / (mx - mn)
    return raw


def score_clients(
    model: nn.Module,
    loaders,
    eval_batches: int,
    device: torch.device,
    server_grad_avg: Optional[torch.Tensor],
    client_grads_prev: Dict[int, Optional[torch.Tensor]],
    sel_counts: Dict[int, int],
    last_selected: Dict[int, int],
    rnd: int,
    fl: dict,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    K = len(loaders)
    V, losses = compute_loss_scores(model, loaders, eval_batches, device)
    if fl.get("no_V"):
        V = np.zeros(len(loaders))
    D = compute_diversity_scores(K, server_grad_avg, client_grads_prev)
    Fc = compute_fairness_scores(K, sel_counts, rnd)
    St = compute_staleness_scores(K, last_selected, rnd, fl["gamma_St"])

    S = (
        fl.get("lambda_V", 1.0) * V
        + fl["lambda_D"]  * D
        + fl["lambda_F"]  * Fc
        + fl["lambda_St"] * St
    )

    mn, mx = float(S.min()), float(S.max())
    if mx - mn > 1e-8:
        S = (S - mn) / (mx - mn + 1e-8)
    else:
        S = np.ones(len(S))

    return S, losses, {"V": V, "D": D, "F": Fc, "St": St}


def softmax_select(
    scores: np.ndarray,
    m: int,
    tau: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    s = scores / tau
    s -= s.max()
    p = np.exp(s)
    p /= p.sum()
    return rng.choice(len(scores), size=m, replace=False, p=p)
