#!/usr/bin/env python3
"""Render the 7 figures of the HeteRo-Select ICDM submission.

Output (all under --out, default figs/paper/):
    fig1_cifar10_bars.png        — resource overhead bar chart (vs. FedCG)
    fig2_mnist_tin.png           — (a) MNIST acc-vs-time, (b) TIN completion vs psi @ 0.30
    fig3_c10_c100_bands.png      — CIFAR-10/100 acc-vs-round, mean ±1σ (seeds 42-44)
    fig4_cifar10_ablations.png   — (a) adaptive vs uniform comp, (b) FedProx mu sensitivity
    fig5_tinyimagenet.png        — (a) TIN acc-vs-round, (b) TIN completion vs psi @ 0.37
    fig6_score_components.png    — (a) V/D/F/St means, (b) per-client selection counts
    fig7_stress_test.png         — inverted-coupling: acc vs cumulative traffic
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "pdf.fonttype": 42,    # TrueType embedding (IEEE/ACM requirement)
    "ps.fonttype":  42,
})

# IEEE 2-column figure widths: ~3.4 in (single column), ~7.0 in (double column).
# Bumped a touch to keep 6-line legends readable.
DOUBLE_W = 8.2
DOUBLE_H = 3.2
SINGLE_W = 3.5
SINGLE_H = 2.7

# ---------------------------------------------------------------------------
# Style — HeteRo-Select highlighted in orange, baselines from InfoCom tables.
# ---------------------------------------------------------------------------

STYLE = {
    "FedAvg":        dict(color="#1a1a1a", ls=(0, (3, 1, 1, 1)), lw=1.7),
    "OptRate":       dict(color="#d62728", ls=(0, (5, 2)),       lw=1.7),
    "FlexCom":       dict(color="#1f4ea1", ls=(0, (3, 1, 1, 1)), lw=1.7),
    "AdaSample":     dict(color="#2ca02c", ls=(0, (1, 1)),       lw=1.7),
    "FedCG":         dict(color="#9c27b0", ls="-",               lw=1.9),
    "HeteRo-Select": dict(color="#ff7f0e", ls="-",               lw=2.4),
}
NIID_MARK = {
    "FedAvg":        ("#1a1a1a", "s"),
    "OptRate":       ("#d62728", "o"),
    "FlexCom":       ("#1f4ea1", "^"),
    "AdaSample":     ("#2ca02c", "v"),
    "FedCG":         ("#9c27b0", "D"),
    "HeteRo-Select": ("#ff7f0e", "*"),
}
BAR_COLORS = {
    "FedAvg":        "#bfbfbf",
    "OptRate":       "#f4b6b6",
    "FlexCom":       "#b6cfe9",
    "AdaSample":     "#b9dfb9",
    "FedCG":         "#d6b8e0",
    "HeteRo-Select": "#ff7f0e",
}

# Paper Table I, row order matches the paper.
TABLE1 = {
    # method: (c10_time_s, c10_traffic_mb, c100_time_s, c100_traffic_mb)
    "FedAvg":    (15932, 15199, 35048, 32551),
    "OptRate":   ( 9697,  6193, 24521, 15582),
    "FlexCom":   ( 5334,  2674, 17726,  8583),
    "AdaSample": ( 6968, 11984, 19723, 34570),
    "FedCG":     ( 5170,  2480, 10069,  8402),
}

# --- Baseline curves (verbatim from figures-combined/.../*.md) ---
ROUND_BASELINES: Dict[str, Dict] = {
    "mnist": dict(
        rounds=list(range(0, 101, 5)),
        FedAvg=[0.100, 0.440, 0.650, 0.720, 0.780, 0.800, 0.820, 0.840, 0.850, 0.855, 0.860, 0.865, 0.870, 0.875, 0.880, 0.885, 0.890, 0.892, 0.895, 0.898, 0.900],
        OptRate=[0.100, 0.500, 0.680, 0.760, 0.800, 0.820, 0.840, 0.850, 0.860, 0.865, 0.870, 0.875, 0.880, 0.885, 0.890, 0.895, 0.900, 0.902, 0.905, 0.908, 0.910],
        FlexCom=[0.100, 0.520, 0.680, 0.780, 0.810, 0.830, 0.850, 0.860, 0.870, 0.875, 0.880, 0.885, 0.890, 0.895, 0.900, 0.905, 0.910, 0.912, 0.915, 0.918, 0.920],
        AdaSample=[0.100, 0.480, 0.650, 0.750, 0.790, 0.810, 0.830, 0.850, 0.860, 0.865, 0.870, 0.875, 0.880, 0.885, 0.890, 0.895, 0.900, 0.903, 0.905, 0.908, 0.910],
        FedCG=[0.100, 0.450, 0.650, 0.780, 0.820, 0.840, 0.860, 0.870, 0.880, 0.885, 0.890, 0.895, 0.900, 0.905, 0.910, 0.912, 0.915, 0.917, 0.920, 0.922, 0.925],
    ),
    "cifar10": dict(
        rounds=list(range(0, 101, 5)),
        FedAvg=[0.100, 0.160, 0.350, 0.420, 0.500, 0.550, 0.580, 0.600, 0.620, 0.630, 0.640, 0.650, 0.670, 0.680, 0.700, 0.710, 0.720, 0.725, 0.730, 0.735, 0.740],
        OptRate=[0.100, 0.180, 0.350, 0.400, 0.450, 0.490, 0.530, 0.560, 0.590, 0.620, 0.650, 0.670, 0.690, 0.700, 0.710, 0.720, 0.730, 0.735, 0.738, 0.739, 0.740],
        FlexCom=[0.100, 0.220, 0.350, 0.390, 0.420, 0.460, 0.500, 0.540, 0.570, 0.600, 0.620, 0.640, 0.660, 0.680, 0.700, 0.710, 0.720, 0.730, 0.735, 0.738, 0.740],
        AdaSample=[0.100, 0.200, 0.350, 0.380, 0.410, 0.450, 0.490, 0.520, 0.550, 0.580, 0.610, 0.630, 0.650, 0.670, 0.690, 0.700, 0.710, 0.720, 0.730, 0.735, 0.740],
        FedCG=[0.100, 0.220, 0.380, 0.480, 0.550, 0.600, 0.640, 0.670, 0.690, 0.700, 0.710, 0.720, 0.730, 0.740, 0.742, 0.744, 0.746, 0.748, 0.749, 0.750, 0.750],
    ),
    "cifar100": dict(
        rounds=list(range(0, 101, 5)),
        FedAvg=[0.010, 0.080, 0.150, 0.220, 0.280, 0.330, 0.380, 0.420, 0.450, 0.470, 0.490, 0.500, 0.510, 0.520, 0.525, 0.530, 0.532, 0.535, 0.538, 0.539, 0.540],
        OptRate=[0.010, 0.100, 0.210, 0.290, 0.360, 0.410, 0.450, 0.480, 0.500, 0.510, 0.520, 0.525, 0.530, 0.533, 0.536, 0.538, 0.539, 0.540, 0.541, 0.542, 0.542],
        FlexCom=[0.010, 0.140, 0.270, 0.370, 0.450, 0.480, 0.510, 0.520, 0.530, 0.535, 0.540, 0.540, 0.541, 0.541, 0.542, 0.542, 0.542, 0.542, 0.542, 0.543, 0.543],
        AdaSample=[0.010, 0.120, 0.240, 0.330, 0.410, 0.450, 0.480, 0.500, 0.515, 0.520, 0.525, 0.530, 0.532, 0.534, 0.536, 0.538, 0.539, 0.540, 0.541, 0.542, 0.542],
        FedCG=[0.010, 0.150, 0.260, 0.350, 0.420, 0.460, 0.490, 0.510, 0.525, 0.535, 0.540, 0.541, 0.542, 0.542, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543, 0.543],
    ),
    "tinyimagenet": dict(
        rounds=list(range(0, 101, 5)),
        FedAvg=[0.000, 0.040, 0.080, 0.120, 0.150, 0.180, 0.210, 0.240, 0.270, 0.290, 0.310, 0.325, 0.340, 0.350, 0.360, 0.365, 0.370, 0.375, 0.380, 0.385, 0.390],
        OptRate=[0.000, 0.050, 0.100, 0.150, 0.190, 0.230, 0.260, 0.290, 0.310, 0.330, 0.340, 0.350, 0.360, 0.365, 0.370, 0.375, 0.380, 0.383, 0.385, 0.388, 0.390],
        FlexCom=[0.000, 0.060, 0.130, 0.190, 0.250, 0.290, 0.320, 0.340, 0.350, 0.360, 0.370, 0.375, 0.380, 0.382, 0.384, 0.386, 0.387, 0.388, 0.389, 0.390, 0.390],
        AdaSample=[0.000, 0.070, 0.140, 0.200, 0.250, 0.290, 0.310, 0.330, 0.340, 0.350, 0.360, 0.370, 0.380, 0.382, 0.385, 0.386, 0.387, 0.388, 0.389, 0.390, 0.390],
        FedCG=[0.000, 0.080, 0.150, 0.230, 0.300, 0.340, 0.360, 0.370, 0.380, 0.382, 0.383, 0.384, 0.385, 0.386, 0.387, 0.388, 0.389, 0.389, 0.390, 0.390, 0.390],
    ),
}

TIME_BASELINES: Dict[str, Dict] = {
    "mnist": dict(
        t1e4=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8],
        FedAvg=[0.100, 0.710, 0.820, 0.870, 0.890, 0.900],
        OptRate=[0.100, 0.810, 0.860, 0.890, 0.900, 0.910],
        FlexCom=[0.100, 0.830, 0.880, 0.900, 0.910, 0.920],
        AdaSample=[0.100, 0.790, 0.850, 0.890, 0.900, 0.910],
        FedCG=[0.100, 0.890, 0.910, 0.920, 0.922, 0.925],
    ),
}

NONIID_BASELINES: Dict[str, Dict] = {
    "tinyimagenet": dict(
        psi=[0, 40, 80, 120],
        FedAvg=[5.13, 5.66, 6.17, 8.59],
        OptRate=[4.13, 4.54, 4.99, 7.28],
        FlexCom=[3.33, 3.49, 4.24, 6.17],
        AdaSample=[2.97, 3.05, 3.43, 4.20],
        FedCG=[1.91, 1.98, 2.14, 2.81],
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FNAME_RE = re.compile(
    r"^(?P<ds>[a-z0-9]+)_psi(?P<psi>[0-9p]+)_mu(?P<mu>[0-9p]+)_seed(?P<seed>\d+)_(?P<variant>adaptive|uniform|stress)(?:_(?P<tag>[A-Za-z0-9]+))?\.json$"
)


def _load(p: Path) -> dict:
    with p.open("r") as f:
        return json.load(f)


def _series(run: dict, key: str) -> np.ndarray:
    return np.asarray([r[key] for r in run["rounds"]], dtype=float)


def _psi(s: str) -> float:
    return float(s.replace("p", "."))


def _smooth(x: np.ndarray, y: np.ndarray, n: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 4:
        return x, y
    cs = CubicSpline(x, y)
    xg = np.linspace(x[0], x[-1], n)
    return xg, cs(xg)


def _baseline_curves(ax, base: dict, x_key: str) -> None:
    x = np.asarray(base[x_key], dtype=float)
    for name in ("FedAvg", "OptRate", "FlexCom", "AdaSample", "FedCG"):
        xs, ys = _smooth(x, np.asarray(base[name], dtype=float))
        ax.plot(xs, ys, label=name, **STYLE[name])


def _collect_seeds(root: Path, ds: str, psi: str, mu: str = "0p1",
                   variant: str = "adaptive", tag: str = "") -> List[Path]:
    out: List[Path] = []
    suff = f"_{tag}" if tag else ""
    pat = f"{ds}_psi{psi}_mu{mu}_seed*_{variant}{suff}.json"
    for p in sorted(root.glob(pat)):
        if tag == "" and "_" + p.stem.split("_")[-1] != f"_{variant}":
            # don't pick up tagged files when tag is empty
            tail = p.stem.split("_")[-1]
            if tail != variant:
                continue
        out.append(p)
    return out


def _mean_curve(runs: List[Path], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounds = None
    cols = []
    for p in runs:
        run = _load(p)
        if rounds is None:
            rounds = _series(run, "round")
        cols.append(_series(run, key))
    arr = np.stack(cols, axis=0)
    return rounds, arr.mean(axis=0), arr.std(axis=0)


def _hetero_t2t(runs: List[Path], target: float) -> Optional[float]:
    """Mean simulated time-to-target across seeds, or None."""
    vals = []
    for p in runs:
        run = _load(p)
        acc = _series(run, "test_acc")
        tt = _series(run, "cum_time_s")
        ok = np.where(acc >= target)[0]
        if ok.size:
            vals.append(float(tt[ok[0]]))
    return float(np.mean(vals)) if vals else None


def _hetero_traffic2t(runs: List[Path], target: float) -> Optional[float]:
    vals = []
    for p in runs:
        run = _load(p)
        acc = _series(run, "test_acc")
        tr = _series(run, "cum_traffic_mb")
        ok = np.where(acc >= target)[0]
        if ok.size:
            vals.append(float(tr[ok[0]]))
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Fig 1 — CIFAR-10 resource bar chart
# ---------------------------------------------------------------------------

def fig1_cifar10_bars(exp: Path, out: Path) -> None:
    runs = _collect_seeds(exp, "cifar10", "0p4")
    hs_time = [_hetero_t2t([p], 0.70) for p in runs]
    hs_traf = [_hetero_traffic2t([p], 0.70) for p in runs]
    hs_time = [t for t in hs_time if t is not None]
    hs_traf = [t for t in hs_traf if t is not None]
    hs_time_mean, hs_time_std = float(np.mean(hs_time)), float(np.std(hs_time))
    hs_traf_mean, hs_traf_std = float(np.mean(hs_traf)), float(np.std(hs_traf))

    methods = list(TABLE1.keys()) + ["HeteRo-Select"]
    traf_mb = [TABLE1[m][1] for m in TABLE1] + [hs_traf_mean]
    time_s  = [TABLE1[m][0] for m in TABLE1] + [hs_time_mean]
    traf_err = [0, 0, 0, 0, 0, hs_traf_std]
    time_err = [0, 0, 0, 0, 0, hs_time_std]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, DOUBLE_H))
    x = np.arange(len(methods))
    colors = [BAR_COLORS[m] for m in methods]
    edges = ["black"] * (len(methods) - 1) + ["#cc6510"]

    axes[0].bar(x, np.asarray(traf_mb) / 1e3, yerr=np.asarray(traf_err) / 1e3,
                color=colors, edgecolor=edges, linewidth=1.0, capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_ylabel(r"Traffic to 70%  ($\times 10^3$ MB)")
    axes[0].set_title("(a) CIFAR-10 traffic to 70 %")
    for xi, v in zip(x, traf_mb):
        axes[0].text(xi, v / 1e3 + 0.5, f"{v/1e3:.2f}", ha="center", fontsize=9)

    axes[1].bar(x, np.asarray(time_s) / 1e4, yerr=np.asarray(time_err) / 1e4,
                color=colors, edgecolor=edges, linewidth=1.0, capsize=4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_ylabel(r"Time to 70%  ($\times 10^4$ s)")
    axes[1].set_title("(b) CIFAR-10 time to 70 %")
    for xi, v in zip(x, time_s):
        axes[1].text(xi, v / 1e4 + 0.05, f"{v/1e4:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_cifar10_bars.png"); fig.savefig(out / "fig1_cifar10_bars.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — (a) MNIST acc vs time   (b) TIN completion vs psi @ 0.30
# ---------------------------------------------------------------------------

def fig2_mnist_tin(exp: Path, ablv: Path, niid: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, DOUBLE_H))

    # ---- (a) MNIST acc vs time ----
    ax = axes[0]
    _baseline_curves(ax, TIME_BASELINES["mnist"], "t1e4")
    runs = _collect_seeds(exp, "mnist", "0p4")
    t_min = None
    for p in runs:
        run = _load(p)
        t = _series(run, "cum_time_s") / 1e4
        a = _series(run, "test_acc")
        t_min = t if t_min is None or t[-1] > t_min[-1] else t_min
    grid = np.linspace(0.0, 0.8, 200)
    seed_acc = []
    for p in runs:
        run = _load(p)
        t = _series(run, "cum_time_s") / 1e4
        a = _series(run, "test_acc")
        seed_acc.append(np.interp(grid, t, a, left=a[0], right=a[-1]))
    arr = np.stack(seed_acc, axis=0)
    ax.plot(grid, arr.mean(axis=0), label="HeteRo-Select", **STYLE["HeteRo-Select"])
    if len(runs) > 1:
        ax.fill_between(grid, arr.mean(axis=0) - arr.std(axis=0),
                        arr.mean(axis=0) + arr.std(axis=0),
                        color=STYLE["HeteRo-Select"]["color"], alpha=0.18)
    ax.axhline(0.90, color="gray", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(0, 0.8); ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Time ($\times 10^4$ s)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(r"(a) MNIST / LR  ($\psi=0.4$)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="0.6")

    # ---- (b) TIN completion vs psi @ 0.30 ----
    ax = axes[1]
    base = NONIID_BASELINES["tinyimagenet"]
    bx = np.asarray(base["psi"], dtype=float)
    for name in ("FedAvg", "OptRate", "FlexCom", "AdaSample", "FedCG"):
        col, mk = NIID_MARK[name]
        ax.plot(bx, base[name], label=name, lw=1.7, color=col, marker=mk, markersize=7)
    # HeteRo-Select completion times at 0.30 for each psi we have
    hs_pts = []
    for psi_str, psi_val in [("0p0", 0.0), ("40p0", 40.0), ("80p0", 80.0), ("120p0", 120.0)]:
        runs = _collect_seeds(exp, "tinyimagenet", psi_str)
        if not runs:
            for d in [niid, ablv]:
                runs = _collect_seeds(d, "tinyimagenet", psi_str)
                if runs: break
        if not runs:
            continue
        ct = _hetero_t2t(runs, 0.30)
        if ct is None:
            continue
        hs_pts.append((psi_val, ct / 1e4))
    if hs_pts:
        hs_pts.sort()
        hx = np.asarray([p for p, _ in hs_pts]); hy = np.asarray([t for _, t in hs_pts])
        col, mk = NIID_MARK["HeteRo-Select"]
        ax.plot(hx, hy, label="HeteRo-Select", lw=2.5, color=col, marker=mk, markersize=12)
    ax.set_xlabel(r"Non-IID level $\psi$ (missing classes)")
    ax.set_ylabel(r"Completion time ($\times 10^4$ s)")
    ax.set_title("(b) Tiny-ImageNet completion to 30 %")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="0.6")

    fig.tight_layout()
    fig.savefig(out / "fig2_mnist_tin.png"); fig.savefig(out / "fig2_mnist_tin.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 — CIFAR-10 / CIFAR-100 acc-vs-round, mean ±1σ
# ---------------------------------------------------------------------------

def fig3_c10_c100_bands(exp: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, DOUBLE_H))
    for ax, ds, psi, target, title in [
        (axes[0], "cifar10",  "0p4", 0.70, r"(a) CIFAR-10 / AlexNet  ($\psi=0.4$)"),
        (axes[1], "cifar100", "40",  0.54, r"(b) CIFAR-100 / ResNet-9  ($\psi=40$)"),
    ]:
        _baseline_curves(ax, ROUND_BASELINES[ds], "rounds")
        runs = _collect_seeds(exp, ds, psi)
        rounds, mu_a, sd_a = _mean_curve(runs, "test_acc")
        ax.plot(rounds, mu_a, label="HeteRo-Select", **STYLE["HeteRo-Select"])
        if len(runs) > 1:
            ax.fill_between(rounds, mu_a - sd_a, mu_a + sd_a,
                            color=STYLE["HeteRo-Select"]["color"], alpha=0.18)
        ax.axhline(target, color="gray", lw=0.9, ls="--", alpha=0.7)
        ax.set_xlim(0, 100); ax.set_xlabel("Communication round")
        ax.set_ylabel("Test accuracy"); ax.set_title(title)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="0.6")
    axes[0].set_ylim(0.0, 0.80); axes[1].set_ylim(0.0, 0.60)
    fig.tight_layout()
    fig.savefig(out / "fig3_c10_c100_bands.png"); fig.savefig(out / "fig3_c10_c100_bands.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 — CIFAR-10 ablations: (a) uniform vs adaptive, (b) mu sensitivity
# ---------------------------------------------------------------------------

def fig4_cifar10_ablations(exp: Path, abl: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, DOUBLE_H))

    ax = axes[0]
    adp = _load(next(exp.glob("cifar10_psi0p4_mu0p1_seed42_adaptive.json")))
    uni_path = next(abl.glob("cifar10_psi0p4_mu0p1_seed42_uniform.json"), None)
    rounds = _series(adp, "round")
    acc_adp = _series(adp, "test_acc")
    ax.plot(rounds, acc_adp,
            label="Adaptive (score-proportional)", color="#ff7f0e", lw=2.4)
    if uni_path is not None:
        uni = _load(uni_path)
        ax.plot(_series(uni, "round"), _series(uni, "test_acc"),
                label=r"Uniform ($\theta_k=\theta_t$)",
                color="#1f4ea1", lw=2.0, ls="--")
    ax.axhline(0.70, color="gray", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(0, 100); ax.set_ylim(0, 0.80)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy")
    ax.set_title(r"(a) Score-proportional vs. uniform compression")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="0.6")
    # Inset: late-round zoom (mid-right of axes), legend in lower-right.
    if uni_path is not None:
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        axins = ax.inset_axes([0.55, 0.35, 0.40, 0.42])  # [x0, y0, w, h]
        ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                  edgecolor="0.6")
        axins.plot(rounds, acc_adp, color="#ff7f0e", lw=2.0)
        axins.plot(_series(uni, "round"), _series(uni, "test_acc"),
                   color="#1f4ea1", lw=1.7, ls="--")
        axins.set_xlim(70, 100); axins.set_ylim(0.66, 0.75)
        axins.set_xticks([70, 85, 100]); axins.set_yticks([0.68, 0.72])
        axins.tick_params(labelsize=8)
        axins.grid(True, alpha=0.3, linestyle=":")
        mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5", lw=0.6)

    ax = axes[1]
    mu_colors = {"0p0": "#1a1a1a", "0p01": "#d62728", "0p1": "#ff7f0e", "0p5": "#9c27b0"}
    mu_styles = {"0p0": "--", "0p01": "-.", "0p1": "-", "0p5": ":"}
    mu_label = {"0p0": r"$\mu=0$", "0p01": r"$\mu=0.01$",
                "0p1": r"$\mu=0.1$ (primary)", "0p5": r"$\mu=0.5$"}
    for mu in ("0p0", "0p01", "0p1", "0p5"):
        candidates = list(abl.glob(f"cifar10_psi0p4_mu{mu}_seed42_adaptive.json"))
        if not candidates and mu == "0p1":
            candidates = list(exp.glob("cifar10_psi0p4_mu0p1_seed42_adaptive.json"))
        if not candidates:
            continue
        run = _load(candidates[0])
        ax.plot(_series(run, "round"), _series(run, "test_acc"),
                label=mu_label[mu], color=mu_colors[mu], lw=2.0,
                ls=mu_styles[mu])
    ax.axhline(0.70, color="gray", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(0, 100); ax.set_ylim(0, 0.80)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy")
    ax.set_title(r"(b) FedProx $\mu$ sensitivity")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="0.6")

    fig.tight_layout()
    fig.savefig(out / "fig4_cifar10_ablations.png"); fig.savefig(out / "fig4_cifar10_ablations.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5 — TIN: (a) acc-vs-round (ψ=80), (b) completion vs ψ @ 0.37
# ---------------------------------------------------------------------------

def fig5_tinyimagenet(exp: Path, niid: Path, long: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, DOUBLE_H))

    ax = axes[0]
    _baseline_curves(ax, ROUND_BASELINES["tinyimagenet"], "rounds")
    runs = _collect_seeds(exp, "tinyimagenet", "80p0")
    rounds, mu_a, sd_a = _mean_curve(runs, "test_acc")
    ax.plot(rounds, mu_a, label=r"HeteRo-Select (ls=50)",
            **STYLE["HeteRo-Select"])
    if len(runs) > 1:
        ax.fill_between(rounds, mu_a - sd_a, mu_a + sd_a,
                        color=STYLE["HeteRo-Select"]["color"], alpha=0.18)
    long_p = next(long.glob("tinyimagenet_psi80p0_mu0p1_seed42_adaptive_long.json"), None)
    if long_p:
        run = _load(long_p)
        r = _series(run, "round"); a = _series(run, "test_acc")
        ax.plot(r[r <= 100], a[r <= 100], label=r"HeteRo-Select (ls=100, projected)",
                color="#ff7f0e", lw=2.0, ls="--", alpha=0.85)
    ax.axhline(0.37, color="gray", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(0, 100); ax.set_ylim(0.0, 0.45)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy")
    ax.set_title(r"(a) ResNet-18 / Tiny-ImageNet  ($\psi=80$)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="0.6")

    ax = axes[1]
    base = NONIID_BASELINES["tinyimagenet"]
    bx = np.asarray(base["psi"], dtype=float)
    for name in ("FedAvg", "OptRate", "FlexCom", "AdaSample", "FedCG"):
        col, mk = NIID_MARK[name]
        ax.plot(bx, base[name], label=name, lw=1.7, color=col, marker=mk, markersize=7)
    # 30% (filled) and 37% (open) completion times — paper Fig 5(b) overlays both.
    for target, marker_kw, label in [
        (0.30, dict(markersize=12), "HeteRo-Select @ 30 %"),
        (0.37, dict(markersize=14, markerfacecolor="white", markeredgewidth=2),
         "HeteRo-Select @ 37 %"),
    ]:
        pts = []
        for psi_str, psi_val in [("0p0", 0.0), ("40p0", 40.0), ("80p0", 80.0), ("120p0", 120.0)]:
            runs = _collect_seeds(exp, "tinyimagenet", psi_str)
            if not runs:
                runs = _collect_seeds(niid, "tinyimagenet", psi_str)
            ct = _hetero_t2t(runs, target) if runs else None
            # For the 37% line, prefer the long-train (ls=100) sweep if present.
            if target == 0.37:
                lp = next(long.glob(f"tinyimagenet_psi{psi_str}_mu0p1_seed42_adaptive_long.json"),
                          None)
                if lp is not None:
                    ct_long = _hetero_t2t([lp], 0.37)
                    if ct_long is not None:
                        ct = ct_long
            if ct is not None:
                pts.append((psi_val, ct / 1e4))
        if pts:
            pts.sort()
            hx = np.asarray([p for p, _ in pts])
            hy = np.asarray([t for _, t in pts])
            col, mk = NIID_MARK["HeteRo-Select"]
            ax.plot(hx, hy, label=label, lw=2.5, color=col, marker=mk, **marker_kw)
    ax.set_xlabel(r"Non-IID level $\psi$ (missing classes)")
    ax.set_ylabel(r"Completion time ($\times 10^4$ s)")
    ax.set_title("(b) Tiny-ImageNet non-IID sweep")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="0.6")

    fig.tight_layout()
    fig.savefig(out / "fig5_tinyimagenet.png"); fig.savefig(out / "fig5_tinyimagenet.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6 — (a) score components, (b) per-client selection counts
# ---------------------------------------------------------------------------

def fig6_score_components(exp: Path, out: Path) -> None:
    run = _load(next(exp.glob("cifar10_psi0p4_mu0p1_seed42_adaptive.json")))
    rounds = _series(run, "round")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, DOUBLE_H))

    ax = axes[0]
    ax.plot(rounds, _series(run, "score_V_mean"),  label=r"$V'_k$ loss",
            color="#1f77b4", lw=1.8)
    ax.plot(rounds, _series(run, "score_D_mean"),  label=r"$D_k$ diversity",
            color="#d62728", lw=1.8)
    ax.plot(rounds, _series(run, "score_F_mean"),  label=r"$F'_k$ fairness",
            color="#2ca02c", lw=1.8)
    ax.plot(rounds, _series(run, "score_St_mean"), label=r"$St'_k$ staleness",
            color="#9c27b0", lw=1.8)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Mean score component")
    ax.set_xlim(0, 100); ax.set_ylim(-0.05, 1.05)
    ax.set_title(r"(a) Score components on CIFAR-10")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.6", ncol=2)

    ax = axes[1]
    counts: Counter = Counter()
    for entry in run["rounds"]:
        for k in entry["selected"]:
            counts[k] += 1
    K = 100
    arr = np.asarray([counts.get(k, 0) for k in range(K)])
    sorted_arr = np.sort(arr)[::-1]
    ax.bar(np.arange(K), sorted_arr, color="#ff7f0e",
           edgecolor="#cc6510", linewidth=0.5)
    ax.axhline(10, color="gray", lw=1.1, ls="--", alpha=0.85,
               label="Uniform expectation = 10")
    ax.set_xlabel("Client rank (sorted by selection count)")
    ax.set_ylabel("Selection count over 100 rounds")
    ax.set_title(rf"(b) Selection histogram (min={arr.min()}, max={arr.max()}, "
                 rf"mean={arr.mean():.1f})")
    ax.set_xlim(-1, K)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.6")
    fig.tight_layout()
    fig.savefig(out / "fig6_score_components.png"); fig.savefig(out / "fig6_score_components.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 7 — Inverted-coupling stress: acc vs cumulative uplink traffic
# ---------------------------------------------------------------------------

def fig7_stress_test(exp: Path, abl: Path, out: Path) -> None:
    base_path = next(exp.glob("cifar10_psi0p4_mu0p1_seed42_adaptive.json"), None)
    stress_path = next(abl.glob("cifar10_psi0p4_mu0p1_seed42_stress.json"), None)
    if base_path is None or stress_path is None:
        return
    base = _load(base_path); stress = _load(stress_path)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, DOUBLE_H))

    # (a) Accuracy vs. round.
    ax = axes[0]
    ax.plot(_series(base,   "round"), _series(base,   "test_acc"),
            label="Normal coupling",   color="#ff7f0e", lw=2.4)
    ax.plot(_series(stress, "round"), _series(stress, "test_acc"),
            label="Inverted coupling", color="#1f4ea1", lw=2.2, ls="--")
    ax.axhline(0.70, color="gray", lw=0.9, ls=":", alpha=0.8)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")
    ax.set_xlim(0, 100); ax.set_ylim(0, 0.80)
    ax.set_title("(a) Accuracy vs. round")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="0.6")

    # (b) Accuracy vs. cumulative uplink traffic.
    ax = axes[1]
    ax.plot(_series(base,   "cum_traffic_mb"), _series(base,   "test_acc"),
            label="Normal coupling",   color="#ff7f0e", lw=2.4)
    ax.plot(_series(stress, "cum_traffic_mb"), _series(stress, "test_acc"),
            label="Inverted coupling", color="#1f4ea1", lw=2.2, ls="--")
    ax.axhline(0.70, color="gray", lw=0.9, ls=":", alpha=0.8)
    ax.set_xlabel("Cumulative uplink traffic (MB)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 0.80)
    ax.set_title("(b) Accuracy vs. cumulative traffic")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="0.6")

    fig.tight_layout()
    fig.savefig(out / "fig7_stress_test.png"); fig.savefig(out / "fig7_stress_test.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp",   type=Path, default=Path("results/experiment"))
    p.add_argument("--abl",   type=Path, default=Path("results/ablation"))
    p.add_argument("--ablv",  type=Path, default=Path("results/ablation_v2"))
    p.add_argument("--niid",  type=Path, default=Path("results/noniid"))
    p.add_argument("--long",  type=Path, default=Path("results/long"))
    p.add_argument("--out",   type=Path, default=Path("figs/paper"))
    return p.parse_args()


def main() -> None:
    args = _parse()
    args.out.mkdir(parents=True, exist_ok=True)
    fig1_cifar10_bars(args.exp, args.out)
    fig2_mnist_tin(args.exp, args.ablv, args.niid, args.out)
    fig3_c10_c100_bands(args.exp, args.out)
    fig4_cifar10_ablations(args.exp, args.abl, args.out)
    fig5_tinyimagenet(args.exp, args.niid, args.long, args.out)
    fig6_score_components(args.exp, args.out)
    fig7_stress_test(args.exp, args.abl, args.out)
    print(f"Figures written to {args.out.resolve()}")
    for f in sorted(args.out.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
