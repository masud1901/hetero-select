#!/usr/bin/env python3
"""Render InfoCom-2023 style 4-panel comparison figures.

Three figures are emitted:
    fig_acc_vs_round.png         — Test Accuracy vs. communication round
    fig_acc_vs_time.png          — Test Accuracy vs. simulated time (x10^4 s)
    fig_completion_vs_noniid.png — Completion Time vs. Non-IID Level

Each figure has four panels: (a) MNIST/LR, (b) CIFAR-10/AlexNet,
(c) CIFAR-100/ResNet-9, (d) TinyImageNet/ResNet-18.

Baselines (FedAvg, OptRate, FlexCom, AdaSample, FedCG) are taken
verbatim from the InfoCom-2023 tables checked in under
figures-combined/. HeteRo-Select is overlaid from the JSON logs
in results/experiment and results/ablation, averaged over seeds
where multiple seeds are available.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
})

# Curve style (color, linestyle, marker) — picked to match the
# InfoCom-2023 paper figures (and HeteRo-Select highlighted in orange).
STYLE = {
    "FedAvg":        dict(color="#1a1a1a", ls=(0, (3, 1, 1, 1)), marker=None, lw=1.7),
    "OptRate":       dict(color="#d62728", ls=(0, (5, 2)),      marker=None, lw=1.7),
    "FlexCom":       dict(color="#1f4ea1", ls=(0, (3, 1, 1, 1)), marker=None, lw=1.7),
    "AdaSample":     dict(color="#2ca02c", ls=(0, (1, 1)),      marker=None, lw=1.7),
    "FedCG":         dict(color="#9c27b0", ls="-",              marker=None, lw=1.9),
    "HeteRo-Select": dict(color="#ff7f0e", ls="-",              marker="o", lw=2.5,
                          markersize=4, markevery=10),
}

NONIID_MARKER = {
    "FedAvg":        dict(color="#1a1a1a", marker="s"),
    "OptRate":       dict(color="#d62728", marker="o"),
    "FlexCom":       dict(color="#1f4ea1", marker="^"),
    "AdaSample":     dict(color="#2ca02c", marker="v"),
    "FedCG":         dict(color="#9c27b0", marker="D"),
    "HeteRo-Select": dict(color="#ff7f0e", marker="*"),
}

DATASETS = ["mnist", "cifar10", "cifar100", "tinyimagenet"]
LABEL = {
    "mnist":        "(a) LR over MNIST",
    "cifar10":      "(b) AlexNet over CIFAR-10",
    "cifar100":     "(c) ResNet9 over CIFAR-100",
    "tinyimagenet": "(d) ResNet18 over Tiny-ImageNet",
}
TARGETS = {"mnist": 0.90, "cifar10": 0.74, "cifar100": 0.54, "tinyimagenet": 0.37}
ACC_YLIM = {"mnist": (0.0, 1.0), "cifar10": (0.0, 0.8),
            "cifar100": (0.0, 0.6), "tinyimagenet": (0.0, 0.45)}
TIME_XMAX_1E4 = {"mnist": 0.8, "cifar10": 2.0, "cifar100": 4.0, "tinyimagenet": 10.0}


# ---------------------------------------------------------------------------
# Baseline data — verbatim from figures-combined/.../*.md
# ---------------------------------------------------------------------------

ROUND_BASELINES: Dict[str, Dict] = {
    "mnist": dict(
        rounds=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
                70, 75, 80, 85, 90, 95, 100],
        FedAvg=[0.100, 0.440, 0.650, 0.720, 0.780, 0.800, 0.820, 0.840,
                0.850, 0.855, 0.860, 0.865, 0.870, 0.875, 0.880, 0.885,
                0.890, 0.892, 0.895, 0.898, 0.900],
        OptRate=[0.100, 0.500, 0.680, 0.760, 0.800, 0.820, 0.840, 0.850,
                 0.860, 0.865, 0.870, 0.875, 0.880, 0.885, 0.890, 0.895,
                 0.900, 0.902, 0.905, 0.908, 0.910],
        FlexCom=[0.100, 0.520, 0.680, 0.780, 0.810, 0.830, 0.850, 0.860,
                 0.870, 0.875, 0.880, 0.885, 0.890, 0.895, 0.900, 0.905,
                 0.910, 0.912, 0.915, 0.918, 0.920],
        AdaSample=[0.100, 0.480, 0.650, 0.750, 0.790, 0.810, 0.830, 0.850,
                   0.860, 0.865, 0.870, 0.875, 0.880, 0.885, 0.890, 0.895,
                   0.900, 0.903, 0.905, 0.908, 0.910],
        FedCG=[0.100, 0.450, 0.650, 0.780, 0.820, 0.840, 0.860, 0.870,
               0.880, 0.885, 0.890, 0.895, 0.900, 0.905, 0.910, 0.912,
               0.915, 0.917, 0.920, 0.922, 0.925],
    ),
    "cifar10": dict(
        rounds=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
                70, 75, 80, 85, 90, 95, 100],
        FedAvg=[0.100, 0.160, 0.350, 0.420, 0.500, 0.550, 0.580, 0.600,
                0.620, 0.630, 0.640, 0.650, 0.670, 0.680, 0.700, 0.710,
                0.720, 0.725, 0.730, 0.735, 0.740],
        OptRate=[0.100, 0.180, 0.350, 0.400, 0.450, 0.490, 0.530, 0.560,
                 0.590, 0.620, 0.650, 0.670, 0.690, 0.700, 0.710, 0.720,
                 0.730, 0.735, 0.738, 0.739, 0.740],
        FlexCom=[0.100, 0.220, 0.350, 0.390, 0.420, 0.460, 0.500, 0.540,
                 0.570, 0.600, 0.620, 0.640, 0.660, 0.680, 0.700, 0.710,
                 0.720, 0.730, 0.735, 0.738, 0.740],
        AdaSample=[0.100, 0.200, 0.350, 0.380, 0.410, 0.450, 0.490, 0.520,
                   0.550, 0.580, 0.610, 0.630, 0.650, 0.670, 0.690, 0.700,
                   0.710, 0.720, 0.730, 0.735, 0.740],
        FedCG=[0.100, 0.220, 0.380, 0.480, 0.550, 0.600, 0.640, 0.670,
               0.690, 0.700, 0.710, 0.720, 0.730, 0.740, 0.742, 0.744,
               0.746, 0.748, 0.749, 0.750, 0.750],
    ),
    "cifar100": dict(
        rounds=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
                70, 75, 80, 85, 90, 95, 100],
        FedAvg=[0.010, 0.080, 0.150, 0.220, 0.280, 0.330, 0.380, 0.420,
                0.450, 0.470, 0.490, 0.500, 0.510, 0.520, 0.525, 0.530,
                0.532, 0.535, 0.538, 0.539, 0.540],
        OptRate=[0.010, 0.100, 0.210, 0.290, 0.360, 0.410, 0.450, 0.480,
                 0.500, 0.510, 0.520, 0.525, 0.530, 0.533, 0.536, 0.538,
                 0.539, 0.540, 0.541, 0.542, 0.542],
        FlexCom=[0.010, 0.140, 0.270, 0.370, 0.450, 0.480, 0.510, 0.520,
                 0.530, 0.535, 0.540, 0.540, 0.541, 0.541, 0.542, 0.542,
                 0.542, 0.542, 0.542, 0.543, 0.543],
        AdaSample=[0.010, 0.120, 0.240, 0.330, 0.410, 0.450, 0.480, 0.500,
                   0.515, 0.520, 0.525, 0.530, 0.532, 0.534, 0.536, 0.538,
                   0.539, 0.540, 0.541, 0.542, 0.542],
        FedCG=[0.010, 0.150, 0.260, 0.350, 0.420, 0.460, 0.490, 0.510,
               0.525, 0.535, 0.540, 0.541, 0.542, 0.542, 0.543, 0.543,
               0.543, 0.543, 0.543, 0.543, 0.543],
    ),
    "tinyimagenet": dict(
        rounds=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
                70, 75, 80, 85, 90, 95, 100],
        FedAvg=[0.000, 0.040, 0.080, 0.120, 0.150, 0.180, 0.210, 0.240,
                0.270, 0.290, 0.310, 0.325, 0.340, 0.350, 0.360, 0.365,
                0.370, 0.375, 0.380, 0.385, 0.390],
        OptRate=[0.000, 0.050, 0.100, 0.150, 0.190, 0.230, 0.260, 0.290,
                 0.310, 0.330, 0.340, 0.350, 0.360, 0.365, 0.370, 0.375,
                 0.380, 0.383, 0.385, 0.388, 0.390],
        FlexCom=[0.000, 0.060, 0.130, 0.190, 0.250, 0.290, 0.320, 0.340,
                 0.350, 0.360, 0.370, 0.375, 0.380, 0.382, 0.384, 0.386,
                 0.387, 0.388, 0.389, 0.390, 0.390],
        AdaSample=[0.000, 0.070, 0.140, 0.200, 0.250, 0.290, 0.310, 0.330,
                   0.340, 0.350, 0.360, 0.370, 0.380, 0.382, 0.385, 0.386,
                   0.387, 0.388, 0.389, 0.390, 0.390],
        FedCG=[0.000, 0.080, 0.150, 0.230, 0.300, 0.340, 0.360, 0.370,
               0.380, 0.382, 0.383, 0.384, 0.385, 0.386, 0.387, 0.388,
               0.389, 0.389, 0.390, 0.390, 0.390],
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
    "cifar10": dict(
        t1e4=[0.0, 0.2, 0.5, 1.0, 1.5, 2.0],
        FedAvg=[0.100, 0.400, 0.550, 0.680, 0.740, 0.750],
        OptRate=[0.100, 0.450, 0.660, 0.740, 0.750, 0.752],
        FlexCom=[0.100, 0.500, 0.700, 0.750, 0.752, 0.753],
        AdaSample=[0.100, 0.450, 0.640, 0.730, 0.750, 0.752],
        FedCG=[0.100, 0.710, 0.750, 0.752, 0.753, 0.753],
    ),
    "cifar100": dict(
        t1e4=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        FedAvg=[0.010, 0.150, 0.280, 0.380, 0.460, 0.510, 0.540],
        OptRate=[0.010, 0.210, 0.360, 0.470, 0.510, 0.540, 0.542],
        FlexCom=[0.010, 0.270, 0.450, 0.510, 0.540, 0.542, 0.543],
        AdaSample=[0.010, 0.240, 0.410, 0.480, 0.520, 0.540, 0.542],
        FedCG=[0.010, 0.480, 0.540, 0.542, 0.542, 0.543, 0.543],
    ),
    "tinyimagenet": dict(
        t1e4=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0],
        FedAvg=[0.000, 0.080, 0.150, 0.210, 0.270, 0.340, 0.370, 0.390],
        OptRate=[0.000, 0.100, 0.190, 0.260, 0.320, 0.370, 0.390, 0.390],
        FlexCom=[0.000, 0.130, 0.250, 0.320, 0.360, 0.380, 0.390, 0.390],
        AdaSample=[0.000, 0.140, 0.250, 0.310, 0.350, 0.380, 0.390, 0.390],
        FedCG=[0.000, 0.270, 0.360, 0.380, 0.382, 0.383, 0.390, 0.390],
    ),
}

NONIID_BASELINES: Dict[str, Dict] = {
    "mnist": dict(
        psi=[0.0, 0.2, 0.4, 0.6, 0.8],
        FedAvg=[0.775, 0.812, 0.814, 0.991, 1.205],
        OptRate=[0.454, 0.461, 0.472, 0.551, 0.748],
        FlexCom=[0.365, 0.388, 0.395, 0.443, 0.662],
        AdaSample=[0.482, 0.460, 0.491, 0.562, 0.650],
        FedCG=[0.206, 0.228, 0.222, 0.236, 0.320],
    ),
    "cifar10": dict(
        psi=[0.0, 0.2, 0.4, 0.6, 0.8],
        FedAvg=[0.712, 0.763, 0.864, 1.085, 1.911],
        OptRate=[0.441, 0.485, 0.510, 0.682, 1.554],
        FlexCom=[0.301, 0.348, 0.361, 0.425, 1.142],
        AdaSample=[0.392, 0.401, 0.412, 0.558, 0.781],
        FedCG=[0.185, 0.188, 0.201, 0.252, 0.410],
    ),
    "cifar100": dict(
        psi=[0, 20, 40, 60],
        FedAvg=[2.65, 3.46, 3.89, 4.81],
        OptRate=[1.78, 2.14, 2.26, 3.01],
        FlexCom=[1.20, 1.36, 1.52, 2.25],
        AdaSample=[1.46, 1.77, 1.77, 2.10],
        FedCG=[0.69, 0.81, 0.73, 1.01],
    ),
    "tinyimagenet": dict(
        psi=[0, 40, 80, 120],
        FedAvg=[5.13, 5.66, 6.17, 8.59],
        OptRate=[4.13, 4.54, 4.99, 7.28],
        FlexCom=[3.33, 3.49, 4.24, 6.17],
        AdaSample=[2.97, 3.05, 3.43, 4.20],
        FedCG=[1.91, 1.98, 2.14, 2.81],
    ),
}

NONIID_XLABEL = {
    "mnist":        "Non-IID Level",
    "cifar10":      "Non-IID Level",
    "cifar100":     "Non-IID Level (missing classes)",
    "tinyimagenet": "Non-IID Level (missing classes)",
}


# ---------------------------------------------------------------------------
# JSON-log handling
# ---------------------------------------------------------------------------

FNAME_RE = re.compile(
    r"^(?P<ds>[a-z0-9]+)_psi(?P<psi>[0-9p]+)_mu(?P<mu>[0-9p]+)_seed(?P<seed>\d+)_(?P<variant>adaptive|uniform|stress)\.json$"
)


def _psi_value(token: str) -> float:
    """'0p4' -> 0.4 ; '40' -> 40.0"""
    return float(token.replace("p", "."))


def _load(p: Path) -> dict:
    with p.open("r") as f:
        return json.load(f)


def _series(run: dict, key: str) -> np.ndarray:
    return np.asarray([r[key] for r in run["rounds"]], dtype=float)


def _discover(paths: Iterable[Path]) -> Dict[Tuple[str, float], List[Path]]:
    """Group adaptive logs (mu=0.1, no `_tag` suffix) by (dataset, psi)."""
    by: Dict[Tuple[str, float], List[Path]] = defaultdict(list)
    for root in paths:
        for p in sorted(root.glob("*.json")):
            m = FNAME_RE.match(p.name)
            if not m or m.group("variant") != "adaptive":
                continue
            if m.group("mu") != "0p1":
                continue
            by[(m.group("ds"), _psi_value(m.group("psi")))].append(p)
    return by


def _discover_long(path: Path) -> Dict[str, Path]:
    """Map dataset -> long-train JSON (rounds=150, ls=100)."""
    out: Dict[str, Path] = {}
    if not path.is_dir():
        return out
    for p in sorted(path.glob("*_long.json")):
        m = re.match(r"^([a-z0-9]+)_psi[0-9p]+_mu[0-9p]+_seed\d+_adaptive_long\.json$",
                     p.name)
        if m:
            out[m.group(1)] = p
    return out


def _hetero_round_curve(runs: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    rounds = None
    acc_seeds = []
    for p in runs:
        run = _load(p)
        if rounds is None:
            rounds = _series(run, "round")
        acc_seeds.append(_series(run, "test_acc"))
    arr = np.stack(acc_seeds, axis=0)
    return rounds, arr.mean(axis=0)


def _hetero_time_curve(runs: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Sample mean accuracy onto a common time grid (union of seeds)."""
    time_seeds, acc_seeds = [], []
    for p in runs:
        run = _load(p)
        time_seeds.append(_series(run, "cum_time_s") / 1e4)
        acc_seeds.append(_series(run, "test_acc"))
    grid = np.linspace(0.0, max(t[-1] for t in time_seeds), 400)
    interp = np.stack(
        [np.interp(grid, t, a, left=a[0], right=a[-1])
         for t, a in zip(time_seeds, acc_seeds)],
        axis=0,
    )
    return grid, interp.mean(axis=0)


def _hetero_completion_time(runs: List[Path], target: float) -> Optional[float]:
    """Mean time-to-target (x10^4 s) computed at the *figure* target across
    seeds. Returns None if no seed hits ``target`` within its training budget.
    """
    vals = []
    for p in runs:
        run = _load(p)
        acc = _series(run, "test_acc")
        tt = _series(run, "cum_time_s")
        ok = np.where(acc >= target)[0]
        if ok.size:
            vals.append(float(tt[ok[0]]) / 1e4)
    if not vals:
        return None
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _smooth(x: np.ndarray, y: np.ndarray, n: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 4:
        return x, y
    cs = CubicSpline(x, y)
    xg = np.linspace(x[0], x[-1], n)
    return xg, cs(xg)


def _draw_baselines_curve(ax, base: dict, x_key: str) -> None:
    x = np.asarray(base[x_key], dtype=float)
    for name in ("FedAvg", "OptRate", "FlexCom", "AdaSample", "FedCG"):
        y = np.asarray(base[name], dtype=float)
        xs, ys = _smooth(x, y)
        ax.plot(xs, ys, label=name, **STYLE[name])


def _draw_hetero_round(ax, runs: List[Path]) -> None:
    rounds, mean_acc = _hetero_round_curve(runs)
    ax.plot(rounds, mean_acc, label="HeteRo-Select", **STYLE["HeteRo-Select"])


def _draw_hetero_time(ax, runs: List[Path], x_max: float) -> None:
    grid, mean_acc = _hetero_time_curve(runs)
    ax.plot(grid, mean_acc, label="HeteRo-Select", **STYLE["HeteRo-Select"])
    if grid[-1] < x_max:
        ax.axvline(grid[-1], color=STYLE["HeteRo-Select"]["color"],
                   lw=0.8, ls=":", alpha=0.55)


LONG_STYLE = dict(color="#ff7f0e", ls="--", marker="s", lw=2.0,
                  markersize=4, markevery=15, alpha=0.85)


def fig_acc_vs_round(results: Dict[Tuple[str, float], List[Path]],
                     long_runs: Dict[str, Path],
                     out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    for ax, ds in zip(axes.ravel(), DATASETS):
        _draw_baselines_curve(ax, ROUND_BASELINES[ds], "rounds")
        candidates = [(p, r) for (d, p), r in results.items() if d == ds]
        if candidates:
            wanted = {"mnist": 0.4, "cifar10": 0.4, "cifar100": 40.0,
                      "tinyimagenet": 80.0}[ds]
            runs = min(candidates, key=lambda kr: abs(kr[0] - wanted))[1]
            _draw_hetero_round(ax, runs)
        if ds in long_runs:
            r_long = _series(_load(long_runs[ds]), "round")
            a_long = _series(_load(long_runs[ds]), "test_acc")
            ax.plot(r_long[r_long <= 100], a_long[r_long <= 100],
                    label="HeteRo-Select (ls=100)", **LONG_STYLE)
        ax.axhline(TARGETS[ds], color="gray", lw=0.9, ls="--", alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_ylim(*ACC_YLIM[ds])
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(LABEL[ds])
        ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                  edgecolor="0.6")
    fig.tight_layout()
    fig.savefig(out / "fig_acc_vs_round.png")
    plt.close(fig)


def fig_acc_vs_time(results: Dict[Tuple[str, float], List[Path]],
                    long_runs: Dict[str, Path],
                    out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    for ax, ds in zip(axes.ravel(), DATASETS):
        _draw_baselines_curve(ax, TIME_BASELINES[ds], "t1e4")
        candidates = [(p, r) for (d, p), r in results.items() if d == ds]
        if candidates:
            wanted = {"mnist": 0.4, "cifar10": 0.4, "cifar100": 40.0,
                      "tinyimagenet": 80.0}[ds]
            runs = min(candidates, key=lambda kr: abs(kr[0] - wanted))[1]
            _draw_hetero_time(ax, runs, TIME_XMAX_1E4[ds])
        if ds in long_runs:
            t_long = _series(_load(long_runs[ds]), "cum_time_s") / 1e4
            a_long = _series(_load(long_runs[ds]), "test_acc")
            mask = t_long <= TIME_XMAX_1E4[ds]
            ax.plot(t_long[mask], a_long[mask],
                    label="HeteRo-Select (ls=100)", **LONG_STYLE)
        ax.axhline(TARGETS[ds], color="gray", lw=0.9, ls="--", alpha=0.7)
        ax.set_xlim(0, TIME_XMAX_1E4[ds])
        ax.set_ylim(*ACC_YLIM[ds])
        ax.set_xlabel(r"Time ($\times 10^4$ s)")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(LABEL[ds])
        ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                  edgecolor="0.6")
    fig.tight_layout()
    fig.savefig(out / "fig_acc_vs_time.png")
    plt.close(fig)


def fig_completion_vs_noniid(results: Dict[Tuple[str, float], List[Path]],
                             long_runs: Dict[str, Path],
                             out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    for ax, ds in zip(axes.ravel(), DATASETS):
        base = NONIID_BASELINES[ds]
        x = np.asarray(base["psi"], dtype=float)
        for name in ("FedAvg", "OptRate", "FlexCom", "AdaSample", "FedCG"):
            y = np.asarray(base[name], dtype=float)
            ax.plot(x, y, label=name, lw=1.7,
                    color=NONIID_MARKER[name]["color"],
                    marker=NONIID_MARKER[name]["marker"], markersize=6)
        # HeteRo-Select points at whatever psi values we have JSONs for
        hpts: List[Tuple[float, float]] = []
        for (d, psi), runs in results.items():
            if d != ds:
                continue
            ct = _hetero_completion_time(runs, TARGETS[ds])
            if ct is None:
                continue
            hpts.append((psi, ct))
        if hpts:
            hpts.sort()
            hx = np.asarray([p for p, _ in hpts])
            hy = np.asarray([t for _, t in hpts])
            ax.plot(hx, hy, label="HeteRo-Select", lw=2.5,
                    color=NONIID_MARKER["HeteRo-Select"]["color"],
                    marker=NONIID_MARKER["HeteRo-Select"]["marker"],
                    markersize=11)
        # If a long-train (ls=100) run beats the paper target where the
        # baseline-budget run did not, plot it as an open star.
        if ds in long_runs:
            ct = _hetero_completion_time([long_runs[ds]], TARGETS[ds])
            cfg = _load(long_runs[ds])["config"]
            psi_long = float(cfg["psi"])
            already = any(abs(p - psi_long) < 1e-6 for p, _ in hpts)
            if ct is not None and not already:
                ax.plot([psi_long], [ct], lw=0,
                        color=NONIID_MARKER["HeteRo-Select"]["color"],
                        marker="*", markersize=15,
                        markerfacecolor="white", markeredgewidth=2,
                        label="HeteRo-Select (ls=100)")
        ax.set_xlabel(NONIID_XLABEL[ds])
        ax.set_ylabel(r"Completion Time ($\times 10^4$ s)")
        ax.set_title(LABEL[ds])
        ax.legend(loc="upper left", frameon=True, framealpha=0.92,
                  edgecolor="0.6")
    fig.tight_layout()
    fig.savefig(out / "fig_completion_vs_noniid.png")
    plt.close(fig)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", type=Path, default=Path("results/experiment"))
    p.add_argument("--noniid", type=Path, default=Path("results/noniid"))
    p.add_argument("--abl", type=Path, default=Path("results/ablation"),
                   help="Older non-IID variants (cifar10 psi=0.2, 0.6) lived "
                        "in results/ablation; pull them in too.")
    p.add_argument("--long", type=Path, default=Path("results/long"))
    p.add_argument("--out", type=Path, default=Path("figs/infocom"))
    return p.parse_args()


def main() -> None:
    args = _parse()
    args.out.mkdir(parents=True, exist_ok=True)
    results = _discover([args.exp, args.noniid, args.abl])
    long_runs = _discover_long(args.long)
    if not results:
        print(f"No adaptive JSON logs found under {args.exp} / {args.abl}.")
        return
    print("Found HeteRo-Select runs:")
    for (ds, psi), runs in sorted(results.items()):
        print(f"  {ds:14s} psi={psi:<6g}  {len(runs)} seed(s)")
    if long_runs:
        print("Long-train (rounds=150, ls=100) runs:")
        for ds, p in long_runs.items():
            print(f"  {ds:14s} -> {p.name}")
    fig_acc_vs_round(results, long_runs, args.out)
    fig_acc_vs_time(results, long_runs, args.out)
    fig_completion_vs_noniid(results, long_runs, args.out)
    print(f"\nFigures written to {args.out.resolve()}")
    for f in sorted(args.out.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
