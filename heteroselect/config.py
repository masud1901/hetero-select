from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict


DEFAULT_FL_CONFIG: Dict[str, Any] = dict(
    num_clients       = 100,
    clients_per_round = 10,
    num_rounds        = 100,
    local_steps       = 50,
    mu                = 0.1,
    bw_min_mbps       = 1.0,
    bw_max_mbps       = 5.0,
    comp_min_s        = 0.10,
    comp_max_s        = 0.50,
    target_acc        = {"mnist": 0.90, "cifar10": 0.70, "cifar100": 0.54, "tinyimagenet": 0.30},
    theta_total       = 0.20,
    theta_floor       = 0.08,
    warmup_rounds     = 1,
    alpha_cos         = 0.4,
    local_lr          = 0.05,
    lr_scale_cap      = 0.15,
    grad_clip         = 2.0,
    batch_size        = 32,
    eval_batches      = 8,
    lambda_V          = 1.0,
    lambda_D          = 0.3,
    lambda_F          = 0.2,
    lambda_St         = 0.2,
    gamma_St          = 0.5,
    tau_0             = 1.0,
    beta_s            = 0.5,
    newton_Q          = 3,
    newton_lambda     = 0.2,
    bn_calib_batches  = 20,
)


def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None) -> Dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_FL_CONFIG)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r") as f:
            overlay = json.load(f)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to read YAML configs. Install with 'pip install pyyaml'."
            ) from exc
        with open(path, "r") as f:
            overlay = yaml.safe_load(f) or {}
    else:
        raise ValueError(f"Unknown config extension '{ext}'. Use .json or .yaml.")
    return _merge(DEFAULT_FL_CONFIG, overlay)


def default_experiments() -> list[dict]:
    return [
        dict(dataset="mnist",        psi=0.4, seed=42),
        dict(dataset="mnist",        psi=0.4, seed=43),
        dict(dataset="mnist",        psi=0.4, seed=44),
        dict(dataset="cifar10",      psi=0.4, seed=42),
        dict(dataset="cifar10",      psi=0.4, seed=43),
        dict(dataset="cifar10",      psi=0.4, seed=44),
        dict(dataset="cifar100",     psi=40,  seed=42),
        dict(dataset="cifar100",     psi=40,  seed=43),
        dict(dataset="cifar100",     psi=40,  seed=44),
        dict(dataset="tinyimagenet", psi=80,  seed=42),
        dict(dataset="tinyimagenet", psi=80,  seed=43),
        dict(dataset="tinyimagenet", psi=80,  seed=44),
    ]


def ablation_experiments() -> list[dict]:
    return [
        dict(dataset="cifar10", psi=0.4, seed=42, variant="uniform"),
        dict(dataset="cifar10", psi=0.4, seed=42, mu=0.0),
        dict(dataset="cifar10", psi=0.4, seed=42, mu=0.01),
        dict(dataset="cifar10", psi=0.4, seed=42, mu=0.5),
        dict(dataset="cifar10", psi=0.2, seed=42),
        dict(dataset="cifar10", psi=0.6, seed=42),
        dict(dataset="cifar10", psi=0.4, seed=42, variant="stress"),
    ]


def component_ablation_experiments() -> list[dict]:
    """Paper Table II: score components and design-choice ablations (seed 42)."""
    base = dict(dataset="cifar10", psi=0.4, seed=42)
    return [
        dict(**base, variant="no_V"),
        dict(**base, variant="no_D"),
        dict(**base, variant="no_FS"),
        dict(**base, variant="no_newton"),
        dict(**base, variant="static_beta"),
        dict(**base, variant="uniform_lr"),
        dict(**base, variant="uniform_agg"),
    ]
