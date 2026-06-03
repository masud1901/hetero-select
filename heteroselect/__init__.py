from __future__ import annotations

from .config import (
    DEFAULT_FL_CONFIG,
    ablation_experiments,
    component_ablation_experiments,
    default_experiments,
    load_config,
)
from .variants import VARIANT_CHOICES, apply_variant

__all__ = [
    "DEFAULT_FL_CONFIG",
    "load_config",
    "default_experiments",
    "ablation_experiments",
    "component_ablation_experiments",
    "run_experiment",
    "VARIANT_CHOICES",
    "apply_variant",
]

__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "run_experiment":
        from .trainer import run_experiment as fn
        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
