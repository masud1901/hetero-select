from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

# All supported experiment variants (paper Table II + protocol ablations).
VARIANT_CHOICES = (
    "adaptive",
    "uniform",
    "stress",
    "no_V",
    "no_D",
    "no_FS",
    "no_newton",
    "static_beta",
    "uniform_lr",
    "uniform_agg",
)


def apply_variant(
    fl: Dict[str, Any],
    variant: str,
    mu: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a copy of *fl* with overrides for the named ablation variant."""
    out = deepcopy(fl)
    if mu is not None:
        out["mu"] = mu

    v = variant or "adaptive"

    if v == "no_V":
        out["no_V"] = True
    elif v == "no_D":
        out["lambda_D"] = 0.0
    elif v == "no_FS":
        out["lambda_F"] = 0.0
        out["lambda_St"] = 0.0
    elif v == "no_newton":
        out["newton_Q"] = 0
    elif v == "static_beta":
        out["static_beta"] = 0.90
    elif v == "uniform_lr":
        out["uniform_lr"] = True
    elif v == "uniform_agg":
        out["uniform_aggregation"] = True

    return out
