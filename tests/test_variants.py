from __future__ import annotations

from heteroselect.config import (
    DEFAULT_FL_CONFIG,
    ablation_experiments,
    component_ablation_experiments,
    load_config,
)
from heteroselect.variants import VARIANT_CHOICES, apply_variant


def test_variant_choices_cover_component_grid() -> None:
    grid_variants = {e["variant"] for e in component_ablation_experiments()}
    assert grid_variants <= set(VARIANT_CHOICES)


def test_apply_variant_no_d_zeros_lambda_d() -> None:
    fl = apply_variant(DEFAULT_FL_CONFIG, "no_D", mu=0.1)
    assert fl["lambda_D"] == 0.0
    assert fl["mu"] == 0.1


def test_apply_variant_no_fs_zeros_fairness_staleness() -> None:
    fl = apply_variant(DEFAULT_FL_CONFIG, "no_FS")
    assert fl["lambda_F"] == 0.0
    assert fl["lambda_St"] == 0.0


def test_apply_variant_no_v_flag() -> None:
    fl = apply_variant(DEFAULT_FL_CONFIG, "no_V")
    assert fl["no_V"] is True


def test_apply_variant_static_beta() -> None:
    fl = apply_variant(DEFAULT_FL_CONFIG, "static_beta")
    assert fl["static_beta"] == 0.90


def test_apply_variant_uniform_lr_and_agg() -> None:
    fl_lr = apply_variant(DEFAULT_FL_CONFIG, "uniform_lr")
    fl_agg = apply_variant(DEFAULT_FL_CONFIG, "uniform_agg")
    assert fl_lr["uniform_lr"] is True
    assert fl_agg["uniform_aggregation"] is True
