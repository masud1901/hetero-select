#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from torch.utils.data import DataLoader

from heteroselect.config import (
    DEFAULT_FL_CONFIG,
    ablation_experiments,
    component_ablation_experiments,
    default_experiments,
    load_config,
)
from heteroselect.data import load_data
from heteroselect.trainer import run_experiment
from heteroselect.utils import seed_everything
from heteroselect.variants import VARIANT_CHOICES


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HeteRo-Select training driver")
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML/JSON overlay on DEFAULT_FL_CONFIG.")
    p.add_argument("--grid", choices=["main", "ablation", "component", "all"],
                   default=None,
                   help="Predefined experiment grid (overrides single-run flags).")
    p.add_argument("--dataset",
                   choices=["mnist", "cifar10", "cifar100", "tinyimagenet"],
                   default="cifar10")
    p.add_argument("--psi", type=float, default=0.4,
                   help="psi-LDA concentration (MNIST/CIFAR-10) or missing classes.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variant", choices=list(VARIANT_CHOICES), default="adaptive")
    p.add_argument("--mu", type=float, default=None,
                   help="Override FedProx proximal coefficient.")
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--local-steps", type=int, default=None)
    p.add_argument("--local-lr", type=float, default=None)
    p.add_argument("--lambda-V", type=float, default=None)
    p.add_argument("--lambda-D", type=float, default=None)
    p.add_argument("--lambda-F", type=float, default=None)
    p.add_argument("--lambda-St", type=float, default=None)
    p.add_argument("--newton-Q", type=int, default=None)
    p.add_argument("--newton-lambda", type=float, default=None)
    p.add_argument("--static-beta", type=float, default=None,
                   help="Fixed error-feedback beta (disables adaptive schedule).")
    p.add_argument("--uniform-lr", action="store_true",
                   help="Use uniform local LR (same as --variant uniform_lr).")
    p.add_argument("--uniform-aggregation", action="store_true",
                   help="FedAvg aggregation (same as --variant uniform_agg).")
    p.add_argument("--tag", type=str, default=None,
                   help="Suffix for output JSON filename.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--zip", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _build_experiments(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.grid == "main":
        return default_experiments()
    if args.grid == "ablation":
        return ablation_experiments()
    if args.grid == "component":
        return component_ablation_experiments()
    if args.grid == "all":
        return (
            default_experiments()
            + ablation_experiments()
            + component_ablation_experiments()
        )
    cfg: Dict[str, Any] = dict(
        dataset=args.dataset, psi=args.psi, seed=args.seed, variant=args.variant,
    )
    if args.mu is not None:
        cfg["mu"] = args.mu
    return [cfg]


def main() -> None:
    args = _parse_args()
    fl = load_config(args.config) if args.config else dict(DEFAULT_FL_CONFIG)
    if args.rounds is not None:
        fl["num_rounds"] = args.rounds
    if args.local_steps is not None:
        fl["local_steps"] = args.local_steps
    if args.local_lr is not None:
        fl["local_lr"] = args.local_lr
    for cli_name, fl_name in (
        ("lambda_V", "lambda_V"),
        ("lambda_D", "lambda_D"),
        ("lambda_F", "lambda_F"),
        ("lambda_St", "lambda_St"),
        ("newton_Q", "newton_Q"),
        ("newton_lambda", "newton_lambda"),
        ("static_beta", "static_beta"),
    ):
        v = getattr(args, cli_name.replace("-", "_"), None)
        if v is not None:
            fl[fl_name] = v
    if args.uniform_lr:
        fl["uniform_lr"] = True
    if args.uniform_aggregation:
        fl["uniform_aggregation"] = True

    experiments = _build_experiments(args)
    os.makedirs(args.results_dir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    cache: Dict[str, Any] = {}

    for cfg in experiments:
        rng = seed_everything(cfg["seed"])
        ds_name = cfg["dataset"]
        if ds_name not in cache:
            train_ds, test_ds = load_data(ds_name)
            test_loader = DataLoader(
                test_ds, batch_size=512, shuffle=False,
                num_workers=2, pin_memory=True,
            )
            cache[ds_name] = (train_ds, test_loader)
        train_ds, test_loader = cache[ds_name]

        result = run_experiment(
            cfg, fl, train_ds, test_loader, rng,
            verbose=not args.quiet,
        )

        psi_str = str(cfg["psi"]).replace(".", "p")
        mu_str = str(cfg.get("mu", fl["mu"])).replace(".", "p")
        variant = cfg.get("variant", "adaptive")
        tag = f"_{args.tag}" if args.tag else ""
        fname = os.path.join(
            args.results_dir,
            f"{ds_name}_psi{psi_str}_mu{mu_str}_seed{cfg['seed']}_{variant}{tag}.json",
        )
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {fname}")

        row: Dict[str, Any] = dict(
            dataset=ds_name, psi=cfg["psi"],
            mu=cfg.get("mu", fl["mu"]), seed=cfg["seed"], variant=variant,
        )
        row.update(result["summary"])
        all_rows.append(row)

    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nWrote {summary_path}  ({len(all_rows)} experiment(s))")

    if args.zip:
        zip_path = "results.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(args.results_dir):
                for fn in files:
                    full = os.path.join(root, fn)
                    zf.write(full, os.path.relpath(full, start=args.results_dir))
        print(f"Packaged {zip_path}")


if __name__ == "__main__":
    main()
