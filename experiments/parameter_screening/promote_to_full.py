#!/usr/bin/env python3
"""Create and optionally run a full-data config from winning screen parameters."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-config", type=Path,
        default=PROJECT_ROOT / "training" / "train_config.yaml",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/data/FOXES_screening/final"),
    )
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        raise ValueError("Unsafe run name")
    if args.learning_rate <= 0 or args.epochs <= 0:
        raise ValueError("Invalid learning-rate or epoch value")
    with args.base_config.open() as handle:
        config = yaml.safe_load(handle)

    run_dir = args.output_root.resolve() / args.name
    config.pop("max_steps", None)
    config["seed"] = args.seed
    config["epochs"] = args.epochs
    # CosineAnnealingLR is stepped once per epoch, so the promoted full run
    # should reach eta_min at its final epoch even when --epochs is overridden.
    config["optimizer"]["scheduler"]["T_max"] = args.epochs
    config["calculate_base_weights"] = False
    config.pop("loss", None)
    uncertainty = config.get("uncertainty", {})
    config["uncertainty"] = {
        "beta_nll_beta": uncertainty.get("beta_nll_beta", 1.0),
        "patch_flux_scale_multiplier": uncertainty.get(
            "patch_flux_scale_multiplier", 1.0
        ),
        "max_abs_log10_patch_multiplier": uncertainty.get(
            "max_abs_log10_patch_multiplier", 8.0
        ),
        "relative_std_floor": uncertainty.get(
            "relative_std_floor", 0.0025
        ),
        "relative_std_max": uncertainty.get(
            "relative_std_max", 20.0
        ),
        "patch_scale_floor_fraction": uncertainty.get(
            "patch_scale_floor_fraction", 1.0
        ),
        "spatial_sparsity": uncertainty.get("spatial_sparsity", {
            "weight": 0.1,
            "top_fraction": 0.05,
            "gate_center_flux": 5e-6,
            "gate_width_dex": 0.25,
        }),
    }
    config["vit_architecture"]["learning_rate"] = args.learning_rate
    config["data"]["checkpoints_dir"] = str(run_dir / "checkpoints")
    config["callbacks"]["attention_enabled"] = True
    config["callbacks"]["sxr_plot_enabled"] = True
    config["callbacks"]["per_class_metrics_enabled"] = True
    config["checkpoint"]["monitor"] = "val/nll"
    config["checkpoint"]["mode"] = "min"
    config["wandb"]["run_name"] = args.name
    tags = list(config["wandb"].get("tags", []))
    config["wandb"]["tags"] = list(dict.fromkeys(tags + ["full-data", "screen-winner"]))
    config["promoted_screening_parameters"] = {
        "learning_rate": args.learning_rate,
        "uncertainty_source": "summed_patch_variance",
        "beta_nll_beta": config["uncertainty"]["beta_nll_beta"],
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "train_config.yaml"
    rendered = yaml.safe_dump(config, sort_keys=False)
    if config_path.exists() and config_path.read_text() != rendered:
        raise FileExistsError(f"Different config already exists: {config_path}")
    config_path.write_text(rendered)
    print(f"Full-data config: {config_path}")
    print(
        "Using detached beta-NLL "
        f"(beta={config['uncertainty']['beta_nll_beta']:g}) "
        "with summed patch variance."
    )
    if args.run:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "training" / "train.py"),
             "--config", str(config_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    else:
        print(
            "Config only; start with:\n"
            f"  {sys.executable} {PROJECT_ROOT / 'training' / 'train.py'} "
            f"--config {config_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
