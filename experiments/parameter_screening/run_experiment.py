#!/usr/bin/env python3
"""Generate and optionally execute one fixed-step FOXES screening run."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--objective", choices=("gaussian", "quantile"),
        default="gaussian",
        help="Probabilistic objective to screen (default: gaussian).",
    )
    parser.add_argument(
        "--gaussian-model-type",
        choices=(
            "gaussian_nll", "gaussian_nll_background_excess",
            "gaussian_nll_og",
        ),
        default="gaussian_nll",
        help=(
            "Gaussian patch-mean parameterization. The background-excess "
            "variant combines a bounded global background with local excess; "
            "gaussian_nll_og uses the original inverse forecast and clamp."
        ),
    )
    parser.add_argument(
        "--base-config", type=Path,
        default=PROJECT_ROOT / "training" / "train_config.yaml",
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--subset-root", type=Path,
        default=None,
    )
    data_group.add_argument(
        "--dataset-root", type=Path, default=None,
        help=(
            "Use a complete AIA_processed/SXR_processed dataset directly "
            "instead of a generated screening subset."
        ),
    )
    parser.add_argument(
        "--aia-pre-normalized", action="store_true",
        help=(
            "Disable lazy AIA normalization because the selected dataset "
            "already contains normalized image arrays."
        ),
    )
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("/data/FOXES_screening/runs"),
    )
    parser.add_argument("--sparsity-weight", type=float, default=0.1)
    parser.add_argument(
        "--beta-nll-beta", type=float, default=None,
        help=(
            "Detached beta-NLL exponent for Gaussian runs. When omitted, "
            "inherit it from --base-config."
        ),
    )
    parser.add_argument(
        "--mean-loss", choices=("beta_nll", "huber", "mse"), default=None,
        help="Mean/backbone objective for Gaussian runs.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=("none", "inverse_frequency", "sqrt_inverse_frequency"),
        default=None,
        help=(
            "Class weighting for the canonical Huber mean loss. Weights are "
            "calculated once from the selected training split."
        ),
    )
    parser.add_argument(
        "--mean-loss-weighting",
        choices=("none", "four_class_macro", "sqrt_four_class_macro"),
        default=None,
        help=(
            "Deprecated alias for --class-weighting used by older sweeps."
        ),
    )
    parser.add_argument(
        "--mean-only", action="store_true",
        help=(
            "Train only the Gaussian model's mean path. This requires a "
            "direct --mean-loss of mse or huber; uncertainty predictions are "
            "still emitted for interface-compatible evaluation but receive "
            "no training gradient."
        ),
    )
    parser.add_argument(
        "--mean-huber-delta", type=float, default=1.0,
        help="Huber transition point in normalized log-flux units.",
    )
    parser.add_argument(
        "--top-fraction", type=float, default=0.05,
        help="Fraction of q50 patches allowed to carry concentrated flux.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler-t-max", type=int, default=None,
        help="Override CosineAnnealingLR T_max for this isolated run.",
    )
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--disable-gradient-clipping", action="store_true",
        help="Set optimizer.gradient_clip_val to null for this run.",
    )
    parser.add_argument(
        "--dropout", type=float, default=None,
        help="Override transformer dropout for this isolated run.",
    )
    parser.add_argument(
        "--mask-mode", choices=("inverted", "local", "none"), default=None,
        help="Override the transformer's attention-mask mode.",
    )
    parser.add_argument(
        "--local-window", type=int, default=None,
        help="Override the local attention neighbourhood side length in patches.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=None,
        help=(
            "Override square patch size in pixels. num_patches is derived "
            "from the base config's input size."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override the DataLoader microbatch size.",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=None,
        help="Override the transformer token-embedding dimension.",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None,
        help="Override the transformer MLP hidden width.",
    )
    parser.add_argument(
        "--num-heads", type=int, default=None,
        help="Override the number of transformer attention heads.",
    )
    parser.add_argument(
        "--num-layers", type=int, default=None,
        help="Override the number of transformer blocks.",
    )
    parser.add_argument(
        "--checkpoint-monitor", default=None,
        help=(
            "Override the validation metric used to retain checkpoints. "
            "For example: val_class/X/rmse_dex."
        ),
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu-id", type=int, default=None,
        help=(
            "Pin this run to one CUDA device. When omitted, inherit gpu_ids "
            "from the base training config."
        ),
    )
    parser.add_argument("--wandb-project", default="FOXES-SDOML-screening")
    parser.add_argument(
        "--run", action="store_true",
        help="Run training after creating the config. Default only writes the config.",
    )
    args = parser.parse_args()
    if args.dataset_root is None and args.subset_root is None:
        args.subset_root = Path("/data/FOXES_screening/subset")
    return args


def validate_args(args: argparse.Namespace) -> None:
    objective = getattr(args, "objective", "gaussian")
    if objective not in {"gaussian", "quantile"}:
        raise ValueError("--objective must be 'gaussian' or 'quantile'")
    gaussian_model_type = getattr(
        args, "gaussian_model_type", "gaussian_nll"
    )
    if gaussian_model_type not in {
        "gaussian_nll", "gaussian_nll_background_excess", "gaussian_nll_og",
    }:
        raise ValueError(
            "--gaussian-model-type must be 'gaussian_nll', "
            "'gaussian_nll_background_excess', or 'gaussian_nll_og'"
        )
    if objective == "gaussian" and gaussian_model_type == "gaussian_nll":
        if getattr(args, "mean_loss", None) not in {None, "huber"}:
            raise ValueError(
                "The canonical gaussian_nll model uses weighted Huber only"
            )
        if getattr(args, "beta_nll_beta", None) not in {None, 0.0}:
            raise ValueError(
                "The canonical gaussian_nll model uses detached ordinary "
                "Gaussian NLL; beta-NLL has been removed"
            )
        if getattr(args, "mean_only", False):
            raise ValueError(
                "The canonical gaussian_nll model always trains its detached "
                "uncertainty head"
            )
        if args.sparsity_weight != 0:
            raise ValueError(
                "The canonical gaussian_nll model no longer has a sparsity "
                "loss; set --sparsity-weight 0"
            )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        raise ValueError("--name may contain only letters, numbers, dot, underscore, and dash")
    if args.sparsity_weight < 0 or args.learning_rate <= 0:
        raise ValueError("Loss weights must be nonnegative and learning rate positive")
    beta_nll_beta = getattr(args, "beta_nll_beta", None)
    if beta_nll_beta is not None and not 0 <= beta_nll_beta <= 1:
        raise ValueError("--beta-nll-beta must be in [0, 1]")
    if getattr(args, "mean_huber_delta", 1.0) <= 0:
        raise ValueError("--mean-huber-delta must be positive")
    if (
        getattr(args, "mean_only", False)
        and getattr(args, "mean_loss", None) not in {"mse", "huber"}
    ):
        raise ValueError("--mean-only requires --mean-loss mse or huber")
    if (
        getattr(args, "mean_loss_weighting", None)
        in {"four_class_macro", "sqrt_four_class_macro"}
        and getattr(args, "mean_loss", None) not in {"mse", "huber"}
    ):
        raise ValueError(
            "class-balanced --mean-loss-weighting requires "
            "--mean-loss mse or huber"
        )
    if (
        getattr(args, "class_weighting", None) is not None
        and getattr(args, "mean_loss_weighting", None) is not None
    ):
        raise ValueError(
            "Use --class-weighting or --mean-loss-weighting, not both"
        )
    if not 0 < getattr(args, "top_fraction", 0.05) <= 1:
        raise ValueError("--top-fraction must be in (0, 1]")
    if args.weight_decay is not None and args.weight_decay < 0:
        raise ValueError("--weight-decay must be nonnegative")
    if args.dropout is not None and not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if (
        getattr(args, "local_window", None) is not None
        and (args.local_window <= 0 or args.local_window % 2 == 0)
    ):
        raise ValueError("--local-window must be a positive odd integer")
    if getattr(args, "patch_size", None) is not None and args.patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    if getattr(args, "batch_size", None) is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for option in ("embed_dim", "hidden_dim", "num_heads", "num_layers"):
        value = getattr(args, option, None)
        if value is not None and value <= 0:
            raise ValueError(f"--{option.replace('_', '-')} must be positive")
    checkpoint_monitor = getattr(args, "checkpoint_monitor", None)
    if (
        checkpoint_monitor is not None
        and not re.fullmatch(r"[A-Za-z0-9_./-]+", checkpoint_monitor)
    ):
        raise ValueError("--checkpoint-monitor contains invalid characters")
    if (
        args.accumulate_grad_batches is not None
        and args.accumulate_grad_batches <= 0
    ):
        raise ValueError("--accumulate-grad-batches must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if (
        getattr(args, "scheduler_t_max", None) is not None
        and args.scheduler_t_max <= 0
    ):
        raise ValueError("--scheduler-t-max must be positive")
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError("--gpu-id must be nonnegative")


def build_config(args: argparse.Namespace) -> tuple[dict, Path]:
    validate_args(args)
    with args.base_config.open() as handle:
        config = yaml.safe_load(handle)

    is_full_dataset = args.dataset_root is not None
    data_root = (
        args.dataset_root if is_full_dataset else args.subset_root
    ).resolve()
    required = [
        data_root / kind / split
        for kind in ("AIA_processed", "SXR_processed")
        for split in ("train", "val", "test")
    ]
    if is_full_dataset:
        required.append(data_root / "SXR_processed" / "normalized_sxr.npy")
    else:
        required.append(data_root / "subset_summary.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Experiment data is incomplete: " + ", ".join(missing)
        )

    run_dir = args.runs_root.resolve() / args.name
    objective = getattr(args, "objective", "gaussian")
    gaussian_model_type = getattr(
        args, "gaussian_model_type", "gaussian_nll"
    )
    top_fraction = getattr(args, "top_fraction", 0.05)
    config["model_type"] = (
        "quantile" if objective == "quantile" else gaussian_model_type
    )
    config["seed"] = args.seed
    if args.gpu_id is not None:
        # One process owns one device. This prevents parallel screening jobs
        # from each interpreting the base config's "all" as a DDP request.
        config["gpu_ids"] = args.gpu_id
    config["epochs"] = max(int(config.get("epochs", 1)), 10_000)
    config["max_steps"] = args.max_steps
    config["calculate_base_weights"] = False
    # The deterministic Huber settings are irrelevant to probabilistic runs.
    config.pop("loss", None)
    config["data"]["aia_dir"] = str(data_root / "AIA_processed")
    config["data"]["sxr_dir"] = str(data_root / "SXR_processed")
    local_sxr_norm_path = (
        data_root / "SXR_processed" / "normalized_sxr.npy"
    )
    if local_sxr_norm_path.is_file():
        # Newly generated subsets link the source dataset's full-training
        # normalization artifact here. Older subsets without the link retain
        # the base config path for backwards compatibility.
        config["data"]["sxr_norm_path"] = str(local_sxr_norm_path)
    if args.aia_pre_normalized:
        config["data"]["aia_norm_path"] = None
    config["data"]["checkpoints_dir"] = str(run_dir / "checkpoints")
    if args.accumulate_grad_batches is not None:
        config["accumulate_grad_batches"] = args.accumulate_grad_batches
    if args.weight_decay is not None:
        config["optimizer"]["weight_decay"] = args.weight_decay
    if getattr(args, "disable_gradient_clipping", False):
        config["optimizer"]["gradient_clip_val"] = None
    if args.dropout is not None:
        config["vit_architecture"]["dropout"] = args.dropout
    patch_size = getattr(args, "patch_size", None)
    if patch_size is not None:
        base_patch_size = int(config["vit_architecture"]["patch_size"])
        base_num_patches = int(config["vit_architecture"]["num_patches"])
        base_grid_size = math.isqrt(base_num_patches)
        if base_grid_size * base_grid_size != base_num_patches:
            raise ValueError(
                "Base vit_architecture.num_patches must be a square"
            )
        input_size = base_grid_size * base_patch_size
        if input_size % patch_size:
            raise ValueError(
                f"Input size {input_size} is not divisible by patch size "
                f"{patch_size}"
            )
        config["vit_architecture"]["patch_size"] = patch_size
        config["vit_architecture"]["num_patches"] = (
            input_size // patch_size
        ) ** 2
    if getattr(args, "batch_size", None) is not None:
        config["batch_size"] = args.batch_size
    for option in ("embed_dim", "hidden_dim", "num_heads", "num_layers"):
        value = getattr(args, option, None)
        if value is not None:
            config["vit_architecture"][option] = value
    embed_dim = int(config["vit_architecture"]["embed_dim"])
    num_heads = int(config["vit_architecture"]["num_heads"])
    if embed_dim % num_heads:
        raise ValueError(
            f"embed_dim {embed_dim} must be divisible by num_heads "
            f"{num_heads}"
        )
    if getattr(args, "mask_mode", None) is not None:
        config["vit_architecture"]["mask_mode"] = args.mask_mode
    if getattr(args, "local_window", None) is not None:
        config["vit_architecture"]["local_window"] = args.local_window
    scheduler_t_max = getattr(args, "scheduler_t_max", None)
    if scheduler_t_max is not None:
        config["optimizer"].setdefault("scheduler", {})["T_max"] = (
            scheduler_t_max
        )
    config["vit_architecture"]["learning_rate"] = args.learning_rate
    config["checkpoint"]["save_top_k"] = min(
        int(config["checkpoint"].get("save_top_k", 2)), 2
    )
    config["checkpoint"]["mode"] = "min"
    # Screening runs are compared at the same optimizer-step budget. Early
    # stopping would otherwise give different configurations different
    # amounts of training.
    config.setdefault("early_stopping", {})["enabled"] = False
    config["callbacks"].update({
        "attention_enabled": False,
        "sxr_plot_enabled": True,
        "sxr_plot_num_samples": 5,
    })
    config["wandb"]["project"] = args.wandb_project
    config["wandb"]["run_name"] = args.name
    stale_exact_tags = {
        "quantile-regression", "spatial-uncertainty",
        "nll-five-class-macro", "global-uncertainty", "full-data",
        "og-patch-mean", "background-excess-patch-mean",
        "multiplier-patch-mean", "train-uncertainty", "mean-only",
    }
    stale_parameter_prefixes = (
        "beta-", "embed-", "heads-", "layers-", "local-", "mlp-",
        "sparsity-", "top-fraction-", "mean-loss-", "mean-weighting-",
    )
    tags = [
        tag for tag in (config["wandb"].get("tags") or [])
        if tag not in stale_exact_tags
        and not tag.startswith(stale_parameter_prefixes)
    ]
    tags.append("full-data" if is_full_dataset else "subset-data")
    screening = {
        "data_root": str(data_root),
        "data_mode": "full_dataset" if is_full_dataset else "subset",
        "aia_pre_normalized": bool(args.aia_pre_normalized),
        "objective": objective,
        "model_type": config["model_type"],
        "weight_decay": config["optimizer"].get("weight_decay", 0.0),
        "gradient_clip_val": config["optimizer"].get("gradient_clip_val"),
        "dropout": config["vit_architecture"].get("dropout", 0.0),
        "mask_mode": config["vit_architecture"].get("mask_mode", "inverted"),
        "local_window": config["vit_architecture"].get("local_window", 9),
        "patch_size": config["vit_architecture"].get("patch_size"),
        "num_patches": config["vit_architecture"].get("num_patches"),
        "batch_size": config.get("batch_size"),
        "embed_dim": config["vit_architecture"].get("embed_dim"),
        "hidden_dim": config["vit_architecture"].get("hidden_dim"),
        "num_heads": config["vit_architecture"].get("num_heads"),
        "num_layers": config["vit_architecture"].get("num_layers"),
        "accumulate_grad_batches": config.get("accumulate_grad_batches", 1),
        "sparsity_weight": args.sparsity_weight,
        "top_fraction": top_fraction,
        "learning_rate": args.learning_rate,
        "scheduler_t_max": config["optimizer"].get(
            "scheduler", {}
        ).get("T_max"),
        "max_steps": args.max_steps,
        "gpu_id": args.gpu_id,
    }
    if objective == "quantile":
        # Stable spatial-quantile baseline, independent of later edits to the
        # full-data configuration.
        config.pop("uncertainty", None)
        quantile = config.setdefault("quantile", {})
        quantile.update({
            "patch_flux_scale_multiplier": 1.0,
            "max_abs_log10_patch_multiplier": 8.0,
            "initial_log10_quantile_step": 0.1,
            "max_log10_quantile_step": 4.0,
            "loss_weighting": "five_class",
            "spatial_sparsity": {
                "weight": args.sparsity_weight,
                "top_fraction": top_fraction,
                "gate_center_flux": 5e-6,
                "gate_width_dex": 0.25,
            },
        })
        quantile.pop("five_class_weights", None)
        config["checkpoint"]["monitor"] = "val_class/macro_pinball"
        config["callbacks"].update({
            "spatial_quantile_enabled": True,
            "spatial_quantile_num_samples": 5,
            "spatial_quantile_log_every_n_epochs": 1,
        })
        tags.extend((
            "screening", "quantile-regression", "spatial-uncertainty",
            "pinball-five-class-macro",
            f"weight-decay-{config['optimizer'].get('weight_decay', 0):g}",
            f"sparsity-{args.sparsity_weight:g}",
            f"top-fraction-{top_fraction:g}",
            f"lr-{args.learning_rate:g}",
        ))
        config["wandb"]["notes"] = (
            "Small-data ordered spatial-quantile experiment with direct "
            "68%/95% interval evaluation"
        )
        screening["loss_weighting"] = "five_class_macro_pinball"
        screening["quantiles"] = [0.025, 0.16, 0.5, 0.84, 0.975]
    else:
        config.pop("quantile", None)
        existing_uncertainty = config.get("uncertainty", {})
        uncertainty = {
            key: existing_uncertainty[key]
            for key in (
                "beta_nll_beta",
                "mean_loss",
                "mean_huber_delta",
                "mean_loss_weighting",
                "train_uncertainty",
                "patch_flux_scale_multiplier",
                "max_abs_log10_patch_multiplier",
                "relative_std_floor",
                "relative_std_max",
                "patch_scale_floor_fraction",
                "spatial_sparsity",
            )
            if key in existing_uncertainty
        }
        uncertainty.setdefault("patch_flux_scale_multiplier", 1.0)
        uncertainty.setdefault("beta_nll_beta", 1.0)
        uncertainty.setdefault("max_abs_log10_patch_multiplier", 8.0)
        uncertainty.setdefault("relative_std_floor", 0.0025)
        uncertainty.setdefault("relative_std_max", 20.0)
        uncertainty.setdefault("patch_scale_floor_fraction", 1.0)
        if getattr(args, "beta_nll_beta", None) is not None:
            uncertainty["beta_nll_beta"] = args.beta_nll_beta
        uncertainty.setdefault("mean_loss", "beta_nll")
        uncertainty.setdefault("train_uncertainty", True)
        uncertainty.setdefault("mean_huber_delta", 1.0)
        if getattr(args, "mean_loss", None) is not None:
            uncertainty["mean_loss"] = args.mean_loss
        mean_loss_weighting = getattr(args, "mean_loss_weighting", None)
        requested_class_weighting = getattr(args, "class_weighting", None)
        if requested_class_weighting is not None:
            mean_loss_weighting = {
                "none": "none",
                "inverse_frequency": "four_class_macro",
                "sqrt_inverse_frequency": "sqrt_four_class_macro",
            }[requested_class_weighting]
        if mean_loss_weighting is not None:
            uncertainty["mean_loss_weighting"] = mean_loss_weighting
        else:
            uncertainty.setdefault("mean_loss_weighting", "none")
        # Always derive these anew from the selected train split in train.py.
        uncertainty.pop("mean_class_weights", None)
        if getattr(args, "mean_only", False):
            uncertainty["train_uncertainty"] = False
        uncertainty["mean_huber_delta"] = getattr(
            args, "mean_huber_delta", 1.0
        )
        uncertainty["spatial_sparsity"] = {
            "weight": args.sparsity_weight,
            "top_fraction": top_fraction,
            "gate_center_flux": 5e-6,
            "gate_width_dex": 0.25,
        }
        if gaussian_model_type == "gaussian_nll_og":
            # These configure only the newer log10-multiplier mean. Keeping
            # them in an OG config would imply that they affect the run.
            uncertainty.pop("patch_flux_scale_multiplier", None)
            uncertainty.pop("max_abs_log10_patch_multiplier", None)
        elif gaussian_model_type == "gaussian_nll_background_excess":
            uncertainty.setdefault("background_flux_max", 5e-6)
            uncertainty.setdefault("background_initial_cap_fraction", 0.5)
            uncertainty.setdefault("excess_initial_fraction", 0.2)
            uncertainty.setdefault("solar_disk_radius_fraction", 0.48)
            uncertainty.setdefault("background_scale_floor_fraction", 0.05)
            uncertainty.setdefault("background_uncertainty_initial_raw", -3.0)
        config["uncertainty"] = uncertainty
        # Select point-prediction quality consistently across beta values;
        # NLL and coverage remain logged for uncertainty comparison.
        config["checkpoint"]["monitor"] = "val/mse"
        config["callbacks"].update({
            "spatial_quantile_enabled": False,
            "spatial_gaussian_enabled": True,
            "spatial_gaussian_num_samples": 5,
            "spatial_gaussian_log_every_n_epochs": 1,
        })
        tags.extend((
            "screening", "patch-gaussian-nll", "patch-uncertainty",
            (
                "og-patch-mean"
                if gaussian_model_type == "gaussian_nll_og"
                else (
                    "background-excess-patch-mean"
                    if gaussian_model_type
                    == "gaussian_nll_background_excess"
                    else "multiplier-patch-mean"
                )
            ),
            f"mean-loss-{uncertainty['mean_loss']}",
            f"mean-weighting-{uncertainty['mean_loss_weighting'].replace('_', '-')}",
            (
                "train-uncertainty"
                if uncertainty["train_uncertainty"]
                else "mean-only"
            ),
            f"weight-decay-{config['optimizer'].get('weight_decay', 0):g}",
            (
                "gradient-clip-off"
                if config['optimizer'].get('gradient_clip_val') is None
                else f"gradient-clip-{config['optimizer']['gradient_clip_val']:g}"
            ),
            f"dropout-{config['vit_architecture'].get('dropout', 0):g}",
            f"mask-{config['vit_architecture'].get('mask_mode', 'inverted')}",
            f"window-{config['vit_architecture'].get('local_window', 9)}",
            f"patch-{config['vit_architecture'].get('patch_size')}",
            f"batch-{config.get('batch_size')}",
            f"embed-{config['vit_architecture'].get('embed_dim')}",
            f"heads-{config['vit_architecture'].get('num_heads')}",
            f"layers-{config['vit_architecture'].get('num_layers')}",
            f"mlp-{config['vit_architecture'].get('hidden_dim')}",
            f"beta-{uncertainty['beta_nll_beta']:g}",
            f"sparsity-{args.sparsity_weight:g}",
            f"top-fraction-{top_fraction:g}",
            f"lr-{args.learning_rate:g}",
        ))
        screening["uncertainty_source"] = "summed_patch_variance"
        screening["beta_nll_beta"] = uncertainty["beta_nll_beta"]
        screening["mean_loss"] = uncertainty["mean_loss"]
        screening["mean_loss_weighting"] = uncertainty[
            "mean_loss_weighting"
        ]
        screening["train_uncertainty"] = uncertainty["train_uncertainty"]
        screening["mean_huber_delta"] = uncertainty["mean_huber_delta"]
        config["wandb"]["notes"] = (
            f"{config['model_type']} screening run with detached "
            f"beta-NLL beta={uncertainty['beta_nll_beta']:g}, "
            f"train_uncertainty={uncertainty['train_uncertainty']}, "
            f"{config['vit_architecture'].get('mask_mode', 'inverted')} "
            "attention, and patch-aggregated uncertainty."
        )
        if gaussian_model_type == "gaussian_nll":
            # Keep newly generated configs focused on the one canonical
            # objective. The constructor still recognizes these removed keys
            # when loading historical checkpoints.
            for removed_key in (
                "beta_nll_beta", "mean_loss", "train_uncertainty",
                "spatial_sparsity",
            ):
                uncertainty.pop(removed_key, None)
            for removed_key in (
                "beta_nll_beta", "train_uncertainty",
            ):
                screening.pop(removed_key, None)
            uncertainty["huber_delta"] = uncertainty.pop(
                "mean_huber_delta"
            )
            uncertainty["class_weighting"] = {
                "none": "none",
                "four_class_macro": "inverse_frequency",
                "sqrt_four_class_macro": "sqrt_inverse_frequency",
            }[uncertainty.pop("mean_loss_weighting")]
            screening["huber_delta"] = screening.pop(
                "mean_huber_delta"
            )
            screening["class_weighting"] = {
                "none": "none",
                "four_class_macro": "inverse_frequency",
                "sqrt_four_class_macro": "sqrt_inverse_frequency",
            }[screening.pop("mean_loss_weighting")]
            tags = [
                tag for tag in tags
                if tag not in {"train-uncertainty", "mean-only"}
                and not tag.startswith((
                    "beta-", "sparsity-", "top-fraction-",
                    "mean-loss-", "mean-weighting-",
                ))
            ]
            tags.extend((
                "canonical-gaussian", "multiplier-patch-mean",
                "weighted-huber", "detached-patch-uncertainty",
                "class-weighting-"
                + uncertainty["class_weighting"].replace("_", "-"),
            ))
            config["wandb"]["notes"] = (
                "Canonical multiplier patch model with class-weighted Huber "
                "and detached patch-aggregated Gaussian uncertainty."
            )
    checkpoint_monitor = getattr(args, "checkpoint_monitor", None)
    if checkpoint_monitor is not None:
        config["checkpoint"]["monitor"] = checkpoint_monitor
        if checkpoint_monitor.startswith("val_class/"):
            config["callbacks"]["per_class_metrics_enabled"] = True
    screening["checkpoint_monitor"] = config["checkpoint"]["monitor"]
    config["wandb"]["tags"] = list(dict.fromkeys(tags))
    config["screening"] = screening
    return config, run_dir


def write_config(config: dict, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "train_config.yaml"
    rendered = yaml.safe_dump(config, sort_keys=False)
    if config_path.exists():
        if config_path.read_text() != rendered:
            raise FileExistsError(
                f"A different config already exists at {config_path}; "
                "choose a new --name"
            )
        return config_path
    config_path.write_text(rendered)
    return config_path


def main() -> int:
    args = parse_args()
    config, run_dir = build_config(args)
    config_path = write_config(config, run_dir)
    print(f"Screening config: {config_path}")
    if config["model_type"] == "quantile":
        print(
            "Parameters: pinball=five-class-macro, "
            "quantiles=0.025/0.16/0.5/0.84/0.975, "
            f"sparsity={args.sparsity_weight:g}, "
            f"learning rate={args.learning_rate:g}, steps={args.max_steps:,}"
        )
    else:
        if config["model_type"] == "gaussian_nll":
            uncertainty = config["uncertainty"]
            print(
                f"Parameters: model=gaussian_nll, "
                f"mean=huber(delta={uncertainty['huber_delta']:g}, "
                f"weighting={uncertainty['class_weighting']}), "
                "uncertainty=detached patch Gaussian NLL, "
                f"gradient clip={config['optimizer'].get('gradient_clip_val')}, "
                f"dropout={config['vit_architecture'].get('dropout', 0):g}, "
                f"weight decay={config['optimizer'].get('weight_decay', 0):g}, "
                f"mask={config['vit_architecture'].get('mask_mode', 'inverted')}, "
                f"window={config['vit_architecture'].get('local_window', 9)}, "
                f"patch={config['vit_architecture'].get('patch_size')}, "
                f"batch={config.get('batch_size')}, "
                f"embed={config['vit_architecture'].get('embed_dim')}, "
                f"heads={config['vit_architecture'].get('num_heads')}, "
                f"layers={config['vit_architecture'].get('num_layers')}, "
                f"mlp={config['vit_architecture'].get('hidden_dim')}, "
                f"learning rate={args.learning_rate:g}, steps={args.max_steps:,}"
            )
        else:
            print(
                f"Parameters: legacy model={config['model_type']}, "
                f"learning rate={args.learning_rate:g}, "
                f"steps={args.max_steps:,}"
            )
    if config["model_type"] == "quantile":
        print("Five-class macro weights will be calculated by training/train.py")
    elif (
        config["uncertainty"].get("class_weighting")
        in {"inverse_frequency", "sqrt_inverse_frequency"}
        or config["uncertainty"].get("mean_loss_weighting")
        in {"four_class_macro", "sqrt_four_class_macro"}
    ):
        print(
            "Quiet/C/M/X Huber weights will be calculated from the training "
            "SXR targets by training/train.py"
        )
    if not args.run:
        print(
            "Config only; start with:\n"
            f"  {sys.executable} {PROJECT_ROOT / 'training' / 'train.py'} "
            f"--config {config_path}"
        )
        return 0
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "training" / "train.py"),
            "--config", str(config_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
