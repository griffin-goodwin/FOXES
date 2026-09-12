#!/usr/bin/env python3
"""Generate or run a staged FOXES parameter screen, optionally in parallel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


def parse_gpu_ids(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list used to assign one GPU per process."""
    try:
        gpu_ids = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--gpu-ids must be comma-separated nonnegative integers"
        ) from error
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError(
            "--gpu-ids must contain at least one nonnegative integer"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError(
            "--gpu-ids must not contain duplicates"
        )
    return gpu_ids


def parse_float_values(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of nonnegative floating-point values."""
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated numbers"
        ) from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "values must contain at least one nonnegative number"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def parse_local_window_sizes(value: str) -> tuple[int, ...]:
    """Parse unique, positive odd local-attention window side lengths."""
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--local-window-sizes must be comma-separated integers"
        ) from error
    if not values or any(item <= 0 or item % 2 == 0 for item in values):
        raise argparse.ArgumentTypeError(
            "--local-window-sizes must contain positive odd integers"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            "--local-window-sizes must not contain duplicates"
        )
    return values


def subset_preflight_error(
    subset_root: Path,
    *,
    source_root: Path | None = None,
    preset: str = "quick",
    require_sxr_norm: bool = False,
) -> str | None:
    """Return an actionable error when a screening subset is incomplete."""
    subset_root = subset_root.resolve()
    required = [
        subset_root / "AIA_processed" / "train",
        subset_root / "SXR_processed" / "train",
        subset_root / "subset_summary.json",
    ]
    if require_sxr_norm:
        required.append(subset_root / "SXR_processed" / "normalized_sxr.npy")
    missing = [path for path in required if not path.exists()]
    if not missing:
        if source_root is None:
            return None
        summary_path = subset_root / "subset_summary.json"
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            return f"Cannot read subset provenance from {summary_path}: {error}"
        expected_aia = (source_root / "AIA_processed").resolve()
        expected_sxr = (source_root / "SXR_processed").resolve()
        actual_aia = Path(str(summary.get("aia_source", ""))).resolve()
        actual_sxr = Path(str(summary.get("sxr_source", ""))).resolve()
        if (actual_aia, actual_sxr) != (expected_aia, expected_sxr):
            return "\n".join((
                f"Wrong dataset behind screening subset: {subset_root}",
                f"Expected AIA source: {expected_aia}",
                f"Actual AIA source:   {actual_aia}",
                f"Expected SXR source: {expected_sxr}",
                f"Actual SXR source:   {actual_sxr}",
            ))
        return None

    source_root = (source_root or Path("/data")).resolve()
    lines = [
        f"Screening subset is missing or incomplete: {subset_root}",
        "Missing:",
        *(f"  - {path}" for path in missing),
        "",
        "Create the quick subset first:",
        f"  {sys.executable} {HERE / 'create_subset.py'} \\",
        f"    --aia-dir {source_root / 'AIA_processed'} \\",
        f"    --sxr-dir {source_root / 'SXR_processed'} \\",
        f"    --output-root {subset_root} --preset {preset}",
    ]
    available = sorted(
        summary.parent
        for summary in subset_root.parent.glob("*/subset_summary.json")
        if (summary.parent / "AIA_processed" / "train").is_dir()
        and (summary.parent / "SXR_processed" / "train").is_dir()
    )
    if available:
        lines.extend((
            "",
            "Existing complete subset roots (use --subset-root to select one):",
            *(f"  - {path}" for path in available),
        ))
    return "\n".join(lines)


def dataset_preflight_error(dataset_root: Path) -> str | None:
    """Validate a complete train/val/test dataset used without subsetting."""
    dataset_root = dataset_root.resolve()
    required = [
        dataset_root / kind / split
        for kind in ("AIA_processed", "SXR_processed")
        for split in ("train", "val", "test")
    ]
    required.append(dataset_root / "SXR_processed" / "normalized_sxr.npy")
    missing = [path for path in required if not path.exists()]
    if not missing:
        return None
    return "\n".join((
        f"Full experiment dataset is incomplete: {dataset_root}",
        "Missing:",
        *(f"  - {path}" for path in missing),
    ))


@dataclass(frozen=True)
class Experiment:
    name: str
    sparsity_weight: float
    learning_rate: float
    objective: str = "gaussian"
    gaussian_model_type: str | None = None
    top_fraction: float = 0.05
    scheduler_t_max: int | None = None
    weight_decay: float | None = None
    dropout: float | None = None
    accumulate_grad_batches: int | None = None
    beta_nll_beta: float | None = None
    mean_loss: str | None = None
    mean_huber_delta: float | None = None
    class_weighting: str | None = None
    mean_loss_weighting: str | None = None
    mask_mode: str | None = None
    local_window: int | None = None
    patch_size: int | None = None
    batch_size: int | None = None
    embed_dim: int | None = None
    hidden_dim: int | None = None
    num_heads: int | None = None
    num_layers: int | None = None
    checkpoint_monitor: str | None = None
    mean_only: bool = False
    disable_gradient_clipping: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "quantile", "patch-gaussian", "global-gaussian",
            "beta-detached", "og-beta-sparsity",
            "local-window-beta",
            "patch-size-beta0",
            "architecture-capacity-beta0",
            "architecture-combos-beta0",
            "og-patch-attention-beta",
            "mean-loss",
            "mean-gradient-controls",
            "contextual-local-depth",
            "inverted-mask-window",
            "local-weighted-huber",
            "canonical-weighted-huber",
            "attention-capacity-weighted-huber",
            "weighted-huber-followup",
            "background-excess-depth",
            "background-excess-depth-completion",
            "learning-rate",
        ),
        required=True,
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--subset-root", type=Path, default=None,
        help=(
            "Generated screening subset. Defaults to subset-og-quick for "
            "og-beta-sparsity and subset for other stages."
        ),
    )
    data_group.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Use a complete train/val/test dataset without creating a subset.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("/data/FOXES_screening/runs"))
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparsity-weight", type=float, default=0.1,
                        help="Sparsity weight held fixed across the stage.")
    parser.add_argument(
        "--sparsity-weights", type=parse_float_values,
        default=(0.1, 0.3, 1.0), metavar="VALUES",
        help=(
            "Comma-separated sparsity weights for og-beta-sparsity "
            "(default: 0.1,0.3,1.0)."
        ),
    )
    parser.add_argument(
        "--local-window-sizes", type=parse_local_window_sizes,
        default=(3, 5, 9), metavar="SIZES",
        help=(
            "Comma-separated odd local-attention side lengths for "
            "local-window-beta (default: 3,5,9)."
        ),
    )
    parser.add_argument(
        "--patch-batch-size", type=int, default=4,
        help="Microbatch size for patch-size-beta0 (default: 4).",
    )
    parser.add_argument(
        "--patch-accumulate-grad-batches", type=int, default=16,
        help=(
            "Gradient accumulation for patch-size-beta0; 4 x 16 preserves "
            "the baseline effective batch size of 64 (default: 16)."
        ),
    )
    parser.add_argument(
        "--aia-pre-normalized", action="store_true",
        help="Disable lazy AIA normalization for already-normalized arrays.",
    )
    parser.add_argument(
        "--top-fraction", type=float, default=0.05,
        help="Fraction of patches used by the spatial concentration loss.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Fixed value for non-learning-rate stages.")
    parser.add_argument(
        "--base-config", type=Path,
        default=PROJECT_ROOT / "training" / "train_config.yaml",
    )
    parser.add_argument(
        "--scheduler-t-max", type=int, default=None,
        help=(
            "CosineAnnealingLR T_max. Defaults to optimizer.scheduler.T_max "
            "from --base-config."
        ),
    )
    parser.add_argument("--run", action="store_true",
                        help="Run training; default only creates configs.")
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Run up to N training processes concurrently (default: 1).",
    )
    parser.add_argument(
        "--gpu-ids", type=parse_gpu_ids, default=None, metavar="IDS",
        help=(
            "Comma-separated CUDA IDs assigned round-robin, one per process. "
            "Use one ID (for example 0) to share a high-memory GPU."
        ),
    )
    args = parser.parse_args()
    og_stages = {
        "og-beta-sparsity", "local-window-beta", "patch-size-beta0",
        "architecture-capacity-beta0", "architecture-combos-beta0",
        "og-patch-attention-beta",
        "mean-gradient-controls",
        "contextual-local-depth",
        "inverted-mask-window",
        "local-weighted-huber",
        "canonical-weighted-huber",
        "attention-capacity-weighted-huber",
        "weighted-huber-followup",
        "background-excess-depth",
        "background-excess-depth-completion",
    }
    if args.stage in og_stages:
        # Hugging Face FOXES_OG AIA arrays are already scaled to [-1, 1].
        args.aia_pre_normalized = True
        if args.subset_root is None:
            args.subset_root = Path("/data/FOXES_screening/subset-og-quick")
    elif args.subset_root is None:
        args.subset_root = Path("/data/FOXES_screening/subset")
    if args.scheduler_t_max is None:
        with args.base_config.open() as handle:
            base_config = yaml.safe_load(handle)
        args.scheduler_t_max = int(
            base_config["optimizer"]["scheduler"]["T_max"]
        )
    if args.parallel <= 0:
        parser.error("--parallel must be positive")
    if not 0 < args.top_fraction <= 1:
        parser.error("--top-fraction must be in (0, 1]")
    if args.scheduler_t_max <= 0:
        parser.error("--scheduler-t-max must be positive")
    if args.patch_batch_size <= 0:
        parser.error("--patch-batch-size must be positive")
    if args.patch_accumulate_grad_batches <= 0:
        parser.error("--patch-accumulate-grad-batches must be positive")
    if args.run and args.parallel > 1 and args.gpu_ids is None:
        parser.error("parallel training requires explicit --gpu-ids")
    return args


def experiment(
    args: argparse.Namespace,
    name: str,
    *,
    objective: str = "gaussian",
    gaussian_model_type: str | None = None,
    top_fraction: float | None = None,
    scheduler_t_max: int | None = None,
    sparsity_weight: float | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    dropout: float | None = None,
    accumulate_grad_batches: int | None = None,
    beta_nll_beta: float | None = None,
    mean_loss: str | None = None,
    mean_huber_delta: float | None = None,
    class_weighting: str | None = None,
    mean_loss_weighting: str | None = None,
    mask_mode: str | None = None,
    local_window: int | None = None,
    patch_size: int | None = None,
    batch_size: int | None = None,
    embed_dim: int | None = None,
    hidden_dim: int | None = None,
    num_heads: int | None = None,
    num_layers: int | None = None,
    checkpoint_monitor: str | None = None,
    mean_only: bool = False,
    disable_gradient_clipping: bool = False,
) -> Experiment:
    """Create a run spec, inheriting values fixed on the command line."""
    return Experiment(
        name=name,
        sparsity_weight=(
            args.sparsity_weight if sparsity_weight is None else sparsity_weight
        ),
        learning_rate=(
            args.learning_rate if learning_rate is None else learning_rate
        ),
        objective=objective,
        gaussian_model_type=gaussian_model_type,
        top_fraction=(
            args.top_fraction if top_fraction is None else top_fraction
        ),
        scheduler_t_max=(
            getattr(args, "scheduler_t_max", None)
            if scheduler_t_max is None else scheduler_t_max
        ),
        weight_decay=weight_decay,
        dropout=dropout,
        accumulate_grad_batches=accumulate_grad_batches,
        beta_nll_beta=beta_nll_beta,
        mean_loss=mean_loss,
        mean_huber_delta=mean_huber_delta,
        class_weighting=class_weighting,
        mean_loss_weighting=mean_loss_weighting,
        mask_mode=mask_mode,
        local_window=local_window,
        patch_size=patch_size,
        batch_size=batch_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        checkpoint_monitor=checkpoint_monitor,
        mean_only=mean_only,
        disable_gradient_clipping=disable_gradient_clipping,
    )


def stage_values(args: argparse.Namespace) -> list[Experiment]:
    if args.stage == "quantile":
        run_name = (
            f"screen-spatial-quantile-s{args.sparsity_weight:g}"
            f"-top{args.top_fraction:g}"
            f"-lr{args.learning_rate:g}-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        return [
            experiment(
                args,
                run_name,
                objective="quantile",
                accumulate_grad_batches=1,
            ),
        ]
    if args.stage in {"patch-gaussian", "global-gaussian"}:
        # Keep objective, data, seed, and patch-variance parameterization fixed.
        # Each non-baseline run changes exactly one regularizer.
        name_suffix = (
            f"s{args.sparsity_weight:g}-top{args.top_fraction:g}"
            f"-lr{args.learning_rate:g}-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        return [
            experiment(
                args,
                f"screen-patch-base-{name_suffix}-do0.1-wd0.0001",
                weight_decay=1e-4,
                dropout=0.1,
            ),
            experiment(
                args,
                f"screen-patch-strong-wd-{name_suffix}-do0.1-wd0.001",
                weight_decay=1e-3,
                dropout=0.1,
            ),
            experiment(
                args,
                f"screen-patch-dropout-{name_suffix}-do0.2-wd0.0001",
                weight_decay=1e-4,
                dropout=0.2,
            ),
        ]
    if args.stage == "beta-detached":
        name_suffix = (
            f"s{args.sparsity_weight:g}-top{args.top_fraction:g}"
            f"-lr{args.learning_rate:g}-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        return [
            experiment(
                args,
                f"screen-detached-beta0-{name_suffix}",
                beta_nll_beta=0.0,
                accumulate_grad_batches=1,
            ),
            experiment(
                args,
                f"screen-detached-beta1-{name_suffix}",
                beta_nll_beta=1.0,
                accumulate_grad_batches=1,
            ),
        ]
    if args.stage == "og-beta-sparsity":
        data_mode = "full" if args.dataset_root is not None else "subset"
        suffix = (
            f"top{args.top_fraction:g}-lr{args.learning_rate:g}"
            f"-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        return [
            experiment(
                args,
                f"screen-og-{data_mode}-beta{beta:g}-s{sparsity:g}-{suffix}",
                beta_nll_beta=beta,
                sparsity_weight=sparsity,
                accumulate_grad_batches=1,
            )
            for sparsity in args.sparsity_weights
            for beta in (0.0, 1.0)
        ]
    if args.stage == "local-window-beta":
        data_mode = "full" if args.dataset_root is not None else "subset"
        suffix = (
            f"lr{args.learning_rate:g}-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        return [
            experiment(
                args,
                f"screen-og-{data_mode}-local-w{window}-beta{beta:g}"
                f"-nosparse-{suffix}",
                beta_nll_beta=beta,
                mean_loss="beta_nll",
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="local",
                local_window=window,
            )
            for window in args.local_window_sizes
            for beta in (0.0, 1.0)
        ]
    if args.stage == "og-patch-attention-beta":
        data_mode = "full" if args.dataset_root is not None else "subset"
        suffix = (
            f"embed512-layers8-p8-nosparse-lr{args.learning_rate:g}"
            f"-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        attention_specs = (
            ("global", "none", None),
            ("local-w1", "local", 1),
            ("local-w3", "local", 3),
            ("local-w5", "local", 5),
        )
        return [
            experiment(
                args,
                f"screen-ogpatch-{data_mode}-{label}-beta{beta:g}-{suffix}",
                gaussian_model_type="gaussian_nll_og",
                beta_nll_beta=beta,
                mean_loss="beta_nll",
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode=mask_mode,
                local_window=window,
                patch_size=8,
                embed_dim=512,
                num_layers=8,
            )
            for label, mask_mode, window in attention_specs
            for beta in (0.0, 1.0)
        ]
    if args.stage == "patch-size-beta0":
        data_mode = "full" if args.dataset_root is not None else "subset"
        suffix = (
            f"bs{args.patch_batch_size}"
            f"-acc{args.patch_accumulate_grad_batches}"
            f"-lr{args.learning_rate:g}-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        return [
            experiment(
                args,
                f"screen-og-{data_mode}-p4-w{window}-beta0-nosparse-"
                f"{suffix}",
                beta_nll_beta=0.0,
                mean_loss="beta_nll",
                sparsity_weight=0.0,
                accumulate_grad_batches=(
                    args.patch_accumulate_grad_batches
                ),
                mask_mode="local",
                local_window=window,
                patch_size=4,
                batch_size=args.patch_batch_size,
            )
            # w5 matches the p8/w3 physical receptive field; w3 is narrower.
            for window in (5, 3)
        ]
    if args.stage == "architecture-capacity-beta0":
        data_mode = "full" if args.dataset_root is not None else "subset"
        suffix = (
            f"p8-w3-beta0-nosparse-lr{args.learning_rate:g}"
            f"-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        # Ordered to balance expected compute across round-robin GPU queues.
        specs = (
            ("embed512", {"embed_dim": 512}),
            ("layers16", {"num_layers": 16}),
            ("embed64", {"embed_dim": 64}),
            ("embed128", {"embed_dim": 128}),
            ("layers2", {"num_layers": 2}),
            ("layers4", {"num_layers": 4}),
            ("layers12", {"num_layers": 12}),
            ("mlp4096", {"hidden_dim": 4096}),
            ("mlp256", {"hidden_dim": 256}),
            ("mlp512", {"hidden_dim": 512}),
            ("mlp2048", {"hidden_dim": 2048}),
        )
        return [
            experiment(
                args,
                f"screen-og-{data_mode}-{label}-{suffix}",
                beta_nll_beta=0.0,
                mean_loss="beta_nll",
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="local",
                local_window=3,
                patch_size=8,
                **overrides,
            )
            for label, overrides in specs
        ]
    if args.stage == "architecture-combos-beta0":
        data_mode = "full" if args.dataset_root is not None else "subset"
        suffix = (
            f"p8-w3-beta0-nosparse-lr{args.learning_rate:g}"
            f"-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        # Start with the two highest-risk configurations so an OOM is found
        # early. The ordering also balances expected compute across two GPUs.
        # Microbatch x accumulation always preserves effective batch size 64.
        specs = (
            (1024, 16, 8, 8),
            (768, 24, 8, 8),
            (512, 16, 32, 2),
            (768, 16, 16, 4),
            (512, 24, 16, 4),
        )
        return [
            experiment(
                args,
                f"screen-og-{data_mode}-embed{embed_dim}-layers{num_layers}"
                f"-bs{batch_size}-acc{accumulate}-{suffix}",
                beta_nll_beta=0.0,
                mean_loss="beta_nll",
                sparsity_weight=0.0,
                accumulate_grad_batches=accumulate,
                mask_mode="local",
                local_window=3,
                patch_size=8,
                batch_size=batch_size,
                embed_dim=embed_dim,
                num_layers=num_layers,
            )
            for embed_dim, num_layers, batch_size, accumulate in specs
        ]
    if args.stage == "mean-loss":
        name_suffix = (
            f"lr{args.learning_rate:g}-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        return [
            experiment(
                args,
                f"screen-mean-beta1-nosparsity-{name_suffix}",
                beta_nll_beta=1.0,
                mean_loss="beta_nll",
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
            ),
            experiment(
                args,
                f"screen-mean-huber-nosparsity-{name_suffix}",
                beta_nll_beta=1.0,
                mean_loss="huber",
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
            ),
        ]
    if args.stage == "mean-gradient-controls":
        data_mode = "full" if args.dataset_root is not None else "subset"
        suffix = (
            f"{data_mode}-embed512-p8-nosparse-lr{args.learning_rate:g}"
            f"-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        return [
            experiment(
                args,
                f"screen-meanonly-beta0-local-w3-{suffix}",
                beta_nll_beta=0.0,
                mean_loss="mse",
                mean_only=True,
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="local",
                local_window=3,
                patch_size=8,
                embed_dim=512,
            ),
            experiment(
                args,
                f"screen-gaussian-beta0-noclip-local-w1-{suffix}",
                beta_nll_beta=0.0,
                mean_loss="beta_nll",
                disable_gradient_clipping=True,
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="local",
                local_window=1,
                patch_size=8,
                embed_dim=512,
            ),
            experiment(
                args,
                f"screen-gaussian-beta0-noclip-local-w3-{suffix}",
                beta_nll_beta=0.0,
                mean_loss="beta_nll",
                disable_gradient_clipping=True,
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="local",
                local_window=3,
                patch_size=8,
                embed_dim=512,
            ),
            experiment(
                args,
                f"screen-gaussian-beta0-noclip-global-{suffix}",
                beta_nll_beta=0.0,
                mean_loss="beta_nll",
                disable_gradient_clipping=True,
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="none",
                patch_size=8,
                embed_dim=512,
            ),
        ]
    if args.stage == "contextual-local-depth":
        data_mode = "full" if args.dataset_root is not None else "subset"
        suffix = (
            f"{data_mode}-embed512-p8-w3-mse-beta0-nosparse-noclip"
            f"-lr{args.learning_rate:g}-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        runs = [
            experiment(
                args,
                f"screen-local-layers{depth}-{suffix}",
                beta_nll_beta=0.0,
                mean_loss="mse",
                disable_gradient_clipping=True,
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="local",
                local_window=3,
                patch_size=8,
                embed_dim=512,
                num_layers=depth,
            )
            for depth in (1, 2, 4, 8)
        ]
        runs.append(experiment(
            args,
            f"screen-background-excess-layers2-{suffix}",
            gaussian_model_type="gaussian_nll_background_excess",
            beta_nll_beta=0.0,
            mean_loss="mse",
            disable_gradient_clipping=True,
            sparsity_weight=0.0,
            accumulate_grad_batches=1,
            mask_mode="local",
            local_window=3,
            patch_size=8,
            embed_dim=512,
            num_layers=2,
        ))
        return runs
    if args.stage == "inverted-mask-window":
        data_mode = "full" if args.dataset_root is not None else "subset"
        suffix = (
            f"layers8-{data_mode}-embed512-p8-mse-beta0-nosparse-noclip"
            f"-lr{args.learning_rate:g}-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        # A centered discrete square has an odd side length. The requested
        # 16-wide arm therefore uses the nearest symmetric window (17x17)
        # and says so in its immutable run name.
        window_specs = (
            ("w3", 3),
            ("w5", 5),
            ("w9", 9),
            ("requestedw16-effectivew17", 17),
        )
        return [
            experiment(
                args,
                f"screen-inverted-{label}-{suffix}",
                beta_nll_beta=0.0,
                mean_loss="mse",
                disable_gradient_clipping=True,
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="inverted",
                local_window=window,
                patch_size=8,
                embed_dim=512,
                num_layers=8,
                checkpoint_monitor="val_class/X/rmse_dex",
            )
            for label, window in window_specs
        ]
    if args.stage == "local-weighted-huber":
        data_mode = "full" if args.dataset_root is not None else "subset"
        name = (
            f"screen-local-layers8-{data_mode}-embed512-p8-w3"
            f"-huber0.3-fourclass-beta0-nosparse-noclip"
            f"-lr{args.learning_rate:g}-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        return [
            experiment(
                args,
                name,
                beta_nll_beta=0.0,
                mean_loss="huber",
                mean_huber_delta=0.3,
                mean_loss_weighting="four_class_macro",
                disable_gradient_clipping=True,
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="local",
                local_window=3,
                patch_size=8,
                embed_dim=512,
                num_layers=8,
                checkpoint_monitor="val_class/X/rmse_dex",
            )
        ]
    if args.stage == "canonical-weighted-huber":
        data_mode = "full" if args.dataset_root is not None else "subset"
        return [
            experiment(
                args,
                (
                    f"screen-canonical-local-layers8-huber0.15-{label}"
                    f"-{data_mode}-embed512-p8-w3"
                    f"-lr{args.learning_rate:g}-steps{args.max_steps}"
                    f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
                ),
                mean_huber_delta=0.15,
                class_weighting=weighting,
                disable_gradient_clipping=True,
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="local",
                local_window=3,
                patch_size=8,
                embed_dim=512,
                num_layers=8,
                checkpoint_monitor="val_class/X/rmse_dex",
            )
            for label, weighting in (
                ("inverse", "inverse_frequency"),
                ("sqrt", "sqrt_inverse_frequency"),
            )
        ]
    if args.stage == "attention-capacity-weighted-huber":
        data_mode = "full" if args.dataset_root is not None else "subset"
        specs = (
            ("a", 8, 8, 2048),
            ("b", 12, 8, 2048),
            ("c", 12, 16, 2048),
            ("d", 12, 8, 4096),
        )
        return [
            experiment(
                args,
                (
                    f"screen-attn-capacity-{label}-layers{layers}"
                    f"-heads{heads}-mlp{hidden}-{data_mode}"
                    f"-embed512-p8-w3-huber0.15-inverse"
                    f"-nosparse-noclip-lr{args.learning_rate:g}"
                    f"-steps{args.max_steps}"
                    f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
                ),
                mean_huber_delta=0.15,
                class_weighting="inverse_frequency",
                disable_gradient_clipping=True,
                sparsity_weight=0.0,
                batch_size=8,
                accumulate_grad_batches=8,
                mask_mode="local",
                local_window=3,
                patch_size=8,
                embed_dim=512,
                hidden_dim=hidden,
                num_heads=heads,
                num_layers=layers,
                checkpoint_monitor="val_class/macro_mse_dex2",
            )
            for label, layers, heads, hidden in specs
        ]
    if args.stage == "weighted-huber-followup":
        data_mode = "full" if args.dataset_root is not None else "subset"
        # Together with the existing layers8/delta0.3/inverse-frequency run,
        # these are one-factor-at-a-time comparisons. Ordering approximately
        # balances two round-robin GPU queues by transformer depth.
        specs = (
            (8, 0.15, "four_class_macro", "inverse", 3),
            (8, 1.0, "four_class_macro", "inverse", 3),
            (8, 0.15, "sqrt_four_class_macro", "sqrt", 3),
            (8, 1.0, "sqrt_four_class_macro", "sqrt", 3),
            (8, 0.3, "four_class_macro", "inverse", 5),
            (8, 0.3, "sqrt_four_class_macro", "sqrt", 3),
            (4, 0.3, "four_class_macro", "inverse", 3),
            (8, 0.3, "four_class_macro", "inverse", 9),
            (4, 0.3, "sqrt_four_class_macro", "sqrt", 3),
            (2, 0.3, "four_class_macro", "inverse", 3),
            (2, 0.3, "sqrt_four_class_macro", "sqrt", 3),
        )
        return [
            experiment(
                args,
                (
                    f"screen-local-layers{depth}-huber{delta:g}"
                    f"-{weight_label}-{data_mode}-embed512-p8-w{window}"
                    f"-beta0-nosparse-noclip"
                    f"-lr{args.learning_rate:g}-steps{args.max_steps}"
                    f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
                ),
                beta_nll_beta=0.0,
                mean_loss="huber",
                mean_huber_delta=delta,
                mean_loss_weighting=weighting,
                disable_gradient_clipping=True,
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="local",
                local_window=window,
                patch_size=8,
                embed_dim=512,
                num_layers=depth,
                checkpoint_monitor="val_class/X/rmse_dex",
            )
            for depth, delta, weighting, weight_label, window in specs
        ]
    if args.stage in {
        "background-excess-depth",
        "background-excess-depth-completion",
    }:
        data_mode = "full" if args.dataset_root is not None else "subset"
        suffix = (
            f"{data_mode}-embed512-p8-w3-mse-beta0-nosparse-noclip"
            f"-lr{args.learning_rate:g}-steps{args.max_steps}"
            f"-tmax{getattr(args, 'scheduler_t_max', 50)}"
        )
        depths = (
            (4, 8, 1)
            if args.stage == "background-excess-depth-completion"
            else (1, 2, 4, 8)
        )
        # The completion ordering balances two GPU queues: GPU 0 runs 4 then
        # 1 while GPU 1 runs 8. Depth 2 already exists in the contextual run.
        return [
            experiment(
                args,
                f"screen-background-excess-layers{depth}-{suffix}",
                gaussian_model_type="gaussian_nll_background_excess",
                beta_nll_beta=0.0,
                mean_loss="mse",
                disable_gradient_clipping=True,
                sparsity_weight=0.0,
                accumulate_grad_batches=1,
                mask_mode="local",
                local_window=3,
                patch_size=8,
                embed_dim=512,
                num_layers=depth,
            )
            for depth in depths
        ]
    return [
        experiment(args, f"screen-lr-{value:g}", learning_rate=value)
        for value in (3e-5, 1e-4, 3e-4)
    ]


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    commands = []
    for index, run in enumerate(stage_values(args)):
        command = [
            sys.executable, str(HERE / "run_experiment.py"),
            "--name", run.name,
            "--base-config", str(getattr(
                args, "base_config",
                PROJECT_ROOT / "training" / "train_config.yaml",
            )),
            "--runs-root", str(args.runs_root),
            "--learning-rate", str(run.learning_rate),
            "--sparsity-weight", str(run.sparsity_weight),
            "--top-fraction", str(run.top_fraction),
            "--max-steps", str(args.max_steps),
            "--seed", str(args.seed),
        ]
        if args.dataset_root is not None:
            command.extend(("--dataset-root", str(args.dataset_root)))
        else:
            command.extend(("--subset-root", str(args.subset_root)))
        if args.aia_pre_normalized:
            command.append("--aia-pre-normalized")
        if run.objective != "gaussian":
            command.extend(("--objective", run.objective))
        if run.gaussian_model_type is not None:
            command.extend((
                "--gaussian-model-type", run.gaussian_model_type,
            ))
        if run.scheduler_t_max is not None:
            command.extend((
                "--scheduler-t-max", str(run.scheduler_t_max),
            ))
        if args.gpu_ids is not None:
            command.extend((
                "--gpu-id", str(args.gpu_ids[index % len(args.gpu_ids)]),
            ))
        if run.weight_decay is not None:
            command.extend(("--weight-decay", str(run.weight_decay)))
        if run.dropout is not None:
            command.extend(("--dropout", str(run.dropout)))
        if run.accumulate_grad_batches is not None:
            command.extend((
                "--accumulate-grad-batches",
                str(run.accumulate_grad_batches),
            ))
        if run.beta_nll_beta is not None:
            command.extend((
                "--beta-nll-beta", str(run.beta_nll_beta),
            ))
        if run.mean_loss is not None:
            command.extend(("--mean-loss", run.mean_loss))
        if run.mean_huber_delta is not None:
            command.extend((
                "--mean-huber-delta", str(run.mean_huber_delta),
            ))
        if run.class_weighting is not None:
            command.extend(("--class-weighting", run.class_weighting))
        if run.mean_loss_weighting is not None:
            command.extend((
                "--mean-loss-weighting", run.mean_loss_weighting,
            ))
        if run.mean_only:
            command.append("--mean-only")
        if run.disable_gradient_clipping:
            command.append("--disable-gradient-clipping")
        if run.mask_mode is not None:
            command.extend(("--mask-mode", run.mask_mode))
        if run.local_window is not None:
            command.extend(("--local-window", str(run.local_window)))
        if run.patch_size is not None:
            command.extend(("--patch-size", str(run.patch_size)))
        if run.batch_size is not None:
            command.extend(("--batch-size", str(run.batch_size)))
        if run.embed_dim is not None:
            command.extend(("--embed-dim", str(run.embed_dim)))
        if run.hidden_dim is not None:
            command.extend(("--hidden-dim", str(run.hidden_dim)))
        if run.num_heads is not None:
            command.extend(("--num-heads", str(run.num_heads)))
        if run.num_layers is not None:
            command.extend(("--num-layers", str(run.num_layers)))
        if run.checkpoint_monitor is not None:
            command.extend((
                "--checkpoint-monitor", run.checkpoint_monitor,
            ))
        if args.run:
            command.append("--run")
        commands.append(command)
    return commands


def group_commands_by_gpu(commands: list[list[str]]) -> list[list[list[str]]]:
    """Build one sequential command queue per assigned GPU."""
    queues: dict[str, list[list[str]]] = {}
    for command in commands:
        try:
            gpu_id = command[command.index("--gpu-id") + 1]
        except (ValueError, IndexError) as error:
            raise ValueError(
                "Every parallel command must have a --gpu-id"
            ) from error
        queues.setdefault(gpu_id, []).append(command)
    return list(queues.values())


def run_command_queue(
    commands: list[list[str]],
) -> list[subprocess.CalledProcessError]:
    """Run one GPU's experiments sequentially, retaining later runs."""
    failures = []
    for command in commands:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            failures.append(error)
            print(
                f"Run failed with exit status {error.returncode}; "
                "continuing this GPU queue: " + " ".join(command),
                file=sys.stderr,
                flush=True,
            )
    return failures


def snapshot_base_config(base_config: Path, runs_root: Path) -> Path:
    """Copy a content-addressed base config so queued runs cannot drift."""
    content = base_config.resolve().read_bytes()
    digest = hashlib.sha256(content).hexdigest()[:12]
    runs_root = runs_root.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    snapshot = runs_root / f"sweep_base_config_{digest}.yaml"
    if snapshot.exists():
        if snapshot.read_bytes() != content:
            raise RuntimeError(f"Base-config snapshot collision: {snapshot}")
    else:
        snapshot.write_bytes(content)
    return snapshot


def main() -> int:
    args = parse_args()
    preflight_error = (
        dataset_preflight_error(args.dataset_root)
        if args.dataset_root is not None
        else subset_preflight_error(
            args.subset_root,
            source_root=(
                Path("/data/FOXES_OG")
                if args.stage in {
                    "og-beta-sparsity", "local-window-beta",
                    "patch-size-beta0",
                    "architecture-capacity-beta0",
                    "architecture-combos-beta0",
                    "og-patch-attention-beta",
                    "mean-gradient-controls",
                    "contextual-local-depth",
                    "inverted-mask-window",
                    "local-weighted-huber",
                    "canonical-weighted-huber",
                    "attention-capacity-weighted-huber",
                    "weighted-huber-followup",
                    "background-excess-depth",
                    "background-excess-depth-completion",
                }
                else None
            ),
            preset=(
                "og-quick"
                if args.stage in {
                    "og-beta-sparsity", "local-window-beta",
                    "patch-size-beta0",
                    "architecture-capacity-beta0",
                    "architecture-combos-beta0",
                    "og-patch-attention-beta",
                    "mean-gradient-controls",
                    "contextual-local-depth",
                    "inverted-mask-window",
                    "local-weighted-huber",
                    "canonical-weighted-huber",
                    "attention-capacity-weighted-huber",
                    "weighted-huber-followup",
                    "background-excess-depth",
                    "background-excess-depth-completion",
                }
                else "quick"
            ),
            require_sxr_norm=(
                args.stage in {
                    "og-beta-sparsity", "local-window-beta",
                    "patch-size-beta0",
                    "architecture-capacity-beta0",
                    "architecture-combos-beta0",
                    "og-patch-attention-beta",
                    "mean-gradient-controls",
                    "contextual-local-depth",
                    "inverted-mask-window",
                    "local-weighted-huber",
                    "canonical-weighted-huber",
                    "attention-capacity-weighted-huber",
                    "weighted-huber-followup",
                    "background-excess-depth",
                    "background-excess-depth-completion",
                }
            ),
        )
    )
    if preflight_error:
        print(preflight_error, file=sys.stderr, flush=True)
        return 2
    if args.run:
        args.base_config = snapshot_base_config(
            args.base_config, args.runs_root
        )
    commands = build_commands(args)
    for command in commands:
        print(" ".join(command), flush=True)

    if args.run:
        # Materialize every immutable run config before allocating a GPU. This
        # catches collisions and prevents later queue entries from silently
        # inheriting edits made while earlier experiments are training.
        for command in commands:
            config_command = [item for item in command if item != "--run"]
            subprocess.run(config_command, check=True)
        print(f"Prepared all {len(commands)} run configs", flush=True)

    if args.run and args.parallel > 1:
        command_queues = group_commands_by_gpu(commands)
        workers = min(args.parallel, len(command_queues))
        print(
            f"Starting {len(commands)} runs on {len(command_queues)} "
            f"GPU queue(s), with up to {workers} in parallel",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Iterating forces every result/exception to be observed. A queue
            # is sequential, so no GPU receives two concurrent model copies.
            queue_failures = list(
                executor.map(run_command_queue, command_queues)
            )
        failures = [
            failure
            for queue in queue_failures
            for failure in queue
        ]
    else:
        failures = run_command_queue(commands) if args.run else []
    if failures:
        print(
            f"{len(failures)} run(s) failed; all remaining queued runs "
            "were still attempted.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
