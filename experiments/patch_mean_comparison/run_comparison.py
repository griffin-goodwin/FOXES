#!/usr/bin/env python3
"""Validate and optionally launch the patch-mean experiment arms."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CONFIGS = {
    "original": HERE / "original.yaml",
    "multiplier": HERE / "multiplier.yaml",
    "positive": HERE / "positive.yaml",
}
SUBSET_ROOT = Path("/data/FOXES_screening/subset-og-quick")


def _merge(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    """Resolve the same relative ``base_config`` inheritance as train.py."""
    path = path.resolve()
    if path in ancestors:
        raise ValueError(f"Circular base_config chain ending at {path}")
    with path.open() as stream:
        config = yaml.safe_load(stream) or {}
    base_reference = config.pop("base_config", None)
    if base_reference is None:
        return config
    base_path = Path(base_reference).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return _merge(load_config(base_path, (*ancestors, path)), config)


def _comparison_payload(config: dict) -> dict:
    """Remove intentionally arm-specific settings and metadata."""
    payload = deepcopy(config)
    payload.pop("mean_parameterization")
    payload["data"].pop("checkpoints_dir")
    payload.pop("wandb")
    return payload


def validate_matched_configs() -> dict[str, dict]:
    """Fail before allocating GPUs if the arms are not controlled."""
    configs = {name: load_config(path) for name, path in CONFIGS.items()}
    for name, config in configs.items():
        if config.get("mean_parameterization") != name:
            raise ValueError(
                f"{CONFIGS[name]} must set mean_parameterization: {name}"
            )
        if config.get("uncertainty", {}).get("enabled") is not False:
            raise ValueError(f"{CONFIGS[name]} must disable uncertainty")
        data = config.get("data", {})
        expected_paths = {
            "aia_dir": SUBSET_ROOT / "AIA_processed",
            "sxr_dir": SUBSET_ROOT / "SXR_processed",
            "sxr_norm_path": (
                SUBSET_ROOT / "SXR_processed" / "normalized_sxr.npy"
            ),
        }
        for key, expected in expected_paths.items():
            if Path(data.get(key, "")) != expected:
                raise ValueError(
                    f"{CONFIGS[name]} must use {key}: {expected}"
                )
        if data.get("aia_norm_path") is not None:
            raise ValueError(
                f"{CONFIGS[name]} must not renormalize FOXES_OG AIA arrays"
            )
    reference = _comparison_payload(configs["original"])
    for name, config in configs.items():
        if _comparison_payload(config) != reference:
            raise ValueError(
                f"Experiment arm {name!r} differs outside "
                "mean_parameterization and output metadata"
            )
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("all", "both", *CONFIGS), default="all",
        help="Run all matched arms or only one arm (default: all).",
    )
    parser.add_argument(
        "--gpu-ids", default=None,
        help=(
            "Comma-separated physical GPU IDs. With --parallel, provide one "
            "per arm; otherwise the first ID is reused."
        ),
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run multiple arms concurrently, with one GPU per arm.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Launch training. Without this flag, only validate and print.",
    )
    parser.add_argument(
        "--python", type=Path, default=None,
        help="Python interpreter with torch installed (auto-detected by default).",
    )
    return parser.parse_args()


def _gpu_ids(value: str | None) -> list[str]:
    if value is None:
        return []
    values = [item.strip() for item in value.split(",")]
    if not values or any(not item.isdigit() for item in values):
        raise ValueError("--gpu-ids must be comma-separated nonnegative integers")
    if len(set(values)) != len(values):
        raise ValueError("--gpu-ids must not contain duplicates")
    return values


def _run(command: list[str], gpu_id: str | None) -> None:
    environment = os.environ.copy()
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _training_python(requested: Path | None) -> Path:
    """Find a Python interpreter that can import the training dependencies."""
    candidates = []
    if requested is not None:
        candidates.append(requested.expanduser())
    candidates.append(Path(sys.executable))
    executable = Path(sys.executable).resolve()
    if len(executable.parents) >= 2:
        candidates.append(
            executable.parents[1] / "envs" / "foxes" / "bin" / "python"
        )
    checked = []
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        checked.append(str(candidate))
        if not candidate.is_file():
            continue
        result = subprocess.run(
            [
                str(candidate), "-c",
                "import torch, pytorch_lightning",
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return candidate
    raise RuntimeError(
        "No Python interpreter with torch and pytorch_lightning was found. "
        "Activate the foxes environment or pass --python. Checked: "
        + ", ".join(checked)
    )


def main() -> int:
    args = parse_args()
    validate_matched_configs()
    training_python = _training_python(args.python)
    arms = list(CONFIGS) if args.arm in {"all", "both"} else [args.arm]
    gpu_ids = _gpu_ids(args.gpu_ids)
    if args.parallel and len(arms) > 1 and len(gpu_ids) < len(arms):
        raise ValueError("Parallel comparison requires one --gpu-ids entry per arm")

    jobs = []
    for index, arm in enumerate(arms):
        command = [
            str(training_python),
            str(PROJECT_ROOT / "training" / "train.py"),
            "--config",
            str(CONFIGS[arm]),
        ]
        gpu_id = (
            gpu_ids[index] if args.parallel and gpu_ids
            else gpu_ids[0] if gpu_ids else None
        )
        prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
        print(f"[{arm}] {prefix}{' '.join(command)}", flush=True)
        jobs.append((command, gpu_id))

    if not args.run:
        print("Validated matched configs; add --run to launch training.")
        return 0
    required_data = [
        SUBSET_ROOT / kind / split
        for kind in ("AIA_processed", "SXR_processed")
        for split in ("train", "val", "test")
    ] + [SUBSET_ROOT / "SXR_processed" / "normalized_sxr.npy"]
    missing = [path for path in required_data if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "FOXES_OG screening subset is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    if args.parallel and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(_run, *job) for job in jobs]
            for future in futures:
                future.result()
    else:
        for job in jobs:
            _run(*job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
