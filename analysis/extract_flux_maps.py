#!/usr/bin/env python3
"""
Extract Flux Maps from AIA Data

Scans a processed AIA directory for .npy files within a specified time range,
runs the trained ViTLocal model, and saves per-patch flux contribution maps.

Output files are saved as comma-delimited CSVs named by timestamp, matching
the format produced by inference.py (64x64 grid of patch-level flux values).

Usage:
    python analysis/extract_flux_maps.py --config analysis/extract_flux_maps_config.yaml
    python analysis/extract_flux_maps.py --config analysis/extract_flux_maps_config.yaml \
        --start 2023-08-01T00:00:00 --end 2023-08-03T00:00:00
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.models.vit_patch_model_local import ViTLocal


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExtractFluxMapsConfig:
    """Configuration for flux map extraction."""

    # Paths
    aia_dir: str = ""
    output_dir: str = ""
    checkpoint_path: str = ""

    # Time range (ISO 8601)
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # Model / grid
    wavelengths: Tuple[int, ...] = (94, 131, 171, 193, 211, 304, 335)
    input_size: int = 512
    patch_size: int = 8

    # Inference
    batch_size: int = 10
    num_workers: int = 4
    skip_existing: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "ExtractFluxMapsConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Flatten one level of nesting (time_range, model, inference sections)
        flat: dict = {}
        for key, val in raw.items():
            if isinstance(val, dict):
                flat.update(val)
            else:
                flat[key] = val

        # Normalize time_range sub-keys
        if "start" in flat:
            flat["start_time"] = flat.pop("start")
        if "end" in flat:
            flat["end_time"] = flat.pop("end")

        # Lists → tuples for tuple-typed fields
        if "wavelengths" in flat and isinstance(flat["wavelengths"], list):
            flat["wavelengths"] = tuple(flat["wavelengths"])

        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in flat.items() if k in valid and v is not None})


# =============================================================================
# Dataset
# =============================================================================

class AIAOnlyDataset(Dataset):
    """
    Minimal dataset that loads AIA .npy files from an explicit list of paths.
    No SXR data required — designed for flux map extraction only.
    """

    ALL_WAVELENGTHS = [94, 131, 171, 193, 211, 304, 335]

    def __init__(self, paths: List[Path], wavelengths: Tuple[int, ...]):
        self.paths = paths
        self.indices = [self.ALL_WAVELENGTHS.index(w) for w in wavelengths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        try:
            arr = np.load(path)           # (7, H, W)
            arr = arr[self.indices]       # select requested wavelengths
        except Exception as e:
            print(f"\nWarning: could not load {path}: {e}. Returning zeros.")
            arr = np.zeros((len(self.indices), 512, 512), dtype=np.float32)

        tensor = torch.tensor(arr, dtype=torch.float32).permute(1, 2, 0)  # (H, W, C)
        return tensor, torch.tensor(0.0)  # dummy SXR scalar


# =============================================================================
# Helpers
# =============================================================================

def _normalize_timestamp(stem: str) -> str:
    """Handle filenames that use underscores instead of colons in the time part."""
    if "T" in stem:
        date_part, time_part = stem.split("T", 1)
        return f"{date_part}T{time_part.replace('_', ':')}"
    return stem


def scan_aia_files(
    aia_dir: Path,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> List[Tuple[Path, str]]:
    """
    Search aia_dir (including train/, val/, test/ subdirectories) for .npy files
    whose timestamp falls in [start, end].
    Returns a sorted list of (path, stem) pairs.
    """
    results = []
    for f in sorted(aia_dir.rglob("*.npy")):
        ts_str = _normalize_timestamp(f.stem)
        try:
            dt = pd.to_datetime(ts_str)
        except Exception:
            continue
        if start is not None and dt < start:
            continue
        if end is not None and dt > end:
            continue
        results.append((f, f.stem))
    return results


def load_model(checkpoint_path: str) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loc = "GPU" if device.type == "cuda" else "CPU"
    print(f"Loading ViTLocal checkpoint from {checkpoint_path} ({loc})...")
    model = ViTLocal.load_from_checkpoint(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.eval()
    return model


# =============================================================================
# Main extraction logic
# =============================================================================

def extract_flux_maps(cfg: ExtractFluxMapsConfig) -> None:
    aia_dir = Path(cfg.aia_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not aia_dir.is_dir():
        raise FileNotFoundError(f"AIA directory not found: {aia_dir}")

    start = pd.Timestamp(cfg.start_time) if cfg.start_time else None
    end = pd.Timestamp(cfg.end_time) if cfg.end_time else None

    time_desc = f"[{start or 'beginning'}, {end or 'end'}]"
    print(f"Scanning {aia_dir} for AIA files in {time_desc}...")

    available = scan_aia_files(aia_dir, start, end)
    print(f"  Found {len(available)} file(s) in range")

    if not available:
        print("No AIA files found in the specified time range. Exiting.")
        return

    if cfg.skip_existing:
        before = len(available)
        available = [
            (p, ts) for p, ts in available
            if not (output_dir / f"{ts}.npy").exists()
        ]
        skipped = before - len(available)
        if skipped:
            print(f"  Skipping {skipped} already-computed flux map(s) (skip_existing=true)")

    if not available:
        print("All flux maps already exist for this time range. Exiting.")
        return

    print(f"  Extracting flux maps for {len(available)} file(s)")

    paths, timestamps = zip(*available)
    dataset = AIAOnlyDataset(list(paths), cfg.wavelengths)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
    )

    model = load_model(cfg.checkpoint_path)
    device = next(model.parameters()).device

    grid_h = cfg.input_size // cfg.patch_size
    grid_w = cfg.input_size // cfg.patch_size

    sample_idx = 0
    failed = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting", unit="batch"):
            aia_imgs, _ = batch
            batch_n = aia_imgs.shape[0]
            aia_imgs = aia_imgs.to(device)

            # ViTLocal with return_attention=False returns (predictions, flux_contributions)
            pred = model(aia_imgs, return_attention=False)

            if isinstance(pred, tuple) and len(pred) >= 2:
                flux_contributions = pred[1]
            else:
                print(
                    f"\nWarning: model did not return flux contributions for batch "
                    f"starting at index {sample_idx}. Skipping."
                )
                sample_idx += batch_n
                failed += batch_n
                continue

            for i in range(batch_n):
                ts = timestamps[sample_idx]
                flux_map = (
                    flux_contributions[i]
                    .detach()
                    .cpu()
                    .reshape(grid_h, grid_w)
                    .numpy()
                )
                out_path = output_dir / f"{ts}.npy"
                np.save(str(out_path), flux_map)
                sample_idx += 1

            del aia_imgs, pred, flux_contributions
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    n_saved = len(available) - failed
    print(f"\nDone. {n_saved} flux map(s) saved to {output_dir}")
    if failed:
        print(f"  {failed} sample(s) failed — check warnings above.")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-patch flux contribution maps from AIA data within a time span"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--start", help="Start time override (ISO 8601, e.g. 2023-08-01T00:00:00)"
    )
    parser.add_argument(
        "--end", help="End time override (ISO 8601, e.g. 2023-08-03T00:00:00)"
    )
    args = parser.parse_args()

    cfg = ExtractFluxMapsConfig.from_yaml(args.config)
    if args.start:
        cfg.start_time = args.start
    if args.end:
        cfg.end_time = args.end

    print("Configuration:")
    print(f"  aia_dir:         {cfg.aia_dir}")
    print(f"  output_dir:      {cfg.output_dir}")
    print(f"  checkpoint_path: {cfg.checkpoint_path}")
    print(f"  start_time:      {cfg.start_time or '(none — full range)'}")
    print(f"  end_time:        {cfg.end_time or '(none — full range)'}")
    print(f"  wavelengths:     {cfg.wavelengths}")
    print(f"  batch_size:      {cfg.batch_size}")
    print(f"  num_workers:     {cfg.num_workers}")
    print(f"  skip_existing:   {cfg.skip_existing}")
    print()

    extract_flux_maps(cfg)


if __name__ == "__main__":
    main()
