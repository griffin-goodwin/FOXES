#!/usr/bin/env python3
"""Validate processed FOXES AIA/SXR pairs and report flare-class balance.

The scan is read-only unless ``--move-invalid-to`` is provided. It checks every
selected ``.npy`` pair for missing/blank/non-finite data and noise-like AIA
dark products, writes a JSON summary plus a CSV of invalid samples, and exits
nonzero when integrity errors are found. By default it scans train, val, and
test in full. When quarantine is enabled, reports are written before any files
are moved.

Example
-------
python data/check_dataset_integrity.py \
    --aia-dir /data/AIA_processed \
    --sxr-dir /data/SXR_processed \
    --report /data/dataset_integrity.json \
    --move-invalid-to /data/FOXES_invalid
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


FLARE_CLASSES = (
    ("quiet", -np.inf, 1e-6),
    ("c", 1e-6, 1e-5),
    ("m", 1e-5, 1e-4),
    ("x", 1e-4, np.inf),
)


@dataclass
class SampleResult:
    split: str
    timestamp: str
    valid: bool
    flare_class: str | None = None
    sxr_flux: float | None = None
    aia_shape: str | None = None
    aia_dtype: str | None = None
    aia_disk_contrast: float | None = None
    negative_pixels: int = 0
    errors: str = ""
    warnings: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aia-dir", type=Path, default=Path("/data/AIA_processed"))
    parser.add_argument("--sxr-dir", type=Path, default=Path("/data/SXR_processed"))
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Split subdirectories to scan (default: train val test).",
    )
    parser.add_argument("--channels", type=int, default=7)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--min-disk-contrast", type=float, default=2.0,
        help=(
            "Reject noise/dark-frame products whose median center-to-corner "
            "contrast across AIA channels is below this value (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--noise-check-downsample", type=int, default=8,
        help="Spatial stride used by the noise morphology check (default: 8).",
    )
    parser.add_argument(
        "--disable-noise-check", action="store_true",
        help="Disable the centered-solar-disk morphology check.",
    )
    parser.add_argument(
        "--workers", type=int, default=min(48, os.cpu_count() or 1),
        help="Concurrent file readers.",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("dataset_integrity_report.json"),
    )
    parser.add_argument(
        "--invalid-csv", type=Path, default=None,
        help="Defaults to <report stem>_invalid.csv.",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Diagnostic only: scan at most this many evenly spaced pairs per split.",
    )
    parser.add_argument(
        "--move-invalid-to", type=Path, default=None, metavar="DIRECTORY",
        help=(
            "After writing the reports, move both files belonging to every invalid "
            "pair into DIRECTORY/{aia,sxr}/<split>. Disabled by default."
        ),
    )
    parser.add_argument(
        "--move-from-csv", type=Path, default=None, metavar="INVALID_CSV",
        help=(
            "Skip the scan and quarantine timestamps listed in an invalid CSV "
            "created by an earlier run. Requires --move-invalid-to."
        ),
    )
    return parser.parse_args()


def flare_class(flux: float) -> str:
    for name, lower, upper in FLARE_CLASSES:
        if lower <= flux < upper:
            return name
    raise ValueError(f"cannot classify flux {flux!r}")


def median_solar_disk_contrast(
    aia: np.ndarray, downsample: int = 8
) -> float:
    """Measure whether a stack contains a centered solar disk rather than noise.

    FOXES products are centered full-disk images. Real AIA images have much
    greater mean signal inside the disk than in the far corners, whereas the
    daily dark-current products are spatially noise-like and have a ratio near
    one. Taking the median over wavelengths tolerates faint 94/131 channels and
    bright off-limb structure without keying the check to a particular time.
    """
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    if aia.ndim != 3 or min(aia.shape[-2:]) < 16:
        return float("nan")

    sampled = np.asarray(
        aia[:, ::downsample, ::downsample], dtype=np.float64
    )
    height, width = sampled.shape[-2:]
    y, x = np.ogrid[:height, :width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    scale = min(height, width)
    disk_mask = radius <= 0.35 * scale
    corner_mask = radius >= 0.48 * scale

    contrasts = []
    for channel in sampled:
        # Keep a few already-reported negative pixels from destabilizing the
        # score while leaving the original product untouched.
        channel = np.where(
            np.isfinite(channel), np.maximum(channel, 0), np.nan
        )
        disk_mean = float(np.nanmean(channel[disk_mask]))
        corner_mean = float(np.nanmean(channel[corner_mask]))
        if not np.isfinite(disk_mean) or not np.isfinite(corner_mean):
            continue
        if corner_mean <= np.finfo(np.float64).tiny:
            contrast = np.inf if disk_mean > 0 else 1.0
        else:
            contrast = disk_mean / corner_mean
        contrasts.append(contrast)

    return float(np.median(contrasts)) if contrasts else float("nan")


def inspect_sample(
    split: str,
    timestamp: str,
    aia_path: Path | None,
    sxr_path: Path | None,
    expected_shape: tuple[int, int, int],
    min_disk_contrast: float | None = 2.0,
    noise_check_downsample: int = 8,
) -> SampleResult:
    errors: list[str] = []
    warnings: list[str] = []
    result = SampleResult(split=split, timestamp=timestamp, valid=False)

    if aia_path is None:
        errors.append("missing_aia")
    if sxr_path is None:
        errors.append("missing_sxr")

    if aia_path is not None:
        try:
            aia = np.load(aia_path, mmap_mode="r", allow_pickle=False)
            result.aia_shape = "x".join(str(value) for value in aia.shape)
            result.aia_dtype = str(aia.dtype)
            if aia.shape != expected_shape:
                errors.append(f"wrong_aia_shape:{aia.shape}")
            elif not np.issubdtype(aia.dtype, np.number):
                errors.append(f"nonnumeric_aia_dtype:{aia.dtype}")
            else:
                for channel_index, channel in enumerate(aia):
                    finite = np.isfinite(channel)
                    if not finite.all():
                        errors.append(
                            f"nonfinite_aia_channel_{channel_index}:{int((~finite).sum())}"
                        )
                    finite_values = channel[finite]
                    if finite_values.size == 0:
                        errors.append(f"blank_aia_channel_{channel_index}:no_finite_pixels")
                        continue
                    channel_min = float(finite_values.min())
                    channel_max = float(finite_values.max())
                    if channel_min == channel_max:
                        errors.append(
                            f"blank_aia_channel_{channel_index}:constant_{channel_min:g}"
                        )
                    if channel_max <= 0:
                        errors.append(f"blank_aia_channel_{channel_index}:nonpositive")
                    result.negative_pixels += int((finite_values < 0).sum())
                if result.negative_pixels:
                    warnings.append(f"negative_aia_pixels:{result.negative_pixels}")
                if min_disk_contrast is not None:
                    result.aia_disk_contrast = median_solar_disk_contrast(
                        aia, downsample=noise_check_downsample
                    )
                    if (
                        np.isfinite(result.aia_disk_contrast)
                        and result.aia_disk_contrast < min_disk_contrast
                    ):
                        errors.append(
                            "noise_like_aia:median_disk_contrast_"
                            f"{result.aia_disk_contrast:.4g}_below_"
                            f"{min_disk_contrast:g}"
                        )
        except (OSError, ValueError, EOFError) as error:
            errors.append(f"unreadable_aia:{type(error).__name__}:{error}")

    if sxr_path is not None:
        try:
            sxr = np.load(sxr_path, allow_pickle=False)
            if sxr.size != 1:
                errors.append(f"nonscalar_sxr:{sxr.shape}")
            elif not np.issubdtype(sxr.dtype, np.number):
                errors.append(f"nonnumeric_sxr_dtype:{sxr.dtype}")
            else:
                flux = float(sxr.reshape(-1)[0])
                result.sxr_flux = flux
                if not np.isfinite(flux):
                    errors.append(f"nonfinite_sxr:{flux}")
                elif flux <= 0:
                    errors.append(f"nonpositive_sxr:{flux}")
                else:
                    result.flare_class = flare_class(flux)
        except (OSError, ValueError, EOFError) as error:
            errors.append(f"unreadable_sxr:{type(error).__name__}:{error}")

    result.errors = ";".join(errors)
    result.warnings = ";".join(warnings)
    result.valid = not errors
    return result


def paths_by_stem(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in directory.glob("*.npy")}


def select_stems(stems: list[str], max_files: int | None) -> list[str]:
    if max_files is None or max_files >= len(stems):
        return stems
    indices = np.linspace(0, len(stems) - 1, max_files, dtype=int)
    return [stems[index] for index in indices]


def validate_quarantine_path(quarantine_dir: Path, source_dirs: tuple[Path, Path]) -> Path:
    """Reject quarantine locations that overlap either source tree."""
    quarantine = quarantine_dir.resolve()
    for source_dir in source_dirs:
        source = source_dir.resolve()
        if (
            quarantine == source
            or quarantine.is_relative_to(source)
            or source.is_relative_to(quarantine)
        ):
            raise ValueError(
                f"Quarantine directory {quarantine} must not contain or be inside "
                f"source directory {source}"
            )
    return quarantine


def quarantine_invalid_pairs(
    invalid_results: list[SampleResult],
    aia_dir: Path,
    sxr_dir: Path,
    quarantine_dir: Path,
) -> list[dict[str, str]]:
    """Move existing members of invalid pairs without overwriting destinations."""
    moves: list[dict[str, str]] = []
    for result in invalid_results:
        paths = (
            ("aia", aia_dir / result.split / f"{result.timestamp}.npy"),
            ("sxr", sxr_dir / result.split / f"{result.timestamp}.npy"),
        )
        destinations = {
            kind: quarantine_dir / kind / result.split / source.name
            for kind, source in paths
        }
        for kind, source in paths:
            destination = destinations[kind]
            if not source.is_file() and destination.is_file():
                moves.append({
                    "split": result.split,
                    "timestamp": result.timestamp,
                    "kind": kind,
                    "source": str(source),
                    "destination": str(destination),
                    "status": "already_quarantined",
                    "error": "",
                })
            elif not source.is_file() and f"missing_{kind}" in result.errors:
                moves.append({
                    "split": result.split,
                    "timestamp": result.timestamp,
                    "kind": kind,
                    "source": str(source),
                    "destination": str(destination),
                    "status": "source_absent",
                    "error": "",
                })
            elif not source.is_file():
                moves.append({
                    "split": result.split,
                    "timestamp": result.timestamp,
                    "kind": kind,
                    "source": str(source),
                    "destination": str(destination),
                    "status": "source_missing",
                    "error": "source no longer exists and was not found in quarantine",
                })
        existing = [
            (kind, source, destinations[kind])
            for kind, source in paths if source.is_file()
        ]
        collisions = [destination for _, _, destination in existing if destination.exists()]
        if collisions:
            error = "destination_exists:" + ",".join(str(path) for path in collisions)
            for kind, source, destination in existing:
                moves.append({
                    "split": result.split,
                    "timestamp": result.timestamp,
                    "kind": kind,
                    "source": str(source),
                    "destination": str(destination),
                    "status": "not_moved",
                    "error": error,
                })
            continue

        for kind, source, destination in existing:
            status = "moved"
            error = ""
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
            except OSError as move_error:
                status = "move_failed"
                error = f"{type(move_error).__name__}:{move_error}"
            moves.append({
                "split": result.split,
                "timestamp": result.timestamp,
                "kind": kind,
                "source": str(source),
                "destination": str(destination),
                "status": status,
                "error": error,
            })
    return moves


def read_invalid_csv(path: Path) -> list[SampleResult]:
    """Load and validate the identifying fields from a prior invalid CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Invalid-sample CSV not found: {path}")
    results: list[SampleResult] = []
    seen: set[tuple[str, str]] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "timestamp", "errors"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            split = row["split"]
            timestamp = row["timestamp"]
            if not split or Path(split).name != split or split in {".", ".."}:
                raise ValueError(f"Unsafe split value on CSV row {row_number}: {split!r}")
            if (
                not timestamp
                or Path(timestamp).name != timestamp
                or timestamp in {".", ".."}
            ):
                raise ValueError(
                    f"Unsafe timestamp value on CSV row {row_number}: {timestamp!r}"
                )
            key = (split, timestamp)
            if key in seen:
                continue
            seen.add(key)
            results.append(SampleResult(
                split=split,
                timestamp=timestamp,
                valid=False,
                errors=row["errors"],
            ))
    return results


def write_moves_csv(path: Path, moves: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = (
            "split", "timestamp", "kind", "source", "destination", "status", "error"
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(moves)


def moves_failed(moves: list[dict[str, str]]) -> bool:
    successful = {"moved", "already_quarantined", "source_absent"}
    return any(move["status"] not in successful for move in moves)


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    if args.min_disk_contrast <= 0:
        raise ValueError("--min-disk-contrast must be positive")
    if args.noise_check_downsample <= 0:
        raise ValueError("--noise-check-downsample must be positive")
    if args.move_from_csv is not None and args.move_invalid_to is None:
        raise ValueError("--move-from-csv requires --move-invalid-to")
    quarantine_dir = None
    if args.move_invalid_to is not None:
        quarantine_dir = validate_quarantine_path(
            args.move_invalid_to, (args.aia_dir, args.sxr_dir)
        )
    if args.move_from_csv is not None:
        invalid_results = read_invalid_csv(args.move_from_csv)
        moves = quarantine_invalid_pairs(
            invalid_results, args.aia_dir, args.sxr_dir, quarantine_dir
        )
        moves_csv = args.move_from_csv.with_name(
            f"{args.move_from_csv.stem}_moves.csv"
        )
        write_moves_csv(moves_csv, moves)
        moved_count = sum(move["status"] == "moved" for move in moves)
        already_count = sum(
            move["status"] == "already_quarantined" for move in moves
        )
        print(f"Loaded {len(invalid_results):,} invalid timestamps from {args.move_from_csv}")
        print(
            f"Quarantine: moved {moved_count:,} files; "
            f"already quarantined {already_count:,}"
        )
        print(f"Move log: {moves_csv}")
        return 2 if moves_failed(moves) else 0

    expected_shape = (args.channels, args.height, args.width)
    all_results: list[SampleResult] = []
    split_reports: dict[str, dict[str, object]] = {}

    for split in args.splits:
        aia_split = args.aia_dir / split
        sxr_split = args.sxr_dir / split
        if not aia_split.is_dir() or not sxr_split.is_dir():
            missing = []
            if not aia_split.is_dir():
                missing.append(str(aia_split))
            if not sxr_split.is_dir():
                missing.append(str(sxr_split))
            split_reports[split] = {
                "valid": False,
                "errors": [f"missing_directory:{path}" for path in missing],
            }
            print(f"{split}: missing directory: {', '.join(missing)}", flush=True)
            continue

        aia_paths = paths_by_stem(aia_split)
        sxr_paths = paths_by_stem(sxr_split)
        all_stems = sorted(set(aia_paths) | set(sxr_paths))
        selected_stems = select_stems(all_stems, args.max_files)
        print(
            f"{split}: scanning {len(selected_stems):,} of {len(all_stems):,} timestamps",
            flush=True,
        )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(tqdm(
                executor.map(
                    lambda stem: inspect_sample(
                        split, stem, aia_paths.get(stem), sxr_paths.get(stem),
                        expected_shape,
                        min_disk_contrast=(
                            None if args.disable_noise_check
                            else args.min_disk_contrast
                        ),
                        noise_check_downsample=args.noise_check_downsample,
                    ),
                    selected_stems,
                ),
                total=len(selected_stems),
                desc=f"Checking {split}",
                unit="pair",
            ))
        all_results.extend(results)

        valid_results = [result for result in results if result.valid]
        invalid_results = [result for result in results if not result.valid]
        warning_results = [result for result in results if result.warnings]
        class_counts = Counter(
            result.flare_class for result in valid_results if result.flare_class
        )
        split_reports[split] = {
            "valid": not invalid_results,
            "available_timestamps": len(all_stems),
            "scanned_timestamps": len(results),
            "valid_samples": len(valid_results),
            "invalid_samples": len(invalid_results),
            "samples_with_warnings": len(warning_results),
            "flare_class_counts": {
                name: class_counts.get(name, 0) for name, _, _ in FLARE_CLASSES
            },
        }
        print(
            f"{split}: valid={len(valid_results):,}, invalid={len(invalid_results):,}, "
            f"warnings={len(warning_results):,}, classes={dict(class_counts)}",
            flush=True,
        )

    invalid_results = [result for result in all_results if not result.valid]
    report = {
        "valid": not invalid_results and all(
            split_report.get("valid", False) for split_report in split_reports.values()
        ),
        "full_scan": args.max_files is None,
        "expected_aia_shape": expected_shape,
        "noise_check": {
            "enabled": not args.disable_noise_check,
            "min_disk_contrast": args.min_disk_contrast,
            "downsample": args.noise_check_downsample,
        },
        "aia_dir": str(args.aia_dir.resolve()),
        "sxr_dir": str(args.sxr_dir.resolve()),
        "splits": split_reports,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    invalid_csv = args.invalid_csv or args.report.with_name(
        f"{args.report.stem}_invalid.csv"
    )
    invalid_csv.parent.mkdir(parents=True, exist_ok=True)
    with invalid_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SampleResult.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(result) for result in invalid_results)

    print(f"JSON report: {args.report}")
    print(f"Invalid-sample CSV: {invalid_csv}")
    move_failed = False
    if quarantine_dir is not None:
        # The report and invalid CSV intentionally exist before this first move.
        moves = quarantine_invalid_pairs(
            invalid_results, args.aia_dir, args.sxr_dir, quarantine_dir
        )
        moves_csv = args.report.with_name(f"{args.report.stem}_moves.csv")
        write_moves_csv(moves_csv, moves)
        move_failed = moves_failed(moves)
        report["quarantine"] = {
            "directory": str(quarantine_dir),
            "files_moved": sum(move["status"] == "moved" for move in moves),
            "files_not_moved": sum(move["status"] != "moved" for move in moves),
            "moves_csv": str(moves_csv),
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"Quarantine: moved {report['quarantine']['files_moved']} files to "
            f"{quarantine_dir}"
        )
        print(f"Move log: {moves_csv}")
    if args.max_files is not None:
        print("NOTE: --max-files was used; this is not a full integrity check.")
    if move_failed:
        return 2
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
