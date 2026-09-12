#!/usr/bin/env python3
"""Create reproducible time-stratified FOXES screening subsets using symlinks.

No image arrays are copied or modified. Selection is stratified jointly by
flare class and calendar month so rare events and the full time span remain
represented. Each split gets a CSV manifest and the output root gets a JSON
summary containing exact counts and provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


CLASSES = ("quiet", "c", "m", "x")
DEFAULT_COUNTS = {
    "train": {"quiet": 12_000, "c": 8_000, "m": 4_000, "x": None},
    "val": {"quiet": 2_000, "c": 2_000, "m": 1_000, "x": None},
    "test": {"quiet": 1_000, "c": 2_000, "m": 1_000, "x": None},
}
QUICK_COUNTS = {
    # Roughly retain the full training-set class proportions. Keeping every X
    # sample while shrinking the majority classes would itself be aggressive
    # oversampling and would make the loss-weight comparison misleading.
    "train": {"quiet": 10_000, "c": 3_200, "m": 1_000, "x": 50},
    # Validation and test are relatively cheap and must keep their natural
    # distribution for aggregate NLL/coverage to remain interpretable.
    "val": {"quiet": None, "c": None, "m": None, "x": None},
    "test": {"quiet": None, "c": None, "m": None, "x": None},
}
OG_QUICK_COUNTS = {
    # A 14,250-sample training subset matching the FOXES_OG training class
    # proportions (11,712 quiet / 47,824 C / 16,469 M / 1,804 X). This keeps
    # the beta/sparsity screen from becoming an accidental resampling study.
    "train": {"quiet": 2_145, "c": 8_759, "m": 3_016, "x": 330},
    "val": {"quiet": None, "c": None, "m": None, "x": None},
    "test": {"quiet": None, "c": None, "m": None, "x": None},
}
COUNT_PRESETS = {
    "standard": DEFAULT_COUNTS,
    "quick": QUICK_COUNTS,
    "og-quick": OG_QUICK_COUNTS,
}


def parse_count_spec(value: str) -> dict[str, int | None]:
    """Parse ``quiet=12000,c=8000,m=4000,x=all``."""
    parsed: dict[str, int | None] = {}
    for item in value.split(","):
        try:
            name, raw_count = item.split("=", 1)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"Invalid count item {item!r}; expected class=count"
            ) from error
        name = name.strip().lower()
        if name not in CLASSES:
            raise argparse.ArgumentTypeError(f"Unknown flare class {name!r}")
        raw_count = raw_count.strip().lower()
        if raw_count == "all":
            parsed[name] = None
        else:
            try:
                count = int(raw_count)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"Invalid count for {name}: {raw_count!r}"
                ) from error
            if count <= 0:
                raise argparse.ArgumentTypeError("Subset counts must be positive")
            parsed[name] = count
    missing = set(CLASSES).difference(parsed)
    if missing:
        raise argparse.ArgumentTypeError(f"Missing class counts: {sorted(missing)}")
    return parsed


def format_count_spec(counts: dict[str, int | None]) -> str:
    return ",".join(
        f"{name}={'all' if counts[name] is None else counts[name]}"
        for name in CLASSES
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aia-dir", type=Path, default=Path("/data/AIA_processed"))
    parser.add_argument("--sxr-dir", type=Path, default=Path("/data/SXR_processed"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/data/FOXES_screening/subset"),
    )
    parser.add_argument(
        "--preset", choices=tuple(COUNT_PRESETS), default="standard",
        help=(
            "Use 'quick' for the new SDOML distribution or 'og-quick' for "
            "FOXES_OG. Both retain complete validation and test splits."
        ),
    )
    for split in ("train", "val", "test"):
        default = format_count_spec(DEFAULT_COUNTS[split])
        parser.add_argument(
            f"--{split}-counts", type=parse_count_spec,
            default=None,
            metavar="COUNTS",
            help=f"Override the preset for this split. Standard: {default}",
        )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    preset_counts = COUNT_PRESETS[args.preset]
    for split in ("train", "val", "test"):
        if getattr(args, f"{split}_counts") is None:
            setattr(args, f"{split}_counts", dict(preset_counts[split]))
    return args


def classify_flux(flux: float) -> str:
    if not np.isfinite(flux) or flux <= 0:
        raise ValueError(f"Invalid SXR flux {flux!r}")
    if flux < 1e-6:
        return "quiet"
    if flux < 1e-5:
        return "c"
    if flux < 1e-4:
        return "m"
    return "x"


def stable_rng(seed: int, split: str, flare_class: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{split}:{flare_class}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def proportional_month_sample(
    rows: list[dict[str, object]], count: int | None, rng: np.random.Generator
) -> list[dict[str, object]]:
    """Sample proportionally within calendar months, distributing rounding fairly."""
    if count is None or count >= len(rows):
        return sorted(rows, key=lambda row: str(row["timestamp"]))

    by_month: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_month[str(row["month"])].append(row)
    months = sorted(by_month)
    exact = {month: count * len(by_month[month]) / len(rows) for month in months}
    allocation = {month: min(len(by_month[month]), int(exact[month])) for month in months}
    remaining = count - sum(allocation.values())
    priority = sorted(
        months,
        key=lambda month: (exact[month] - int(exact[month]), len(by_month[month])),
        reverse=True,
    )
    while remaining:
        added = False
        for month in priority:
            if allocation[month] < len(by_month[month]) and remaining:
                allocation[month] += 1
                remaining -= 1
                added = True
        if not added:
            raise RuntimeError("Could not allocate requested time-stratified sample")

    selected: list[dict[str, object]] = []
    for month in months:
        candidates = by_month[month]
        indices = rng.choice(len(candidates), size=allocation[month], replace=False)
        selected.extend(candidates[int(index)] for index in np.sort(indices))
    return sorted(selected, key=lambda row: str(row["timestamp"]))


def safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to replace existing subset entry: {destination}"
        )
    destination.symlink_to(source)


def discover_split(aia_dir: Path, sxr_dir: Path, split: str) -> list[dict[str, object]]:
    print(f"{split}: indexing AIA/SXR filenames...", flush=True)
    aia_paths = {path.stem: path for path in (aia_dir / split).glob("*.npy")}
    sxr_paths = {path.stem: path for path in (sxr_dir / split).glob("*.npy")}
    if not aia_paths or not sxr_paths:
        raise RuntimeError(f"No paired source data found for split {split!r}")
    missing_sxr = sorted(set(aia_paths).difference(sxr_paths))
    missing_aia = sorted(set(sxr_paths).difference(aia_paths))
    if missing_sxr or missing_aia:
        raise RuntimeError(
            f"Unpaired {split} data: {len(missing_sxr)} missing SXR and "
            f"{len(missing_aia)} missing AIA. Run the integrity checker first."
        )

    rows = []
    timestamps = sorted(aia_paths)
    for timestamp in tqdm(
        timestamps,
        desc=f"{split}: reading SXR",
        unit="sample",
        dynamic_ncols=True,
    ):
        target = np.load(sxr_paths[timestamp], allow_pickle=False)
        if target.size != 1:
            raise ValueError(f"Expected scalar SXR target: {sxr_paths[timestamp]}")
        flux = float(target.reshape(-1)[0])
        # The first seven characters of ISO timestamps are YYYY-MM. Both colon
        # and underscore time separators therefore produce the same month key.
        if len(timestamp) < 10 or timestamp[4] != "-" or timestamp[7] != "-":
            raise ValueError(f"Timestamp does not begin YYYY-MM: {timestamp!r}")
        rows.append({
            "split": split,
            "timestamp": timestamp,
            "month": timestamp[:7],
            "flare_class": classify_flux(flux),
            "sxr_flux": flux,
            "aia_source": str(aia_paths[timestamp].resolve()),
            "sxr_source": str(sxr_paths[timestamp].resolve()),
        })
    return rows


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    source_roots = (args.aia_dir.resolve(), args.sxr_dir.resolve())
    if any(
        output_root == source
        or output_root.is_relative_to(source)
        or source.is_relative_to(output_root)
        for source in source_roots
    ):
        raise ValueError("Subset output must not overlap either source directory")

    summary: dict[str, object] = {
        "seed": args.seed,
        "preset": args.preset,
        "aia_source": str(source_roots[0]),
        "sxr_source": str(source_roots[1]),
        "output_root": str(output_root),
        "splits": {},
    }
    requested = {
        "train": args.train_counts,
        "val": args.val_counts,
        "test": args.test_counts,
    }
    summary["requested"] = requested

    # Screening must use the same target transform as its source dataset.
    # Link the full-training normalization artifact rather than recalculating
    # it from the smaller sample, which also makes the subset self-contained.
    sxr_norm_source = args.sxr_dir / "normalized_sxr.npy"
    if sxr_norm_source.is_file():
        sxr_norm_destination = (
            output_root / "SXR_processed" / "normalized_sxr.npy"
        )
        safe_symlink(sxr_norm_source, sxr_norm_destination)
        summary["sxr_norm_source"] = str(sxr_norm_source.resolve())
    else:
        summary["sxr_norm_source"] = None
        print(
            f"Warning: no SXR normalization artifact at {sxr_norm_source}",
            flush=True,
        )

    for split in ("train", "val", "test"):
        rows = discover_split(args.aia_dir, args.sxr_dir, split)
        by_class = {
            name: [row for row in rows if row["flare_class"] == name]
            for name in CLASSES
        }
        selected = []
        for name in CLASSES:
            selected.extend(proportional_month_sample(
                by_class[name], requested[split][name],
                stable_rng(args.seed, split, name),
            ))
        selected.sort(key=lambda row: str(row["timestamp"]))

        for row in tqdm(
            selected,
            desc=f"{split}: creating links",
            unit="pair",
            dynamic_ncols=True,
        ):
            timestamp = str(row["timestamp"])
            safe_symlink(
                Path(str(row["aia_source"])),
                output_root / "AIA_processed" / split / f"{timestamp}.npy",
            )
            safe_symlink(
                Path(str(row["sxr_source"])),
                output_root / "SXR_processed" / split / f"{timestamp}.npy",
            )

        manifest = output_root / f"{split}_manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="") as handle:
            fieldnames = (
                "split", "timestamp", "month", "flare_class", "sxr_flux",
                "aia_source", "sxr_source",
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)

        available_counts = Counter(str(row["flare_class"]) for row in rows)
        selected_counts = Counter(str(row["flare_class"]) for row in selected)
        summary["splits"][split] = {
            "available": {name: available_counts[name] for name in CLASSES},
            "selected": {name: selected_counts[name] for name in CLASSES},
            "manifest": str(manifest),
        }
        print(
            f"{split}: selected {len(selected):,} symlink pairs "
            f"{dict(selected_counts)}",
            flush=True,
        )

    summary_path = output_root / "subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Subset summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
