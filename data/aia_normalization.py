#!/home/ubuntu/miniconda3/envs/fox/bin/python
"""Compute AIA normalization parameters with bounded RAM.

Each finite pixel from every selected training file contributes to the global
per-wavelength statistics. Negative finite values are counted and clamped to
zero; NaN and infinite values are counted and excluded.

This script keeps only the statistics needed for normalization:

    Q90     -> asinh scale
    Q99.999 -> relaxed clipping threshold
    asinh_mean, asinh_std -> transformed normalization moments

Percentiles are approximated with a fine log-intensity histogram so that
hundreds of billions of pixels never need to be held in memory. Large batches
and parallel shards keep the CPUs busy, while atomic checkpoints make a long
run safe to resume after interruption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


# Channel order expected in each input array: (7, height, width).
WAVELENGTHS = (94, 131, 171, 193, 211, 304, 335)

# Normalization depends only on the 90th percentile and the relaxed clipping
# threshold at 99.999%.
PERCENTILES = (90.0, 99.999)


def available_cpu_count() -> int:
    """Return CPUs available to this process, respecting CPU affinity."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def parse_args() -> argparse.Namespace:
    cpu_count = available_cpu_count()
    parser = argparse.ArgumentParser(
        description=(
            "Stream AIA training NPY files and estimate the normalization "
            "parameters Q90, Q99.999, and asinh mean/std."
        )
    )
    parser.add_argument(
        "--train-dir", type=Path, default=Path("/data/AIA_processed/train")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/aia_q90_asinh_q99999_norm_streamed.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/aia_streaming_stats_checkpoint.npz"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/data/aia_q90_asinh_q99999_norm_streamed.csv"),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Randomly select this many files; default is every file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4048,
        help=(
            "Images per batch. 2048 is a 14 GiB base buffer for 7x512x512 "
            "float32 images; parallel temporaries use substantially more RAM."
        ),
    )
    parser.add_argument("--load-workers", type=int, default=cpu_count)
    parser.add_argument("--reduce-workers", type=int, default=cpu_count)
    parser.add_argument(
        "--shards-per-channel",
        type=int,
        default=None,
        help="Default creates about twice as many reduction tasks as CPU slots.",
    )
    parser.add_argument("--histogram-bins", type=int, default=131_072)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Checkpoint every N completed batches.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and start from the beginning.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "batch-size": args.batch_size,
        "load-workers": args.load_workers,
        "reduce-workers": args.reduce_workers,
        "histogram-bins": args.histogram_bins,
        "checkpoint-every": args.checkpoint_every,
    }
    if args.shards_per_channel is not None:
        positive_values["shards-per-channel"] = args.shards_per_channel
    if args.max_files is not None:
        positive_values["max-files"] = args.max_files
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")


def atomic_savez(path: Path, **payload: object) -> None:
    """Write completely before replacing an existing checkpoint or result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def selection_digest(paths: list[Path]) -> str:
    """Identify the exact ordered file selection used by a checkpoint."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()

    # 1. Discover all training files, optionally taking a reproducible sample.
    # With the default --max-files=None, no files or pixels are sampled.
    paths = sorted(args.train_dir.glob("*.npy"))
    if not paths:
        raise RuntimeError(f"No .npy files found in {args.train_dir}")
    if args.max_files is not None and args.max_files < len(paths):
        rng = np.random.default_rng(args.seed)
        selected = np.sort(
            rng.choice(len(paths), size=args.max_files, replace=False)
        )
        paths = [paths[index] for index in selected]

    # Inspect one memory-mapped file without loading all its pixels into RAM.
    probe = np.load(paths[0], mmap_mode="r", allow_pickle=False)
    if probe.ndim != 3 or probe.shape[0] != len(WAVELENGTHS):
        raise ValueError(
            f"Expected (7, H, W), got {probe.shape} in {paths[0]}"
        )
    image_shape = tuple(probe.shape)
    image_dtype = probe.dtype
    del probe
    if image_dtype != np.float32:
        print(
            f"Warning: input dtype is {image_dtype}; the batch buffer converts it "
            "to float32.",
            flush=True,
        )

    # More reduction tasks than workers helps prevent CPUs sitting idle when
    # individual shards finish at slightly different times.
    cpu_count = available_cpu_count()
    shards_per_channel = args.shards_per_channel or math.ceil(
        2 * args.reduce_workers / len(WAVELENGTHS)
    )
    file_bytes = paths[0].stat().st_size
    input_bytes = len(paths) * file_bytes
    buffer_bytes = args.batch_size * int(np.prod(image_shape)) * 4
    selected_digest = selection_digest(paths)

    print(f"Training directory: {args.train_dir}", flush=True)
    print(f"Files to stream: {len(paths):,}", flush=True)
    print(f"Input to read: {input_bytes / 2**40:.3f} TiB", flush=True)
    print(f"Batch buffer: {buffer_bytes / 2**30:.2f} GiB", flush=True)
    print(f"Available logical CPUs: {cpu_count}", flush=True)
    print(
        f"Parallelism: {args.load_workers} loaders, {args.reduce_workers} reducers, "
        f"up to {shards_per_channel * len(WAVELENGTHS)} reduction tasks/batch",
        flush=True,
    )

    # 2. Allocate the global accumulators needed for the histogram-based
    # normalization parameters.
    channel_count = len(WAVELENGTHS)

    # Uniform log1p bins provide roughly constant relative intensity precision
    # while covering every possible nonnegative float32 value.
    log_max = float(np.log1p(np.finfo(np.float32).max))
    log_width = log_max / args.histogram_bins
    histograms = np.zeros(
        (channel_count, args.histogram_bins), dtype=np.int64
    )
    counts = np.zeros(channel_count, dtype=np.int64)
    zero_counts = np.zeros(channel_count, dtype=np.int64)
    processed_files = 0

    def save_checkpoint() -> None:
        """Persist global accumulators at a completed-batch boundary."""
        atomic_savez(
            args.checkpoint,
            processed_files=np.asarray(processed_files),
            selected_file_count=np.asarray(len(paths)),
            selection_digest=np.asarray(selected_digest),
            first_file=np.asarray(paths[0].name),
            last_file=np.asarray(paths[-1].name),
            histogram_bins=np.asarray(args.histogram_bins),
            log_max=np.asarray(log_max),
            histograms=histograms,
            counts=counts,
            zero_counts=zero_counts,
        )

    # Resume only when both the exact file selection and histogram layout match.
    # Batch size and worker counts can safely differ from the previous process.
    if not args.no_resume and args.checkpoint.exists():
        with np.load(args.checkpoint, allow_pickle=False) as checkpoint:
            required = {
                "processed_files",
                "selected_file_count",
                "selection_digest",
                "histogram_bins",
                "log_max",
                "histograms",
                "counts",
                "zero_counts",
            }
            missing = required.difference(checkpoint.files)
            if missing:
                raise RuntimeError(
                    f"Checkpoint {args.checkpoint} is missing fields: "
                    f"{', '.join(sorted(missing))}. Use --no-resume or a new path."
                )
            compatible = (
                int(checkpoint["selected_file_count"]) == len(paths)
                and str(checkpoint["selection_digest"]) == selected_digest
                and int(checkpoint["histogram_bins"]) == args.histogram_bins
                and float(checkpoint["log_max"]) == log_max
            )
            if not compatible:
                raise RuntimeError(
                    f"Checkpoint {args.checkpoint} does not match this file "
                    "selection or histogram configuration. Use --no-resume or "
                    "choose a new checkpoint path."
                )
            processed_files = int(checkpoint["processed_files"])
            if not 0 <= processed_files <= len(paths):
                raise RuntimeError("Checkpoint has an invalid processed file count")
            histograms[:] = checkpoint["histograms"]
            counts[:] = checkpoint["counts"]
            zero_counts[:] = checkpoint["zero_counts"]
        print(f"Resuming after {processed_files:,} files", flush=True)

    # 3. Stream pixels in bounded batches. At the default size this persistent
    # buffer is ~14 GiB; masks and reducer temporaries raise peak RAM further.
    batch_buffer = np.empty(
        (args.batch_size,) + image_shape, dtype=np.float32
    )

    def load_item(item: tuple[int, Path]) -> None:
        """Load one complete (7, H, W) file into its batch-buffer slot."""
        slot, path = item
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != image_shape:
            raise ValueError(
                f"Expected shape {image_shape}, got {array.shape} in {path}"
            )
        batch_buffer[slot] = array

    def reduce_shard(
        task: tuple[int, int, int],
    ) -> tuple[int, int, int, np.ndarray | None]:
        """Scan one channel slice and return the histogram contribution."""
        channel, shard_start, shard_stop = task
        values = batch_buffer[shard_start:shard_stop, channel].reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (channel, 0, 0, None)

        clean = np.maximum(finite, 0.0)
        zero_count = int(np.count_nonzero(clean == 0))
        shard_count = int(clean.size)

        log_values = np.log1p(clean)
        bin_indices = np.floor(log_values / log_width).astype(np.int64)
        np.clip(bin_indices, 0, args.histogram_bins - 1, out=bin_indices)
        histogram = np.bincount(
            bin_indices, minlength=args.histogram_bins
        )
        return (channel, shard_count, zero_count, histogram)

    def merge_shard(
        result: tuple[int, int, int, np.ndarray | None]
    ) -> None:
        """Merge one worker result into the per-wavelength global state."""
        channel, shard_count, zero_count, histogram = result
        if shard_count == 0:
            return

        counts[channel] += shard_count
        zero_counts[channel] += zero_count
        if histogram is None:
            raise RuntimeError("Missing histogram for a nonempty shard")
        histograms[channel] += histogram

    state_arrays = (histograms, counts, zero_counts)

    progress = tqdm(
        total=len(paths),
        initial=processed_files,
        desc="Streaming full AIA images",
        unit="file",
        dynamic_ncols=True,
    )
    batch_number = math.ceil(processed_files / args.batch_size)
    try:
        with (
            ThreadPoolExecutor(max_workers=args.load_workers) as load_executor,
            ThreadPoolExecutor(max_workers=args.reduce_workers) as reduce_executor,
        ):
            for batch_start in range(
                processed_files, len(paths), args.batch_size
            ):
                batch_paths = paths[
                    batch_start : batch_start + args.batch_size
                ]
                # Phase A: parallel disk reads into disjoint buffer slots.
                list(load_executor.map(load_item, enumerate(batch_paths)))

                # Phase B: divide every wavelength into enough independent
                # shards to occupy all reduction workers.
                shard_count = min(shards_per_channel, len(batch_paths))
                shard_bounds = np.linspace(
                    0, len(batch_paths), shard_count + 1, dtype=np.int64
                )
                reduction_tasks = [
                    (
                        channel,
                        int(shard_bounds[shard]),
                        int(shard_bounds[shard + 1]),
                    )
                    for channel in range(channel_count)
                    for shard in range(shard_count)
                    if shard_bounds[shard] < shard_bounds[shard + 1]
                ]

                # Restoring this small snapshot makes an interrupted partial batch
                # safe to checkpoint and process again on resume.
                stable_state = tuple(array.copy() for array in state_arrays)
                try:
                    for result in reduce_executor.map(
                        reduce_shard, reduction_tasks
                    ):
                        merge_shard(result)
                except BaseException:
                    for array, stable in zip(state_arrays, stable_state):
                        array[:] = stable
                    raise

                processed_files = batch_start + len(batch_paths)
                batch_number += 1
                progress.update(len(batch_paths))
                if batch_number % args.checkpoint_every == 0:
                    save_checkpoint()
    except BaseException:
        save_checkpoint()
        print(
            f"Saved a resumable checkpoint after {processed_files:,} complete "
            f"files: {args.checkpoint}",
            flush=True,
        )
        raise
    finally:
        progress.close()

    save_checkpoint()

    # 4. Convert histogram counts into the two percentile values used for
    # normalization: Q90 (the scale) and Q99.999 (the clipped tail cap).
    quantiles = np.empty(
        (channel_count, len(PERCENTILES)), dtype=np.float64
    )
    for channel in range(channel_count):
        histogram = histograms[channel]
        cumulative = np.cumsum(histogram)
        total = int(cumulative[-1])
        for percentile_index, percentile in enumerate(PERCENTILES):
            # Cumulative histogram entries are counts, so use a count target.
            target_count = percentile / 100.0 * total
            if target_count <= zero_counts[channel]:
                quantiles[channel, percentile_index] = 0.0
                continue
            bin_index = int(
                np.searchsorted(cumulative, target_count, side="left")
            )
            bin_index = min(bin_index, len(histogram) - 1)
            previous = 0 if bin_index == 0 else cumulative[bin_index - 1]
            fraction = (target_count - previous) / max(
                int(histogram[bin_index]), 1
            )
            log_value = (
                bin_index + np.clip(fraction, 0.0, 1.0)
            ) * log_width
            quantiles[channel, percentile_index] = float(
                np.expm1(log_value)
            )

    q90 = quantiles[:, PERCENTILES.index(90.0)]
    clip_q99999 = quantiles[:, PERCENTILES.index(99.999)]
    if np.any(q90 <= 0):
        bad_channels = np.asarray(WAVELENGTHS)[q90 <= 0].tolist()
        raise RuntimeError(f"Q90 is zero for wavelengths {bad_channels}")

    # 5. Estimate transformed mean/std using histogram-bin representatives.
    # This avoids rereading the entire ~1.75 TiB dataset. Unlike the exact raw
    # moments above, these transformed moments are fine approximations.
    asinh_mean = np.empty(channel_count, dtype=np.float64)
    asinh_std = np.empty(channel_count, dtype=np.float64)
    log_centers = (np.arange(args.histogram_bins) + 0.5) * log_width
    raw_representatives = np.expm1(log_centers)
    for channel in range(channel_count):
        weights = histograms[channel].astype(np.float64)
        zero_weight = float(zero_counts[channel])

        positive_weights = weights.copy()
        positive_weights[0] -= zero_weight

        transformed = np.arcsinh(
            np.minimum(raw_representatives, clip_q99999[channel])
            / q90[channel]
        )

        transformed_mean = np.dot(positive_weights, transformed) / counts[channel]
        positive_variance = np.dot(
            positive_weights,
            (transformed - transformed_mean) ** 2,
        )
        zero_variance = zero_weight * transformed_mean**2
        transformed_variance = (
            positive_variance + zero_variance
        ) / counts[channel]

        asinh_mean[channel] = transformed_mean
        asinh_std[channel] = np.sqrt(transformed_variance)

    # 6. Save both the machine-readable parameters and a readable CSV summary.
    atomic_savez(
        args.output,
        wavelengths=np.asarray(WAVELENGTHS, dtype=np.int32),
        percentiles=np.asarray(PERCENTILES),
        q90=q90,
        clip_q99999=clip_q99999,
        asinh_mean=asinh_mean,
        asinh_std=asinh_std,
    )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "wavelength_A",
        "q90",
        "q99.999",
        "asinh_mean",
        "asinh_std",
    ]
    with args.summary_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for channel, wavelength in enumerate(WAVELENGTHS):
            writer.writerow(
                {
                    "wavelength_A": wavelength,
                    "q90": q90[channel],
                    "q99.999": clip_q99999[channel],
                    "asinh_mean": asinh_mean[channel],
                    "asinh_std": asinh_std[channel],
                }
            )

    elapsed = time.perf_counter() - started
    print("\nNormalization parameters", flush=True)
    print("wave       Q90       Q99.999   asinh_mean  asinh_std", flush=True)
    for channel, wavelength in enumerate(WAVELENGTHS):
        print(
            f"{wavelength:>4}  {q90[channel]:>10.5g}  "
            f"{clip_q99999[channel]:>10.5g}  "
            f"{asinh_mean[channel]:>10.6f}  {asinh_std[channel]:>9.6f}",
            flush=True,
        )
    print(f"\nSaved parameters: {args.output}", flush=True)
    print(f"Saved summary: {args.summary_csv}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(
        f"Completed {len(paths):,} files in {elapsed / 3600:.2f} hours "
        f"({len(paths) / elapsed:.2f} files/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
