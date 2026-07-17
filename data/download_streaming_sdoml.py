"""
Materialize FOXES-format AIA .npy files by pulling matched timestamps
directly from the SDOML v2 Zarr store on S3, as an alternative to the
raw-FITS route (download/download_sdo.py + data/convert_aia.py).

AIA-only -- this does not touch SXR data at all. Output layout matches
data/convert_aia.py's (one <timestamp>.npy AIA stack per matched sample),
so the rest of the existing pipeline runs completely unmodified against it:
    python data/align_aia_sxr.py ...          (match against GOES CSVs)
    python data/split_train_val_test.py ...   (on both aia_dir and the
                                                resulting sxr_dir)
    python training/train.py --config training/train_config.yaml

Usage:
    python data/download_streaming_sdoml.py --config data/download_streaming_sdoml_config.yaml
"""
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import s3fs
import yaml
from astropy.visualization import AsinhStretch, ImageNormalize
from tqdm import tqdm

from build_streaming_index import (
    DEFAULT_S3_URL, WAVELENGTH_MAP, build_matched_index,
    get_zarray_meta, make_chunk_cache, open_sdoml_store, read_channel_slice,
)

# Same per-wavelength asinh-stretch normalization itipy's AIADataset applies
# (itipy.data.editor.sdo_norms / NormalizeEditor) to [-1, 1]. Applied here
# directly to SDOML's raw intensities -- SDOML is itself raw AIA data (not
# run through itipy's AIAPrepEditor), so this is a new preprocessing path
# for these samples, not a reproduction of the released checkpoints'
# original training data.
SDO_NORMS = {
    94:  ImageNormalize(vmin=0, vmax=340,  stretch=AsinhStretch(0.005), clip=True),
    131: ImageNormalize(vmin=0, vmax=1400, stretch=AsinhStretch(0.005), clip=True),
    171: ImageNormalize(vmin=0, vmax=8600, stretch=AsinhStretch(0.005), clip=True),
    193: ImageNormalize(vmin=0, vmax=9800, stretch=AsinhStretch(0.005), clip=True),
    211: ImageNormalize(vmin=0, vmax=5800, stretch=AsinhStretch(0.005), clip=True),
    304: ImageNormalize(vmin=0, vmax=8800, stretch=AsinhStretch(0.001), clip=True),
    335: ImageNormalize(vmin=0, vmax=600,  stretch=AsinhStretch(0.005), clip=True),
}


def normalize_aia(img, wavelength):
    return (SDO_NORMS[wavelength](img).data * 2 - 1).astype(np.float32)


def _download_one(row, wavelengths, s3_url, aia_dir, normalize, fs, zarray_cache, cache_lock, chunk_cache):
    timestamp_str = row.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    aia_path = aia_dir / f"{timestamp_str}.npy"
    if aia_path.exists():
        return "skipped", timestamp_str, None

    try:
        channels = []
        for wl in wavelengths:
            base_path = f"{s3_url}/{row.year}/{WAVELENGTH_MAP[wl]}"
            cache_key = (row.year, wl)
            if cache_key not in zarray_cache:
                # get_zarray_meta itself only fetches a few hundred bytes,
                # but guard against every thread racing to fetch the same
                # (year, wavelength) the first time it's seen.
                with cache_lock:
                    if cache_key not in zarray_cache:
                        zarray_cache[cache_key] = get_zarray_meta(fs, base_path)
            img = read_channel_slice(fs, base_path, zarray_cache[cache_key], getattr(row, f"idx_{wl}"),
                                      chunk_cache=chunk_cache)
            if normalize:
                img = normalize_aia(img, wl)
            channels.append(img)
        stack = np.stack(channels, axis=0).astype(np.float32)
        np.save(aia_path, stack)
        return "written", timestamp_str, None
    except Exception as e:
        return "failed", timestamp_str, str(e)


def download_index(index_df, wavelengths, s3_url, aia_dir, normalize=True, max_workers=32, chunk_cache_size=64):
    """
    Fetch each matched sample's 7 channels directly from S3 (bypassing
    zarr's own array-opening path -- see read_channel_slice) and write a
    (7, 512, 512) AIA .npy file per sample, named by timestamp exactly
    like data/convert_aia.py's output.

    Downloads are I/O-bound (waiting on S3, not CPU), so this parallelizes
    with a thread pool rather than data/convert_aia.py's process pool --
    s3fs's sync calls run on a shared internal asyncio event loop, so
    concurrent calls from multiple threads against one S3FileSystem
    interleave for real concurrency without needing a separate connection
    per thread. A shared chunk_cache (see make_chunk_cache) means samples
    that share a chunk with an already-fetched one -- common, since
    build_matched_index's default cadence is close to SDOML's native
    cadence -- are served from memory instead of re-fetched.
    """
    fs = s3fs.S3FileSystem(anon=True)
    aia_dir = Path(aia_dir)
    aia_dir.mkdir(parents=True, exist_ok=True)

    # .zarray metadata (shape/dtype/chunks/compressor) per (year, wavelength)
    # actually touched by this index -- a few hundred bytes each, fetched
    # once, not once per sample. Shared/written across worker threads.
    zarray_cache = {}
    cache_lock = threading.Lock()
    chunk_cache = make_chunk_cache(fs, maxsize=chunk_cache_size)

    rows = list(index_df.itertuples(index=False))
    written = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_download_one, row, wavelengths, s3_url, aia_dir,
                                normalize, fs, zarray_cache, cache_lock, chunk_cache)
                   for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures)):
            status, timestamp_str, err = future.result()
            if status == "written":
                written += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"Failed on {timestamp_str}: {err}")

    print(f"Done: {written} written, {skipped} already existed, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Download SDOML-matched AIA samples to FOXES-format .npy files.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    wavelengths = config["wavelengths"]
    matching = config.get("matching", {})
    s3_url = config.get("s3_url", DEFAULT_S3_URL)

    root = open_sdoml_store(s3_url)
    index_df = build_matched_index(
        root, config["years"], wavelengths, config["cache_dir"],
        cadence_minutes=matching.get("cadence_minutes", 6),
        max_gap_minutes=matching.get("max_gap_minutes", 3),
    )
    print(f"{len(index_df)} AIA-matched samples across {config['years']}")

    download_index(
        index_df, wavelengths, s3_url, config["aia_dir"],
        normalize=config.get("normalize", True),
        max_workers=config.get("max_workers", 32),
        chunk_cache_size=config.get("chunk_cache_size", 64),
    )


if __name__ == "__main__":
    main()
