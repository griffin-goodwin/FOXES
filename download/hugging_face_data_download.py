"""
Download FOXES dataset from HuggingFace Hub
============================================
Reconstructs the local AIA / SXR directory layout expected by the pipeline:
    {aia_dir}/{split}/{filename}   — shape (7, 512, 512) float32 .npy
    {sxr_dir}/{split}/{filename}   — shape (N,) float32 .npy

Uses streaming so parquet shards are fetched on-the-fly without loading the
full dataset into RAM. A ThreadPoolExecutor overlaps disk writes with network
fetches so the stream never stalls waiting for np.save to finish.

Usage:
    python download/hugging_face_data_download.py \\
        --config download/hf_download_config.yaml
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yaml
from datasets import load_dataset
from huggingface_hub import login


# HuggingFace uses "validation"; local pipeline directories use "val"
HF_TO_LOCAL = {"validation": "val"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_n(repo_id: str, hf_split: str, cfg: dict) -> int | None:
    """
    Resolve how many rows to download. Returns an exact count, or None for all rows.
    Frac is resolved here (before streaming starts) using dataset info so we can
    always use .take(n) and never need to break mid-stream.
    """
    if not cfg.get("subsample", False):
        return None

    n = cfg.get("subsample_n")
    if n is not None:
        return int(n)

    frac = float(cfg.get("subsample_frac", 0.1))
    try:
        # Dataset info is available without downloading any data
        info = load_dataset(repo_id, split=hf_split, streaming=True).info
        total = info.splits[hf_split].num_examples
        return max(1, int(total * frac))
    except Exception:
        print(f"[warn] Could not resolve total size for frac subsampling; downloading all rows.")
        return None


def build_dataset(repo_id: str, hf_split: str, cfg: dict, n: int | None):
    """
    Return a streaming IterableDataset capped at n rows (or unlimited if n is None).
    Always uses .take(n) so the iterator reaches its natural end — no mid-stream
    break, which would leave the prefetch thread holding a dead file descriptor.
    """
    ds = load_dataset(repo_id, split=hf_split, streaming=True)

    if n is not None:
        seed = cfg.get("subsample_seed", 42)
        buffer_size = int(cfg.get("shuffle_buffer_size", max(n * 3, 500)))
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size).take(n)

    return ds


def _write_arrays(filename: str, aia_arr: np.ndarray, sxr_arr: np.ndarray,
                  aia_split_dir: str, sxr_split_dir: str) -> bool:
    """Save pre-materialized arrays to disk. Returns True if written, False if already exists."""
    aia_path = os.path.join(aia_split_dir, filename)
    sxr_path = os.path.join(sxr_split_dir, filename)

    if os.path.exists(aia_path) and os.path.exists(sxr_path):
        return False

    np.save(aia_path, aia_arr)
    np.save(sxr_path, sxr_arr)
    return True


def download_split(hf_split: str, cfg: dict):
    local_split = HF_TO_LOCAL.get(hf_split, hf_split)
    aia_split_dir = os.path.join(cfg["aia_dir"], local_split)
    sxr_split_dir = os.path.join(cfg["sxr_dir"], local_split)
    os.makedirs(aia_split_dir, exist_ok=True)
    os.makedirs(sxr_split_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Downloading split: {hf_split} -> local dir: {local_split}")
    print(f"{'='*50}")
    print(f"  AIA -> {aia_split_dir}")
    print(f"  SXR -> {sxr_split_dir}")

    n = resolve_n(cfg["repo_id"], hf_split, cfg)
    if n is not None:
        print(f"[{hf_split}] Subsampling {n} rows")
    ds = build_dataset(cfg["repo_id"], hf_split, cfg, n)

    num_workers = cfg.get("num_workers", 8)
    print_every = cfg.get("print_every", 500)
    saved = skipped = submitted = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {}

        for i, row in enumerate(ds):
            # Materialize arrays in the main thread before submitting — this
            # detaches them from the pyarrow buffer so threads never touch the
            # parquet file descriptor (which causes [Errno 9] Bad file descriptor).
            aia_arr = np.array(row["aia_stack"], dtype=np.float32)
            sxr_arr = np.array(row["sxr_value"], dtype=np.float32)
            filename = row["filename"]

            fut = pool.submit(_write_arrays, filename, aia_arr, sxr_arr, aia_split_dir, sxr_split_dir)
            futures[fut] = i
            submitted += 1

            # Drain completed futures periodically to track progress and free memory
            if submitted % print_every == 0:
                done = [f for f in futures if f.done()]
                for f in done:
                    if f.result():
                        saved += 1
                    else:
                        skipped += 1
                    del futures[f]

                elapsed = time.time() - start
                rate = submitted / elapsed if elapsed > 0 else 0
                print(
                    f"[{hf_split}] submitted={submitted} | saved={saved} skipped={skipped} | "
                    f"{rate:.1f} rows/sec",
                    flush=True,
                )

        # Wait for all remaining saves to finish
        for fut in as_completed(futures):
            if fut.result():
                saved += 1
            else:
                skipped += 1

    elapsed = time.time() - start
    print(f"[{hf_split}] Done — {saved} saved, {skipped} skipped | {elapsed/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description="Download FOXES data from HuggingFace Hub")
    parser.add_argument("--config", default="download/hf_download_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    login()

    splits = cfg.get("splits", ["train", "validation", "test"])
    for split in splits:
        download_split(split, cfg)

    print("\nAll splits downloaded successfully.")
    # Silence HF/fsspec connection pool teardown — it throws [Errno 9] Bad file
    # descriptor during GC which is harmless but noisy.
    for name in ("huggingface_hub", "urllib3", "fsspec", "datasets"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


if __name__ == "__main__":
    main()
