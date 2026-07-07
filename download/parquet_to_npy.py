"""
Convert Local HuggingFace Parquet Files to .npy
================================================
Same output layout as hugging_face_data_download.py, but reads from
parquet files you've already downloaded instead of streaming from HF.

Expected parquet columns: filename, aia_stack, sxr_value

Usage:
    # Convert one split at a time (parquet files flat in a directory)
    python download/parquet_to_npy.py \\
        --parquet_dir /path/to/parquet/train \\
        --split train \\
        --config download/hf_download_config.yaml

    # Or specify output dirs directly
    python download/parquet_to_npy.py \\
        --parquet_dir /path/to/parquet/validation \\
        --split validation \\
        --aia_dir /Volumes/T9/AIA_hg_processed \\
        --sxr_dir /Volumes/T9/SXR_hg_processed

    # Auto-discover split subdirs (train/, validation/, test/) under a root
    python download/parquet_to_npy.py \\
        --parquet_root /path/to/parquet \\
        --config download/hf_download_config.yaml
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


HF_TO_LOCAL = {"validation": "val"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _write_arrays(filename: str, aia_arr: np.ndarray, sxr_arr: np.ndarray,
                  aia_split_dir: str, sxr_split_dir: str) -> bool:
    """Save arrays to disk. Returns True if written, False if already exists."""
    aia_path = os.path.join(aia_split_dir, filename)
    sxr_path = os.path.join(sxr_split_dir, filename)

    if os.path.exists(aia_path) and os.path.exists(sxr_path):
        return False

    np.save(aia_path, aia_arr)
    np.save(sxr_path, sxr_arr)
    return True


def convert_split(parquet_dir: str, hf_split: str, aia_base: str, sxr_base: str,
                  num_workers: int = 8, print_every: int = 500, delete_after: bool = False):
    local_split = HF_TO_LOCAL.get(hf_split, hf_split)
    aia_split_dir = os.path.join(aia_base, local_split)
    sxr_split_dir = os.path.join(sxr_base, local_split)
    os.makedirs(aia_split_dir, exist_ok=True)
    os.makedirs(sxr_split_dir, exist_ok=True)

    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}", file=sys.stderr)
        return

    print(f"\n{'='*50}")
    print(f"Converting split: {hf_split} -> local dir: {local_split}")
    print(f"{'='*50}")
    print(f"  Parquet dir: {parquet_dir} ({len(parquet_files)} files)")
    print(f"  AIA -> {aia_split_dir}")
    print(f"  SXR -> {sxr_split_dir}")

    saved = skipped = submitted = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {}

        for pq_file in parquet_files:
            table = pq.read_table(pq_file, columns=["filename", "aia_stack", "sxr_value"])
            n_rows = len(table)

            # Extract all columns in bulk — avoids creating a new Arrow Table per
            # row (table.slice) and per-row .as_py() calls on large nested arrays.
            filenames = table.column("filename").to_pylist()

            # Fast path: flatten nested fixed-size list columns directly to numpy,
            # then reshape to (n, *element_shape). Avoids per-row Python overhead.
            # Falls back to to_pylist() if the column type doesn't support it.
            def _to_numpy_bulk(col, n):
                chunk = col.combine_chunks()
                # Walk the array levels to recover per-element shape.
                # FixedSizeList exposes list_size directly; regular List
                # (which HF uses even for fixed-size arrays) stores a uniform
                # stride in its offsets buffer — both give the same answer.
                shape = []
                vals = chunk
                while hasattr(vals, "values"):
                    if hasattr(vals, "list_size"):   # FixedSizeList
                        shape.append(vals.list_size)
                    elif hasattr(vals, "offsets"):   # List — stride from offsets
                        shape.append(vals.offsets[1].as_py())
                    vals = vals.values
                arr = vals.to_numpy(zero_copy_only=False).astype(np.float32)
                return arr.reshape(n, *shape) if shape else arr.reshape(n, arr.size // n)

            try:
                aia_bulk = _to_numpy_bulk(table.column("aia_stack"), n_rows)
                use_bulk_aia = True
            except Exception:
                aia_pylist = table.column("aia_stack").to_pylist()
                use_bulk_aia = False

            try:
                sxr_bulk = _to_numpy_bulk(table.column("sxr_value"), n_rows)
                use_bulk_sxr = True
            except Exception:
                sxr_pylist = table.column("sxr_value").to_pylist()
                use_bulk_sxr = False

            file_futures = []
            for i, filename in enumerate(filenames):
                aia_arr = aia_bulk[i] if use_bulk_aia else np.array(aia_pylist[i], dtype=np.float32)
                sxr_arr = sxr_bulk[i] if use_bulk_sxr else np.array(sxr_pylist[i], dtype=np.float32)

                fut = pool.submit(_write_arrays, filename, aia_arr, sxr_arr,
                                  aia_split_dir, sxr_split_dir)
                file_futures.append(fut)
                futures[fut] = submitted
                submitted += 1

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

            # Wait for all writes from this file to finish before deleting it
            if delete_after:
                for fut in as_completed(file_futures):
                    pass
                pq_file.unlink()
                print(f"  Deleted {pq_file.name}", flush=True)

        for fut in as_completed(futures):
            if fut.result():
                saved += 1
            else:
                skipped += 1

    elapsed = time.time() - start
    print(f"[{hf_split}] Done — {saved} saved, {skipped} skipped | {elapsed/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(
        description="Convert locally-downloaded HF parquet files to .npy arrays"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to hf_download_config.yaml (provides aia_dir, sxr_dir, num_workers)")
    parser.add_argument("--aia_dir", type=str, default=None,
                        help="Output base dir for AIA .npy files (overrides config)")
    parser.add_argument("--sxr_dir", type=str, default=None,
                        help="Output base dir for SXR .npy files (overrides config)")
    parser.add_argument("--parquet_dir", type=str, default=None,
                        help="Directory containing parquet files for a single split")
    parser.add_argument("--split", type=str, default=None,
                        help="Split name for --parquet_dir (train, validation, test)")
    parser.add_argument("--parquet_root", type=str, default=None,
                        help="Root dir with split subdirs (train/, validation/, test/)")
    parser.add_argument("--splits", type=str, default="train,validation,test",
                        help="Comma-separated splits to process when using --parquet_root")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Parallel write threads (default: from config or 8)")
    parser.add_argument("--print_every", type=int, default=500,
                        help="Log progress every N rows")
    parser.add_argument("--delete", action="store_true",
                        help="Delete each parquet file after its rows are fully written to disk")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

    aia_dir = args.aia_dir or cfg.get("aia_dir")
    sxr_dir = args.sxr_dir or cfg.get("sxr_dir")
    num_workers = args.num_workers or cfg.get("num_workers", 8)

    if not aia_dir or not sxr_dir:
        parser.error("Provide --aia_dir and --sxr_dir, or --config with those keys set.")

    if args.parquet_root:
        # Allow both HF names ("validation") and local aliases ("val")
        LOCAL_TO_HF = {v: k for k, v in HF_TO_LOCAL.items()}  # {"val": "validation"}
        splits = [s.strip() for s in args.splits.split(",")]
        for split in splits:
            split_dir = os.path.join(args.parquet_root, split)
            if not os.path.isdir(split_dir):
                # Try the alias (e.g. "val" if "validation" was requested, or vice versa)
                alias = HF_TO_LOCAL.get(split) or LOCAL_TO_HF.get(split)
                if alias:
                    alt_dir = os.path.join(args.parquet_root, alias)
                    if os.path.isdir(alt_dir):
                        print(f"[info] Using '{alias}/' for split '{split}'")
                        split_dir = alt_dir
                    else:
                        print(f"[warn] Split dir not found, skipping: {split_dir}")
                        continue
                else:
                    print(f"[warn] Split dir not found, skipping: {split_dir}")
                    continue
            convert_split(split_dir, split, aia_dir, sxr_dir, num_workers, args.print_every,
                          delete_after=args.delete)
    elif args.parquet_dir:
        if not args.split:
            parser.error("--split is required when using --parquet_dir")
        convert_split(args.parquet_dir, args.split, aia_dir, sxr_dir, num_workers, args.print_every,
                      delete_after=args.delete)
    else:
        parser.error("Provide either --parquet_dir + --split, or --parquet_root")

    print("\nDone.")


if __name__ == "__main__":
    main()
