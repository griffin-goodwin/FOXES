"""
Split a flat folder of timestamp-named .npy files (AIA or SXR) into
train/val/test subfolders, so forecasting/training's AIAGOESDataModule can
find them.

Called by build_dataset.py — not meant to be run standalone.
"""
import os
import shutil

import pandas as pd


def _normalize_timestamp(ts: str) -> str:
    """Normalize timestamp strings with underscores instead of colons (cross-platform filenames)."""
    if 'T' in ts:
        date_part, time_part = ts.split('T', 1)
        return f"{date_part}T{time_part.replace('_', ':')}"
    return ts


def _assign_split(file_time, train_range, val_range, test_range):
    """Return 'train'/'val'/'test' for this timestamp, or None if no range matches."""
    ranges = {'train': train_range, 'val': val_range, 'test': test_range}
    if any(ranges.values()):
        for split_name, rng in ranges.items():
            if rng is None:
                continue
            start = pd.to_datetime(rng[0])
            end = pd.to_datetime(rng[1]).replace(hour=23, minute=59, second=59, microsecond=999999)
            if start <= file_time <= end:
                return split_name
        return None

    # Default: month-based split (August held out for test, Jan-Mar for val)
    month = file_time.month
    if month == 8:
        return 'test'
    if month in (1, 2, 3):
        return 'val'
    return 'train'


def split_train_val_test(input_dir, output_dir, train_range=None, val_range=None, test_range=None,
                          copy_files=False):
    """
    Split .npy files in input_dir into train/val/test subfolders under output_dir,
    based on each file's timestamp filename.

    Parameters
    ----------
    input_dir : str
        Flat folder of .npy files named by timestamp (e.g. from align_aia_sxr.py
        or convert_aia.py). Can be the same path as output_dir to split in place.
    output_dir : str
        Destination folder; train/val/test subfolders are created under it.
    train_range, val_range, test_range : [start, end] date strings, optional
        Inclusive date ranges ("YYYY-MM-DD") for each split. If none are given,
        falls back to a month-based default: August -> test, Jan-Mar -> val,
        everything else -> train.
    copy_files : bool
        Copy instead of move (default: move).
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input folder does not exist: {input_dir}")

    for split_name in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))
    print(f"Splitting {len(files)} files from {input_dir}")

    moved = skipped = 0
    for filename in files:
        try:
            file_time = pd.to_datetime(_normalize_timestamp(filename[:-len(".npy")]))
        except ValueError:
            print(f"Skipping {filename}: invalid timestamp")
            skipped += 1
            continue

        split_name = _assign_split(file_time, train_range, val_range, test_range)
        if split_name is None:
            print(f"Skipping {filename}: no matching date range ({file_time.date()})")
            skipped += 1
            continue

        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, split_name, filename)
        if os.path.exists(dst):
            skipped += 1
            continue

        (shutil.copy2 if copy_files else shutil.move)(src, dst)
        moved += 1

    action = "copied" if copy_files else "moved"
    print(f"Done: {moved} files {action}, {skipped} skipped")
    for split_name in ("train", "val", "test"):
        n = len(os.listdir(os.path.join(output_dir, split_name)))
        print(f"  {split_name}: {n} files")
