import pandas as pd
import os
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

def check_and_move_file(filename, directory, extras_dir, valid_filenames, dry_run=True):
    """Worker function to check and move a single file."""
    if filename not in valid_filenames:
        src = directory / filename
        dst = extras_dir / filename
        try:
            if not dry_run:
                shutil.move(src, dst)
            return 1 # Increment moved count
        except Exception as e:
            print(f"Error moving {filename}: {e}")
            return 0
    return 0

def process_directory(directory, extras_sub_dir, valid_filenames, dry_run=True):
    """Processes a single directory using a pool of workers."""
    if not directory.exists():
        print(f"Directory {directory} does not exist, skipping.")
        return 0
    
    print(f"\nProcessing {directory}...")
    extras_sub_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .npy files in the directory
    files = [f.name for f in os.scandir(directory) if f.name.endswith(".npy") and f.is_file()]
    print(f"Found {len(files)} files in {directory.name}")

    # Partial function with fixed arguments
    worker = partial(check_and_move_file, 
                    directory=directory, 
                    extras_dir=extras_sub_dir, 
                    valid_filenames=valid_filenames, 
                    dry_run=dry_run)
    
    # Use multiprocessing to check and move files
    num_workers = cpu_count()
    chunksize = max(1, len(files) // (num_workers * 4))
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(worker, files, chunksize=chunksize), total=len(files), desc=f"Cleaning {directory.name}"))
    
    moved_count = sum(results)
    status = "[DRY RUN] Would move" if dry_run else "Moved"
    print(f"{status} {moved_count} files to {extras_sub_dir}")
    return moved_count

if __name__ == "__main__":
    # Paths
    base_data_path = Path("/Volumes/T9/FOXES_Data")
    train_csv = base_data_path / "train_data_combined.csv"
    val_csv = base_data_path / "val_data_combined.csv"
    extras_base_dir = base_data_path / "extras"

    # Set dry_run to False to actually move files
    DRY_RUN = False

    # Load combined filenames
    print("Loading CSVs...")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    valid_filenames = set(df_train['filename'].tolist()) | set(df_val['filename'].tolist())
    print(f"Total valid files in CSVs: {len(valid_filenames)}")

    # Directories to process and their corresponding extras sub-directories
    tasks = [
        (base_data_path / "SXR" / "train", extras_base_dir / "SXR"),
        (base_data_path / "SXR" / "val", extras_base_dir / "SXR"),
        (base_data_path / "AIA" / "train", extras_base_dir / "AIA"),
        (base_data_path / "AIA" / "val", extras_base_dir / "AIA")
    ]

    total_moved = 0
    for directory, extras_sub_dir in tasks:
        total_moved += process_directory(directory, extras_sub_dir, valid_filenames, dry_run=DRY_RUN)

    if DRY_RUN:
        print(f"\nDry run complete. Total files that would be moved: {total_moved}")
        print("To actually move files, set DRY_RUN = False in the script.")
    else:
        print(f"\nProcessing complete. Total files moved: {total_moved}")
