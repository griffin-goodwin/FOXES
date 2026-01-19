import pandas as pd
import os
import shutil
from pathlib import Path

# Paths
base_data_path = Path("/Volumes/T9/FOXES_Data")
train_csv = base_data_path / "train_data_combined.csv"
val_csv = base_data_path / "val_data_combined.csv"

# Load combined filenames
print("Loading CSVs...")
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)

valid_filenames = set(df_train['filename'].tolist()) | set(df_val['filename'].tolist())
print(f"Total valid files: {len(valid_filenames)}")

# Directories to process
dirs_to_clean = [
    base_data_path / "SXR" / "train",
    base_data_path / "SXR" / "val",
    base_data_path / "AIA" / "train",
    base_data_path / "AIA" / "val"
]

for directory in dirs_to_clean:
    if not directory.exists():
        print(f"Directory {directory} does not exist, skipping.")
        continue
    
    print(f"\nProcessing {directory}...")
    extras_dir = directory / "extras"
    extras_dir.mkdir(exist_ok=True)
    
    # Get all .npy files in the directory
    files = [f for f in os.listdir(directory) if f.endswith(".npy") and os.path.isfile(directory / f)]
    
    moved_count = 0
    for filename in files:
        if filename not in valid_filenames:
            src = directory / filename
            dst = extras_dir / filename
            # shutil.move(src, dst) # Dry run first
            moved_count += 1
            if moved_count <= 5:
                print(f"  [DRY RUN] Would move: {filename}")
    
    print(f"  [DRY RUN] Would move {moved_count} files to {extras_dir}")

print("\nDone with dry run.")
