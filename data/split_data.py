import os
import pandas as pd
import shutil
import argparse
from datetime import datetime
from pathlib import Path

def split_data(input_folder, output_dir, data_type, flare_events_csv=None, repartition=False):
    """
    Split data from input folder into train/val/test based on month.
    Optionally use flare events for additional classification.
    
    Args:
        input_folder (str): Path to the input folder containing data files
        output_dir (str): Path to the output directory where split data will be saved
        data_type (str): Type of data ('aia' or 'sxr')
        flare_events_csv (str, optional): Path to the flare events CSV file
        repartition (bool): If True, treat input_folder as already partitioned (has train/val/test subdirs)
    """
    
    # Validate input folder
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    # Validate data type
    if data_type.lower() not in ['aia', 'sxr']:
        raise ValueError("data_type must be 'aia' or 'sxr'")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    print(f"Processing {data_type.upper()} data from: {input_folder}")
    print(f"Output directory: {output_dir}")
    
    # Load flare events if provided
    flaring_eve_list = []
    if flare_events_csv and os.path.exists(flare_events_csv):
        print(f"Loading flare events from: {flare_events_csv}")
        flare_event = pd.read_csv(flare_events_csv)
        
        # Create list of flare event time ranges
        for i, row in flare_event.iterrows():
            start_time = pd.to_datetime(row['event_starttime'])
            end_time = pd.to_datetime(row['event_endtime'])
            flaring_eve_list.append((start_time, end_time))
        print(f"Loaded {len(flaring_eve_list)} flare events")
    else:
        print("No flare events CSV provided or file not found. Skipping flare classification.")
    
    # Get list of files in input folder
    if repartition:
        # For repartitioning, collect files from train/val/test subdirectories
        data_list = []
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(input_folder, split)
            if os.path.exists(split_dir):
                split_files = os.listdir(split_dir)
                # Add split information to each file for tracking
                for file in split_files:
                    data_list.append((file, split))
                print(f"Found {len(split_files)} files in {split}/ directory")
        print(f"Total files to repartition: {len(data_list)}")
    else:
        # For normal splitting, get files directly from input folder
        data_list = os.listdir(input_folder)
        print(f"Found {len(data_list)} files to process")
    
    moved_count = 0
    skipped_count = 0
    
    for file_info in data_list:
        if repartition:
            file, original_split = file_info
        else:
            file = file_info
            original_split = None
            
        try:
            # Extract timestamp from filename (assuming format like "2012-01-01T00:00:00.npy")
            file_time = pd.to_datetime(file.split(".")[0])
        except ValueError:
            print(f"Skipping file {file}: Invalid timestamp format")
            skipped_count += 1
            continue
        
        # Determine if the file is during a flare event (if flare events are available)
        is_flaring = False
        if flaring_eve_list:
            is_flaring = any(start <= file_time <= end for start, end in flaring_eve_list)
        
        # Determine split based on month
        month = file_time.month
        
        if month in [4, 5, 6, 7, 9, 10, 11, 12]:
            new_split_dir = "train"
        elif month in [1,2,3]:
            new_split_dir = "val"
        elif month == 8:
            new_split_dir = "test"
        else:
            print(f"Skipping file {file}: Unexpected month {month}")
            skipped_count += 1
            continue
        
        # Determine source and destination paths
        if repartition:
            src_path = os.path.join(input_folder, original_split, file)
        else:
            src_path = os.path.join(input_folder, file)
        
        dst_path = os.path.join(output_dir, new_split_dir, file)
        
        # Skip if file is already in the correct split and we're repartitioning
        if repartition and original_split == new_split_dir and os.path.exists(dst_path):
            print(f"File {file} already in correct split ({new_split_dir}), skipping.")
            skipped_count += 1
            continue
        
        if not os.path.exists(dst_path):
            try:
                shutil.move(src_path, dst_path)
                if repartition:
                    if flaring_eve_list:
                        print(f"Moved {file} from {original_split}/ to {new_split_dir}/ (flaring: {is_flaring})")
                    else:
                        print(f"Moved {file} from {original_split}/ to {new_split_dir}/")
                else:
                    if flaring_eve_list:
                        print(f"Moved {file} to {new_split_dir}/ (flaring: {is_flaring})")
                    else:
                        print(f"Moved {file} to {new_split_dir}/")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {file}: {e}")
                skipped_count += 1
        else:
            print(f"File {dst_path} already exists, skipping move.")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Files moved: {moved_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Total files processed: {moved_count + skipped_count}")

def main():
    parser = argparse.ArgumentParser(description='Split AIA or SXR data into train/val/test sets based on month')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Path to the input folder containing data files (or partitioned folder for repartition)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to the output directory where split data will be saved')
    parser.add_argument('--data_type', type=str, choices=['aia', 'sxr'], required=True,
                       help='Type of data: "aia" or "sxr"')
    parser.add_argument('--flare_events_csv', type=str, default=None,
                       help='Path to the flare events CSV file (optional)')
    parser.add_argument('--repartition', action='store_true',
                       help='Repartition an already partitioned folder (input_folder should have train/val/test subdirs)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_folder = os.path.abspath(args.input_folder)
    output_dir = os.path.abspath(args.output_dir)
    flare_events_csv = os.path.abspath(args.flare_events_csv) if args.flare_events_csv else None
    
    # Validate repartition mode
    if args.repartition:
        # Check if input folder has train/val/test subdirectories
        expected_dirs = ['train', 'val', 'test']
        missing_dirs = []
        for dir_name in expected_dirs:
            if not os.path.exists(os.path.join(input_folder, dir_name)):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"Warning: Input folder is missing expected subdirectories: {missing_dirs}")
            print("Continuing with available directories...")
    
    split_data(input_folder, output_dir, args.data_type, flare_events_csv, args.repartition)

if __name__ == "__main__":
    main()
