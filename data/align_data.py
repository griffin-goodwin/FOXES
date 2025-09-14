import glob
import os
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
import re

warnings.filterwarnings('ignore')


def process_timestamp(args):
    """
    Process a single timestamp: find the most recent available GOES data,
    extract SXR data, and save to disk.
    """
    timestamp, goes_data_dict = args
    try:
        # Convert timestamp to datetime for comparison
        target_time = pd.to_datetime(timestamp)
        
        # Try to find GOES data for this timestamp, starting with the most recent instrument
        sxr_a = None
        sxr_b = None
        used_instrument = None
        
        # Sort instruments by G-number (most recent first)
        for g_number in sorted(goes_data_dict.keys(), reverse=True):
            goes_data = goes_data_dict[g_number]
            
            # Look for exact match first
            exact_match = goes_data[goes_data['time'] == target_time]
            if not exact_match.empty:
                sxr_a = exact_match['xrsa_flux'].values[0]
                sxr_b = exact_match['xrsb_flux'].values[0]
                used_instrument = f"GOES-{g_number}"
                break
            

        if sxr_a is None or sxr_b is None or np.isnan(sxr_a) or np.isnan(sxr_b):
            return (timestamp, False, f"No valid SXR data found for timestamp {timestamp}")

        # Initialize arrays
        sxr_a_data = np.zeros(1, dtype=np.float32)
        sxr_b_data = np.zeros(1, dtype=np.float32)
        sxr_a_data[0] = sxr_a
        sxr_b_data[0] = sxr_b

        # Save data to disk
        np.save(f"/mnt/data/NEW-FLARE/GOES-18-SXR-A/{timestamp}.npy", sxr_a_data)
        np.save(f"/mnt/data/NEW-FLARE/GOES-18-SXR-B/{timestamp}.npy", sxr_b_data)

        return (timestamp, True, f"Success using {used_instrument}")

    except Exception as e:
        return (timestamp, False, f"Error processing timestamp {timestamp}: {e}")


def update_progress(result):
    """Callback function to update progress bar"""
    global pbar, successful_count, failed_count
    timestamp, success, message = result

    if success:
        successful_count += 1
        pbar.set_postfix(success=successful_count, failed=failed_count)
        # Only show success message if it includes instrument info
        if "using" in message:
            tqdm.write(f"Success: {timestamp} - {message}")
    else:
        failed_count += 1
        pbar.set_postfix(success=successful_count, failed=failed_count)
        tqdm.write(f"Failed: {timestamp} - {message}")

    pbar.update(1)


def main():
    global pbar, successful_count, failed_count

    # Load GOES data from multiple instruments
    print("Loading GOES data from multiple instruments...")
    # Directory containing GOES files
    directory = "/mnt/data/NEW-FLARE/combined"

    # Regex to match filenames and extract G-number
    pattern = re.compile(r"combined_g(\d+)_avg1m_\d+_\d+\.csv")

    # Find all files matching the pattern and extract G-numbers
    goes_files = []
    for fname in os.listdir(directory):
        match = pattern.match(fname)
        if match:
            g_number = int(match.group(1))
            goes_files.append((g_number, fname))

    if not goes_files:
        raise FileNotFoundError("No GOES CSV files found in directory.")

    # Load all available GOES instruments
    goes_data_dict = {}
    print(f"Found {len(goes_files)} GOES instrument files:")
    
    for g_number, filename in sorted(goes_files, reverse=True):  # Most recent first
        print(f"  Loading GOES-{g_number} from {filename}")
        try:
            goes_df = pd.read_csv(os.path.join(directory, filename))
            goes_df['time'] = pd.to_datetime(goes_df['time'], format='%Y-%m-%d %H:%M:%S')
            goes_data_dict[g_number] = goes_df
            print(f"    Loaded {len(goes_df)} records from {goes_df['time'].min()} to {goes_df['time'].max()}")
        except Exception as e:
            print(f"    Warning: Failed to load {filename}: {e}")
            continue
    
    if not goes_data_dict:
        raise FileNotFoundError("No valid GOES data files could be loaded.")
    
    print(f"Successfully loaded {len(goes_data_dict)} GOES instruments: {sorted(goes_data_dict.keys())}")
    
    # Analyze timestamp coverage across instruments
    print("\nAnalyzing timestamp coverage...")
    for g_number in sorted(goes_data_dict.keys(), reverse=True):
        goes_data = goes_data_dict[g_number]
        time_range = f"{goes_data['time'].min()} to {goes_data['time'].max()}"
        print(f"  GOES-{g_number}: {len(goes_data)} records, {time_range}")

    # Create output directories if they don't exist
    os.makedirs("/mnt/data/NEW-FLARE/GOES-18-SXR-A", exist_ok=True)
    os.makedirs("/mnt/data/NEW-FLARE/GOES-18-SXR-B", exist_ok=True)

    aia_files = sorted(glob.glob('/mnt/data/NEW-FLARE/SDO-AIA-flaring/AIA_processed/*.npy', recursive=True))
    #print(aia_files)
    aia_files_split = []
    for file in aia_files:
        aia_files_split.append(file.split('/')[-1].split('.')[0])
    #print(aia_files_split)
    common_timestamps = [
        datetime.fromisoformat(date_str).strftime('%Y-%m-%dT%H:%M:%S')
        for date_str in aia_files_split]

    # Use all available CPU cores
    num_processes = cpu_count()
    print(f"Using {num_processes} CPU cores for processing")
    print(f"Processing {len(common_timestamps)} timestamps...")

    # Initialize global counters for progress tracking
    successful_count = 0
    failed_count = 0

    # Create arguments for multiprocessing (timestamp, goes_data_dict pairs)
    args_list = [(timestamp, goes_data_dict) for timestamp in common_timestamps]

    # Start timing
    start_time = time.time()

    # Create progress bar
    pbar = tqdm(total=len(common_timestamps), desc="Processing timestamps",
                unit="timestamp", dynamic_ncols=True)

    # Process timestamps in parallel with progress tracking
    with Pool(processes=num_processes) as pool:
        # Use map with callback for real-time progress updates
        results = []
        for args in args_list:
            result = pool.apply_async(process_timestamp, (args,), callback=update_progress)
            results.append(result)

        # Wait for all processes to complete
        for result in results:
            result.wait()

    # Close progress bar
    pbar.close()

    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per timestamp: {total_time / len(common_timestamps):.2f} seconds")
    print(f"Successfully processed: {successful_count}/{len(common_timestamps)} timestamps")
    print(f"Failed processes: {failed_count}")
    print(f"Processing rate: {len(common_timestamps) / total_time:.2f} timestamps/second")
    print(f"Available GOES instruments: {sorted(goes_data_dict.keys())}")

    if failed_count > 0:
        print(f"\n{failed_count} timestamps failed processing (see messages above)")
        print("This may be due to:")
        print("  - Timestamps outside the coverage range of all GOES instruments")
        print("  - Missing or invalid SXR data in the GOES files")
        print("  - Time gaps between different GOES instruments")


if __name__ == "__main__":
    main()
