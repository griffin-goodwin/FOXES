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


def load_and_prepare_goes_data(directory):
    """
    Load all GOES data and prepare it for efficient lookups.
    """
    print("Loading GOES data from multiple instruments...")
    
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
            
            goes_df.set_index('time', inplace=True)
            goes_df.sort_index(inplace=True)  # Ensure sorted for faster lookups
            
            goes_data_dict[g_number] = goes_df
            print(f"    Loaded {len(goes_df)} records from {goes_df.index.min()} to {goes_df.index.max()}")
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
        time_range = f"{goes_data.index.min()} to {goes_data.index.max()}"
        print(f"  GOES-{g_number}: {len(goes_data)} records, {time_range}")
    
    return goes_data_dict


def create_combined_lookup_table(goes_data_dict, target_timestamps):
    """
    Create a single lookup table with the best available data for each timestamp.
    This eliminates the need to search through multiple DataFrames during processing.
    """
    print("Creating optimized lookup table...")
    
    target_times = pd.to_datetime(target_timestamps)
    lookup_data = []
    
    # For each target timestamp, find the best available data
    for target_time in tqdm(target_times, desc="Building lookup table"):
        # Try instruments in priority order (most recent first)
        for g_number in sorted(goes_data_dict.keys(), reverse=True):
            goes_data = goes_data_dict[g_number]
            
            if target_time in goes_data.index:
                row = goes_data.loc[target_time]
                sxr_a = row['xrsa_flux']
                sxr_b = row['xrsb_flux']
                
                # Check if data is valid
                if not (pd.isna(sxr_b)):
                    lookup_data.append({
                        'timestamp': target_time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'sxr_a': float(sxr_a),
                        'sxr_b': float(sxr_b),
                        'instrument': f"GOES-{g_number}"
                    })
                    break
    
    print(f"Found valid data for {len(lookup_data)}/{len(target_timestamps)} timestamps")
    return lookup_data


def process_batch(batch_data):
    """
    Process a batch of timestamps efficiently.
    This is much more efficient than processing one timestamp per process.
    """
    successful_count = 0
    failed_count = 0
    results = []
    
    for data in batch_data:
        try:
            timestamp = data['timestamp']
            sxr_a = data['sxr_a']
            sxr_b = data['sxr_b']
            instrument = data['instrument']
            
            # Create arrays
            sxr_a_data = np.array([sxr_a], dtype=np.float32)
            sxr_b_data = np.array([sxr_b], dtype=np.float32)
            
            # Save data to disk
            np.save(f"/mnt/data/GOES-flaring/GOES-SXR-A/{timestamp}.npy", sxr_a_data)
            np.save(f"/mnt/data/GOES-flaring/GOES-SXR-B/{timestamp}.npy", sxr_b_data)
            
            successful_count += 1
            results.append((timestamp, True, f"Success using {instrument}"))
            
        except Exception as e:
            failed_count += 1
            results.append((timestamp, False, f"Error processing timestamp {timestamp}: {e}"))
    
    return results, successful_count, failed_count


def split_into_batches(data, batch_size):
    """Split data into batches for parallel processing."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def main():
    # Directory containing GOES files
    directory = "/mnt/data/GOES-flaring/combined"
    
    # Make output directories if they don't exist
    os.makedirs("/mnt/data/GOES-flaring/GOES-SXR-A", exist_ok=True)
    os.makedirs("/mnt/data/GOES-flaring/GOES-SXR-B", exist_ok=True)
    
    # Load and prepare GOES data with optimizations
    goes_data_dict = load_and_prepare_goes_data(directory)
    
    # Get target timestamps from AIA files
    print("Finding target timestamps from AIA files...")
    aia_files = sorted(glob.glob('/mnt/data/AIA_ITI/*.npy', recursive=True))
    aia_files_split = [file.split('/')[-1].split('.')[0] for file in aia_files]
    common_timestamps = [
        datetime.fromisoformat(date_str).strftime('%Y-%m-%dT%H:%M:%S')
        for date_str in aia_files_split
    ]
    
    print(f"Found {len(common_timestamps)} target timestamps")
    
    # Create optimized lookup table
    lookup_data = create_combined_lookup_table(goes_data_dict, common_timestamps)
    
    if not lookup_data:
        print("No valid data found for any timestamps!")
        return
    
    # Start timing the processing phase
    start_time = time.time()
    
    # Determine optimal batch size and number of processes
    num_processes = min(cpu_count(), max(1, len(lookup_data) // 100))  # Don't create too many processes
    batch_size = max(1, len(lookup_data) // (num_processes * 4))  # 4 batches per process
    
    print(f"Processing {len(lookup_data)} valid timestamps...")
    print(f"Using {num_processes} processes with batch size {batch_size}")
    
    # Split data into batches
    batches = list(split_into_batches(lookup_data, batch_size))
    
    # Process batches in parallel
    total_successful = 0
    total_failed = 0
    
    if num_processes == 1:
        # Single-threaded processing for small datasets
        pbar = tqdm(batches, desc="Processing batches")
        for batch in pbar:
            results, successful, failed = process_batch(batch)
            total_successful += successful
            total_failed += failed
            pbar.set_postfix(success=total_successful, failed=total_failed)
    else:
        # Multi-threaded processing
        with Pool(processes=num_processes) as pool:
            # Process all batches
            results = []
            for batch in tqdm(batches, desc="Submitting batches"):
                result = pool.apply_async(process_batch, (batch,))
                results.append(result)
            
            # Collect results with progress bar
            pbar = tqdm(total=len(results), desc="Processing batches")
            for result in results:
                batch_results, successful, failed = result.get()
                total_successful += successful
                total_failed += failed
                pbar.set_postfix(success=total_successful, failed=total_failed)
                pbar.update(1)
            pbar.close()
    
    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per timestamp: {total_time / len(lookup_data):.4f} seconds")
    print(f"Successfully processed: {total_successful}/{len(lookup_data)} timestamps")
    print(f"Failed processes: {total_failed}")
    print(f"Processing rate: {len(lookup_data) / total_time:.2f} timestamps/second")
    print(f"Available GOES instruments: {sorted(goes_data_dict.keys())}")
    
    # Report on timestamps that couldn't be processed
    missing_count = len(common_timestamps) - len(lookup_data)
    if missing_count > 0:
        print(f"\n{missing_count} timestamps had no valid GOES data available")
        print("This may be due to:")
        print("  - Timestamps outside the coverage range of all GOES instruments")
        print("  - Missing or invalid SXR data in the GOES files")
        print("  - Time gaps between different GOES instruments")

    # For AIA data that has missing GOES data, we will move the file from the AIA_ITI directory to the AIA_ITI_MISSING directory
    print(f"\nChecking for AIA files with missing GOES data...")
    os.makedirs("/mnt/data/AIA_ITI_MISSING", exist_ok=True)
    
    # Create a set of timestamps that have valid GOES data for faster lookup
    valid_timestamps = {data['timestamp'] for data in lookup_data}
    
    moved_count = 0
    for file in aia_files:
        # Extract timestamp from filename
        filename = file.split('/')[-1].split('.')[0]
        timestamp = datetime.fromisoformat(filename).strftime('%Y-%m-%dT%H:%M:%S')
        
        if timestamp not in valid_timestamps:
            try:
                target_path = f"/mnt/data/AIA_ITI_MISSING/{file.split('/')[-1]}"
                os.rename(file, target_path)
                moved_count += 1
                print(f"Moved {file} to AIA_ITI_MISSING directory")
            except Exception as e:
                print(f"Failed to move {file}: {e}")
    
    print(f"Moved {moved_count} files to AIA_ITI_MISSING directory")

    print("Done")

if __name__ == "__main__":
    main()