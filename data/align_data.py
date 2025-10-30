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

# =============================================================================
# CONFIGURATION - Load from environment or use defaults
# =============================================================================
# 
# Configuration is loaded from environment variables set by the pipeline orchestrator
# or falls back to default values if running standalone
#
# =============================================================================

import os
import json

def load_config():
    """Load configuration from environment or use defaults."""
    if 'PIPELINE_CONFIG' in os.environ:
        try:
            config = json.loads(os.environ['PIPELINE_CONFIG'])
            return config
        except:
            pass
    
    # Default configuration
    return {
        'alignment': {
            'goes_data_dir': "/mnt/data/PAPER/GOES-timespan/combined",
            'aia_processed_dir': "/mnt/data/PAPER/SDOITI",
            'output_sxr_a_dir': "/mnt/data/PAPER/GOES-SXR-A",
            'output_sxr_b_dir': "/mnt/data/PAPER/GOES-SXR-B",
            'aia_missing_dir': "/mnt/data/PAPER/AIA_ITI_MISSING"
        },
        'processing': {
            'batch_size_multiplier': 4,
            'min_batch_size': 1,
            'max_processes': None
        }
    }

config = load_config()

# Input directories
GOES_DATA_DIR = config['alignment']['goes_data_dir']
AIA_PROCESSED_DIR = config['alignment']['aia_processed_dir']

# Output directories
OUTPUT_SXR_A_DIR = config['alignment']['output_sxr_a_dir']
OUTPUT_SXR_B_DIR = config['alignment']['output_sxr_b_dir']
AIA_MISSING_DIR = config['alignment']['aia_missing_dir']

# Processing configuration
BATCH_SIZE_MULTIPLIER = config['processing']['batch_size_multiplier']
MIN_BATCH_SIZE = config['processing']['min_batch_size']
MAX_PROCESSES = config['processing']['max_processes']

# =============================================================================


def load_and_prepare_goes_data(goes_data_dir):
    """
    Load all GOES data and prepare it for efficient lookups.
    """
    print(f"Loading GOES data from: {goes_data_dir}")
    
    # Regex to match filenames and extract G-number
    pattern = re.compile(r"combined_g(\d+)_avg1m_\d+_\d+\.csv")
    
    # Find all files matching the pattern and extract G-numbers
    goes_files = []
    for fname in os.listdir(goes_data_dir):
        match = pattern.match(fname)
        if match:
            g_number = int(match.group(1))
            goes_files.append((g_number, fname))
    
    if not goes_files:
        raise FileNotFoundError(f"No GOES CSV files found in directory: {goes_data_dir}")
    
    # Load all available GOES instruments
    goes_data_dict = {}
    print(f"Found {len(goes_files)} GOES instrument files:")
    
    for g_number, filename in sorted(goes_files, reverse=True):  # Most recent first
        print(f"  Loading GOES-{g_number} from {filename}")
        try:
            goes_df = pd.read_csv(os.path.join(goes_data_dir, filename))
            goes_df['time'] = pd.to_datetime(goes_df['time'], format='%Y-%m-%d %H:%M:%S')
            
            goes_df.set_index('time', inplace=True)
            goes_df.sort_index(inplace=True)  # Ensure sorted for faster lookups
            #Make sure quality flag requirement is in place:
            goes_df = goes_df[goes_df['xrsb_flag']==0]

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
    
    # For each target timestamp, average over all available instruments at that time
    for target_time in tqdm(target_times, desc="Building lookup table"):
        sxr_a_values = []
        sxr_b_values = []
        available_instruments = []

        for g_number in sorted(goes_data_dict.keys(), reverse=True):
            goes_data = goes_data_dict[g_number]
            if target_time in goes_data.index:
                row = goes_data.loc[target_time]
                sxr_a = row['xrsa_flux']
                sxr_b = row['xrsb_flux']
                # Only care about xrsb_flux for validity
                if not pd.isna(sxr_b):
                    sxr_b_values.append(float(sxr_b))
                    if not pd.isna(sxr_a):
                        sxr_a_values.append(float(sxr_a))
                    available_instruments.append(f"GOES-{g_number}")

        if sxr_b_values:
            avg_sxr_b = float(np.mean(sxr_b_values))
            avg_sxr_a = float(np.mean(sxr_a_values)) if sxr_a_values else float('nan')
            lookup_data.append({
                'timestamp': target_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'sxr_a': avg_sxr_a,
                'sxr_b': avg_sxr_b,
                'instrument': ",".join(available_instruments)
            })
    
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
            
            # Save data to disk using configured directories
            np.save(f"{OUTPUT_SXR_A_DIR}/{timestamp}.npy", sxr_a_data)
            np.save(f"{OUTPUT_SXR_B_DIR}/{timestamp}.npy", sxr_b_data)
            
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
    print("=" * 60)
    print("GOES Data Alignment Tool")
    print("=" * 60)
    print(f"GOES data directory: {GOES_DATA_DIR}")
    print(f"AIA processed directory: {AIA_PROCESSED_DIR}")
    print(f"Output SXR-A directory: {OUTPUT_SXR_A_DIR}")
    print(f"Output SXR-B directory: {OUTPUT_SXR_B_DIR}")
    print(f"AIA missing directory: {AIA_MISSING_DIR}")
    print("=" * 60)
    
    # Make output directories if they don't exist
    os.makedirs(OUTPUT_SXR_A_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SXR_B_DIR, exist_ok=True)
    os.makedirs(AIA_MISSING_DIR, exist_ok=True)
    
    # Load and prepare GOES data with optimizations
    goes_data_dict = load_and_prepare_goes_data(GOES_DATA_DIR)
    
    # Get target timestamps from AIA files
    print(f"\nFinding target timestamps from AIA files in: {AIA_PROCESSED_DIR}")
    aia_files = sorted(glob.glob(f"{AIA_PROCESSED_DIR}/*.npy", recursive=True))
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
    max_procs = MAX_PROCESSES if MAX_PROCESSES is not None else cpu_count()
    num_processes = min(max_procs, max(1, len(lookup_data) // 100))  # Don't create too many processes
    batch_size = max(MIN_BATCH_SIZE, len(lookup_data) // (num_processes * BATCH_SIZE_MULTIPLIER))
    
    print(f"\nProcessing {len(lookup_data)} valid timestamps...")
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
    
    print(f"\n" + "=" * 60)
    print(f"PROCESSING COMPLETE!")
    print(f"=" * 60)
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

    # For AIA data that has missing GOES data, move files to missing directory
    print(f"\nChecking for AIA files with missing GOES data...")
    print(f"Moving files with missing GOES data to: {AIA_MISSING_DIR}")
    
    # Create a set of timestamps that have valid GOES data for faster lookup
    valid_timestamps = {data['timestamp'] for data in lookup_data}
    
    moved_count = 0
    for file in aia_files:
        # Extract timestamp from filename
        filename = file.split('/')[-1].split('.')[0]
        timestamp = datetime.fromisoformat(filename).strftime('%Y-%m-%dT%H:%M:%S')
        
        if timestamp not in valid_timestamps:
            try:
                target_path = f"{AIA_MISSING_DIR}/{file.split('/')[-1]}"
                os.rename(file, target_path)
                moved_count += 1
                print(f"Moved {file} to {AIA_MISSING_DIR}")
            except Exception as e:
                print(f"Failed to move {file}: {e}")
    
    print(f"Moved {moved_count} files to {AIA_MISSING_DIR}")
    print("\nDone!")

if __name__ == "__main__":
    main()