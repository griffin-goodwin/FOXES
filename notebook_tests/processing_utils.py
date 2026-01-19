import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def classify_flare(max_flux):
    if max_flux is None or np.isnan(max_flux):
        return 'Unknown'
    if max_flux < 1e-6:
        return 'Weaker than C'
    elif 1e-6 <= max_flux < 1e-5:
        return 'C-class'
    elif 1e-5 <= max_flux < 1e-4:
        return 'M-class'
    elif max_flux >= 1e-4:
        return 'X-class'
    else:
        return 'Unknown'

def process_single_file(entry_tuple):
    split, fname, fpath = entry_tuple
    base, ext = os.path.splitext(fname)
    
    # Try to parse the ISO-like date string
    try:
        date_found = pd.to_datetime(base, format="%Y-%m-%dT%H:%M:%S")
    except Exception:
        date_found = None
        
    # Attempt to load the npy file and find max flux for classification
    max_flux = None
    try:
        data = np.load(fpath)
        if isinstance(data, np.ndarray):
            max_flux = np.nanmax(data)
    except Exception:
        pass
        
    flare_class = classify_flare(max_flux)
    return {
        "filename": fname,
        "split": split,
        "date": date_found,
        "flare_class": flare_class
    }

def run_processing(data_dir, num_workers=None, chunksize=100):
    if num_workers is None:
        num_workers = cpu_count()
        
    files_to_process = []
    for split in ["train", "val", "test"]:
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            continue
        
        print(f"Listing files in {split}...")
        with os.scandir(split_path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith('.npy'):
                    files_to_process.append((split, entry.name, entry.path))

    print(f"Total files to process: {len(files_to_process)}")

    file_infos = []
    with Pool(num_workers) as pool:
        # Use imap for progress bar
        for result in tqdm(pool.imap(process_single_file, files_to_process, chunksize=chunksize), 
                          total=len(files_to_process), 
                          desc="Processing files"):
            file_infos.append(result)

    return pd.DataFrame(file_infos)

def process_file_sum(file_info):
    path, filename = file_info
    try:
        data = np.load(os.path.join(path, filename))
        # Sum along spatial dimensions (axis 1 and 2), keeping channel dimension (axis 0)
        # Expected data shape: (7, H, W)
        if data.ndim == 3 and data.shape[0] == 7:
            return np.sum(data, axis=(1, 2))
        elif data.ndim == 3:
            # If not 7 channels, print warning and return None or handle as needed
            print(f"Warning: {filename} has unexpected number of channels: {data.shape[0]}")
            return None
        elif data.ndim == 1:
            # SXR data often (1,)
            return data[0]
        else:
            # Fallback or single channel
            return np.sum(data)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None
