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

def analyze_all_channels_radial(aia_data):
    """
    Analyze radial distribution for all 7 channels
    """
    channels = ['94', '131', '171', '193', '211', '304', '335']
    h, w = aia_data[0].shape
    center_x, center_y = w // 2, h // 2
    
    y, x = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_radius = min(center_x, center_y)
    normalized_distance = distance_from_center / max_radius
    
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    
    results = {}
    
    for idx, chan_name in enumerate(channels):
        channel = aia_data[idx]
        radial_intensity = []
        radial_bright_pixels = []
        threshold = np.percentile(channel, 90)
        
        for i in range(n_bins):
            mask = (normalized_distance >= bins[i]) & (normalized_distance < bins[i+1])
            radial_intensity.append(np.mean(channel[mask]))
            radial_bright_pixels.append(np.sum(channel[mask] > threshold))
        
        results[chan_name] = {
            'intensities': np.array(radial_intensity),
            'bright_pixels': np.array(radial_bright_pixels)
        }
        
    return bins[:-1], results

def analyze_ar_radial_distribution(aia_data, channel_idx=2):  # channel 2 = 171Å
    """
    Analyze if ARs are more concentrated near limb vs center
    """
    channel = aia_data[channel_idx]
    h, w = channel.shape
    center_x, center_y = w // 2, h // 2
    
    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Normalize distances to solar radius (0=center, 1=limb)
    max_radius = min(center_x, center_y)
    normalized_distance = distance_from_center / max_radius
    
    # Define radial bins
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Calculate mean intensity in each radial bin
    radial_intensity = []
    radial_bright_pixels = []
    
    # Threshold for "bright" pixels (adjust based on your data)
    threshold = np.percentile(channel, 90)
    
    for i in range(n_bins):
        mask = (normalized_distance >= bins[i]) & (normalized_distance < bins[i+1])
        radial_intensity.append(np.mean(channel[mask]))
        radial_bright_pixels.append(np.sum(channel[mask] > threshold))
    
    return bins[:-1], radial_intensity, radial_bright_pixels

def process_single_file_radial(filename, data_path, channels, n_bins):
    """Worker function for multiprocessing radial analysis"""
    try:
        aia_data = np.load(os.path.join(data_path, filename))
        bins, results = analyze_all_channels_radial(aia_data)
        return results
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def process_single_file_radial_ratio(filename, data_path, channels, n_bins):
    """Worker function for multiprocessing radial analysis - returns limb/center ratio"""
    try:
        aia_data = np.load(os.path.join(data_path, filename))
        # Use 171A (idx 2) for AR distribution by default if multiple channels provided
        # or analyze all and average if needed. Here we use idx 2 as in previous examples.
        bins, radial_intensity, radial_bright_pixels = analyze_ar_radial_distribution(aia_data, channel_idx=2)
        
        # Compare inner half (0-0.5 radius) vs outer half (0.5-1.0 radius)
        # n_bins is 10, so first 5 are center, last 5 are limb
        center_activity = np.sum(radial_bright_pixels[:5])
        limb_activity = np.sum(radial_bright_pixels[5:])
        
        ratio = limb_activity / center_activity if center_activity > 0 else np.nan
        
        return {
            'filename': filename,
            'center_activity': center_activity,
            'limb_activity': limb_activity,
            'limb_center_ratio': ratio
        }
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None
