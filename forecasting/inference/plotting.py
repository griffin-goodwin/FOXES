"""

---------------------------------
This script generates a time-lapse movie combining SDO/AIA EUV images, attention heatmaps from a trained Vision Transformer (ViT),
and corresponding GOES soft X-ray (SXR) flux predictions over time.

It performs the following:
- Loads AIA images, attention maps, and predicted SXR data.
- Generates composite visualizations showing attention overlayed on AIA wavelengths with synchronized SXR plots.
- Uses multiprocessing to parallelize frame generation for efficiency.
- Compiles generated frames into an MP4 movie.

"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from datetime import datetime, timedelta
import imageio.v2 as imageio
import pandas as pd
from scipy.ndimage import zoom
from multiprocessing import Pool, cpu_count
import time

# Configuration paths
aia_dir = "/mnt/data/ML-Ready/mixed_data/AIA/test/"
weight_path = "/mnt/data/ML-Ready/mixed_data/weights2/"
sxr_data_path = "/mnt/data/ML-Ready/mixed_data/outputs/deep-vit-weighted-clean.csv"
output_dir = "/mnt/data/ML-Ready/mixed_data/movie/"
output_video = "aia_attention_sxr_movie4.mp4"
os.makedirs(output_dir, exist_ok=True)

# Global variable accessible by worker processes
sxr_df = None


def init_worker(sxr_data_path):
    """
    Initialize worker process with shared SXR data.

    Parameters
    ----------
    sxr_data_path : str
        Path to the CSV file containing timestamps, predictions, and ground truth SXR flux values.
    """
    global sxr_df
    print(f"Worker {os.getpid()}: Loading SXR data...")
    try:
        sxr_df = pd.read_csv(sxr_data_path)
        sxr_df['Timestamp'] = pd.to_datetime(sxr_df['Timestamp'])
        print(f"Worker {os.getpid()}: Loaded SXR data with {len(sxr_df)} records")
    except Exception as e:
        print(f"Worker {os.getpid()}: Warning: Could not load SXR data: {e}")
        sxr_df = None


def load_aia_image(timestamp):
    """
    Load an AIA image corresponding to a given timestamp.

    Parameters
    ----------
    timestamp : str
        Timestamp string (ISO format) used to locate the correct AIA image.

    Returns
    -------
    np.ndarray or None
        Loaded AIA image array or None if not found.
    """
    pattern = f"{aia_dir}/*{timestamp}*"
    files = glob.glob(pattern)
    if files:
        return np.load(files[0])
    return None


def load_attention_map(timestamp):
    """
    Load and resize the attention map for a given timestamp.

    Parameters
    ----------
    timestamp : str
        Timestamp corresponding to the model output weight file.

    Returns
    -------
    np.ndarray or None
        512x512 resized attention map as a numpy array or None if not found.
    """
    filepath = f"{weight_path}{timestamp}"
    try:
        attention = np.loadtxt(filepath, delimiter=",")
        target_shape = [512, 512]
        zoom_factors = (target_shape[0] / attention.shape[0],
                        target_shape[1] / attention.shape[1])
        return zoom(attention, zoom_factors, order=1)
    except:
        return None


def get_sxr_data_for_timestamp(timestamp, window_hours=12):
    """
    Retrieve SXR prediction data within a time window around the target timestamp.

    Parameters
    ----------
    timestamp : str
        Center timestamp around which to extract SXR data.
    window_hours : int, default=12
        Time window (± hours) around the target timestamp to extract context data.

    Returns
    -------
    tuple (pd.DataFrame, pd.Series, pd.Timestamp)
        - Data window containing SXR flux data.
        - Current data row corresponding to the timestamp.
        - Converted datetime object of target timestamp.
    """
    global sxr_df
    if sxr_df is None:
        return None, None, None

    target_time = pd.to_datetime(timestamp)
    time_start = target_time - timedelta(hours=window_hours)
    time_end = target_time + timedelta(hours=window_hours)

    mask = (sxr_df['Timestamp'] >= time_start) & (sxr_df['Timestamp'] <= time_end)
    window_data = sxr_df[mask].copy()

    if window_data.empty:
        return None, None, None

    current_idx = (window_data['Timestamp'] - target_time).abs().idxmin()
    current_data = sxr_df.loc[current_idx]

    return window_data, current_data, target_time


def generate_frame_worker(timestamp):
    """
    Generate a single composite visualization frame for the movie.

    Parameters
    ----------
    timestamp : str
        Timestamp identifier used to fetch AIA images, attention maps, and SXR data.

    Returns
    -------
    str or None
        Path to the saved frame image if successful, otherwise None.
    """
    try:
        print(f"Worker {os.getpid()}: Processing {timestamp}")

        aia_data = load_aia_image(timestamp)
        attention_data = load_attention_map(timestamp)

        if aia_data is None or attention_data is None:
            print(f"Worker {os.getpid()}: Skipping {timestamp} (missing data)")
            return None

        sxr_window, sxr_current, target_time = get_sxr_data_for_timestamp(timestamp)
        save_path = os.path.join(output_dir, f"{timestamp}.png")

        # Create visualization layout
        fig = plt.figure(figsize=(19, 8))
        fig.patch.set_facecolor('#1a1a2e')
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 2.5], hspace=0.2, wspace=0.2)

        wavs = ['94', '131', '171', '193', '211', '304']
        att_max = np.percentile(attention_data, 100)
        att_min = np.percentile(attention_data, 0)
        att_norm = AsinhNorm(vmin=att_min, vmax=att_max, clip=False)

        # Overlay attention heatmaps on AIA wavelengths
        for i in range(6):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            aia_img = aia_data[i]
            ax.imshow(aia_img, cmap="gray", origin='lower')
            ax.imshow(attention_data, cmap='hot', origin='lower', alpha=0.35, norm=att_norm)
            ax.set_title(f'AIA {wavs[i]} Å', fontsize=10, color='white')
            ax.axis('off')

        # Plot SXR data over time
        sxr_ax = fig.add_subplot(gs[:, 3])
        sxr_ax.set_facecolor('#2a2a3e')

        if sxr_window is not None and not sxr_window.empty:
            sxr_ax.plot(sxr_window['Timestamp'], sxr_window['ground_truth'], 'b-', label='Ground Truth', linewidth=2, alpha=0.8)
            sxr_ax.plot(sxr_window['Timestamp'], sxr_window['Predictions'], 'r-', label='Prediction', linewidth=2, alpha=0.8)

            if sxr_current is not None:
                sxr_ax.axvline(target_time, color='white', linestyle='--', linewidth=2, alpha=0.8, label='Current Time')
                sxr_ax.plot(target_time, sxr_current['ground_truth'], 'bo', markersize=8, markerfacecolor='lightblue')
                sxr_ax.plot(target_time, sxr_current['Predictions'], 'ro', markersize=8, markerfacecolor='lightcoral')

            sxr_ax.set_ylabel('SXR Flux', fontsize=10, color='white')
            sxr_ax.set_xlabel('Time', fontsize=10, color='white')
            sxr_ax.set_title('SXR Data Context', fontsize=12, color='white')
            sxr_ax.legend(fontsize=8, loc='upper right')
            sxr_ax.grid(True, alpha=0.3)
            sxr_ax.tick_params(axis='x', rotation=45, labelsize=8, colors='white')
            sxr_ax.tick_params(axis='y', labelsize=8, colors='white')
            sxr_ax.set_yscale('log')

            if sxr_current is not None:
                info_text = f"Current Values:\nGT: {sxr_current['ground_truth']:.2e}\nPred: {sxr_current['Predictions']:.2e}"
                sxr_ax.text(0.02, 0.98, info_text, transform=sxr_ax.transAxes,
                            fontsize=8, color='white', verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        else:
            sxr_ax.text(0.5, 0.5, 'No SXR Data\nAvailable',
                        transform=sxr_ax.transAxes, fontsize=12, color='white',
                        horizontalalignment='center', verticalalignment='center')
            sxr_ax.set_title('SXR Data Context', fontsize=12, color='white')

        for spine in sxr_ax.spines.values():
            spine.set_color('white')

        plt.suptitle(f'Timestamp: {timestamp}', color='white', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e')
        plt.close()

        print(f"Worker {os.getpid()}: Completed {timestamp}")
        return save_path

    except Exception as e:
        print(f"Worker {os.getpid()}: Error processing {timestamp}: {e}")
        plt.close('all')
        return None


def main():
    """
    Main function to generate attention visualizations and compile them into a movie.

    Steps
    -----
    1. Generate timestamps between a start and end date.
    2. Spawn worker processes to create visualization frames in parallel.
    3. Combine generated frames into a time-ordered MP4 video.
    4. Optionally delete frame images after movie creation.
    """
    start_time = datetime(2023, 8, 1)
    end_time = datetime(2023, 8, 14)
    interval = timedelta(minutes=15)
    timestamps = []
    while start_time <= end_time:
        timestamps.append(start_time.strftime("%Y-%m-%dT%H:%M:%S"))
        start_time += interval

    print(f"Generated {len(timestamps)} timestamps to process")

    num_processes = min(cpu_count(), len(timestamps))
    num_processes = max(1, num_processes - 1)
    print(f"Using {num_processes} processes")

    start_time = time.time()
    with Pool(processes=num_processes, initializer=init_worker, initargs=(sxr_data_path,)) as pool:
        results = pool.map(generate_frame_worker, timestamps)

    frame_paths = [path for path in results if path is not None]

    processing_time = time.time() - start_time
    print(f"Generated {len(frame_paths)} frames in {processing_time:.2f} seconds")
    print(f"Average: {processing_time / len(frame_paths):.2f} seconds per frame")

    print("Creating movie...")
    video_start = time.time()

    frame_paths.sort(key=lambda x: os.path.basename(x))

    with imageio.get_writer(output_video, fps=30) as writer:
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                image = imageio.imread(frame_path)
                writer.append_data(image)

    video_time = time.time() - video_start
    total_time = time.time() - start_time

    print(f"Video creation took {video_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"✅ Movie saved to: {output_video}")

    cleanup = input("Delete individual frame files? (y/n): ").lower().strip()
    if cleanup == 'y':
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        print("Frame files deleted")


if __name__ == "__main__":
    main()
