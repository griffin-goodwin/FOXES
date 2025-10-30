import numpy as np
from pathlib import Path
import glob
import os
import argparse

def compute_sxr_norm(sxr_dir):
    """
    Compute the mean and standard deviation of log10-transformed Soft X-Ray (SXR) flux values.

    This function scans a given directory containing `.npy` SXR data files,
    loads each file, filters out invalid or non-finite values, applies a logarithmic
    transformation (`log10(SXR + 1e-8)`), and computes the mean and standard deviation
    for normalization purposes. These normalization statistics are typically used
    during model training and inference to ensure consistent SXR scaling.

    Parameters
    ----------
    sxr_dir : str or Path
        Path to the directory containing `.npy` SXR flux files.

    Returns
    -------
    tuple of (float, float)
        - mean : Mean of log10-transformed SXR flux values.
        - std : Standard deviation of log10-transformed SXR flux values.

    Raises
    ------
    FileNotFoundError
        If the specified SXR directory does not exist.
    ValueError
        If no valid `.npy` files or no valid SXR values are found.

    Notes
    -----
    - Files are expected to contain scalar SXR flux values in W/m².
    - Invalid (non-finite or negative) values are automatically skipped.
    - The logarithmic transform helps stabilize the variance and normalize scale differences.
    """
    sxr_dir = Path(sxr_dir).resolve()
    print(f"Checking SXR directory: {sxr_dir}")
    if not sxr_dir.is_dir():
        raise FileNotFoundError(f"SXR directory does not exist or is not a directory: {sxr_dir}")

    # Use glob for case-insensitive matching
    sxr_files = sorted(glob.glob(os.path.join(sxr_dir, "*.npy")))
    print(f"Found {len(sxr_files)} SXR files in {sxr_dir}")
    if len(sxr_files) == 0:
        print(f"No files matching '*.npy' found. Listing directory contents:")
        print(os.listdir(sxr_dir)[:10])  # Show first 10 files
        raise ValueError(f"No SXR files found in {sxr_dir}")

    sxr_values = []
    for f in sxr_files:
        try:
            sxr = np.load(f)
            sxr = np.atleast_1d(sxr).flatten()[0]
            if not np.isfinite(sxr) or sxr < 0:
                print(f"Skipping invalid SXR value in {f}: {sxr}")
                continue
            sxr_values.append(np.log10(sxr))
        except Exception as e:
            print(f"Failed to load SXR file {f}: {e}")
            continue

    sxr_values = np.array(sxr_values)
    if len(sxr_values) == 0:
        raise ValueError(f"No valid SXR values found in {sxr_dir}. All files failed to load or contained invalid data.")

    mean = np.mean(sxr_values)
    std = np.std(sxr_values)
    print(f"Computed SXR normalization: mean={mean}, std={std}")
    return mean, std


if __name__ == "__main__":
    """
    Command-line interface for calculating and saving SXR normalization parameters.

    When run as a script, this entry point:
      1. Computes the normalization statistics (mean and standard deviation of log10(SXR flux))
         for all valid `.npy` SXR files in the specified directory.
      2. Saves the resulting (mean, std) tuple as a NumPy `.npy` file at the user-specified output path.

    Usage
    -----
    python sxr_normalization.py --sxr_dir /path/to/SXR/dir --output_path /where/to/save/normalized_sxr.npy

    Arguments
    ---------
    --sxr_dir : str
        Path to the directory containing SXR `.npy` data files.
    --output_path : str
        Path where the computed normalization statistics (mean, std) will be saved.

    Notes
    -----
    - By default, --sxr_dir and --output_path are set for the paper dataset example.
    - The output file `normalized_sxr.npy` contains a two-element array: [mean, std]
    - These statistics are expected to be used for normalizing GOES SXR flux data in the ML pipeline.
    """

    parser = argparse.ArgumentParser(description='Compute and save SXR normalization statistics.')
    parser.add_argument('--sxr_dir', type=str, default='/mnt/data/PAPER/SXR/train', help='Path to the SXR data directory.')
    parser.add_argument('--output_path', type=str, default='/mnt/data/PAPER/SXR/normalized_sxr.npy', help='Path to save the normalized SXR data.')
    args = parser.parse_args()

    sxr_norm = compute_sxr_norm(args.sxr_dir)
    np.save(args.output_path, sxr_norm)
    print(f"Saved SXR normalization to {args.output_path}")
