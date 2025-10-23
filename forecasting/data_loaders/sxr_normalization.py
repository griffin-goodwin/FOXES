import numpy as np
from pathlib import Path
import glob
import os

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
    Command-line entry point for computing and saving SXR normalization statistics.

    This block allows the script to be executed directly. It:
      1. Computes log10(SXR) normalization parameters (mean, std) for the dataset.
      2. Saves the computed normalization values as a NumPy `.npy` file for later use.

    Notes
    -----
    - Update the `sxr_dir` variable below to point to your actual SXR data directory.
    - The resulting file `normalized_sxr.npy` will be saved in the same SXR directory.
    """
    # Update this path to your real data SXR directory
    sxr_dir = "/mnt/data/PAPER_DATA_B/SXR/train"  # Replace with actual path
    sxr_norm = compute_sxr_norm(sxr_dir)
    np.save("/mnt/data/PAPER_DATA_B/SXR/normalized_sxr.npy", sxr_norm)
    # print(f"Saved SXR normalization to /mnt/data/ML-Ready-Data-No-Intensity-Cut/normalized_sxr")
