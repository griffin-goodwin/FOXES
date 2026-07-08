"""
Remove AIA FITS files that are unusable: filename timestamp doesn't match the
DATE-OBS header (more than 60s apart, indicating a mis-tagged/corrupt download),
the image itself is blank (all-zero/constant/NaN), or QUALITY is nonzero (AIA's
own bitmask flagging eclipse/calibration/off-point/etc. frames). download_sdo.py
only warns and falls back to the closest available frame when no quality-0
frame exists nearby, so a bad-quality frame can still make it to disk — this
is the backstop that catches it.

Called by build_dataset.py — not meant to be run standalone.
"""
import collections.abc
import os
import shutil
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

# itipy needs these aliases done manually on newer Python versions.
collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]
collections.Mapping = collections.abc.Mapping  # type: ignore[attr-defined]
collections.MutableSet = collections.abc.MutableSet  # type: ignore[attr-defined]
collections.MutableMapping = collections.abc.MutableMapping  # type: ignore[attr-defined]
from itipy.data.dataset import get_intersecting_files
from astropy.io import fits


def _is_blank(data):
    """True if the image has no usable signal (all-zero, constant, or all-NaN)."""
    if data is None:
        return True
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return True
    return bool(np.nanmax(finite) == np.nanmin(finite))


def process_fits_file(file_path):
    try:
        with fits.open(file_path) as hdu:
            header = hdu[1].header  # type: ignore[union-attr]
            date_obs = pd.to_datetime(header['DATE-OBS'])
            # Ensure timezone-naive datetime
            if date_obs.tz is not None:
                date_obs = date_obs.tz_localize(None)
            wavelength = header['WAVELNTH']
            filename = pd.to_datetime(os.path.basename(file_path).split('.')[0])
            blank = _is_blank(hdu[1].data)  # type: ignore[union-attr]
            quality = pd.to_numeric(pd.Series([header.get('QUALITY')]), errors='coerce').iloc[0]
            bad_quality = bool(pd.notna(quality) and quality != 0)
            return {'DATE-OBS': date_obs, 'WAVELNTH': wavelength, 'FILENAME': filename,
                    'BLANK': blank, 'BAD_QUALITY': bad_quality}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def clean_aia_data(input_folder, bad_files_dir, wavelengths):
    """Move AIA FITS files with bad DATE-OBS timestamps or blank images out of input_folder."""
    aia_files = get_intersecting_files(input_folder, wavelengths)
    file_list = aia_files[0]  # List of FITS file paths

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_fits_file, file_list), total=len(file_list)))

    # Filter out None results (in case of failed files)
    results = [r for r in results if r is not None]

    if not results:
        print("No readable AIA files found — nothing to clean.")
        return

    # Convert to DataFrame
    aia_header = pd.DataFrame(results)
    aia_header['DATE-OBS'] = pd.to_datetime(aia_header['DATE-OBS'])

    # Add a column for date difference between DATE-OBS and FILENAME
    aia_header['DATE_DIFF'] = (
        pd.to_datetime(aia_header['FILENAME']) - pd.to_datetime(aia_header['DATE-OBS'])
    ).dt.total_seconds()

    # Remove rows where DATE_DIFF is greater than ±60 seconds, the image is
    # blank, or QUALITY flags a known instrumental issue
    bad_timing = (aia_header['DATE_DIFF'] <= -60) | (aia_header['DATE_DIFF'] >= 60)
    files_to_remove = aia_header[bad_timing | aia_header['BLANK'] | aia_header['BAD_QUALITY']]
    print(f"{len(files_to_remove)} bad files found "
          f"({bad_timing.sum()} bad timing, {aia_header['BLANK'].sum()} blank, "
          f"{aia_header['BAD_QUALITY'].sum()} bad quality)")

    for wavelength in wavelengths:
        print(f"\nProcessing wavelength: {wavelength}")
        for names in files_to_remove['FILENAME'].to_numpy():
            filename = pd.to_datetime(names).strftime('%Y-%m-%dT%H:%M:%S') + ".fits"
            file_path = os.path.join(input_folder, f"{wavelength}/{filename}")
            destination_folder = os.path.join(bad_files_dir, str(wavelength))
            os.makedirs(destination_folder, exist_ok=True)
            if os.path.exists(file_path):
                shutil.move(file_path, destination_folder)
                print(f"Moved: {file_path}")
            else:
                print(f"Not found: {file_path}")
