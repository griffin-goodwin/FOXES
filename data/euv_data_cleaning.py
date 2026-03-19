import collections.abc
import shutil

import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool

# hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
from itipy.data.dataset import get_intersecting_files
from astropy.io import fits

import json


def load_config():
    """Load configuration from environment or use defaults."""
    try:
        config = json.loads(os.environ['PIPELINE_CONFIG'])
        return config
    except:
        pass


def process_fits_file(file_path):
    try:
        with fits.open(file_path) as hdu:
            header = hdu[1].header
            date_obs = pd.to_datetime(header['DATE-OBS'])
            # Ensure timezone-naive datetime
            if date_obs.tz is not None:
                date_obs = date_obs.tz_localize(None)
            wavelength = header['WAVELNTH']
            filename = pd.to_datetime(os.path.basename(file_path).split('.')[0])
            return {'DATE-OBS': date_obs, 'WAVELNTH': wavelength, 'FILENAME': filename}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


if __name__ == '__main__':
    config = load_config()
    wavelengths = config['euv']['wavelengths']
    base_input_folder = config['euv']['input_folder']

    aia_files = get_intersecting_files(base_input_folder, wavelengths)
    file_list = aia_files[0]  # List of FITS file paths

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_fits_file, file_list), total=len(file_list)))

    # Filter out None results (in case of failed files)
    results = [r for r in results if r is not None]

    # Convert to DataFrame
    aia_header = pd.DataFrame(results)
    aia_header['DATE-OBS'] = pd.to_datetime(aia_header['DATE-OBS'])

    # Add a column for date difference between DATE-OBS and FILENAME
    aia_header['DATE_DIFF'] = (
        pd.to_datetime(aia_header['FILENAME']) - pd.to_datetime(aia_header['DATE-OBS'])
    ).dt.total_seconds()

    # Remove rows where DATE_DIFF is greater than ±60 seconds
    files_to_remove = aia_header[(aia_header['DATE_DIFF'] <= -60) | (aia_header['DATE_DIFF'] >= 60)]
    print(f"{len(files_to_remove)} bad files found")

    for wavelength in wavelengths:
        print(f"\nProcessing wavelength: {wavelength}")
        for names in files_to_remove['FILENAME'].to_numpy():
            filename = pd.to_datetime(names).strftime('%Y-%m-%dT%H:%M:%S') + ".fits"
            file_path = os.path.join(base_input_folder, f"{wavelength}/{filename}")
            destination_folder = os.path.join(config['euv']['bad_files_dir'], str(wavelength))
            os.makedirs(destination_folder, exist_ok=True)
            if os.path.exists(file_path):
                shutil.move(file_path, destination_folder)
                print(f"Moved: {file_path}")
            else:
                print(f"Not found: {file_path}")
