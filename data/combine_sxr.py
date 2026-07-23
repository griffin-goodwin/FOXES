"""
Combine raw multi-satellite GOES XRS netCDF files into one interpolated CSV
per satellite (combined_g<N>_avg1m_<start>_<end>.csv).

Called by build_dataset.py — not meant to be run standalone.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


class SXRDataProcessor:
    """Class to combine raw multi-satellite GOES X-ray data and interpolate it in log space.
    This class combines data from different satellites and applies interpolation
    in log space to the X-ray flux data.
    It also tracks which files were used in the processing.
    Parameters
    ----------
    data_dir : str
        Directory containing the raw GOES netCDF files to combine.
    output_dir : str
        Directory where the combined per-satellite CSVs will be saved.
    """

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.used_g13_files = []
        self.used_g14_files = []
        self.used_g15_files = []
        self.used_g16_files = []
        self.used_g17_files = []
        self.used_g18_files = []

    def combine_goes_data(self, columns_to_interp=["xrsb_flux", "xrsa_flux"], apply_pre_goes16_scaling=True):
        """
        Combine GOES-16 and GOES-18 files and track source files used.
        Parameters
        ----------
        apply_pre_goes16_scaling : bool
            Whether to apply the legacy GOES-13/14/15 (pre-GOES-16) scaling
            correction. Set to False if your GOES-13/14/15 files are already
            scaled to match GOES-16 (as with NOAA's newest reprocessed data).
        """
        print("Scanning for GOES data files...")

        def _glob(pattern):
            # Skip macOS AppleDouble sidecar files (e.g. "._sci_xrsf...nc")
            # that external/non-native filesystems litter alongside real files.
            return sorted(p for p in self.data_dir.glob(pattern) if not p.name.startswith('._'))

        g13_files = _glob("*g13*.nc")
        g14_files = _glob("*g14*.nc")
        g15_files = _glob("*g15*.nc")
        g16_files = _glob("*g16*.nc")
        g17_files = _glob("*g17*.nc")
        g18_files = _glob("*g18*.nc")
        
        total_files = len(g13_files) + len(g14_files) + len(g15_files) + len(g16_files) + len(g17_files) + len(g18_files)
        logging.info(
            f"Found {len(g13_files)} GOES-13 files, {len(g14_files)} GOES-14 files, {len(g15_files)} GOES-15 files, {len(g16_files)} GOES-16 files, {len(g17_files)} GOES-17 files, and {len(g18_files)} GOES-18 files.")
        print(f"Total files found: {total_files}")
        
        if total_files == 0:
            print("No GOES data files found in the specified directory.")
            return

        def process_files(files, satellite_name, output_file, used_file_list):
            datasets = []
            successful_files = 0
            failed_files = 0

            print(f"Processing {satellite_name} ({len(files)} files)...")
            
            # Progress bar for file loading
            with tqdm(files, desc=f"Loading {satellite_name}", unit="file", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for file_path in pbar:
                    pbar.set_postfix_str(f"Loading {file_path.name}")
                    try:
                        ds = xr.open_dataset(str(file_path))
                        datasets.append(ds)
                        used_file_list.append(file_path)  # Track file used
                        successful_files += 1
                        logging.info(f"Loaded {file_path.name}")
                    except Exception as e:
                        failed_files += 1
                        logging.error(f"Could not load {file_path.name}: {e}")
                        continue
                    finally:
                        if 'ds' in locals():
                            ds.close() # type: ignore[attr-defined]

            if not datasets:
                print(f"No valid datasets for {satellite_name}")
                logging.warning(f"No valid datasets for {satellite_name}")
                return

            print(f"Processing {len(datasets)} datasets for {satellite_name}...")
            
            try:
                print(f"Concatenating datasets...")
                combined_ds = xr.concat(datasets, dim='time').sortby('time')
                
                # Scaling factors for GOES-13, GOES-14, and GOES-15
                if satellite_name in ['GOES-13', 'GOES-14', 'GOES-15']:
                    if apply_pre_goes16_scaling:
                        print(f"Applying scaling factors for {satellite_name}...")
                        combined_ds['xrsa_flux'] = combined_ds['xrsa_flux'] / .85
                        combined_ds['xrsb_flux'] = combined_ds['xrsb_flux'] / .7
                    else:
                        print(f"Skipping pre-GOES-16 scaling factors for {satellite_name}...")
                
                print(f"Converting to DataFrame...")
                df = combined_ds.to_dataframe().reset_index()
                
                if 'quad_diode' in df.columns:
                    print(f"Filtering quad diode data...")
                    df = df[df['quad_diode'] == 0]  # Filter out quad diode data

                #Filter out data where xrsb_flux has a quality flag of >0
                print(f"Filtering out data where xrsb_flux has a quality flag of >0...")
                df = df[df['xrsb_flag'] == 0]
                
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                print(f"Applying log interpolation...")
                df_log = np.log10(df[columns_to_interp].replace(0, np.nan))

                # Step 3: Interpolate in log space
                df_log_interp = df_log.interpolate(method="time", limit_direction="both")

                # Step 4: Back-transform to linear space
                df[columns_to_interp] = 10 ** df_log_interp

                # Add min and max dates to filename
                min_date = df.index.min().strftime('%Y%m%d')
                max_date = df.index.max().strftime('%Y%m%d')
                filename = f"{str(output_file)}_{min_date}_{max_date}.csv"
                
                print(f"Saving to {filename}...")
                df.to_csv(filename, index=True)

                print(f"Successfully processed {satellite_name}: {successful_files} files loaded, {failed_files} failed")
                logging.info(f"Saved combined file: {output_file}")
                
            except Exception as e:
                print(f"Failed to process {satellite_name}: {e}")
                logging.error(f"Failed to write {output_file}: {e}")
            finally:
                for ds in datasets:
                    ds.close()

        # Create list of satellites to process
        satellites_to_process = []
        if len(g13_files) != 0:
            satellites_to_process.append((g13_files, "GOES-13", self.output_dir / "combined_g13_avg1m", self.used_g13_files))
        if len(g14_files) != 0:
            satellites_to_process.append((g14_files, "GOES-14", self.output_dir / "combined_g14_avg1m", self.used_g14_files))
        if len(g15_files) != 0:
            satellites_to_process.append((g15_files, "GOES-15", self.output_dir / "combined_g15_avg1m", self.used_g15_files))
        if len(g16_files) != 0:
            satellites_to_process.append((g16_files, "GOES-16", self.output_dir / "combined_g16_avg1m", self.used_g16_files))
        if len(g17_files) != 0:
            satellites_to_process.append((g17_files, "GOES-17", self.output_dir / "combined_g17_avg1m", self.used_g17_files))
        if len(g18_files) != 0:
            satellites_to_process.append((g18_files, "GOES-18", self.output_dir / "combined_g18_avg1m", self.used_g18_files))

        print(f"\nStarting processing of {len(satellites_to_process)} satellites...")
        
        # Process each satellite with overall progress tracking
        successful_satellites = 0
        failed_satellites = 0
        
        for i, (files, satellite_name, output_file, used_file_list) in enumerate(satellites_to_process, 1):
            print(f"\n{'='*60}")
            print(f"Processing satellite {i}/{len(satellites_to_process)}: {satellite_name}")
            print(f"{'='*60}")
            
            try:
                process_files(files, satellite_name, output_file, used_file_list)
                successful_satellites += 1
            except Exception as e:
                print(f"Failed to process {satellite_name}: {e}")
                failed_satellites += 1
                logging.error(f"Failed to process {satellite_name}: {e}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed: {successful_satellites} satellites")
        print(f"Failed: {failed_satellites} satellites")
        print(f"Total files processed: {total_files}")
        print(f"Output directory: {self.output_dir}")
        
        # Print file usage statistics
        total_used_files = (len(self.used_g13_files) + len(self.used_g14_files) + 
                           len(self.used_g15_files) + len(self.used_g16_files) + 
                           len(self.used_g17_files) + len(self.used_g18_files))
        print(f"Files used in processing: {total_used_files}")
        
        if successful_satellites > 0:
            print(f"\nSXR data processing completed successfully!")
        else:
            print(f"\n⚠No satellites were processed successfully.")
