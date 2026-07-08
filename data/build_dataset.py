"""
End-to-end raw -> processed dataset builder for FOXES.

Runs, in order: clean_aia -> convert_aia -> combine_sxr -> align -> split ->
(optional) sxr_normalization. Each step can be skipped via the `steps` section
of the config if you already ran it. Everything is driven by one YAML config
— see build_dataset_config.yaml for the fields.

Usage:
    python data/build_dataset.py --config data/build_dataset_config.yaml
"""
import argparse
import os

import numpy as np
import yaml

from clean_aia import clean_aia_data
from convert_aia import process_aia_to_npy
from combine_sxr import SXRDataProcessor
from align_aia_sxr import align_aia_sxr
from split_train_val_test import split_train_val_test
from sxr_normalization import compute_sxr_norm


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Build a FOXES-ready dataset from raw AIA/GOES data.')
    parser.add_argument('--config', type=str, default='data/build_dataset_config.yaml',
                        help='Path to build_dataset config YAML file')
    args = parser.parse_args()

    config = load_config(args.config)

    wavelengths = config['wavelengths']
    aia = config['aia']
    sxr = config['sxr']
    output = config['output']
    processing = config.get('processing', {})
    steps = config.get('steps', {})
    split = config.get('split', {})
    norm = config.get('sxr_normalization', {})

    print("=== FOXES Dataset Build ===")

    if steps.get('clean_aia', True):
        print("\n--- Step 1/5: Clean AIA FITS files ---")
        clean_aia_data(aia['raw_dir'], aia['bad_files_dir'], wavelengths)
    else:
        print("\n--- Step 1/5: Clean AIA FITS files (skipped) ---")

    if steps.get('convert_aia', True):
        print("\n--- Step 2/5: Convert AIA FITS -> .npy ---")
        process_aia_to_npy(aia['raw_dir'], aia['processed_dir'], wavelengths)
    else:
        print("\n--- Step 2/5: Convert AIA FITS -> .npy (skipped) ---")

    if steps.get('combine_sxr', True):
        print("\n--- Step 3/5: Combine raw GOES satellite files ---")
        SXRDataProcessor(data_dir=sxr['raw_dir'], output_dir=sxr['combined_dir']).combine_goes_data()
    else:
        print("\n--- Step 3/5: Combine raw GOES satellite files (skipped) ---")

    if steps.get('align', True):
        print("\n--- Step 4/5: Align AIA timestamps with GOES SXR data ---")
        align_aia_sxr(
            goes_data_dir=sxr['combined_dir'],
            aia_processed_dir=aia['processed_dir'],
            output_sxr_dir=output['sxr_dir'],
            aia_missing_dir=output['aia_missing_dir'],
            max_processes=processing.get('max_processes'),
            batch_size_multiplier=processing.get('batch_size_multiplier', 4),
            min_batch_size=processing.get('min_batch_size', 1),
        )
    else:
        print("\n--- Step 4/5: Align AIA timestamps with GOES SXR data (skipped) ---")

    # Off by default: only needed if you're training (inference/evaluation
    # work fine against the flat processed_dir/sxr_dir from the align step).
    if steps.get('split', False):
        print("\n--- Step 5/5: Split AIA/SXR into train/val/test ---")
        split_kwargs = dict(train_range=split.get('train_range'), val_range=split.get('val_range'),
                            test_range=split.get('test_range'), copy_files=split.get('copy_files', False))
        split_train_val_test(aia['processed_dir'], aia['processed_dir'], **split_kwargs)
        split_train_val_test(output['sxr_dir'], output['sxr_dir'], **split_kwargs)
    else:
        print("\n--- Step 5/5: Split AIA/SXR into train/val/test (skipped) ---")

    if norm.get('compute', False):
        print("\n--- Optional: Compute SXR normalization stats (from the train split) ---")
        sxr_norm = compute_sxr_norm(os.path.join(output['sxr_dir'], 'train'))
        np.save(norm['output_path'], sxr_norm)
        print(f"Saved SXR normalization to {norm['output_path']}")

    print("\nDataset build complete!")


if __name__ == '__main__':
    main()
