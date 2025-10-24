#!/usr/bin/env python3
"""
Script to analyze flare class distribution across training, validation, and testing sets.
This script examines SXR data files and counts flare classes based on GOES X-ray flux thresholds.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def classify_flare_intensity(flux_value):
    """
    Classify flare intensity based on GOES X-ray flux value.
    
    Args:
        flux_value (float): X-ray flux value in W/m²
        
    Returns:
        str: Flare class ('Quiet', 'A', 'B', 'C', 'M', 'X')
    """
    if flux_value < 1e-8:
        return 'Quiet'
    elif flux_value < 1e-7:
        return 'A'
    elif flux_value < 1e-6:
        return 'B'
    elif flux_value < 1e-5:
        return 'C'
    elif flux_value < 1e-4:
        return 'M'
    else:
        return 'X'

def analyze_split_directory(split_dir, split_name):
    """
    Analyze flare class distribution in a single split directory.
    
    Args:
        split_dir (str): Path to split directory (train/val/test)
        split_name (str): Name of the split
        
    Returns:
        dict: Dictionary with flare class counts
    """
    if not os.path.exists(split_dir):
        print(f"Warning: {split_name} directory does not exist: {split_dir}")
        return {}
    
    flare_counts = defaultdict(int)
    total_files = 0
    processed_files = 0
    
    print(f"Analyzing {split_name} directory: {split_dir}")
    
    # Get all .nc files in the directory
    nc_files = list(Path(split_dir).glob("*.npy"))
    print(f"Found {len(nc_files)} .npy files")
    
    for nc_file in nc_files:
        total_files += 1
        try:
            # Load the .npy file and extract 0th index
            ds = np.load(nc_file)
            flux_data = ds[0]
            
            # Classify the flare intensity
            flare_class = classify_flare_intensity(flux_data)
            flare_counts[flare_class] += 1
            processed_files += 1
            
            
            
        except Exception as e:
            print(f"Error processing {nc_file}: {e}")
            continue
    
    print(f"Processed {processed_files}/{total_files} files in {split_name}")
    return dict(flare_counts)

def create_summary_table(results):
    """
    Create a summary table of flare class distribution.
    
    Args:
        results (dict): Results from analyze_split_directory for each split
        
    Returns:
        pd.DataFrame: Summary table
    """
    # Get all unique flare classes
    all_classes = set()
    for split_results in results.values():
        all_classes.update(split_results.keys())
    
    all_classes = sorted(all_classes, key=lambda x: ['Quiet', 'A', 'B', 'C', 'M', 'X'].index(x) if x in ['Quiet', 'A', 'B', 'C', 'M', 'X'] else 999)
    
    # Create summary data
    summary_data = []
    for split_name, split_results in results.items():
        total_files = sum(split_results.values())
        for flare_class in all_classes:
            count = split_results.get(flare_class, 0)
            percentage = (count / total_files * 100) if total_files > 0 else 0
            summary_data.append({
                'Split': split_name,
                'Flare_Class': flare_class,
                'Count': count,
                'Percentage': round(percentage, 2)
            })
    
    return pd.DataFrame(summary_data)


def main():
    parser = argparse.ArgumentParser(description='Analyze flare class distribution across train/val/test splits')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the directory containing train/val/test subdirectories')
    parser.add_argument('--output_dir', type=str, default='flare_analysis_results',
                       help='Output directory for results and plots')
    parser.add_argument('--save_csv', action='store_true',
                       help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("FLARE CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Analyze each split
    results = {}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(args.data_dir, split)
        results[split] = analyze_split_directory(split_dir, split)
        print()
    
    # Create summary table
    summary_df = create_summary_table(results)
    print("SUMMARY TABLE:")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # Save CSV if requested
    if args.save_csv:
        csv_path = os.path.join(args.output_dir, 'flare_class_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("=" * 60)
    for split in splits:
        if split in results:
            total = sum(results[split].values())
            print(f"{split.upper()}: {total} total files")
            for flare_class in ['Quiet', 'A', 'B', 'C', 'M', 'X']:
                count = results[split].get(flare_class, 0)
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {flare_class}: {count} ({pct:.1f}%)")
        print()

if __name__ == "__main__":
    main()
