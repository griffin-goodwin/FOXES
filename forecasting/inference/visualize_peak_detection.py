#!/usr/bin/env python3
"""
Visualization script to demonstrate peak-based clustering vs DBSCAN.

This script loads a single timestamp and shows:
1. The flux map with detected peaks marked
2. Region assignments for peak-based clustering
3. Region assignments for DBSCAN (for comparison)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys
from scipy.ndimage import maximum_filter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from patch_analysis_v6 import FluxContributionAnalyzer


def visualize_peaks_and_regions(analyzer, timestamp, output_dir=None):
    """
    Visualize peak detection and region clustering for a single timestamp.
    
    Args:
        analyzer: FluxContributionAnalyzer instance
        timestamp: Timestamp to visualize
        output_dir: Directory to save output images (None = display only)
    """
    # Load flux contributions
    flux_contrib = analyzer.load_flux_contributions(timestamp)
    if flux_contrib is None:
        print(f"Could not load flux for {timestamp}")
        return
    
    # Get prediction data
    pred_data = analyzer.predictions_df[analyzer.predictions_df['timestamp'] == timestamp]
    if pred_data.empty:
        print(f"No prediction data for {timestamp}")
        return
    pred_data = pred_data.iloc[0]
    
    # Detect regions with both methods
    print(f"\nAnalyzing timestamp: {timestamp}")
    
    # Peak-based clustering
    original_setting = analyzer.flare_config.get('use_peak_clustering', False)
    analyzer.flare_config['use_peak_clustering'] = True
    regions_peak = analyzer._detect_regions_with_peak_clustering(flux_contrib, timestamp, pred_data)
    
    # DBSCAN
    analyzer.flare_config['use_peak_clustering'] = False
    regions_dbscan = analyzer._detect_regions_with_dbscan(flux_contrib, timestamp, pred_data)
    
    # Restore original setting
    analyzer.flare_config['use_peak_clustering'] = original_setting
    
    print(f"Peak clustering found {len(regions_peak)} regions")
    print(f"DBSCAN found {len(regions_dbscan)} regions")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. Flux map with peaks
    ax = axes[0, 0]
    im = ax.imshow(np.log10(flux_contrib + 1e-10), cmap='hot', origin='lower')
    ax.set_title('Flux Map with Detected Peaks', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='log10(flux)')
    
    # Mark peaks
    if regions_peak:
        peak_y = [r['peak_y'] for r in regions_peak]
        peak_x = [r['peak_x'] for r in regions_peak]
        ax.scatter(peak_x, peak_y, c='cyan', marker='*', s=300, 
                  edgecolors='blue', linewidths=2, label='Peaks', zorder=10)
        
        # Add peak IDs
        for i, (py, px) in enumerate(zip(peak_y, peak_x)):
            ax.text(px, py, str(i+1), color='white', fontsize=10, 
                   fontweight='bold', ha='center', va='center')
    
    ax.legend()
    ax.set_xlabel('Patch X')
    ax.set_ylabel('Patch Y')
    
    # 2. Peak-based clustering result
    ax = axes[0, 1]
    region_map_peak = np.zeros_like(flux_contrib)
    for i, region in enumerate(regions_peak):
        region_map_peak[region['mask']] = i + 1
    
    im = ax.imshow(region_map_peak, cmap='tab20', origin='lower', vmin=0)
    ax.set_title(f'Peak-Based Clustering ({len(regions_peak)} regions)', 
                fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Region ID')
    
    # Mark peaks
    if regions_peak:
        peak_y = [r['peak_y'] for r in regions_peak]
        peak_x = [r['peak_x'] for r in regions_peak]
        ax.scatter(peak_x, peak_y, c='white', marker='*', s=200, 
                  edgecolors='black', linewidths=1.5, zorder=10)
    
    ax.set_xlabel('Patch X')
    ax.set_ylabel('Patch Y')
    
    # 3. DBSCAN result
    ax = axes[1, 0]
    region_map_dbscan = np.zeros_like(flux_contrib)
    for i, region in enumerate(regions_dbscan):
        region_map_dbscan[region['mask']] = i + 1
    
    im = ax.imshow(region_map_dbscan, cmap='tab20', origin='lower', vmin=0)
    ax.set_title(f'DBSCAN Clustering ({len(regions_dbscan)} regions)', 
                fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Region ID')
    
    ax.set_xlabel('Patch X')
    ax.set_ylabel('Patch Y')
    
    # 4. Difference map
    ax = axes[1, 1]
    # Show where the methods disagree
    difference = (region_map_peak > 0).astype(int) - (region_map_dbscan > 0).astype(int)
    im = ax.imshow(difference, cmap='RdBu', origin='lower', vmin=-1, vmax=1)
    ax.set_title('Difference (Peak - DBSCAN)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['DBSCAN only', 'Both/Neither', 'Peak only'])
    
    ax.set_xlabel('Patch X')
    ax.set_ylabel('Patch Y')
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_path = Path(output_dir) / f"peak_detection_comparison_{timestamp}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print region statistics
    print("\n=== Peak-Based Clustering ===")
    for i, region in enumerate(regions_peak):
        print(f"Region {i+1}: size={region['size']:3d}, "
              f"sum_flux={region['sum_flux']:.2e}, "
              f"peak=({region['peak_y']:.1f}, {region['peak_x']:.1f})")
    
    print("\n=== DBSCAN ===")
    for i, region in enumerate(regions_dbscan):
        print(f"Region {i+1}: size={region['size']:3d}, "
              f"sum_flux={region['sum_flux']:.2e}, "
              f"centroid=({region['centroid_patch_y']:.1f}, {region['centroid_patch_x']:.1f})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize peak-based clustering vs DBSCAN'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to analysis configuration YAML file'
    )
    parser.add_argument(
        '--timestamp', 
        type=str,
        help='Specific timestamp to analyze (ISO format: 2024-08-01T12:00:00)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save output images (if not provided, images will be displayed)'
    )
    
    args = parser.parse_args()
    
    # Load analyzer
    print(f"Loading configuration from: {args.config}")
    analyzer = FluxContributionAnalyzer(config_path=args.config)
    
    # Get timestamp
    if args.timestamp:
        timestamp = args.timestamp
    else:
        # Use first available timestamp
        timestamps = sorted(analyzer.predictions_df['timestamp'].unique())
        if not timestamps:
            print("No timestamps found in predictions data")
            return
        timestamp = timestamps[0]
        print(f"Using first timestamp: {timestamp}")
    
    # Create output directory if needed
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Visualize
    visualize_peaks_and_regions(analyzer, timestamp, args.output_dir)


if __name__ == '__main__':
    main()

