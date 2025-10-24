#!/usr/bin/env python3
"""
Example script demonstrating how to create contour evolution movies
using the patch_analysis.py functionality.

This script shows how to:
1. Create a contour movie from command line
2. Create a contour movie using a config file
3. Customize movie parameters
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the description"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Command completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with return code {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr)
        return False

def main():
    """Main function demonstrating different ways to create contour movies"""
    
    # Path to the patch analysis script
    script_path = Path(__file__).parent / "patch_analysis.py"
    config_path = Path(__file__).parent / "patch_analysis_config.yaml"
    
    print("Contour Movie Creation Examples")
    print("=" * 50)
    
    # Example 1: Create movie using config file
    print("\n1. Creating contour movie using config file...")
    cmd1 = [
        sys.executable, str(script_path),
        "--config", str(config_path)
    ]
    
    if not run_command(cmd1, "Create contour movie using config file"):
        print("Note: This will fail if the data paths in the config don't exist")
    
    # Example 2: Create movie with command line overrides
    print("\n2. Creating contour movie with command line overrides...")
    cmd2 = [
        sys.executable, str(script_path),
        "--config", str(config_path),
        "--create_contour_movie",
        "--movie_fps", "10",
        "--movie_interval_minutes", "2",
        "--show_sxr_timeseries",
        "--max_tracking_distance", "75",
        "--start_time", "2023-08-05T00:00:00",
        "--end_time", "2023-08-05T12:00:00"
    ]
    
    if not run_command(cmd2, "Create contour movie with command line overrides"):
        print("Note: This will fail if the data paths don't exist")
    
    # Example 3: Show help for all available options
    print("\n3. Available command line options...")
    cmd3 = [sys.executable, str(script_path), "--help"]
    run_command(cmd3, "Show help for patch_analysis.py")
    
    print("\n" + "="*60)
    print("Summary of contour movie functionality:")
    print("="*60)
    print("✅ Added generate_contour_frame_worker() - creates individual frames")
    print("✅ Added create_contour_movie() - orchestrates movie creation")
    print("✅ Added --create_contour_movie command line option")
    print("✅ Added --movie_fps command line option")
    print("✅ Added --movie_interval_minutes command line option")
    print("✅ Added --show_sxr_timeseries command line option")
    print("✅ Added --max_tracking_distance command line option")
    print("✅ Added create_contour_movie config option")
    print("✅ Added movie_interval_minutes config option")
    print("✅ Added show_sxr_timeseries config option")
    print("✅ Added max_tracking_distance config option")
    print("✅ Created sample config file: patch_analysis_config.yaml")
    print("\nThe contour movie shows:")
    print("  - Flux contribution heatmaps over time")
    print("  - Detected flare regions with colored contours")
    print("  - Region labels with flux values")
    print("  - AIA images (if available)")
    print("  - Prediction and ground truth values")
    print("  - SXR time series for tracked regions (when enabled)")
    print("  - Region tracking across time with spatial continuity")
    print("\nUsage examples:")
    print("  python patch_analysis.py --config patch_analysis_config.yaml")
    print("  python patch_analysis.py --config patch_analysis_config.yaml --create_contour_movie --movie_fps 10 --movie_interval_minutes 2")
    print("  python patch_analysis.py --config patch_analysis_config.yaml --create_contour_movie --show_sxr_timeseries --max_tracking_distance 75")
    print("  python patch_analysis.py --config patch_analysis_config.yaml --create_contour_movie --movie_fps 30 --movie_interval_minutes 0  # Use all available data")

if __name__ == "__main__":
    main()
