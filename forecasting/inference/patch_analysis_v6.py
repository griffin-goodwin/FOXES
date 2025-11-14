import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import pandas as pd
from pathlib import Path
import argparse
import cv2
from scipy import ndimage as nd
from sklearn.cluster import DBSCAN
import warnings
import yaml
from datetime import datetime
import imageio.v2 as imageio
import os
from multiprocessing import Pool
import time
from tqdm import tqdm


def setup_barlow_font():
    """Setup Barlow font for matplotlib plots"""
    try:
        # Try to find Barlow font with more specific search
        barlow_fonts = []
        for font in fm.fontManager.ttflist:
            if 'barlow' in font.name.lower() or 'barlow' in font.fname.lower():
                barlow_fonts.append(font.name)
        
        if barlow_fonts:
            rcParams['font.family'] = 'Barlow'
            print(f"Using Barlow font: {barlow_fonts[0]}")
        else:
            # Try alternative approach - directly specify font file
            barlow_path = '/usr/share/fonts/truetype/barlow/Barlow-Regular.ttf'
            if os.path.exists(barlow_path):
                # Add the font file directly to matplotlib
                fm.fontManager.addfont(barlow_path)
                rcParams['font.family'] = 'Barlow'
                print(f"Using Barlow font from: {barlow_path}")
            else:
                # Fallback to sans-serif
                rcParams['font.family'] = 'sans-serif'
                print("Barlow font not found, using default sans-serif")
    except Exception as e:
        print(f"Font setup error: {e}, using default font")

# Setup Barlow font
setup_barlow_font()


"""
Flux Contribution Analysis and Flare Detection Script - Version 6

This script analyzes flux contributions from different patches to identify 
potential flaring events and visualize their spatial and temporal patterns.

Version 6: Uses DBSCAN clustering for region detection (simplified from v5)
"""



warnings.filterwarnings('ignore')


class FluxContributionAnalyzer:
    def __init__(self, config_path=None, flux_path=None, predictions_csv=None, aia_path=None,
                 grid_size=(32, 32), patch_size=16, input_size=512, time_period=None):
        """
        Initialize the flux contribution analyzer

        Args:
            config_path: Path to YAML config file (optional, overrides other parameters)
            flux_path: Path to directory containing flux contribution files
            predictions_csv: Path to CSV file with predictions and timestamps
            aia_path: Path to directory containing AIA numpy files
            grid_size: Size of the flux contribution grid
            patch_size: Size of each patch in pixels
            input_size: Input image size
            time_period: Dict with 'start_time' and 'end_time' for filtering data
        """
        # Load config if provided
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Extract paths from config
            base_dir = self.config['base_data_dir']
            flux_path = self.config['flux_path'].replace('${base_data_dir}', base_dir)
            predictions_csv = self.config['output_path'].replace('${base_data_dir}', base_dir)
            aia_path = self.config['aia_path'].replace('${base_data_dir}', base_dir)
            
            # Extract analysis parameters
            self.analysis_config = self.config.get('analysis', {})
            self.flare_config = self.analysis_config.get('flare_detection', {})
            self.output_config = self.analysis_config.get('output', {})
            
            # Extract time period
            time_period_config = self.analysis_config.get('time_period', {})
            if time_period_config.get('start_time') and time_period_config.get('end_time'):
                time_period = {
                    'start_time': time_period_config['start_time'],
                    'end_time': time_period_config['end_time']
                }
        else:
            self.config = {}
            self.analysis_config = {}
            self.flare_config = {}
            self.output_config = {}
        
        self.flux_path = Path(flux_path)
        self.aia_path = Path(aia_path) if aia_path else None
        self.predictions_df = pd.read_csv(predictions_csv)
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.input_size = input_size
        self.time_period = time_period

        # Convert timestamps to datetime
        self.predictions_df['datetime'] = pd.to_datetime(self.predictions_df['timestamp'])
        self.predictions_df = self.predictions_df.sort_values('datetime')

        # Filter by time period if specified
        if self.time_period:
            start_time = pd.to_datetime(self.time_period['start_time'])
            end_time = pd.to_datetime(self.time_period['end_time'])
            mask = (self.predictions_df['datetime'] >= start_time) & (self.predictions_df['datetime'] <= end_time)
            self.predictions_df = self.predictions_df[mask].reset_index(drop=True)
            print(f"Filtered data to time period: {start_time} to {end_time}")

        print(f"Loaded {len(self.predictions_df)} predictions")
        print(f"Time range: {self.predictions_df['datetime'].min()} to {self.predictions_df['datetime'].max()}")

        if self.aia_path and self.aia_path.exists():
            print(f"AIA data path: {self.aia_path}")
        elif self.aia_path:
            print(f"Warning: AIA data path does not exist: {self.aia_path}")

    def load_aia_image(self, timestamp):
        """Load AIA image as RGB composite from 94, 131, 171 Angstrom channels"""
        if self.aia_path is None:
            return None

        # Try different possible filename formats and subdirectories
        # Search in subdirectories (test, train, val) and root
        possible_dirs = [self.aia_path]
        for subdir in ['test', 'train', 'val']:
            subdir_path = self.aia_path / subdir
            if subdir_path.exists():
                possible_dirs.append(subdir_path)
        
        possible_files = []
        for aia_dir in possible_dirs:
            possible_files.extend([
                aia_dir / f"{timestamp}.npy"
            ])

        for aia_file in possible_files:
            if aia_file.exists():
                try:
                    if aia_file.suffix == '.npy':
                        aia_data = np.load(aia_file)
                    else:  # .npz
                        aia_data = np.load(aia_file)['arr_0']

                    # Get 94, 131, 171 Angstrom channels (dimensions 0, 1, 2)
                    aia_94 = aia_data[1]   # 94 Angstrom
                    aia_131 = aia_data[2]  # 131 Angstrom
                    aia_171 = aia_data[5]  # 171 Angstrom
                    
                    # Stack channels to create RGB image (94 -> Red, 131 -> Green, 171 -> Blue)
                    # Normalize each channel to 0-1 range for visualization
                    aia_stacked = np.stack([
                        (aia_94 - np.min(aia_94)) / (np.max(aia_94) - np.min(aia_94) + 1e-10),
                        (aia_131 - np.min(aia_131)) / (np.max(aia_131) - np.min(aia_131) + 1e-10),
                        (aia_171 - np.min(aia_171)) / (np.max(aia_171) - np.min(aia_171) + 1e-10)
                    ], axis=-1)

                    # Ensure correct size
                    if aia_stacked.shape[:2] != (512, 512):
                        print(f"Warning: AIA image shape is {aia_stacked.shape}, expected (512, 512, 3)")

                    return aia_stacked
                except Exception as e:
                    print(f"Error loading {aia_file}: {e}")
                    continue

        return None

    def load_flux_contributions(self, timestamp):
        """Load flux contributions for a specific timestamp"""
        flux_file = self.flux_path / f"{timestamp}"
        if flux_file.exists():
            return np.loadtxt(flux_file, delimiter=',')
        return None


    def detect_flare_events(self, timestamps=None):
        """
        Detect potential flare events based on tracked regions with temporal-spatial coherence.
        
        This method uses the tracked regions from track_regions_over_time() instead of
        doing independent detection at each timestamp, ensuring temporal-spatial coherence.

        Args:
            timestamps: List of timestamps to analyze (if None, uses all timestamps from predictions_df)
        """
        if timestamps is None:
            timestamps = self.predictions_df['timestamp'].tolist()
        
        print("Detecting flare events using tracked regions with temporal-spatial coherence...")
        
        # Track regions across time (this does the detection and tracking)
        region_tracks = self.track_regions_over_time(timestamps)
        
        # Convert tracked regions to flare events DataFrame format
        flare_events = []
        
        for track_id, track_history in region_tracks.items():
            for timestamp, region_data in track_history:
                # Get prediction data for this timestamp
                pred_data = self.predictions_df[self.predictions_df['timestamp'] == timestamp]
                if pred_data.empty:
                    continue
                pred_data = pred_data.iloc[0]
                
                flare_events.append({
                    'timestamp': timestamp,
                    'datetime': pred_data['datetime'],
                    'prediction': pred_data['predictions'],
                    'groundtruth': pred_data.get('groundtruth', None),
                    'region_size': region_data.get('size', 0),
                    'max_flux': region_data.get('max_flux', 0.0),
                    'mean_flux': region_data.get('sum_flux', 0.0) / max(region_data.get('size', 1), 1),
                    'sum_flux': region_data.get('sum_flux', 0.0),
                    'centroid_patch_y': region_data.get('centroid_patch_y', 0.0),
                    'centroid_patch_x': region_data.get('centroid_patch_x', 0.0),
                    'centroid_img_y': region_data.get('centroid_img_y', 0.0),
                    'centroid_img_x': region_data.get('centroid_img_x', 0.0),
                    'track_id': track_id  # Add track ID for reference
                })
        
        self.flare_events_df = pd.DataFrame(flare_events)
        print(f"Detected {len(flare_events)} potential flare events from {len(region_tracks)} tracked regions")
        return self.flare_events_df

    def detect_simultaneous_flares(self, threshold=1e-5, sequence_window_hours=1.0):
        """
        Detect simultaneous flaring events - multiple distinct regions within the same flux prediction
        where each region has a sum of flux above the threshold. Groups are then clustered into
        flare sequences if they occur within ±sequence_window_hours of each other.
        
        Args:
            threshold: Sum of flux threshold for considering a region as a flare
            sequence_window_hours: Time window in hours for clustering groups into sequences (default: 1.0)
        
        Returns:
            DataFrame with simultaneous flare events, including group_id and sequence_id
        """
        if not hasattr(self, 'flare_events_df') or len(self.flare_events_df) == 0:
            print("Please run detect_flare_events() first")
            return pd.DataFrame()
        
        # Filter regions by sum_flux threshold (not prediction threshold)
        high_flux_regions = self.flare_events_df[self.flare_events_df['sum_flux'] >= threshold].copy()
        
        if len(high_flux_regions) == 0:
            print(f"No regions found with sum_flux above threshold {threshold}")
            return pd.DataFrame()
        
        # Step 1: Group regions by timestamp to find simultaneous flares within the same flux prediction
        print("Step 1/3: Grouping regions by timestamp...")
        simultaneous_groups = []
        unique_timestamps = high_flux_regions['timestamp'].unique()
        for timestamp in tqdm(unique_timestamps, desc="Grouping timestamps", unit="timestamp"):
            group = high_flux_regions[high_flux_regions['timestamp'] == timestamp]
            if len(group) >= 2:  # Multiple distinct regions at the same timestamp
                simultaneous_groups.append(group)
        
        if len(simultaneous_groups) == 0:
            print("No simultaneous flare events detected")
            return pd.DataFrame()
        
        # Step 2: Cluster groups into sequences based on temporal proximity
        # Each group has a timestamp - cluster groups that are within sequence_window_hours
        print(f"Step 2/3: Clustering {len(simultaneous_groups)} groups into sequences...")
        sequence_clusters = []
        used_group_indices = set()
        
        for group_idx, group in tqdm(enumerate(simultaneous_groups), desc="Clustering groups", total=len(simultaneous_groups), unit="group"):
            if group_idx in used_group_indices:
                continue
            
            # Start a new sequence with this group
            sequence_groups = [group_idx]
            used_group_indices.add(group_idx)
            
            # Find all groups within sequence_window_hours of any group in this sequence
            changed = True
            while changed:
                changed = False
                for other_idx, other_group in enumerate(simultaneous_groups):
                    if other_idx in used_group_indices:
                        continue
                    
                    other_datetime = pd.to_datetime(other_group['datetime'].iloc[0])
                    
                    # Check if this group is within sequence_window_hours of any group in the sequence
                    for seq_group_idx in sequence_groups:
                        seq_group = simultaneous_groups[seq_group_idx]
                        seq_datetime = pd.to_datetime(seq_group['datetime'].iloc[0])
                        time_diff_hours = abs((other_datetime - seq_datetime).total_seconds() / 3600)
                        
                        if time_diff_hours <= sequence_window_hours:
                            sequence_groups.append(other_idx)
                            used_group_indices.add(other_idx)
                            changed = True
                            break
            
            sequence_clusters.append(sequence_groups)
        
        # Step 3: Create results DataFrame with both group_id and sequence_id
        print(f"Step 3/3: Creating results DataFrame from {len(sequence_clusters)} sequences...")
        simultaneous_events = []
        for sequence_id, sequence_group_indices in tqdm(enumerate(sequence_clusters), desc="Building DataFrame", total=len(sequence_clusters), unit="sequence"):
            for group_idx in sequence_group_indices:
                group = simultaneous_groups[group_idx]
                for idx, event in group.iterrows():
                    simultaneous_events.append({
                        'sequence_id': sequence_id,
                        'group_id': group_idx,  # Original group ID (same timestamp)
                        'timestamp': event['timestamp'],
                        'datetime': event['datetime'],
                        'prediction': event['prediction'],
                        'region_size': event['region_size'],
                        'max_flux': event['max_flux'],
                        'sum_flux': event['sum_flux'],
                        'centroid_img_y': event['centroid_img_y'],
                        'centroid_img_x': event['centroid_img_x'],
                        'group_size': len(group),
                        'sequence_size': len(sequence_group_indices)  # Number of groups in this sequence
                    })
        
        simultaneous_df = pd.DataFrame(simultaneous_events)
        
        if len(simultaneous_df) > 0:
            print(f"Detected {len(simultaneous_groups)} timestamps with simultaneous flares")
            print(f"Clustered into {len(sequence_clusters)} flare sequences (within ±{sequence_window_hours} hours)")
            print(f"Total simultaneous events: {len(simultaneous_df)}")
            
            # Print summary by sequence
            for sequence_id in sorted(simultaneous_df['sequence_id'].unique()):
                sequence_events = simultaneous_df[simultaneous_df['sequence_id'] == sequence_id]
                timestamps = sorted(sequence_events['datetime'].unique())
                print(f"\nFlare Sequence {sequence_id}:")
                print(f"  Number of groups: {sequence_events['sequence_size'].iloc[0]}")
                print(f"  Time span: {timestamps[0]} to {timestamps[-1]}")
                print(f"  Total events: {len(sequence_events)}")
                
                # Print each group in the sequence
                for group_id in sorted(sequence_events['group_id'].unique()):
                    group_events = sequence_events[sequence_events['group_id'] == group_id]
                    timestamp = group_events['timestamp'].iloc[0]
                    prediction = group_events['prediction'].iloc[0]
                    print(f"    Group {group_id} at {timestamp}:")
                    print(f"      Flux prediction: {prediction:.2e}")
                    print(f"      Number of regions: {len(group_events)}")
        
        self.simultaneous_flares_df = simultaneous_df
        return simultaneous_df

    def create_flare_event_summary(self, output_path=None):
        """Create a comprehensive summary of detected flare events"""
        if not hasattr(self, 'flare_events_df'):
            print("Please run detect_flare_events() first")
            return

        if len(self.flare_events_df) == 0:
            print("No flare events detected")
            return

        # Sort by prediction strength
        flare_events = self.flare_events_df.sort_values('prediction', ascending=False)

        print(f"\n{'=' * 80}")
        print(f"FLARE EVENT SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total events detected: {len(flare_events)}")
        print(f"Time range: {flare_events['datetime'].min()} to {flare_events['datetime'].max()}")

        # Top 10 strongest events
        print(f"\nTop 10 Strongest Flare Events:")
        print("-" * 100)
        print(f"{'Rank':<4} {'Timestamp':<20} {'Prediction':<12} {'Actual':<12} {'Region Size':<12} {'Max Flux':<12} {'Location':<15}")
        print("-" * 100)

        for i, (_, event) in enumerate(flare_events.head(10).iterrows()):
            location = f"({event['centroid_img_y']:.0f},{event['centroid_img_x']:.0f})"
            actual_value = f"{event['groundtruth']:.2e}" if 'groundtruth' in event and not pd.isna(event['groundtruth']) else "N/A"
            print(f"{i + 1:<4} {event['datetime'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{event['prediction']:<12.2e} {actual_value:<12} {event['region_size']:<12} "
                  f"{event['max_flux']:<12.2e} {location:<15}")

        # Statistics
        print(f"\nEvent Statistics:")
        print(f"  Mean region size: {flare_events['region_size'].mean():.1f} patches")
        print(f"  Mean prediction: {flare_events['prediction'].mean():.2e}")
        print(f"  Max prediction: {flare_events['prediction'].max():.2e}")
        print(f"  Min prediction: {flare_events['prediction'].min():.2e}")
        
        # Ground truth statistics if available
        if 'groundtruth' in flare_events.columns:
            valid_groundtruth = flare_events.dropna(subset=['groundtruth'])
            if len(valid_groundtruth) > 0:
                print(f"  Mean actual: {valid_groundtruth['groundtruth'].mean():.2e}")
                print(f"  Max actual: {valid_groundtruth['groundtruth'].max():.2e}")
                print(f"  Min actual: {valid_groundtruth['groundtruth'].min():.2e}")
                print(f"  Ground truth available for: {len(valid_groundtruth)}/{len(flare_events)} events")

        if output_path:
            flare_events.to_csv(output_path, index=False)
            print(f"\nFlare events saved to: {output_path}")

        return flare_events

    def _detect_regions_worker(self, timestamp):
        """Worker function for parallel region detection"""
        try:
            flux_contrib = self.load_flux_contributions(timestamp)
            if flux_contrib is None:
                return (timestamp, None)
                
            # Get prediction data
            pred_data = self.predictions_df[self.predictions_df['timestamp'] == timestamp]
            if pred_data.empty:
                return (timestamp, None)
            pred_data = pred_data.iloc[0]
            
            # Detect regions for this timestamp
            regions = self._detect_regions_with_dbscan(flux_contrib, timestamp, pred_data)
            return (timestamp, regions)
        except Exception as e:
            print(f"Error detecting regions for {timestamp}: {e}")
            return (timestamp, None)

    def _constrain_region_size_change(self, prev_region, new_region, max_growth_factor=2.0, max_shrinkage_factor=0.5):
        """
        Constrain how much a tracked region can grow or shrink between frames.
        
        Args:
            prev_region: Region dict from previous timestamp (same track)
            new_region: Region dict for current timestamp (same track)
            max_growth_factor: Maximum allowed size ratio (new_size / prev_size), e.g., 2.0 = can double (None to disable)
            max_shrinkage_factor: Minimum allowed size ratio (new_size / prev_size), e.g., 0.5 = can halve (None to disable)
            
        Returns:
            Modified new_region with constrained size
        """
        if 'mask' not in prev_region or 'mask' not in new_region:
            return new_region  # Can't constrain without masks
        
        prev_mask = prev_region['mask'].astype(bool)
        new_mask = new_region['mask'].astype(bool)
        
        prev_size = int(prev_mask.sum())
        new_size = int(new_mask.sum())
        
        # If no previous size, allow any size
        if prev_size == 0:
            return new_region
        
        # Calculate size ratio
        size_ratio = new_size / prev_size
        
        # Check if size change is within allowed limits (None disables the constraint)
        min_ratio = max_shrinkage_factor if max_shrinkage_factor is not None else 0
        max_ratio = max_growth_factor if max_growth_factor is not None else float('inf')
        
        if min_ratio <= size_ratio <= max_ratio:
            return new_region  # Size change is acceptable
        
        # Need to constrain the mask
        structure_8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        
        if max_growth_factor is not None and size_ratio > max_growth_factor:
            # Growing too fast: constrain to max_growth_factor * prev_size
            target_size = int(prev_size * max_growth_factor)
            
            # Start with previous mask and grow it iteratively
            constrained_mask = prev_mask.copy()
            current_size = prev_size
            
            # Grow by dilation until we reach target size or can't grow more
            iterations = 0
            max_iterations = 10  # Safety limit
            while current_size < target_size and iterations < max_iterations:
                # Dilate the mask
                dilated = nd.binary_dilation(constrained_mask, structure=structure_8, iterations=1)
                # Only add pixels that are in the new_mask (the detected region)
                new_pixels = dilated & new_mask & ~constrained_mask
                
                if np.sum(new_pixels) == 0:
                    break  # Can't grow more
                
                # Add pixels up to target size
                pixels_to_add = min(np.sum(new_pixels), target_size - current_size)
                if pixels_to_add > 0:
                    # Get indices of new pixels and add the closest ones
                    new_pixel_coords = np.where(new_pixels)
                    if len(new_pixel_coords[0]) > 0:
                        # Add pixels starting from those closest to existing mask
                        # Simple approach: add all available pixels up to limit
                        constrained_mask = constrained_mask | new_pixels
                        current_size = int(constrained_mask.sum())
                        if current_size >= target_size:
                            break
                
                iterations += 1
            
            # If still too large, randomly remove pixels to reach target
            if current_size > target_size:
                excess = current_size - target_size
                all_pixels = np.where(constrained_mask)
                if len(all_pixels[0]) > excess:
                    # Remove excess pixels (prefer removing from edges)
                    # Simple approach: remove pixels furthest from centroid
                    centroid_y = np.mean(all_pixels[0])
                    centroid_x = np.mean(all_pixels[1])
                    distances = np.sqrt((all_pixels[0] - centroid_y)**2 + (all_pixels[1] - centroid_x)**2)
                    # Remove pixels with largest distances
                    remove_indices = np.argsort(distances)[-excess:]
                    constrained_mask[all_pixels[0][remove_indices], all_pixels[1][remove_indices]] = False
        
        elif max_shrinkage_factor is not None and size_ratio < max_shrinkage_factor:
            # Shrinking too fast: constrain to max_shrinkage_factor * prev_size
            target_size = int(prev_size * max_shrinkage_factor)
            
            # Start with intersection of prev and new masks
            constrained_mask = prev_mask & new_mask
            current_size = int(constrained_mask.sum())
            
            # If intersection is too small, add pixels from new_mask closest to prev_mask
            if current_size < target_size:
                # Get pixels in new_mask but not in constrained_mask
                available_pixels = new_mask & ~constrained_mask
                pixels_needed = target_size - current_size
                
                if np.sum(available_pixels) > 0 and pixels_needed > 0:
                    # Add pixels closest to the previous mask
                    available_coords = np.where(available_pixels)
                    prev_coords = np.where(prev_mask)
                    
                    if len(available_coords[0]) > 0 and len(prev_coords[0]) > 0:
                        # Calculate distances from available pixels to previous mask centroid
                        prev_centroid_y = np.mean(prev_coords[0])
                        prev_centroid_x = np.mean(prev_coords[1])
                        distances = np.sqrt(
                            (available_coords[0] - prev_centroid_y)**2 + 
                            (available_coords[1] - prev_centroid_x)**2
                        )
                        # Add closest pixels
                        add_indices = np.argsort(distances)[:pixels_needed]
                        constrained_mask[available_coords[0][add_indices], available_coords[1][add_indices]] = True
        
        # Recompute region properties from constrained mask
        if np.sum(constrained_mask) > 0:
            # Need to get flux_contrib to recalculate flux
            # Try to get it from the region data or load it
            timestamp = new_region.get('timestamp', None)
            if timestamp:
                flux_contrib = self.load_flux_contributions(timestamp)
                if flux_contrib is not None:
                    region_flux = flux_contrib[constrained_mask]
                    new_region['sum_flux'] = float(np.sum(region_flux))
                    new_region['max_flux'] = float(np.max(region_flux)) if region_flux.size > 0 else 0.0
            
            # Recalculate centroid
            coords = np.where(constrained_mask)
            if len(coords[0]) > 0:
                centroid_y = np.mean(coords[0])
                centroid_x = np.mean(coords[1])
                new_region['centroid_patch_y'] = centroid_y
                new_region['centroid_patch_x'] = centroid_x
                new_region['centroid_img_y'] = centroid_y * self.patch_size + self.patch_size // 2
                new_region['centroid_img_x'] = centroid_x * self.patch_size + self.patch_size // 2
            
            new_region['mask'] = constrained_mask
            new_region['size'] = int(np.sum(constrained_mask))
        
        return new_region

    def track_regions_over_time(self, timestamps, max_distance=None):
        """
        Track regions across time using spatial proximity and temporal continuity.
        
        Args:
            timestamps: List of timestamps to analyze
            max_distance: Maximum distance (in pixels) to consider regions as the same
                         (if None, uses config value)
            
        Returns:
            Dictionary mapping region_id to list of (timestamp, region_data) tuples
        """
        if max_distance is None:
            max_distance = self.output_config.get('max_tracking_distance', 75)
        
        print("Tracking regions across time...")
        
        # Store all regions from all timestamps
        all_regions = {}  # timestamp -> list of regions
        region_tracks = {}  # track_id -> list of (timestamp, region_data)
        next_track_id = 1
        
        # First pass: collect all regions using multiprocessing
        print("Phase 1/2: Detecting regions at each timestamp (parallel)...")
        num_processes = min(os.cpu_count(), len(timestamps))
        num_processes = max(1, num_processes - 1)  # Leave one CPU free
        print(f"Using {num_processes} processes for region detection")
        
        with Pool(processes=num_processes) as pool:
            # Use imap for progress tracking
            results = list(tqdm(pool.imap(self._detect_regions_worker, timestamps),
                               desc="Detecting regions", unit="timestamp", total=len(timestamps)))
        
        # Collect results
        for timestamp, regions in results:
            if regions is not None:
                all_regions[timestamp] = regions
            
        # Second pass: track regions across time
        print("Phase 2/2: Tracking regions across timestamps...")
        
        # Track active tracks (those updated recently) for optimization
        active_tracks = set()
        max_time_gap = 60 * 60  # 60 minutes in seconds
        
        for i, timestamp in tqdm(enumerate(timestamps), desc="Tracking regions", unit="timestamp", total=len(timestamps)):
            current_time = pd.to_datetime(timestamp)
            
            # Get recently active tracks (within max_time_gap) for faster matching
            recently_active_tracks = set()
            for track_id in list(active_tracks):
                if track_id in region_tracks and region_tracks[track_id]:
                    last_timestamp, _ = region_tracks[track_id][-1]
                    time_diff = abs((current_time - pd.to_datetime(last_timestamp)).total_seconds())
                    if time_diff <= max_time_gap:
                        recently_active_tracks.add(track_id)
            
            # Process regions detected at this timestamp
            if timestamp in all_regions:
                current_regions = all_regions[timestamp]
                
                for region in current_regions:
                    region_copy = region.copy()
                    
                    # Find best matching existing track based on spatial proximity
                    best_track_id = None
                    best_distance = float('inf')
                    
                    for track_id in recently_active_tracks:
                        if not region_tracks[track_id]:
                            continue
                            
                        last_timestamp, last_region = region_tracks[track_id][-1]
                        
                        # Calculate Euclidean distance between centroids
                        distance = np.sqrt(
                            (region_copy['centroid_img_x'] - last_region['centroid_img_x'])**2 + 
                            (region_copy['centroid_img_y'] - last_region['centroid_img_y'])**2
                        )
                        
                        if distance < max_distance and distance < best_distance:
                            best_distance = distance
                            best_track_id = track_id
                    
                    # Add region to existing track or create new track
                    if best_track_id is not None:
                        region_copy['id'] = best_track_id
                        region_copy['timestamp'] = timestamp  # Add timestamp for constraint method
                        
                        # Constrain region size change to prevent rapid growth/shrinkage
                        last_timestamp, last_region = region_tracks[best_track_id][-1]
                        max_growth_factor = self.flare_config.get('max_size_growth_factor', 2.0)
                        max_shrinkage_factor = self.flare_config.get('max_size_shrinkage_factor', 0.5)
                        # Allow None or 0 to disable constraints
                        if max_growth_factor == 0:
                            max_growth_factor = None
                        if max_shrinkage_factor == 0:
                            max_shrinkage_factor = None
                        
                        # Only apply constraint if both factors are enabled
                        if max_growth_factor is not None or max_shrinkage_factor is not None:
                            region_copy = self._constrain_region_size_change(
                                prev_region=last_region,
                                new_region=region_copy,
                                max_growth_factor=max_growth_factor,
                                max_shrinkage_factor=max_shrinkage_factor
                            )
                        
                        region_tracks[best_track_id].append((timestamp, region_copy))
                        active_tracks.add(best_track_id)
                    else:
                        region_copy['id'] = next_track_id
                        region_copy['timestamp'] = timestamp
                        region_tracks[next_track_id] = [(timestamp, region_copy)]
                        active_tracks.add(next_track_id)
                        next_track_id += 1
        
        # Filter out tracks with only one region (no temporal continuity)
        region_tracks = {k: v for k, v in region_tracks.items() if len(v) > 1}
        
        print(f"Found {len(region_tracks)} region tracks across {len(timestamps)} timestamps")
        
        # Apply temporal smoothing to flux values
        smoothing_window = self.flare_config.get('flux_smoothing_window', 3)
        if smoothing_window > 1:
            region_tracks = self._apply_temporal_smoothing(region_tracks, smoothing_window)
        
        
        return region_tracks
    
    def _apply_temporal_smoothing(self, region_tracks, window_size=3):
        """
        Apply temporal smoothing to region flux values and size to reduce unrealistic jumps.
        
        Args:
            region_tracks: Dictionary of track_id -> list of (timestamp, region_data)
            window_size: Size of moving average window
            
        Returns:
            Updated region_tracks with smoothed flux values and sizes
        """
        print(f"Applying temporal smoothing with window size {window_size}...")
        
        for track_id, track_history in region_tracks.items():
            # Extract flux values and sizes
            flux_values = [r['sum_flux'] for t, r in track_history]
            sizes = [r.get('size', 0) for t, r in track_history]
            
            # Apply moving average to flux
            smoothed_flux = pd.Series(flux_values).rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).mean().tolist()
            
            # Apply moving average to size (to reduce fluctuations)
            smoothed_sizes = pd.Series(sizes).rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).mean().tolist()
            
            # Update regions with smoothed values
            for i, (t, r) in enumerate(track_history):
                r['sum_flux_original'] = r['sum_flux']  # Keep original
                r['size_original'] = r.get('size', 0)  # Keep original size
                r['sum_flux'] = smoothed_flux[i]  # Use smoothed for display
                r['size'] = int(round(smoothed_sizes[i]))  # Use smoothed size (rounded to integer)
                r['prediction'] = smoothed_flux[i]  # Update prediction too
        
        return region_tracks

    def _merge_close_cores(self, labeled_cores, num_cores, merge_distance, structure):
        """
        Merge core regions that are close based on centroid distance.
        
        This method calculates centroids for each core and merges cores that are
        within merge_distance patches of each other, preventing unrelated distant
        cores from being merged.
        
        Args:
            labeled_cores: Labeled array of core regions
            num_cores: Number of distinct core regions
            merge_distance: Maximum distance (in patches) to merge cores
            structure: Structuring element for labeling (not used in new approach)
            
        Returns:
            Tuple of (merged_labeled_cores, num_merged_cores)
        """
        if num_cores == 0 or merge_distance <= 0:
            return labeled_cores, num_cores
        
        # Calculate centroids for each core
        core_centroids = {}
        for core_id in range(1, num_cores + 1):
            core_mask = labeled_cores == core_id
            coords = np.where(core_mask)
            if len(coords[0]) > 0:
                core_centroids[core_id] = (np.mean(coords[0]), np.mean(coords[1]))
        
        if len(core_centroids) == 0:
            return labeled_cores, num_cores
        
        # Merge cores that are within merge_distance patches
        used_cores = set()
        new_label = 1
        label_map = {}  # Map old core_id to new merged label
        
        for core_id in range(1, num_cores + 1):
            if core_id in used_cores or core_id not in core_centroids:
                continue
            
            # Start a new merged region
            used_cores.add(core_id)
            label_map[core_id] = new_label
            
            # Find nearby cores to merge
            cy, cx = core_centroids[core_id]
            for other_id in range(core_id + 1, num_cores + 1):
                if other_id in used_cores or other_id not in core_centroids:
                    continue
                
                oy, ox = core_centroids[other_id]
                distance = np.sqrt((cy - oy)**2 + (cx - ox)**2)
                
                if distance <= merge_distance:
                    # Merge this core into the current merged region
                    used_cores.add(other_id)
                    label_map[other_id] = new_label
            
            new_label += 1
        
        # Create final merged labeled array
        merged_labeled_cores = np.zeros_like(labeled_cores, dtype=int)
        for old_id, new_id in label_map.items():
            merged_labeled_cores[labeled_cores == old_id] = new_id
        
        num_merged_cores = len(set(label_map.values()))
        
        return merged_labeled_cores, num_merged_cores

    def _detect_regions_with_dbscan(self, flux_contrib, timestamp, pred_data):
        """
        Region detection using DBSCAN clustering.
        
        This method:
        1. Filters patches using median + std threshold
        2. Clusters significant patches using DBSCAN
        3. Creates regions from clusters
        """
        # Get DBSCAN config values
        threshold_std_multiplier = self.flare_config.get('threshold_std_multiplier', 2.0)
        dbscan_eps = self.flare_config.get('dbscan_eps', 2.0)
        dbscan_min_samples = self.flare_config.get('dbscan_min_samples', 5)
        dbscan_use_flux = self.flare_config.get('dbscan_use_flux', False)
        dbscan_flux_weight = self.flare_config.get('dbscan_flux_weight', 1.0)
        min_patches = self.flare_config.get('min_patches', 20)
        max_patches = self.flare_config.get('max_patches', 300)
        min_flux_threshold = self.flare_config.get('min_flux_threshold', None)
        
        # Allow None to disable constraints
        if min_patches == 0:
            min_patches = None
        if max_patches == 0:
            max_patches = None
        
        # Step 1: Filter patches using median + std threshold
        flux_values = flux_contrib.flatten()
        # Filter out zero/negative values for log calculation
        positive_flux = flux_values[flux_values > 0]
        if len(positive_flux) == 0:
            return []
        
        log_flux = np.log(positive_flux)
        median_log_flux = np.median(log_flux)
        std_log_flux = np.std(log_flux)
        threshold = np.exp(median_log_flux + threshold_std_multiplier * std_log_flux)
        
        # Find significant patches
        significant_mask = flux_contrib > threshold
        significant_patches = np.where(significant_mask)
        
        if len(significant_patches[0]) == 0:
            return []
        
        # Get patch coordinates and flux values
        patch_y = significant_patches[0]
        patch_x = significant_patches[1]
        patch_flux = flux_contrib[significant_patches]
        
        # Step 2: Prepare features for DBSCAN
        if dbscan_use_flux:
            # Use spatial coordinates + log flux
            # Filter out zero/negative flux
            valid_mask = patch_flux > 0
            if np.sum(valid_mask) == 0:
                return []
            
            patch_y = patch_y[valid_mask]
            patch_x = patch_x[valid_mask]
            patch_flux = patch_flux[valid_mask]
            
            log_flux_values = np.log(patch_flux)
            # Normalize log flux (zero mean, unit std) and apply weight
            log_flux_normalized = (log_flux_values - np.mean(log_flux_values)) / (np.std(log_flux_values) + 1e-10)
            log_flux_scaled = log_flux_normalized * dbscan_flux_weight
            
            features = np.column_stack([patch_y, patch_x, log_flux_scaled])
        else:
            # Use only spatial coordinates
            features = np.column_stack([patch_y, patch_x])
        
        # Step 3: Apply DBSCAN clustering
        if len(features) < dbscan_min_samples:
            return []
        
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        cluster_labels = dbscan.fit_predict(features)
        
        # Get unique cluster IDs (excluding noise: -1)
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        if len(unique_labels) == 0:
            return []
        
        # Step 4: Create regions from clusters
        regions = []
        structure_8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        
        for cluster_id in unique_labels:
            cluster_mask = cluster_labels == cluster_id
            
            # Get patches in this cluster
            cluster_y = patch_y[cluster_mask]
            cluster_x = patch_x[cluster_mask]
            cluster_flux = patch_flux[cluster_mask]
            
            # Create binary mask for this cluster
            region_mask = np.zeros_like(flux_contrib, dtype=bool)
            region_mask[cluster_y, cluster_x] = True
            
            # Apply morphological closing to fill small gaps
            region_mask = nd.binary_closing(region_mask, structure=structure_8, iterations=2)
            
            region_size = np.sum(region_mask)
            
            # Size filtering
            min_check = (min_patches is None) or (region_size >= min_patches)
            max_check = (max_patches is None) or (region_size <= max_patches)
            if not (min_check and max_check):
                continue
            
            # Calculate region properties
            region_flux_values = flux_contrib[region_mask]
            sum_flux = np.sum(region_flux_values)
            max_flux = np.max(region_flux_values)
            
            # Filter by minimum flux threshold if enabled
            if min_flux_threshold is not None and sum_flux < min_flux_threshold:
                continue
            
            # Get region centroid
            coords = np.where(region_mask)
            centroid_y, centroid_x = np.mean(coords[0]), np.mean(coords[1])
            
            # Convert to image coordinates
            img_y = centroid_y * self.patch_size + self.patch_size // 2
            img_x = centroid_x * self.patch_size + self.patch_size // 2
            
            # Calculate region-based prediction
            region_prediction = sum_flux
            
            # Calculate label position
            min_y, max_y = np.min(coords[0]), np.max(coords[0])
            min_x, max_x = np.min(coords[1]), np.max(coords[1])
            label_y = max(0, min_y - 2)
            label_x = centroid_x
            
            regions.append({
                'id': len(regions) + 1,
                'size': region_size,
                'sum_flux': sum_flux,
                'max_flux': max_flux,
                'centroid_patch_y': centroid_y,
                'centroid_patch_x': centroid_x,
                'centroid_img_y': img_y,
                'centroid_img_x': img_x,
                'label_y': label_y,
                'label_x': label_x,
                'mask': region_mask,
                'prediction': region_prediction,
                'groundtruth': pred_data.get('groundtruth', None)
            })
        
        if len(regions) > 0:
            print(f"  {timestamp}: Found {len(regions)} regions using DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples})")
        
        return regions
    
    def _verify_no_overlaps(self, regions, timestamp):
        """Verify that regions don't overlap with each other"""
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions[i+1:], i+1):
                overlap = np.any(region1['mask'] & region2['mask'])
                if overlap:
                    overlap_pixels = np.sum(region1['mask'] & region2['mask'])
                    print(f"    WARNING: Regions {region1['id']} and {region2['id']} overlap by {overlap_pixels} pixels!")

    def create_contour_movie(self, timestamps, auto_cleanup=True, fps=2, show_sxr_timeseries=True, 
                           all_timestamps_for_tracking=None, movie_filename=None):
        """
        Create a movie showing the evolution of contour plots over time with SXR time series
        
        Args:
            timestamps: Timestamps to generate frames for (subsampled for visualization)
            auto_cleanup: Whether to delete frame files after movie creation
            fps: Frames per second for the movie
            show_sxr_timeseries: Whether to show SXR time series plot
            all_timestamps_for_tracking: Full resolution timestamps for accurate region tracking
                                        (if None, uses timestamps parameter)
            movie_filename: Custom filename for the movie (if None, uses default naming)
        """
        print(f"Creating contour movie with {len(timestamps)} frame timestamps...")

        # Always track regions across time for consistent region IDs and colors
        tracking_timestamps = all_timestamps_for_tracking if all_timestamps_for_tracking is not None else timestamps
        print(f"Tracking regions across {len(tracking_timestamps)} timestamps (full resolution)...")
        region_tracks = self.track_regions_over_time(tracking_timestamps)

        # Create frames directory
        self.frames_dir = Path("temp_contour_frames")
        self.frames_dir.mkdir(exist_ok=True)

        # Determine number of processes
        num_processes = min(os.cpu_count(), len(timestamps))  # Don't use more processes than timestamps
        num_processes = max(1, num_processes - 1)  # Leave one CPU free
        print(f"Using {num_processes} processes")
        

        # Process frames in parallel
        start_time = time.time()

        # Always use tracked regions for consistent region IDs and colors
        from functools import partial
        frame_worker = partial(
            self.generate_contour_frame_with_sxr_worker, 
            region_tracks=region_tracks,
            show_sxr_timeseries=show_sxr_timeseries
        )
        
        with Pool(processes=num_processes) as pool:
            results = []
            for result in tqdm(pool.imap(frame_worker, timestamps, chunksize=1), 
                              desc="Generating frames", unit="frame", total=len(timestamps)):
                results.append(result)

        # Filter out failed frames
        frame_paths = [path for path in results if path is not None]

        processing_time = time.time() - start_time
        print(f"Generated {len(frame_paths)} frames in {processing_time:.2f} seconds")
        if len(frame_paths) > 0:
            print(f"Average: {processing_time / len(frame_paths):.2f} seconds per frame")

        if not frame_paths:
            print("No frames were successfully generated. Cannot create movie.")
            return

        # Compile into video
        print("Creating movie...")
        video_start = time.time()

        # Sort frame paths by timestamp to ensure correct order
        frame_paths.sort(key=lambda x: os.path.basename(x))

        # Use custom filename if provided, otherwise use default
        if movie_filename:
            movie_path = os.path.join(self.output_dir, movie_filename)
        else:
            movie_path = os.path.join(self.output_dir, f"contour_evolution_{timestamps[0].split('T')[0]}.mp4")
        # Use faster encoding settings: preset=faster, crf=23 for good quality at faster speed
        with imageio.get_writer(movie_path, fps=fps, codec='libx264', format='ffmpeg',
                                pixelformat='yuv420p', 
                                output_params=['-preset', 'faster', '-crf', '23']) as writer:
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    image = imageio.imread(frame_path)
                    writer.append_data(image)

        video_time = time.time() - video_start
        total_time = time.time() - start_time

        print(f"Video creation took {video_time:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"✅ Contour movie saved to: {movie_path}")

        # Optional: Clean up frame files
        if auto_cleanup:
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            print("Frame files cleaned up")
        else:
            cleanup = input("Delete individual frame files? (y/n): ").lower().strip()
            if cleanup == 'y':
                for frame_path in frame_paths:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                print("Frame files deleted")

        return movie_path

    def generate_contour_frame_with_sxr_worker(self, timestamp, region_tracks=None, show_sxr_timeseries=True):
        """Worker function to generate a single contour frame with optional SXR time series for tracked regions"""
        try:
            # Load flux contributions
            flux_contrib = self.load_flux_contributions(timestamp)
            aia = self.load_aia_image(timestamp)

            if flux_contrib is None:
                print(f"Worker {os.getpid()}: Skipping {timestamp} (missing flux data)")
                return None

            # Get prediction data for this timestamp
            pred_data = self.predictions_df[self.predictions_df['timestamp'] == timestamp]
            if pred_data.empty:
                print(f"Worker {os.getpid()}: Skipping {timestamp} (missing prediction data)")
                return None
            
            pred_data = pred_data.iloc[0]

            # Generate frame
            save_path = os.path.join(self.frames_dir, f"{timestamp}.png")
            os.makedirs(self.frames_dir, exist_ok=True)

            # Create figure - adjust layout based on whether SXR plot is shown
            if show_sxr_timeseries:
                fig = plt.figure(figsize=(16, 8), dpi=100)
                gs = fig.add_gridspec(1, 2, left=0.05, right=0.95, bottom=0.1, top=0.9, 
                                      height_ratios=[1], width_ratios=[1, 1], hspace=0, wspace=0.1)
            else:
                fig = plt.figure(figsize=(8, 8), dpi=100)
                gs = fig.add_gridspec(1, 1, left=0.05, right=0.95, bottom=0.1, top=0.9)

            # Get regions that exist at this timestamp from the tracked regions
            detected_regions = []
            if region_tracks:
                # Find regions that exist at this timestamp
                for track_id, track_history in region_tracks.items():
                    for track_timestamp, region_data in track_history:
                        if track_timestamp == timestamp:
                            # Ensure all required keys are present for plotting
                            region_data_copy = region_data.copy()
                            
                            # Add label coordinates if not present
                            if 'label_x' not in region_data_copy or 'label_y' not in region_data_copy:
                                centroid_y = region_data_copy.get('centroid_patch_y', 
                                                                  region_data_copy.get('centroid_y', 0))
                                centroid_x = region_data_copy.get('centroid_patch_x', 
                                                                  region_data_copy.get('centroid_x', 0))
                                region_data_copy['label_y'] = max(0, centroid_y - 2)
                                region_data_copy['label_x'] = centroid_x
                            
                            detected_regions.append(region_data_copy)
                            break

            # Set up color mapping for regions (consistent colors across frames)
            region_colors = [
                '#000000', '#004949', '#009292', '#FF6DB6', '#FFB6DB', '#490092', 
                '#006DDB', '#B66DFF', '#6DB6FF', '#B6DBFF', '#920000', '#924900', 
                '#DB6D00', '#24FF24', '#D82632'
            ]
            region_to_color = {}
            
            if region_tracks and detected_regions:
                # Use track IDs for consistent color assignment across frames
                track_ids = sorted(region_tracks.keys())
                for i, track_id in enumerate(track_ids):
                    region_to_color[track_id] = region_colors[i % len(region_colors)]
            else:
                # Fallback if no tracks (shouldn't happen, but handle gracefully)
                fallback_colors = plt.cm.Set3(np.linspace(0, 1, max(len(detected_regions), 1)))
                region_to_color = {region['id']: fallback_colors[i % len(fallback_colors)] 
                                 for i, region in enumerate(detected_regions)}

            # Plot AIA image with flux overlay
            if aia is not None:
                ax_aia = fig.add_subplot(gs[0, 0])
                
                # Display AIA image (RGB composite from 94, 131, 171 channels)
                im2 = ax_aia.imshow(aia, interpolation='nearest', origin='lower')
                
                # Add contours for detected regions with colors that match SXR plot
                if detected_regions:
                    # Draw contours with matching colors (already set up in region_to_color)
                    for region in detected_regions:
                        region_mask_resized = cv2.resize(region['mask'].astype(np.float32), 
                                                        (aia.shape[1], aia.shape[0]), 
                                                        interpolation=cv2.INTER_NEAREST)
                        color = region_to_color.get(region['id'], 'gray')
                        ax_aia.contour(region_mask_resized, levels=[0.5], colors=[color], 
                                     linewidths=3, alpha=0.9)
                        
                        # Add region labels
                        # Find the bottommost pixel of the region mask in image coordinates
                        region_mask_resized_y, region_mask_resized_x = np.where(region_mask_resized)
                        if len(region_mask_resized_y) > 0:
                            # Get the bottommost and topmost y coordinates
                            bottom_y_img = int(np.max(region_mask_resized_y))
                            top_y_img = int(np.min(region_mask_resized_y))
                            # Calculate region height for dynamic offset
                            region_height = bottom_y_img - top_y_img
                            # Use region height as offset (with minimum of 20 pixels)
                            offset = max(region_height * 0.3, 10)
                            
                            # Center horizontally on the region
                            label_x_img = int(np.mean(region_mask_resized_x))
                            # Place label below the bottommost pixel
                            label_y_img = int(bottom_y_img + offset)
                        else:
                            # Fallback to centroid if mask is empty
                            label_y_img = int(region['centroid_img_y'] * aia.shape[0] / self.input_size)
                            label_x_img = int(region['centroid_img_x'] * aia.shape[1] / self.input_size)
                        
                        ax_aia.text(label_x_img, label_y_img,
                                   f"R{region['id']}\n{region['sum_flux']:.1e}",
                                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color),
                                   color='black')
                
                ax_aia.set_title(f'AIA 131/171/304 Å Composite\n{timestamp}', fontsize=12)
                ax_aia.axis('off')
                


            # Plot SXR time series if requested
            if show_sxr_timeseries and region_tracks:
                # Create single SXR plot in right column
                ax_sxr = fig.add_subplot(gs[0, 1])
                
                current_time = pd.to_datetime(timestamp)
                
                # Collect all track data for integrated plotting
                all_track_data = []
                for track_id, track_history in region_tracks.items():
                    # Skip empty tracks
                    if not track_history:
                        continue
                    
                    track_timestamps = [t for t, r in track_history]
                    track_predictions = [r['prediction'] for t, r in track_history]
                    track_datetimes = pd.to_datetime(track_timestamps)
                    
                    # Check for any issues with track data
                    if len(track_timestamps) != len(track_predictions):
                        continue
                    
                    # Calculate max flux safely
                    try:
                        max_flux = max((r['sum_flux'] for t, r in track_history), default=0)
                    except (KeyError, ValueError):
                        max_flux = 0
                    
                    all_track_data.append({
                        'track_id': track_id,
                        'timestamps': track_timestamps,
                        'datetimes': track_datetimes,
                        'predictions': track_predictions,
                        'max_flux': max_flux
                    })
                
                # Sort by maximum flux to get top regions
                all_track_data.sort(key=lambda x: x['max_flux'], reverse=True)

                # 1. Plot integrated ground truth (if available) - RED
                if pred_data.get('groundtruth') is not None:
                    # Get ground truth for all timestamps in the time window
                    time_window = pd.Timedelta(hours=2)
                    window_start = current_time - time_window
                    window_end = current_time + time_window
                    
                    # Get ground truth data for this window
                    window_data = self.predictions_df[
                        (self.predictions_df['datetime'] >= window_start) & 
                        (self.predictions_df['datetime'] <= window_end)
                    ]
                    
                    if not window_data.empty:
                        ax_sxr.plot(window_data['datetime'], window_data['groundtruth'], 
                                   's-', color='#fb8072', linewidth=3, markersize=4,
                                   label='Ground Truth (Actual SXR)', alpha=0.8)
                
                # 2. Plot integrated model prediction - BLUE
                # Get model predictions from predictions_df
                time_window = pd.Timedelta(hours=2)
                window_start = current_time - time_window
                window_end = current_time + time_window
                
                window_predictions = self.predictions_df[
                    (self.predictions_df['datetime'] >= window_start) & 
                    (self.predictions_df['datetime'] <= window_end)
                ]
                
                if not window_predictions.empty:
                    ax_sxr.plot(window_predictions['datetime'], window_predictions['predictions'], 
                               'D-', color='#80b1d3', linewidth=3, markersize=4,
                               label='Model Prediction', alpha=0.8)
                
                # 3. Plot individual region flux contributions - each region gets its own colored line
                # Use consistent colors for each region (matching the contours on AIA image)
                # Extended color palette to avoid rapid repetition with many regions
                region_colors_list = [
                  '#000000', '#004949','#009292', '#FF6DB6', '#FFB6DB', '#490092', '#006DDB', '#B66DFF', '#6DB6FF', '#B6DBFF', '#920000', '#924900', '#DB6D00', '#24FF24','#D82632'      
                ]
                
                # Get sorted track IDs for consistent color assignment
                track_ids = sorted(region_tracks.keys())
                
                current_datetime = pd.to_datetime(timestamp)
                
                # Get IDs of currently visible regions
                current_region_ids = set(r['id'] for r in detected_regions)
                
                for i, track_id in enumerate(track_ids):
                    track_history = region_tracks[track_id]
                    
                    # Extract timestamps and sum_flux values for this region
                    track_timestamps = []
                    track_flux_values = []
                    
                    for track_timestamp, region_data in track_history:
                        ts_datetime = pd.to_datetime(track_timestamp)
                        # Only include data up to current time
                        if ts_datetime <= current_datetime:
                            track_timestamps.append(ts_datetime)
                            track_flux_values.append(region_data.get('sum_flux', 0))
                    
                    if len(track_timestamps) > 0:
                        color = region_colors_list[i % len(region_colors_list)]
                        # Only show in legend if region is currently visible
                        if track_id in current_region_ids:
                            ax_sxr.plot(track_timestamps, track_flux_values, 
                                       'o-', color=color, linewidth=2, markersize=4,
                                       label=f'Region {track_id}', alpha=0.7)
                        else:
                            # Plot without label (won't appear in legend)
                            ax_sxr.plot(track_timestamps, track_flux_values, 
                                       'o-', color=color, linewidth=2, markersize=4,
                                       alpha=0.3)  # Dimmer for inactive regions
                
                # Mark current time
                ax_sxr.axvline(current_time, color='black', linestyle='--', alpha=0.7, linewidth=2)
                
                ax_sxr.set_title('SXR Time Series', fontsize=12)
                ax_sxr.set_ylabel('SXR Flux (W/m²)', fontsize=12)
                ax_sxr.set_xlabel('Time', fontsize=12)
                
                # Only use log scale if we have positive values
                try:
                    y_data = []
                    for line in ax_sxr.get_lines():
                        y_data.extend(line.get_ydata())
                    y_data = [y for y in y_data if y > 0]  # Filter out non-positive values
                    
                    if len(y_data) > 0 and min(y_data) > 0:
                        ax_sxr.set_yscale('log')
                    else:
                        ax_sxr.set_yscale('linear')
                except Exception:
                    ax_sxr.set_yscale('linear')
                
                ax_sxr.grid(True, alpha=0.3)
                ax_sxr.legend(fontsize=12, loc='upper right')
                
                # Set x-axis limits to show reasonable time window
                time_window = pd.Timedelta(hours=2)
                ax_sxr.set_xlim([current_time - time_window, current_time + time_window])
                
                # Rotate x-axis labels
                ax_sxr.tick_params(axis='x', rotation=0, labelsize=10)

            # Don't use tight_layout or bbox_inches='tight' - keeps frame size consistent
            plt.savefig(save_path, dpi=100)  # Reduced DPI for speed
            plt.close(fig)  # Close specific figure instead of all
            return save_path

        except Exception as e:
            print(f"Worker {os.getpid()}: Error processing {timestamp}: {e}")
            plt.close('all')
            return None

    def generate_flux_heatmap_frame_worker(self, timestamp, region_tracks=None, show_sxr_timeseries=True):
        """Worker function to generate a single flux heat map frame"""
        try:
            # Load flux contributions
            flux_contrib = self.load_flux_contributions(timestamp)
            
            if flux_contrib is None:
                print(f"Worker {os.getpid()}: Skipping {timestamp} (missing flux data)")
                return None

            # Get prediction data for this timestamp
            pred_data = self.predictions_df[self.predictions_df['timestamp'] == timestamp]
            if pred_data.empty:
                print(f"Worker {os.getpid()}: Skipping {timestamp} (missing prediction data)")
                return None
            
            pred_data = pred_data.iloc[0]

            # Generate frame
            save_path = os.path.join(self.frames_dir, f"flux_{timestamp}.png")
            os.makedirs(self.frames_dir, exist_ok=True)

            # Create figure - use same size as contour frames for consistent movie dimensions
            if show_sxr_timeseries:
                fig = plt.figure(figsize=(16, 8), dpi=100)
                gs = fig.add_gridspec(1, 2, left=0.05, right=0.95, bottom=0.1, top=0.9, 
                                      height_ratios=[1], width_ratios=[1, 1], hspace=0, wspace=0.1)
            else:
                fig = plt.figure(figsize=(8, 8), dpi=100)
                gs = fig.add_gridspec(1, 1, left=0.05, right=0.95, bottom=0.1, top=0.9)

            # Get regions that exist at this timestamp from the tracked regions
            detected_regions = []
            if region_tracks:
                for track_id, track_history in region_tracks.items():
                    for track_timestamp, region_data in track_history:
                        if track_timestamp == timestamp:
                            detected_regions.append(region_data.copy())
                            break

            # Set up color mapping for regions
            region_colors = [
                '#000000', '#004949', '#009292', '#FF6DB6', '#FFB6DB', '#490092', 
                '#006DDB', '#B66DFF', '#6DB6FF', '#B6DBFF', '#920000', '#924900', 
                '#DB6D00', '#24FF24', '#D82632'
            ]
            region_to_color = {}
            
            if region_tracks and detected_regions:
                track_ids = sorted(region_tracks.keys())
                for i, track_id in enumerate(track_ids):
                    region_to_color[track_id] = region_colors[i % len(region_colors)]

            # Plot flux heat map
            ax_flux = fig.add_subplot(gs[0, 0])
            
            # Use log scale for better visualization with fixed range
            flux_log = np.log10(flux_contrib + 1e-10)  # Add small value to avoid log(0)
            
            # Create heat map with fixed colorbar range (1e-10 to 1e-7)
            vmin_fixed = np.log10(1e-10)
            vmax_fixed = np.log10(1e-7)
            im = ax_flux.imshow(flux_log, cmap='hot', interpolation='nearest', origin='lower',
                              aspect='auto', vmin=vmin_fixed, vmax=vmax_fixed)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_flux, label='Log10(Flux Contribution)')
            
            # Add contours for detected regions
            if detected_regions:
                for region in detected_regions:
                    if 'mask' in region:
                        color = region_to_color.get(region['id'], 'cyan')
                        ax_flux.contour(region['mask'], levels=[0.5], colors=[color], 
                                       linewidths=2, alpha=0.8)
                        
                        # Add region labels
                        if 'centroid_patch_y' in region and 'centroid_patch_x' in region:
                            ax_flux.text(region['centroid_patch_x'], region['centroid_patch_y'],
                                       f"R{region['id']}\n{region.get('sum_flux', 0):.1e}",
                                       ha='center', va='center', fontsize=10, fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                                alpha=0.8, edgecolor=color),
                                       color='black')
            
            ax_flux.set_title(f'Flux Contribution Heat Map\n{timestamp}', fontsize=14, fontweight='bold')
            ax_flux.set_xlabel('Patch X', fontsize=12)
            ax_flux.set_ylabel('Patch Y', fontsize=12)
            
            # Plot SXR time series if requested
            current_time = pd.to_datetime(timestamp)
            if show_sxr_timeseries and region_tracks:
                ax_sxr = fig.add_subplot(gs[0, 1])
                
                # Collect all track data
                all_track_data = []
                for track_id, track_history in region_tracks.items():
                    if not track_history:
                        continue
                    
                    track_timestamps = [t for t, r in track_history]
                    track_flux_values = [r.get('sum_flux', 0) for t, r in track_history]
                    track_datetimes = pd.to_datetime(track_timestamps)
                    
                    try:
                        max_flux = max((r.get('sum_flux', 0) for t, r in track_history), default=0)
                    except (KeyError, ValueError):
                        max_flux = 0
                    
                    all_track_data.append({
                        'track_id': track_id,
                        'timestamps': track_timestamps,
                        'datetimes': track_datetimes,
                        'flux_values': track_flux_values,
                        'max_flux': max_flux
                    })
                
                all_track_data.sort(key=lambda x: x['max_flux'], reverse=True)
                
                # Plot ground truth and prediction
                time_window = pd.Timedelta(hours=2)
                window_start = current_time - time_window
                window_end = current_time + time_window
                
                if pred_data.get('groundtruth') is not None:
                    window_data = self.predictions_df[
                        (self.predictions_df['datetime'] >= window_start) & 
                        (self.predictions_df['datetime'] <= window_end)
                    ]
                    if not window_data.empty:
                        ax_sxr.plot(window_data['datetime'], window_data['groundtruth'], 
                                   's-', color='#fb8072', linewidth=2, markersize=3,
                                   label='Ground Truth', alpha=0.8)
                
                window_predictions = self.predictions_df[
                    (self.predictions_df['datetime'] >= window_start) & 
                    (self.predictions_df['datetime'] <= window_end)
                ]
                if not window_predictions.empty:
                    ax_sxr.plot(window_predictions['datetime'], window_predictions['predictions'], 
                               'D-', color='#80b1d3', linewidth=2, markersize=3,
                               label='Model Prediction', alpha=0.8)
                
                # Plot individual region flux contributions
                region_colors_list = [
                    '#000000', '#004949', '#009292', '#FF6DB6', '#FFB6DB', '#490092', 
                    '#006DDB', '#B66DFF', '#6DB6FF', '#B6DBFF', '#920000', '#924900', 
                    '#DB6D00', '#24FF24', '#D82632'
                ]
                
                track_ids = sorted(region_tracks.keys())
                current_region_ids = set(r['id'] for r in detected_regions)
                
                for i, track_id in enumerate(track_ids):
                    track_history = region_tracks[track_id]
                    track_timestamps = []
                    track_flux_values = []
                    
                    for track_timestamp, region_data in track_history:
                        ts_datetime = pd.to_datetime(track_timestamp)
                        if ts_datetime <= current_time:
                            track_timestamps.append(ts_datetime)
                            track_flux_values.append(region_data.get('sum_flux', 0))
                    
                    if len(track_timestamps) > 0:
                        color = region_colors_list[i % len(region_colors_list)]
                        if track_id in current_region_ids:
                            ax_sxr.plot(track_timestamps, track_flux_values, 
                                       'o-', color=color, linewidth=1.5, markersize=3,
                                       label=f'Region {track_id}', alpha=0.7)
                        else:
                            ax_sxr.plot(track_timestamps, track_flux_values, 
                                       'o-', color=color, linewidth=1.5, markersize=3,
                                       alpha=0.3)
                
                ax_sxr.axvline(current_time, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
                ax_sxr.set_title('SXR Time Series', fontsize=12)
                ax_sxr.set_ylabel('SXR Flux (W/m²)', fontsize=10)
                ax_sxr.set_xlabel('Time', fontsize=10)
                ax_sxr.grid(True, alpha=0.3)
                ax_sxr.legend(fontsize=8, loc='upper right', ncol=2)
                ax_sxr.set_xlim([current_time - time_window, current_time + time_window])
                
                try:
                    y_data = []
                    for line in ax_sxr.get_lines():
                        y_data.extend(line.get_ydata())
                    y_data = [y for y in y_data if y > 0]
                    if len(y_data) > 0 and min(y_data) > 0:
                        ax_sxr.set_yscale('log')
                except Exception:
                    ax_sxr.set_yscale('linear')

            # Don't use tight_layout or bbox_inches='tight' - keeps frame size consistent
            plt.savefig(save_path, dpi=100)
            plt.close(fig)
            return save_path

        except Exception as e:
            print(f"Worker {os.getpid()}: Error processing {timestamp}: {e}")
            plt.close('all')
            return None

    def create_flux_heatmap_movie(self, timestamps, auto_cleanup=True, fps=2,
                                  all_timestamps_for_tracking=None, movie_filename=None, show_sxr_timeseries=True):
        """
        Create a movie showing flux contribution heat maps over time
        
        Args:
            timestamps: Timestamps to generate frames for (subsampled for visualization)
            auto_cleanup: Whether to delete frame files after movie creation
            fps: Frames per second for the movie
            all_timestamps_for_tracking: Full resolution timestamps for accurate region tracking
            movie_filename: Custom filename for the movie (if None, uses default naming)
        """
        print(f"Creating flux heat map movie with {len(timestamps)} frame timestamps...")

        # Track regions across time for consistent region IDs and colors
        tracking_timestamps = all_timestamps_for_tracking if all_timestamps_for_tracking is not None else timestamps
        print(f"Tracking regions across {len(tracking_timestamps)} timestamps (full resolution)...")
        region_tracks = self.track_regions_over_time(tracking_timestamps)

        # Create frames directory
        self.frames_dir = Path("temp_flux_heatmap_frames")
        self.frames_dir.mkdir(exist_ok=True)

        # Determine number of processes
        num_processes = min(os.cpu_count(), len(timestamps))
        num_processes = max(1, num_processes - 1)
        print(f"Using {num_processes} processes")

        # Process frames in parallel
        start_time = time.time()

        from functools import partial
        frame_worker = partial(
            self.generate_flux_heatmap_frame_worker,
            region_tracks=region_tracks,
            show_sxr_timeseries=show_sxr_timeseries
        )

        with Pool(processes=num_processes) as pool:
            results = []
            for result in tqdm(pool.imap(frame_worker, timestamps, chunksize=1),
                              desc="Generating flux heatmap frames", unit="frame", total=len(timestamps)):
                results.append(result)

        # Filter out failed frames
        frame_paths = [path for path in results if path is not None]

        processing_time = time.time() - start_time
        print(f"Generated {len(frame_paths)} frames in {processing_time:.2f} seconds")
        if len(frame_paths) > 0:
            print(f"Average: {processing_time / len(frame_paths):.2f} seconds per frame")

        if not frame_paths:
            print("No frames were successfully generated. Cannot create movie.")
            return

        # Compile into video
        print("Creating flux heat map movie...")
        video_start = time.time()

        frame_paths.sort(key=lambda x: os.path.basename(x))

        if movie_filename:
            # Replace .mp4 with _flux_heatmap.mp4 or add it
            if movie_filename.endswith('.mp4'):
                movie_filename = movie_filename.replace('.mp4', '_flux_heatmap.mp4')
            else:
                movie_filename = movie_filename + '_flux_heatmap.mp4'
            movie_path = os.path.join(self.output_dir, movie_filename)
        else:
            movie_path = os.path.join(self.output_dir, f"flux_heatmap_{timestamps[0].split('T')[0]}.mp4")

        with imageio.get_writer(movie_path, fps=fps, codec='libx264', format='ffmpeg',
                                pixelformat='yuv420p',
                                output_params=['-preset', 'faster', '-crf', '23']) as writer:
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    image = imageio.imread(frame_path)
                    writer.append_data(image)

        video_time = time.time() - video_start
        total_time = time.time() - start_time

        print(f"Video creation took {video_time:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"✅ Flux heat map movie saved to: {movie_path}")

        # Optional: Clean up frame files
        if auto_cleanup:
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            print("Frame files cleaned up")

        return movie_path


def _setup_analyzer(args):
    """Initialize analyzer and apply command line overrides"""
    analyzer = FluxContributionAnalyzer(config_path=args.config)
    
    if args.flux_path:
        analyzer.flux_path = Path(args.flux_path)
    if args.predictions_csv:
        analyzer.predictions_df = pd.read_csv(args.predictions_csv)
        analyzer.predictions_df['datetime'] = pd.to_datetime(analyzer.predictions_df['timestamp'])
        analyzer.predictions_df = analyzer.predictions_df.sort_values('datetime')
    if args.start_time and args.end_time:
        analyzer.time_period = {
            'start_time': args.start_time,
            'end_time': args.end_time
        }
        start_time = pd.to_datetime(analyzer.time_period['start_time'])
        end_time = pd.to_datetime(analyzer.time_period['end_time'])
        mask = (analyzer.predictions_df['datetime'] >= start_time) & (analyzer.predictions_df['datetime'] <= end_time)
        analyzer.predictions_df = analyzer.predictions_df[mask].reset_index(drop=True)
        print(f"Filtered data to time period: {start_time} to {end_time}")
    
    return analyzer


def _get_output_dir(analyzer, args):
    """Get output directory from config or command line args"""
    if args.output_dir:
        return Path(args.output_dir)
    
    output_dir_str = analyzer.output_config.get('output_dir', 'flux_analysis_output')
    if '${base_data_dir}' in output_dir_str:
        base_dir = analyzer.config.get('base_data_dir', '/mnt/data/COMBINED')
        output_dir_str = output_dir_str.replace('${base_data_dir}', base_dir)
    return Path(output_dir_str)



def _detect_and_save_flares(analyzer, output_dir):
    """Detect flares and save summaries to CSV files"""
    print("Detecting flare events...")
    flare_events = analyzer.detect_flare_events()
    
    print("\nDetecting simultaneous flares...")
    simultaneous_threshold = analyzer.flare_config.get('simultaneous_flare_threshold', 5e-6)
    sequence_window_hours = analyzer.flare_config.get('sequence_window_hours', 1.0)
    simultaneous_flares = analyzer.detect_simultaneous_flares(
        threshold=simultaneous_threshold,
        sequence_window_hours=sequence_window_hours
    )
    
    flare_summary_path = output_dir / 'flare_events_summary.csv'
    analyzer.create_flare_event_summary(flare_summary_path)
    
    if len(simultaneous_flares) > 0:
        simultaneous_summary_path = output_dir / 'simultaneous_flares_summary.csv'
        simultaneous_flares.to_csv(simultaneous_summary_path, index=False)
        print(f"Simultaneous flares summary saved to: {simultaneous_summary_path}")
    
    return flare_events, simultaneous_flares


def _filter_timestamps_by_interval(timestamps, interval_seconds):
    """Filter timestamps to keep only those at specified interval"""
    if interval_seconds <= 0:
        return timestamps
    
    datetimes = pd.to_datetime(timestamps)
    filtered = []
    last_time = None
    
    for i, dt in enumerate(datetimes):
        if last_time is None or (dt - last_time).total_seconds() >= interval_seconds:
            filtered.append(timestamps[i])
            last_time = dt
    
    return filtered


def _get_flare_sequences(simultaneous_flares, window_days):
    """
    Get flare sequences from simultaneous_flares DataFrame using sequence_id.
    
    For each sequence, calculates the time window as:
    - sequence_min_time = minimum timestamp in the sequence
    - sequence_max_time = maximum timestamp in the sequence
    - video_start_time = sequence_min_time - window_days
    - video_end_time = sequence_max_time + window_days
    
    This ensures the video includes window_days of context before and after
    the actual flare sequence.
    """
    if len(simultaneous_flares) == 0:
        return []
    
    sequences = []
    for sequence_id in sorted(simultaneous_flares['sequence_id'].unique()):
        sequence_events = simultaneous_flares[simultaneous_flares['sequence_id'] == sequence_id]
        
        # Get all unique timestamps in this sequence (sorted)
        timestamps = sorted(sequence_events['timestamp'].unique())
        datetimes = [pd.to_datetime(ts) for ts in timestamps]
        
        # Find the actual sequence time range (min to max)
        sequence_min_time = min(datetimes)
        sequence_max_time = max(datetimes)
        
        # Calculate video window: extend window_days above and below the sequence range
        video_start_time = sequence_min_time - pd.Timedelta(days=window_days)
        video_end_time = sequence_max_time + pd.Timedelta(days=window_days)
        
        # Center timestamp for filename (not used for window calculation)
        center_timestamp = pd.Series(datetimes).median()
        
        sequences.append({
            'sequence_id': sequence_id,
            'timestamps': timestamps,
            'sequence_min_time': sequence_min_time,
            'sequence_max_time': sequence_max_time,
            'center_timestamp': center_timestamp,  # Only for filename
            'start_time': video_start_time,  # Video window start
            'end_time': video_end_time,  # Video window end
            'num_groups': sequence_events['group_id'].nunique(),
            'num_events': len(sequence_events)
        })
    
    return sequences


def _create_simultaneous_flare_movies(analyzer, simultaneous_flares, output_dir, args):
    """Create movies for simultaneous flare events using sequence_id"""
    print("\nCreating contour evolution movies for simultaneous flare sequences...")
    
    movie_fps = args.movie_fps if args.movie_fps != 2 else analyzer.output_config.get('movie_fps', 2)
    movie_interval_seconds = args.movie_interval_seconds if args.movie_interval_seconds is not None else analyzer.output_config.get('movie_interval_seconds', 15)
    simultaneous_window_days = analyzer.output_config.get('simultaneous_flare_window_days', 1)
    
    # Use sequence_id from detect_simultaneous_flares instead of re-clustering
    flare_sequences = _get_flare_sequences(simultaneous_flares, simultaneous_window_days)
    print(f"Creating movies for {len(flare_sequences)} flare sequences")
    
    analyzer.output_dir = str(output_dir)
    show_sxr = args.show_sxr_timeseries or analyzer.output_config.get('show_sxr_timeseries', False)
    
    movies_created = 0
    for seq_idx, sequence in enumerate(flare_sequences):
        sequence_id = sequence['sequence_id']
        center_timestamp = sequence['center_timestamp']
        center_str = center_timestamp.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Video window extends window_days above and below the sequence range
        window_start = sequence['start_time']
        window_end = sequence['end_time']
        sequence_min = sequence['sequence_min_time']
        sequence_max = sequence['sequence_max_time']
        
        window_start_str = window_start.strftime('%Y-%m-%dT%H:%M:%S')
        window_end_str = window_end.strftime('%Y-%m-%dT%H:%M:%S')
        sequence_min_str = sequence_min.strftime('%Y-%m-%dT%H:%M:%S')
        sequence_max_str = sequence_max.strftime('%Y-%m-%dT%H:%M:%S')
        
        print(f"\n  Movie {seq_idx + 1}/{len(flare_sequences)}: Flare Sequence {sequence_id}")
        print(f"    Sequence contains {sequence['num_groups']} groups with {sequence['num_events']} events")
        print(f"    Sequence time range: {sequence_min_str} to {sequence_max_str}")
        print(f"    Video window (with ±{simultaneous_window_days} days padding): {window_start_str} to {window_end_str}")
        
        available_timestamps = analyzer.predictions_df[
            (analyzer.predictions_df['datetime'] >= window_start) & 
            (analyzer.predictions_df['datetime'] <= window_end)
        ]['timestamp'].tolist()
        
        if len(available_timestamps) == 0:
            print(f"    No data found in time window, skipping...")
            continue
        
        all_timestamps_for_tracking = available_timestamps
        filtered_timestamps = _filter_timestamps_by_interval(available_timestamps, movie_interval_seconds)
        
        print(f"    Found {len(available_timestamps)} timestamps, generating {len(filtered_timestamps)} frames")
        print(f"    Movie will be {len(filtered_timestamps)/movie_fps:.1f} seconds long at {movie_fps} FPS")
        
        safe_timestamp = center_str.replace(':', '-').replace(' ', '_')
        base_movie_filename = f'flare_sequence_{sequence_id}_{safe_timestamp}.mp4'
        
        # Check if individual movie types are enabled
        create_contour_movies = analyzer.output_config.get('create_contour_movies_for_sequences', True)
        create_flux_heatmap_movies = analyzer.output_config.get('create_flux_heatmap_movies_for_sequences', True)
        
        # Create contour movie
        if create_contour_movies:
            movie_path = analyzer.create_contour_movie(
                timestamps=filtered_timestamps,
                auto_cleanup=True,
                fps=movie_fps,
                show_sxr_timeseries=show_sxr,
                all_timestamps_for_tracking=all_timestamps_for_tracking,
                movie_filename=base_movie_filename
            )
            
            if movie_path:
                print(f"    ✅ Contour movie created: {movie_path}")
                movies_created += 1
        else:
            print(f"    ⏭️  Skipping contour movie (disabled in config)")
        
        # Create flux heat map movie
        if create_flux_heatmap_movies:
            print(f"    Creating flux heat map movie...")
            flux_movie_path = analyzer.create_flux_heatmap_movie(
                timestamps=filtered_timestamps,
                auto_cleanup=True,
                fps=movie_fps,
                all_timestamps_for_tracking=all_timestamps_for_tracking,
                movie_filename=base_movie_filename,
                show_sxr_timeseries=show_sxr
            )
            
            if flux_movie_path:
                print(f"    ✅ Flux heat map movie created: {flux_movie_path}")
        else:
            print(f"    ⏭️  Skipping flux heat map movie (disabled in config)")
    
    print(f"\n✅ Created {movies_created}/{len(flare_sequences)} flare sequence movies")


def _create_full_period_movie(analyzer, output_dir, args):
    """Create movie for full time period"""
    print("\nCreating contour evolution movie...")
    
    start_time = pd.to_datetime(analyzer.time_period['start_time'])
    end_time = pd.to_datetime(analyzer.time_period['end_time'])
    
    movie_fps = args.movie_fps if args.movie_fps != 2 else analyzer.output_config.get('movie_fps', 2)
    movie_interval_seconds = args.movie_interval_seconds if args.movie_interval_seconds is not None else analyzer.output_config.get('movie_interval_seconds', 15)
    
    available_timestamps = analyzer.predictions_df[
        (analyzer.predictions_df['datetime'] >= start_time) & 
        (analyzer.predictions_df['datetime'] <= end_time)
    ]['timestamp'].tolist()
    
    all_timestamps_for_tracking = available_timestamps
    filtered_timestamps = _filter_timestamps_by_interval(available_timestamps, movie_interval_seconds)
    
    print(f"Found {len(available_timestamps)} total timestamps in time period")
    print(f"Tracking regions at full resolution: {len(all_timestamps_for_tracking)} timestamps")
    print(f"Generating frames at {movie_interval_seconds}-second intervals: {len(filtered_timestamps)} frames")
    print(f"Movie will be {len(filtered_timestamps)/movie_fps:.1f} seconds long at {movie_fps} FPS")
    
    analyzer.output_dir = str(output_dir)
    show_sxr = args.show_sxr_timeseries or analyzer.output_config.get('show_sxr_timeseries', False)
    
    movie_path = analyzer.create_contour_movie(
        timestamps=filtered_timestamps,
        auto_cleanup=True,
        fps=movie_fps,
        show_sxr_timeseries=show_sxr,
        all_timestamps_for_tracking=all_timestamps_for_tracking
    )
    
    if movie_path:
        print(f"Contour evolution movie created: {movie_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze flux contributions for flare detection')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--flux_path', help='Path to flux contributions directory (overrides config)')
    parser.add_argument('--predictions_csv', help='Path to predictions CSV file (overrides config)')
    parser.add_argument('--output_dir', help='Output directory for results (overrides config)')
    parser.add_argument('--start_time', help='Start time for analysis (overrides config)')
    parser.add_argument('--end_time', help='End time for analysis (overrides config)')
    parser.add_argument('--create_contour_movie', action='store_true', help='Create contour evolution movie')
    parser.add_argument('--movie_fps', type=int, default=2, help='Frames per second for movie (default: 2)')
    parser.add_argument('--movie_interval_seconds', type=int, help='Seconds between frames for movie (overrides config)')
    parser.add_argument('--show_sxr_timeseries', action='store_true', help='Show SXR time series for tracked regions in movie')
    parser.add_argument('--max_tracking_distance', type=int, default=50, help='Maximum distance for region tracking (pixels)')

    args = parser.parse_args()

    analyzer = _setup_analyzer(args)
    output_dir = _get_output_dir(analyzer, args)
    output_dir.mkdir(exist_ok=True, parents=True)

    flare_events, simultaneous_flares = _detect_and_save_flares(analyzer, output_dir)

    create_contour_movie = args.create_contour_movie or analyzer.output_config.get('create_contour_movie', False)
    create_simultaneous_movies = analyzer.output_config.get('create_simultaneous_flare_movies', False)
    
    # Create simultaneous flare movies if requested and available
    if create_simultaneous_movies and len(simultaneous_flares) > 0:
        _create_simultaneous_flare_movies(analyzer, simultaneous_flares, output_dir, args)
    elif create_simultaneous_movies and len(simultaneous_flares) == 0:
        print("Warning: create_simultaneous_flare_movies is enabled but no simultaneous flares were detected")
    
    # Create full period contour movie if requested (and not already creating simultaneous movies)
    if create_contour_movie and not (create_simultaneous_movies and len(simultaneous_flares) > 0):
        if analyzer.time_period:
            _create_full_period_movie(analyzer, output_dir, args)
        else:
            print("Warning: Cannot create contour movie without time period specified in config or command line")

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Found {len(flare_events)} potential flare events")
    if len(simultaneous_flares) > 0:
        num_sequences = simultaneous_flares['sequence_id'].nunique() if 'sequence_id' in simultaneous_flares.columns else 0
        num_groups = simultaneous_flares['group_id'].nunique() if 'group_id' in simultaneous_flares.columns else 0
        print(f"Found {len(simultaneous_flares)} simultaneous flare events in {num_groups} groups, clustered into {num_sequences} flare sequences")


if __name__ == "__main__":
    main()