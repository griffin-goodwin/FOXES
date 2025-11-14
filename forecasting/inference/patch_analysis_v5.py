import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import pandas as pd
from pathlib import Path
import argparse
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
import cv2


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
Flux Contribution Analysis and Flare Detection Script (Version 5 - Simplified DBSCAN)

This script analyzes flux contributions from different patches to identify 
potential flaring events and visualize their spatial and temporal patterns.

Version 5 features:
1. Simplified DBSCAN Detection: Simple, intuitive DBSCAN clustering on flux maps.
   Filters patches by threshold, then clusters by spatial position + flux value.
   Easy to tune and understand - mimics how humans identify regions visually.
2. DBSCAN Track Merging: Uses DBSCAN clustering to merge fragmented tracks that
   belong to the same physical region, improving tracking robustness across temporal gaps.
3. Traditional Thresholding: Uses median + standard deviation thresholding
   (same as v1) for automatic threshold determination, calculated in log space.
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
            group_datetime = pd.to_datetime(group['datetime'].iloc[0])
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
            
            # Check if DBSCAN-based detection is enabled
            use_dbscan_detection = self.flare_config.get('use_dbscan_for_detection', False)
            if use_dbscan_detection:
                regions = self._detect_regions_with_dbscan(flux_contrib, timestamp, pred_data, previous_regions=None)
            else:
                # Use traditional threshold-based detection
                regions = self._detect_regions_for_timestamp(flux_contrib, timestamp, pred_data)
            return (timestamp, regions)
        except Exception as e:
            print(f"Error detecting regions for {timestamp}: {e}")
            return (timestamp, None)
    
    def _detect_regions_worker_with_previous(self, timestamp, previous_regions):
        """Worker function for sequential region detection with previous region awareness"""
        try:
            flux_contrib = self.load_flux_contributions(timestamp)
            if flux_contrib is None:
                return None
                
            # Get prediction data
            pred_data = self.predictions_df[self.predictions_df['timestamp'] == timestamp]
            if pred_data.empty:
                return None
            pred_data = pred_data.iloc[0]
            
            # Check if DBSCAN-based detection is enabled
            use_dbscan_detection = self.flare_config.get('use_dbscan_for_detection', False)
            if use_dbscan_detection:
                regions = self._detect_regions_with_dbscan(flux_contrib, timestamp, pred_data, previous_regions=previous_regions)
            else:
                # Use traditional threshold-based detection
                regions = self._detect_regions_for_timestamp(flux_contrib, timestamp, pred_data)
            return regions
        except Exception as e:
            print(f"Error detecting regions for {timestamp}: {e}")
            return None

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
        
        # Check if previous region awareness is enabled
        use_previous_regions = self.flare_config.get('dbscan_detection_use_previous_regions', False)
        
        # First pass: collect all regions
        # For previous region awareness, we need to track which regions belong to which tracks
        # so we can pass track_ids to detection
        detection_id_to_track_id = {}  # Maps (timestamp, detection_id) -> track_id
        
        if use_previous_regions:
            # Sequential detection to have access to previous regions
            print("Phase 1/2: Detecting regions at each timestamp (sequential, with previous region awareness)...")
            previous_regions_with_tracks = None  # Will store (region, track_id) tuples
            for i, timestamp in enumerate(tqdm(timestamps, desc="Detecting regions", unit="timestamp")):
                # Pass previous regions with their track IDs
                previous_regions = None
                if previous_regions_with_tracks is not None:
                    previous_regions = [reg for reg, _ in previous_regions_with_tracks]
                
                regions = self._detect_regions_worker_with_previous(timestamp, previous_regions)
                if regions is not None:
                    all_regions[timestamp] = regions
                    
                    # After detection, do quick tracking to get track_ids for next iteration
                    if i > 0:  # Can't track first timestamp
                        prev_timestamp = timestamps[i-1]
                        if prev_timestamp in all_regions:
                            # Quick tracking: match current regions to previous tracks
                            current_regions_with_tracks = []
                            for region in regions:
                                preferred_id = region.get('preferred_previous_region_id', None)
                                if preferred_id is not None and previous_regions_with_tracks is not None:
                                    # Find the track_id for this preferred detection_id
                                    track_id = None
                                    for prev_reg, prev_track_id in previous_regions_with_tracks:
                                        if prev_reg.get('id') == preferred_id:
                                            track_id = prev_track_id
                                            break
                                    
                                    if track_id is not None:
                                        # Store mapping and add track_id to region
                                        detection_id_to_track_id[(timestamp, region.get('id'))] = track_id
                                        region['preferred_track_id'] = track_id
                                        current_regions_with_tracks.append((region, track_id))
                                        continue
                                
                                # No preferred track, will be assigned during full tracking
                                current_regions_with_tracks.append((region, None))
                            
                            previous_regions_with_tracks = current_regions_with_tracks
                        else:
                            previous_regions_with_tracks = [(reg, None) for reg in regions]
                    else:
                        # First timestamp - no tracks yet
                        previous_regions_with_tracks = [(reg, None) for reg in regions]
        else:
            # Parallel detection (faster, but no previous region awareness)
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
        max_time_gap = 30 * 60  # 30 minutes in seconds

        # Tracking is based on spatial proximity only

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
                    
                    # FIRST: Check if this region is close to multiple tracks (overlap or proximity)
                    # and assign to the one where the flux ratio is highest (the flaring one)
                    # This handles the case where DBSCAN merges regions
                    region_mask = region_copy.get('mask', None)
                    current_flux = region_copy.get('sum_flux', 0)
                    candidate_tracks = []
                    best_track_id = None
                    best_distance = float('inf')
                    
                    # Check all tracks for overlap/proximity
                    all_track_ids = list(region_tracks.keys())
                    
                    if region_mask is not None and current_flux > 0:
                        prev_timestamp = timestamps[i-1] if i > 0 else None
                        
                        for track_id in all_track_ids:
                            if not region_tracks[track_id]:
                                continue
                            
                            # Get the region from the previous timestamp (not just last in track)
                            prev_region_in_track = None
                            for ts, reg in reversed(region_tracks[track_id]):
                                if ts == prev_timestamp:
                                    prev_region_in_track = reg
                                    break
                            
                            # If no previous timestamp region, use last region in track
                            if prev_region_in_track is None:
                                last_timestamp, prev_region_in_track = region_tracks[track_id][-1]
                            
                            last_mask = prev_region_in_track.get('mask', None)
                            last_flux = prev_region_in_track.get('sum_flux', 0)
                            
                            # Calculate distance
                            distance = np.sqrt(
                                (region_copy['centroid_img_x'] - prev_region_in_track['centroid_img_x'])**2 + 
                                (region_copy['centroid_img_y'] - prev_region_in_track['centroid_img_y'])**2
                            )
                            
                            # Check for overlap or proximity (within 2x max_distance for merge detection)
                            has_overlap = False
                            if last_mask is not None:
                                overlap = np.logical_and(region_mask, last_mask)
                                has_overlap = np.any(overlap)
                            
                            proximity_threshold = max_distance * 2.0  # Be lenient for merge detection
                            is_nearby = distance < proximity_threshold
                            
                            if has_overlap or is_nearby:
                                # Calculate flux ratio: current / previous
                                # Higher ratio means this region flared more
                                flux_ratio = current_flux / (last_flux + 1e-12)
                                
                                # Get flux of overlapping patches to see which previous region
                                # the high-flux area belongs to
                                overlap_flux_sum = 0.0
                                if has_overlap:
                                    # Load flux to get overlap flux values
                                    flux_contrib = self.load_flux_contributions(timestamp)
                                    if flux_contrib is not None:
                                        overlap_y, overlap_x = np.where(overlap)
                                        if len(overlap_y) > 0:
                                            overlap_flux = flux_contrib[overlap_y, overlap_x]
                                            overlap_flux_sum = np.sum(overlap_flux)
                                
                                # Score: prioritize based on flux ratio (flaring) and overlap flux
                                overlap_bonus = 10.0 if has_overlap else 1.0
                                # Use overlap flux if available, otherwise use total flux ratio
                                if has_overlap and overlap_flux_sum > 0:
                                    # Compare overlap flux to previous region's flux
                                    overlap_flux_ratio = overlap_flux_sum / (last_flux + 1e-12)
                                    score = overlap_flux_sum * overlap_flux_ratio * overlap_bonus / (distance + 1.0)
                                else:
                                    # No overlap, use total flux ratio - prioritize high flux ratio (flaring)
                                    score = flux_ratio * overlap_bonus / (distance + 1.0)
                                
                                candidate_tracks.append((track_id, score, flux_ratio, current_flux, distance, has_overlap, overlap_flux_sum))
                    
                    # If close to multiple tracks, assign to the one with highest score (flaring region)
                    if len(candidate_tracks) > 1:
                        # Sort by score (descending) - highest score = most flaring + closest
                        candidate_tracks.sort(key=lambda x: x[1], reverse=True)
                        best_track_id = candidate_tracks[0][0]
                        best_distance = candidate_tracks[0][4]  # Distance from the chosen track
                    elif len(candidate_tracks) == 1:
                        # Only one candidate, use it if within reasonable distance
                        if candidate_tracks[0][4] < max_distance * 2.0:
                            best_track_id = candidate_tracks[0][0]
                            best_distance = candidate_tracks[0][4]
                    
                    # SECOND: If no merge detected, check preferred track IDs from DBSCAN detection
                    if best_track_id is None:
                        preferred_previous_region_id = region_copy.get('preferred_previous_region_id', None)
                        preferred_track_id = region_copy.get('preferred_track_id', None)
                        
                        # If preferred_track_id is set (from quick tracking), use it directly
                        if preferred_track_id is not None:
                            if preferred_track_id in region_tracks and region_tracks[preferred_track_id]:
                                last_timestamp, last_region = region_tracks[preferred_track_id][-1]
                                distance = np.sqrt(
                                    (region_copy['centroid_img_x'] - last_region['centroid_img_x'])**2 + 
                                    (region_copy['centroid_img_y'] - last_region['centroid_img_y'])**2
                                )
                                # Use preferred track - be more lenient with distance since this is a merge
                                preferred_max_distance = max_distance * 2.0
                                if distance < preferred_max_distance:
                                    best_track_id = preferred_track_id
                                    best_distance = distance
                        
                        # If preferred_previous_region_id is set but no track_id yet, find the track
                        elif preferred_previous_region_id is not None and i > 0:
                            prev_timestamp = timestamps[i-1]
                            # Find which track the previous region with this detection ID belongs to
                            for track_id, track_history in region_tracks.items():
                                if not track_history:
                                    continue
                                # Check if any region in this track matches the preferred detection ID
                                for ts, reg in track_history:
                                    if ts == prev_timestamp:
                                        # Use detection_id if stored, otherwise use id (for backward compatibility)
                                        last_detection_id = reg.get('detection_id', reg.get('id'))
                                        if last_detection_id == preferred_previous_region_id:
                                            # Found the track! Use it
                                            last_ts, last_reg = track_history[-1]
                                            distance = np.sqrt(
                                                (region_copy['centroid_img_x'] - last_reg['centroid_img_x'])**2 + 
                                                (region_copy['centroid_img_y'] - last_reg['centroid_img_y'])**2
                                            )
                                            preferred_max_distance = max_distance * 2.0
                                            if distance < preferred_max_distance:
                                                best_track_id = track_id
                                                best_distance = distance
                                            break
                                if best_track_id is not None:
                                    break
                    
                    # THIRD: If still no track found, find best match by spatial proximity only
                    if best_track_id is None:
                        for track_id in recently_active_tracks:
                            if not region_tracks[track_id]:
                                continue
                                
                            last_timestamp, last_region = region_tracks[track_id][-1]

                            # Calculate spatial distance
                            distance = np.sqrt(
                                (region_copy['centroid_img_x'] - last_region['centroid_img_x'])**2 + 
                                (region_copy['centroid_img_y'] - last_region['centroid_img_y'])**2
                            )

                            if distance < max_distance and distance < best_distance:
                                best_distance = distance
                                best_track_id = track_id
                    
                    # Add region to existing track or create new track
                    if best_track_id is not None:
                        # Store original detection ID before overwriting with track_id
                        original_detection_id = region_copy.get('id')
                        region_copy['id'] = best_track_id
                        region_copy['detection_id'] = original_detection_id  # Keep original for lookup
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
                        
                        # Only apply constraint if at least one factor is enabled
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
                        # Store original detection ID before overwriting with track_id
                        original_detection_id = region_copy.get('id')
                        region_copy['id'] = next_track_id
                        region_copy['detection_id'] = original_detection_id  # Keep original for lookup
                        region_copy['timestamp'] = timestamp
                        region_tracks[next_track_id] = [(timestamp, region_copy)]
                        active_tracks.add(next_track_id)
                        next_track_id += 1
        
        # Filter out tracks with only one region (no temporal continuity)
        region_tracks = {k: v for k, v in region_tracks.items() if len(v) > 1}
        
        print(f"Found {len(region_tracks)} region tracks after initial tracking")
        
        # Apply DBSCAN clustering to merge fragmented tracks
        use_dbscan = self.flare_config.get('use_dbscan_clustering', True)
        if use_dbscan and len(region_tracks) > 1:
            dbscan_eps = self.flare_config.get('dbscan_eps', 100.0)  # Spatial distance threshold (pixels)
            dbscan_min_samples = self.flare_config.get('dbscan_min_samples', 2)  # Minimum tracks in cluster
            print(f"Applying DBSCAN clustering (eps={dbscan_eps}, min_samples={dbscan_min_samples})...")
            region_tracks = self._cluster_and_merge_tracks(region_tracks, dbscan_eps, dbscan_min_samples)
            print(f"Found {len(region_tracks)} region tracks after DBSCAN clustering")
        
        # Apply temporal smoothing to flux values
        smoothing_window = self.flare_config.get('flux_smoothing_window', 3)
        if smoothing_window > 1:
            region_tracks = self._apply_temporal_smoothing(region_tracks, smoothing_window)
        
        
        return region_tracks
    
    def _cluster_and_merge_tracks(self, region_tracks, eps, min_samples):
        """
        Use DBSCAN clustering to merge fragmented tracks that belong to the same physical region.
        
        Extracts features from each track (spatial position, temporal span, flux) and clusters
        them using DBSCAN. Tracks in the same cluster are merged.
        
        Args:
            region_tracks: Dictionary of track_id -> list of (timestamp, region_data)
            eps: Maximum distance between tracks to be in same cluster (pixels)
            min_samples: Minimum number of tracks required to form a cluster
            
        Returns:
            Merged region_tracks dictionary
        """
        if len(region_tracks) < 2:
            return region_tracks
        
        # Extract features for each track
        track_features = []
        track_ids = list(region_tracks.keys())
        
        for track_id in track_ids:
            track_data = region_tracks[track_id]
            
            # Calculate average centroid position (spatial feature)
            centroids_x = [r['centroid_img_x'] for _, r in track_data]
            centroids_y = [r['centroid_img_y'] for _, r in track_data]
            avg_x = np.mean(centroids_x)
            avg_y = np.mean(centroids_y)
            
            # Calculate temporal center (convert to seconds since first timestamp)
            timestamps = [pd.to_datetime(ts) for ts, _ in track_data]
            if len(timestamps) > 1:
                time_span = (timestamps[-1] - timestamps[0]).total_seconds()
                time_center = (timestamps[0] + pd.Timedelta(seconds=time_span/2)).timestamp()
            else:
                time_span = 0
                time_center = timestamps[0].timestamp()
            
            # Normalize time to be on similar scale as spatial coordinates
            # Use a scaling factor: 1 pixel per second (adjust based on your data)
            time_scale = self.flare_config.get('dbscan_time_scale', 1.0)  # pixels per second
            normalized_time = time_center * time_scale
            
            # Average flux (log scale for better distribution)
            fluxes = [r['sum_flux'] for _, r in track_data]
            avg_flux = np.mean(fluxes)
            log_flux = np.log10(avg_flux + 1e-10)  # Add small value to avoid log(0)
            
            # Normalize flux to spatial scale (flux values are typically much smaller)
            flux_scale = self.flare_config.get('dbscan_flux_scale', 0.0)  # 0 = don't use flux in clustering
            normalized_flux = log_flux * flux_scale
            
            # Feature vector: [x, y, normalized_time, normalized_flux]
            # If flux_scale is 0, only use [x, y, normalized_time]
            if flux_scale > 0:
                features = np.array([avg_x, avg_y, normalized_time, normalized_flux])
            else:
                features = np.array([avg_x, avg_y, normalized_time])
            
            track_features.append(features)
        
        track_features = np.array(track_features)
        
        # Apply DBSCAN clustering
        if len(track_features) < min_samples:
            # Not enough tracks to cluster
            return region_tracks
        
        # Normalize features for DBSCAN (important for mixed units)
        feature_means = np.mean(track_features, axis=0)
        feature_stds = np.std(track_features, axis=0)
        feature_stds[feature_stds == 0] = 1  # Avoid division by zero
        normalized_features = (track_features - feature_means) / feature_stds
        
        # Adjust eps for normalized features
        # eps is in original units, need to scale it
        normalized_eps = eps / np.mean(feature_stds[:2])  # Use spatial std for scaling
        
        clustering = DBSCAN(eps=normalized_eps, min_samples=min_samples, metric='euclidean')
        cluster_labels = clustering.fit_predict(normalized_features)
        
        # Merge tracks in the same cluster
        merged_tracks = {}
        cluster_groups = {}
        
        for idx, (track_id, label) in enumerate(zip(track_ids, cluster_labels)):
            if label == -1:
                # Noise point - keep as separate track
                merged_tracks[track_id] = region_tracks[track_id]
            else:
                # Add to cluster group
                if label not in cluster_groups:
                    cluster_groups[label] = []
                cluster_groups[label].append(track_id)
        
        # Merge tracks in each cluster
        next_new_id = max(track_ids) + 1 if track_ids else 1
        for cluster_id, track_ids_in_cluster in cluster_groups.items():
            if len(track_ids_in_cluster) == 1:
                # Single track in cluster, keep as is
                merged_tracks[track_ids_in_cluster[0]] = region_tracks[track_ids_in_cluster[0]]
            else:
                # Merge multiple tracks
                # Combine all regions from all tracks, sort by timestamp
                all_regions = []
                for tid in track_ids_in_cluster:
                    all_regions.extend(region_tracks[tid])
                
                # Sort by timestamp
                all_regions.sort(key=lambda x: pd.to_datetime(x[0]))
                
                # Update region IDs to the new merged track ID (create new tuples)
                merged_regions = []
                for timestamp, region_data in all_regions:
                    region_data_copy = region_data.copy()
                    region_data_copy['id'] = next_new_id
                    merged_regions.append((timestamp, region_data_copy))
                
                merged_tracks[next_new_id] = merged_regions
                next_new_id += 1
        
        num_merged = len(cluster_groups)
        num_noise = np.sum(cluster_labels == -1)
        if num_merged > 0:
            print(f"  DBSCAN: Merged {num_merged} clusters, {num_noise} tracks kept separate (noise)")
        
        return merged_tracks
    
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

    def _detect_regions_with_dbscan(self, flux_contrib, timestamp, pred_data, previous_regions=None):
        """
        Detect active regions using DBSCAN clustering directly on flux maps.
        
        This method clusters patches based on their spatial position and flux values,
        creating active regions from the clusters. This approach can find irregularly
        shaped regions and is less sensitive to threshold tuning.
        
        Args:
            flux_contrib: 2D array of flux contributions
            timestamp: Current timestamp
            pred_data: Prediction data dictionary
            previous_regions: Optional list of previous timestamp's regions (for temporal awareness)
            
        Returns:
            List of region dictionaries (same format as threshold-based detection)
        """
        # Get DBSCAN detection parameters
        dbscan_eps = self.flare_config.get('dbscan_detection_eps', 2.0)  # Spatial distance in patches
        dbscan_min_samples = self.flare_config.get('dbscan_detection_min_samples', 3)  # Min patches per region
        dbscan_threshold_std_multiplier = self.flare_config.get('dbscan_detection_threshold_std_multiplier', 1.0)  # Threshold multiplier for filtering patches
        
        # Filter patches using median + std threshold (same as v1 thresholding method)
        # Calculate median and std in log space (more appropriate for flux data)
        flux_values = flux_contrib.flatten()
        # Filter out zero/negative values for log calculation
        positive_flux = flux_values[flux_values > 0]
        if len(positive_flux) == 0:
            return []  # No positive flux values
        log_flux = np.log(positive_flux)
        median_log_flux = np.median(log_flux)
        std_log_flux = np.std(log_flux)
        # Convert back to linear space for threshold
        min_flux_for_clustering = np.exp(median_log_flux + dbscan_threshold_std_multiplier * std_log_flux)
        
        # Get patches above minimum flux threshold
        significant_mask = flux_contrib > min_flux_for_clustering
        significant_patches = np.where(significant_mask)
        
        if len(significant_patches[0]) == 0:
            return []
        
        # Extract features: (y, x) or (y, x, log_flux)
        # Since we already filtered by flux threshold, we may not need flux in clustering
        patch_y = significant_patches[0]
        patch_x = significant_patches[1]
        # Always get flux values for region property calculations (sum_flux, max_flux)
        patch_flux = flux_contrib[significant_patches]
        
        # Simple DBSCAN clustering: spatial position + optional flux value, gradient, or previous region proximity
        # This mimics how humans identify regions - patches that are close in space
        # and have similar flux characteristics belong to the same region
        use_flux_in_clustering = self.flare_config.get('dbscan_detection_use_flux', False)
        use_gradient_in_clustering = self.flare_config.get('dbscan_detection_use_gradient', False)
        use_previous_regions = self.flare_config.get('dbscan_detection_use_previous_regions', False)
        
        # Warn if both flux and gradient are enabled (they're mutually exclusive)
        if use_flux_in_clustering and use_gradient_in_clustering:
            print("Warning: Both use_flux and use_gradient are enabled. Using gradient (takes precedence).")
            use_flux_in_clustering = False
        
        # Start with spatial features
        features = np.column_stack([patch_y, patch_x])
        
        # Add optional features
        if use_gradient_in_clustering:
            # Calculate flux gradient (magnitude of spatial change) at each patch
            # This helps identify region boundaries and edges
            flux_gradient = self._calculate_flux_gradient(flux_contrib, significant_mask, patch_y, patch_x)
            # Use log gradient for better distribution
            log_gradient = np.log10(flux_gradient + 1e-10)
            features = np.column_stack([features, log_gradient])
        elif use_flux_in_clustering:
            # Use log flux for better distribution (flux values span many orders of magnitude)
            log_flux = np.log10(patch_flux + 1e-10)
            features = np.column_stack([features, log_flux])
        
        # Add previous region proximity feature if enabled and previous regions are available
        if use_previous_regions and previous_regions is not None and len(previous_regions) > 0:
            # Calculate distance to nearest previous region centroid for each patch
            prev_region_proximity = self._calculate_previous_region_proximity(
                patch_y, patch_x, previous_regions
            )
            features = np.column_stack([features, prev_region_proximity])
        
        # Normalize features for DBSCAN (important for mixed units: patches vs log flux)
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        feature_stds[feature_stds == 0] = 1  # Avoid division by zero
        normalized_features = (features - feature_means) / feature_stds
        
        # Adjust eps for normalized features
        # eps is in original units (patches), need to scale it for normalized space
        # Use spatial std for scaling (flux dimension handled separately if used)
        normalized_eps = dbscan_eps / np.mean(feature_stds[:2])
        
        # Apply weights to optional feature dimensions
        feature_idx = 2  # Start after spatial dimensions (y, x)
        if use_gradient_in_clustering:
            # Get gradient weight to control how much gradient similarity matters
            gradient_weight = self.flare_config.get('dbscan_detection_gradient_weight', 0.3)
            # Scale the gradient dimension by its weight
            normalized_features[:, feature_idx] = normalized_features[:, feature_idx] * gradient_weight
            feature_idx += 1
        elif use_flux_in_clustering:
            # Get flux weight to control how much flux similarity matters
            flux_weight = self.flare_config.get('dbscan_detection_flux_weight', 0.3)
            # Scale the flux dimension by its weight
            normalized_features[:, feature_idx] = normalized_features[:, feature_idx] * flux_weight
            feature_idx += 1
        
        # Apply weight to previous region proximity feature
        if use_previous_regions and previous_regions is not None and len(previous_regions) > 0:
            prev_region_weight = self.flare_config.get('dbscan_detection_previous_region_weight', 0.3)
            normalized_features[:, feature_idx] = normalized_features[:, feature_idx] * prev_region_weight
        
        # Apply DBSCAN clustering
        if len(normalized_features) < dbscan_min_samples:
            return []
        
        clustering = DBSCAN(eps=normalized_eps, min_samples=dbscan_min_samples, metric='euclidean')
        cluster_labels = clustering.fit_predict(normalized_features)
        
        # Extract regions from clusters
        regions = []
        accepted_region_id = 0
        unique_labels = np.unique(cluster_labels)
        
        # DBSCAN already handles minimum cluster size via min_samples parameter
        # No need for additional size constraints - let DBSCAN do its job
        min_flux_threshold = self.flare_config.get('min_flux_threshold', None)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points
            
            # Get patches in this cluster
            cluster_mask = cluster_labels == label
            cluster_y = patch_y[cluster_mask]
            cluster_x = patch_x[cluster_mask]
            cluster_flux = patch_flux[cluster_mask]
            
            cluster_size = len(cluster_y)
            
            # Create binary mask for this region
            region_mask = np.zeros_like(flux_contrib, dtype=bool)
            region_mask[cluster_y, cluster_x] = True
            
            # Calculate region properties
            sum_flux = np.sum(cluster_flux)
            max_flux = np.max(cluster_flux)
            
            # Filter by minimum flux threshold if enabled
            if min_flux_threshold is not None and sum_flux < min_flux_threshold:
                continue
            
            # Calculate centroid
            centroid_y = np.mean(cluster_y)
            centroid_x = np.mean(cluster_x)
            
            # Convert to image coordinates
            img_y = centroid_y * self.patch_size + self.patch_size // 2
            img_x = centroid_x * self.patch_size + self.patch_size // 2
            
            # Calculate label position
            min_y, max_y = np.min(cluster_y), np.max(cluster_y)
            min_x, max_x = np.min(cluster_x), np.max(cluster_x)
            label_y = max(0, min_y - 2)
            label_x = centroid_x
            
            # Check if this cluster is close to multiple previous regions (overlap or proximity)
            # If so, determine which previous region it should be assigned to based on flux
            # This ensures the flaring region (current high flux) gets assigned correctly
            preferred_previous_region_id = None
            if previous_regions is not None and len(previous_regions) > 0:
                # Check both overlapping and nearby previous regions
                # Use a proximity threshold (in patches) - regions within this distance are considered candidates
                proximity_threshold_patches = 5  # Regions within 5 patches are considered close
                candidate_previous = []
                
                for prev_region in previous_regions:
                    prev_mask = prev_region.get('mask', None)
                    prev_centroid_y = prev_region.get('centroid_patch_y', None)
                    prev_centroid_x = prev_region.get('centroid_patch_x', None)
                    
                    if prev_mask is None or prev_centroid_y is None or prev_centroid_x is None:
                        continue
                    
                    # Check for overlap
                    overlap = np.logical_and(region_mask, prev_mask)
                    has_overlap = np.any(overlap)
                    
                    # Check for proximity (centroid distance)
                    centroid_distance = np.sqrt(
                        (centroid_y - prev_centroid_y)**2 + 
                        (centroid_x - prev_centroid_x)**2
                    )
                    is_nearby = centroid_distance <= proximity_threshold_patches
                    
                    if has_overlap or is_nearby:
                        prev_id = prev_region.get('id', None)
                        if prev_id is not None:
                            candidate_previous.append((prev_id, prev_region, has_overlap, centroid_distance))
                
                # If close to multiple previous regions, determine which one to assign to
                if len(candidate_previous) > 1:
                    # Strategy: Assign to the previous region where the current merged region
                    # has the highest flux (indicating it's the one that's flaring)
                    # Simple approach: just use the current merged region's total flux
                    # The region with highest current flux is the one that's flaring
                    
                    # Sort by current merged region's flux (it's the same for all, but we want to ensure
                    # we're assigning to the region that makes sense)
                    # Actually, we should compare: which previous region would this merged region belong to?
                    # The answer: the one where the merged region's flux is highest relative to that previous region
                    
                    prev_region_scores = []
                    for prev_id, prev_region, has_overlap, centroid_distance in candidate_previous:
                        prev_flux = prev_region.get('sum_flux', 0)
                        
                        # Score: prioritize based on current merged region's flux
                        # The merged region with highest flux should be assigned to the previous region
                        # that it's closest to AND where the flux increase is largest
                        flux_ratio = sum_flux / (prev_flux + 1e-12)  # Avoid division by zero
                        
                        # Score: higher current flux + higher flux ratio + closer distance = better match
                        # Weight overlap more heavily
                        overlap_bonus = 10.0 if has_overlap else 1.0
                        score = sum_flux * flux_ratio * overlap_bonus / (centroid_distance + 1.0)
                        prev_region_scores.append((prev_id, score))
                    
                    # Assign to previous region with highest score
                    if len(prev_region_scores) > 0:
                        prev_region_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score (descending)
                        preferred_previous_region_id = prev_region_scores[0][0]  # Highest score region ID
                elif len(candidate_previous) == 1:
                    # Only one candidate, use it
                    preferred_previous_region_id = candidate_previous[0][0]
            
            accepted_region_id += 1
            
            region_data = {
                'id': accepted_region_id,
                'size': cluster_size,
                'core_size': cluster_size,  # For DBSCAN, core_size = total size
                'sum_flux': sum_flux,
                'max_flux': max_flux,
                'centroid_patch_y': centroid_y,
                'centroid_patch_x': centroid_x,
                'centroid_img_y': img_y,
                'centroid_img_x': img_x,
                'label_y': label_y,
                'label_x': label_x,
                'mask': region_mask,
                'core_mask': region_mask,  # Same as mask for DBSCAN
                'prediction': sum_flux,
                'groundtruth': pred_data.get('groundtruth', None)
            }
            
            # Store preferred previous region ID if found (for tracking to use)
            if preferred_previous_region_id is not None:
                region_data['preferred_previous_region_id'] = preferred_previous_region_id
            
            regions.append(region_data)
        
        if len(regions) > 0:
            print(f"  {timestamp}: Found {len(regions)} active regions using DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples})")
        
        return regions

    def _calculate_previous_region_proximity(self, patch_y, patch_x, previous_regions):
        """
        Calculate proximity to previous regions for each patch.
        
        Returns the distance (in patches) to the nearest previous region centroid.
        Patches closer to previous regions will have lower values, making them
        more likely to cluster together with patches from the same previous region.
        
        Args:
            patch_y, patch_x: Patch coordinates (in patch space)
            previous_regions: List of previous timestamp's region dictionaries
            
        Returns:
            Array of distances to nearest previous region (in patches)
        """
        if not previous_regions or len(previous_regions) == 0:
            # No previous regions, return large distance for all patches
            return np.full(len(patch_y), 1000.0)
        
        # Get centroids of previous regions (in patch space)
        prev_centroids = []
        for region in previous_regions:
            centroid_y = region.get('centroid_patch_y', None)
            centroid_x = region.get('centroid_patch_x', None)
            if centroid_y is not None and centroid_x is not None:
                prev_centroids.append((centroid_y, centroid_x))
        
        if len(prev_centroids) == 0:
            # No valid centroids, return large distance
            return np.full(len(patch_y), 1000.0)
        
        # Calculate distance from each patch to nearest previous region centroid
        prev_centroids = np.array(prev_centroids)
        
        # Fully vectorized calculation: for each patch, find distance to nearest previous centroid
        patch_coords = np.column_stack([patch_y, patch_x])  # Shape: (n_patches, 2)
        
        # Compute distances from all patches to all centroids
        # Shape: (n_patches, n_centroids)
        distances_to_all = np.sqrt(
            (patch_coords[:, 0:1] - prev_centroids[:, 0])**2 + 
            (patch_coords[:, 1:2] - prev_centroids[:, 1])**2
        )
        
        # Find minimum distance for each patch
        distances = np.min(distances_to_all, axis=1)
        
        return distances

    def _calculate_flux_gradient(self, flux_contrib, significant_mask, patch_y, patch_x):
        """
        Calculate the magnitude of flux gradient (spatial change) at each patch.
        
        The gradient represents how much flux changes spatially, which helps identify
        region boundaries and edges. Patches with similar gradients are likely in
        similar regions (e.g., both on boundaries or both in interiors).
        
        Args:
            flux_contrib: Full flux contribution map
            significant_mask: Mask of significant patches
            patch_y, patch_x: Coordinates of significant patches
            
        Returns:
            Array of gradient magnitudes for each significant patch
        """
        # Calculate spatial gradients using numpy
        # Gradient in y and x directions
        grad_y, grad_x = np.gradient(flux_contrib)
        
        # Calculate gradient magnitude: sqrt(grad_y^2 + grad_x^2)
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # Extract gradient values for significant patches only
        patch_gradients = gradient_magnitude[patch_y, patch_x]
        
        return patch_gradients

    def _detect_regions_for_timestamp(self, flux_contrib, timestamp, pred_data):
        """
        Robust region detection with dual-threshold approach and morphological operations.
        
        This method uses:
        1. High-confidence cores (strict threshold) - ensures stable region centers
        2. Growth from cores (permissive threshold) - captures full region extent
        3. Morphological closing - fills small gaps for stability
        """
        # Get config values with new dual-threshold parameters
        core_std_multiplier = self.flare_config.get('core_threshold_std_multiplier', 2.0)
        growth_std_multiplier = self.flare_config.get('growth_threshold_std_multiplier', 1.5)
        min_core_patches = self.flare_config.get('min_core_patches', 6)
        min_patches = self.flare_config.get('min_patches', 20)
        max_patches = self.flare_config.get('max_patches', 300)
        # Allow None to disable constraints (0 also treated as disabled for min, but None is preferred)
        if min_patches == 0:
            min_patches = None
        if max_patches == 0:
            max_patches = None
        closing_iterations = self.flare_config.get('closing_iterations', 5)
        dilation_iterations = self.flare_config.get('dilation_iterations', 3)
        prevent_overlap = self.flare_config.get('prevent_overlap', True)
        
        # Stage 1: Find high-confidence cores (strict threshold)
        # Calculate median and std in log space (more appropriate for flux data)
        flux_values = flux_contrib.flatten()
        # Filter out zero/negative values for log calculation
        positive_flux = flux_values[flux_values > 0]
        if len(positive_flux) == 0:
            return []  # No positive flux values
        log_flux = np.log(positive_flux)
        median_log_flux = np.median(log_flux)
        std_log_flux = np.std(log_flux)
        # Convert back to linear space for thresholds
        # Core threshold (higher) - regions must exceed this to appear
        core_threshold = np.exp(median_log_flux + core_std_multiplier * std_log_flux)
        # Growth threshold (lower) - regions can grow to this extent
        growth_threshold = np.exp(median_log_flux + growth_std_multiplier * std_log_flux)
        
        # Use core threshold for core detection
        core_mask = flux_contrib > core_threshold
        
        # Stage 2: Define growth region (more permissive threshold)
        growth_mask = flux_contrib > growth_threshold
        
        # Apply morphological closing to growth mask to fill small gaps
        structure_8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        if closing_iterations > 0:
            growth_mask = nd.binary_closing(growth_mask, structure=structure_8, iterations=closing_iterations)
        
        # Find connected core regions
        labeled_cores, num_cores = nd.label(core_mask, structure=structure_8)
        
        # Merge close/overlapping cores if enabled
        core_merge_distance = self.flare_config.get('core_merge_distance', 0)  # 0 = disabled
        if num_cores > 1 and core_merge_distance > 0:
            labeled_cores, num_cores = self._merge_close_cores(
                labeled_cores, num_cores, core_merge_distance, structure_8
            )
        
        if num_cores > 0:
            overlap_mode = "non-overlapping" if prevent_overlap else "overlapping"
            print(f"  {timestamp}: Found {num_cores} cores at {core_std_multiplier}σ threshold ({core_threshold:.2e}) - {overlap_mode} growth")
        
        regions = []
        accepted_region_id = 0
        
        # Track all claimed pixels to prevent overlap (if enabled)
        claimed_mask = np.zeros_like(growth_mask, dtype=bool) if prevent_overlap else None
        
        for core_id in range(1, num_cores + 1):
            core_region_mask = labeled_cores == core_id
            core_size = np.sum(core_region_mask)
            
            # Require minimum core size for stability
            if core_size < min_core_patches:
                continue
            
            if prevent_overlap and np.any(core_region_mask & claimed_mask):
                continue
            
            # Mark core pixels as claimed immediately (if overlap prevention enabled)
            if prevent_overlap:
                claimed_mask = claimed_mask | core_region_mask
            
            # Grow region from core
            grown_mask = core_region_mask.copy()
            if dilation_iterations > 0:
                if prevent_overlap:
                    # Iterative dilation: grow one step at a time, checking for overlaps
                    initial_size = np.sum(grown_mask)
                    for iteration in range(dilation_iterations):
                        # Find pixels that can be added (in growth mask and not claimed)
                        available_mask = growth_mask & ~claimed_mask
                        
                        # Dilate by one step
                        new_growth = nd.binary_dilation(
                            grown_mask,
                            structure=structure_8,
                            iterations=1
                        ) & available_mask
                        
                        # Only add new pixels that don't overlap with existing regions
                        new_pixels = new_growth & ~grown_mask
                        if np.sum(new_pixels) == 0:
                            break  # No more growth possible
                        
                        # Add new pixels to this region
                        grown_mask = grown_mask | new_pixels
                        
                        # Immediately mark new pixels as claimed to prevent overlap
                        claimed_mask = claimed_mask | new_pixels
                else:
                    # Original behavior: grow all at once (may overlap)
                    grown_mask = nd.binary_dilation(
                        core_region_mask,
                        structure=structure_8,
                        iterations=dilation_iterations,
                        mask=growth_mask
                    )
            
            region_size = np.sum(grown_mask)
            
            # Size filtering on grown region (None disables the constraint)
            min_check = (min_patches is None) or (region_size >= min_patches)
            max_check = (max_patches is None) or (region_size <= max_patches)
            if min_check and max_check:
                accepted_region_id += 1
                
                # Mark ALL pixels as claimed to prevent future overlap (if enabled)
                if prevent_overlap:
                    claimed_mask = claimed_mask | grown_mask
                
                region_flux = flux_contrib[grown_mask]
                sum_flux = np.sum(region_flux)
                max_flux = np.max(region_flux)
                
                # Filter by minimum flux threshold if enabled
                min_flux_threshold = self.flare_config.get('min_flux_threshold', None)
                if min_flux_threshold is not None and sum_flux < min_flux_threshold:
                    continue  # Skip regions with insufficient flux
                
                # Get region centroid from grown region
                coords = np.where(grown_mask)
                centroid_y, centroid_x = np.mean(coords[0]), np.mean(coords[1])
                
                # Convert to image coordinates
                img_y = centroid_y * self.patch_size + self.patch_size // 2
                img_x = centroid_x * self.patch_size + self.patch_size // 2
                
                # Calculate region-based prediction (sum of flux contributions in this region)
                region_prediction = sum_flux
                
                # Calculate label position
                min_y, max_y = np.min(coords[0]), np.max(coords[0])
                min_x, max_x = np.min(coords[1]), np.max(coords[1])
                label_y = max(0, min_y - 2)
                label_x = centroid_x
                
                regions.append({
                    'id': accepted_region_id,
                    'size': region_size,
                    'core_size': core_size,  # Track core size separately
                    'sum_flux': sum_flux,
                    'max_flux': max_flux,
                    'centroid_patch_y': centroid_y,
                    'centroid_patch_x': centroid_x,
                    'centroid_img_y': img_y,
                    'centroid_img_x': img_x,
                    'label_y': label_y,
                    'label_x': label_x,
                    'mask': grown_mask,
                    'core_mask': core_region_mask,  # Keep core mask for reference
                    'prediction': region_prediction,
                    'groundtruth': pred_data.get('groundtruth', None)
                })
        
        if prevent_overlap and len(regions) > 1:
            self._verify_no_overlaps(regions, timestamp)
        
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