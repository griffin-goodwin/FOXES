import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import cv2
from scipy import ndimage as nd
import warnings
import yaml
from datetime import datetime
import imageio.v2 as imageio
import os
from multiprocessing import Pool
import time
from tqdm import tqdm


"""
Flux Contribution Analysis and Flare Detection Script

This script analyzes flux contributions from different patches to identify 
potential flaring events and visualize their spatial and temporal patterns.
"""



warnings.filterwarnings('ignore')


class FluxContributionAnalyzer:
    def __init__(self, config_path=None, flux_path=None, predictions_csv=None, aia_path=None, attention_path=None,
                 grid_size=(32, 32), patch_size=16, input_size=512, time_period=None):
        """
        Initialize the flux contribution analyzer

        Args:
            config_path: Path to YAML config file (optional, overrides other parameters)
            flux_path: Path to directory containing flux contribution files
            predictions_csv: Path to CSV file with predictions and timestamps
            aia_path: Path to directory containing AIA numpy files
            attention_path: Optional path to attention weights directory
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
        self.attention_path = Path(attention_path) if attention_path else None
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
        """Load AIA 94 Angstrom image for a specific timestamp"""
        if self.aia_path is None:
            return None

        # Try different possible filename formats
        possible_files = [
            self.aia_path / f"{timestamp}.npy",
            self.aia_path / f"{timestamp}.npz",
        ]

        for aia_file in possible_files:
            if aia_file.exists():
                try:
                    if aia_file.suffix == '.npy':
                        aia_data = np.load(aia_file)
                    else:  # .npz
                        aia_data = np.load(aia_file)['arr_0']

                    # Get 94 Angstrom channel (dimension 0) and ensure 512x512
                    aia_94 = aia_data[0]  # First channel is 94 Angstrom


                    # Ensure correct size
                    if aia_94.shape != (512, 512):
                        print(f"Warning: AIA image shape is {aia_94.shape}, expected (512, 512)")

                    return aia_94
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


    def resize_flux_to_image_size(self, flux_contrib):
        """Resize flux contribution map from patch grid to full image resolution"""
        # Use bicubic interpolation to smoothly upscale flux contributions
        return cv2.resize(flux_contrib, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_CUBIC)

    def detect_flare_events(self, threshold_percentile=None, min_patches=None, max_patches=None):
        """
        Detect potential flare events based on flux contribution patterns

        Args:
            threshold_percentile: Percentile threshold for high contribution patches
            min_patches: Minimum number of connected high-contribution patches
            max_patches: Maximum number of connected high-contribution patches
        """
        # Use config values if not provided
        if threshold_percentile is None:
            threshold_percentile = self.flare_config.get('threshold_percentile', 95)
        if min_patches is None:
            min_patches = self.flare_config.get('min_patches', 1)
        if max_patches is None:
            max_patches = self.flare_config.get('max_patches', 25)
        flare_events = []

        print("Analyzing flux contributions for flare detection...")

        for idx, row in self.predictions_df.iterrows():
            timestamp = row['timestamp']
            flux_contrib = self.load_flux_contributions(timestamp)

            if flux_contrib is None:
                continue

            # Calculate threshold for this timestamp
            threshold = np.percentile(flux_contrib.flatten(), threshold_percentile)
            # Find high contribution regions
            high_contrib_mask = flux_contrib > threshold

            # Use connected components to find flare regions
            # structure_4 = np.array([[0, 1, 0],
            #                         [1, 1, 1],
            #                         [0, 1, 0]])

            # 8-connectivity: patches connected by edges or corners
            structure_8 = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])

            # Use connected components to find flare regions
            labeled_regions, num_regions = nd.label(high_contrib_mask, structure=structure_8)

            for region_id in range(1, num_regions + 1):
                region_mask = labeled_regions == region_id
                region_size = np.sum(region_mask)

                if min_patches <= region_size <= max_patches:
                    # Calculate region properties
                    region_flux = flux_contrib[region_mask]
                    max_flux = np.max(region_flux)
                    mean_flux = np.mean(region_flux)
                    #calculate sum
                    sum_flux = np.sum(region_flux)

                    # Get region centroid
                    coords = np.where(region_mask)
                    centroid_y, centroid_x = np.mean(coords[0]), np.mean(coords[1])

                    # Convert to image coordinates
                    img_y = centroid_y * self.patch_size + self.patch_size // 2
                    img_x = centroid_x * self.patch_size + self.patch_size // 2

                    flare_events.append({
                        'timestamp': timestamp,
                        'datetime': row['datetime'],
                        'prediction': row['predictions'],
                        'groundtruth': row.get('groundtruth', None),
                        'region_size': region_size,
                        'max_flux': max_flux,
                        'mean_flux': mean_flux,
                        'sum_flux': sum_flux,
                        'centroid_patch_y': centroid_y,
                        'centroid_patch_x': centroid_x,
                        'centroid_img_y': img_y,
                        'centroid_img_x': img_x,
                        'threshold': threshold
                    })

        self.flare_events_df = pd.DataFrame(flare_events)
        print(f"Detected {len(flare_events)} potential flare events")
        return self.flare_events_df

    def detect_simultaneous_flares(self, threshold=1e-5):
        """
        Detect simultaneous flaring events - multiple distinct regions within the same flux prediction
        where each region has a sum of flux above the threshold
        
        Args:
            threshold: Sum of flux threshold for considering a region as a flare
        
        Returns:
            DataFrame with simultaneous flare events
        """
        if not hasattr(self, 'flare_events_df') or len(self.flare_events_df) == 0:
            print("Please run detect_flare_events() first")
            return pd.DataFrame()
        
        # Filter regions by sum_flux threshold (not prediction threshold)
        high_flux_regions = self.flare_events_df[self.flare_events_df['sum_flux'] >= threshold].copy()
        
        if len(high_flux_regions) == 0:
            print(f"No regions found with sum_flux above threshold {threshold}")
            return pd.DataFrame()
        
        # Group regions by timestamp to find simultaneous flares within the same flux prediction
        simultaneous_groups = []
        for timestamp, group in high_flux_regions.groupby('timestamp'):
            if len(group) >= 2:  # Multiple distinct regions at the same timestamp
                simultaneous_groups.append(group)
        
        # Create results DataFrame
        simultaneous_events = []
        for group_id, group in enumerate(simultaneous_groups):
            for idx, event in group.iterrows():
                simultaneous_events.append({
                    'group_id': group_id,
                    'timestamp': event['timestamp'],
                    'datetime': event['datetime'],
                    'prediction': event['prediction'],
                    'region_size': event['region_size'],
                    'max_flux': event['max_flux'],
                    'sum_flux': event['sum_flux'],
                    'centroid_img_y': event['centroid_img_y'],
                    'centroid_img_x': event['centroid_img_x'],
                    'group_size': len(group)
                })
        
        simultaneous_df = pd.DataFrame(simultaneous_events)
        
        if len(simultaneous_df) > 0:
            print(f"Detected {len(simultaneous_groups)} timestamps with simultaneous flares")
            print(f"Total simultaneous events: {len(simultaneous_df)}")
            
            # Print summary
            for group_id in simultaneous_df['group_id'].unique():
                group_events = simultaneous_df[simultaneous_df['group_id'] == group_id]
                timestamp = group_events['timestamp'].iloc[0]
                prediction = group_events['prediction'].iloc[0]
                print(f"\nSimultaneous Group {group_id} at {timestamp}:")
                print(f"  Flux prediction: {prediction:.2e}")
                print(f"  Number of distinct regions: {len(group_events)}")
                print(f"  Region sizes: {group_events['region_size'].values}")
                print(f"  Sum fluxes: {group_events['sum_flux'].values}")
        else:
            print("No simultaneous flare events detected")
        
        self.simultaneous_flares_df = simultaneous_df
        return simultaneous_df

    def load_attention_weights(self, timestamp):
        """Load attention weights for a specific timestamp if available"""
        if self.attention_path is None:
            return None
        attention_file = self.attention_path / f"{timestamp}"
        if attention_file.exists():
            return np.loadtxt(attention_file, delimiter=',')
        return None


    def plot_flux_contribution_heatmap(self, timestamp, save_path=None, show_attention=True, 
                                       threshold_percentile=None, min_patches=None, max_patches=None):
        """Plot flux contribution heatmap for a specific timestamp with detected regions highlighted"""
        flux_contrib = self.load_flux_contributions(timestamp)
        aia = self.load_aia_image(timestamp) if show_attention else None

        if flux_contrib is None:
            print(f"No flux contributions found for {timestamp}")
            return

        # Use config values if not provided
        if threshold_percentile is None:
            threshold_percentile = self.flare_config.get('threshold_percentile', 97)
        if min_patches is None:
            min_patches = self.flare_config.get('min_patches', 2)
        if max_patches is None:
            max_patches = self.flare_config.get('max_patches', 50)

        # Get prediction data for this timestamp
        pred_data = self.predictions_df[self.predictions_df['timestamp'] == timestamp].iloc[0]

        fig, axes = plt.subplots(1, 2 if aia is not None else 1,
                                 figsize=(15 if aia is not None else 8, 6))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Calculate threshold and find connected regions (same logic as detect_flare_events)
        threshold = np.percentile(flux_contrib.flatten(), threshold_percentile)
        high_contrib_mask = flux_contrib > threshold

        # Use connected components to find flare regions
        # 8-connectivity: patches connected by edges or corners
        structure_8 = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

        labeled_regions, num_regions = nd.label(high_contrib_mask, structure=structure_8)

        # Create a copy for display and collect region info
        flux_contrib_display = flux_contrib.copy()
        # Don't hide patches below threshold - show all patches that are part of detected regions
        # flux_contrib_display[flux_contrib < threshold] = np.nan

        detected_regions = []
        region_colors = plt.cm.Set3(np.linspace(0, 1, max(num_regions, 1)))
        accepted_region_id = 0

        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            region_size = np.sum(region_mask)

            if min_patches <= region_size <= max_patches:
                accepted_region_id += 1
                region_flux = flux_contrib[region_mask]
                sum_flux = np.sum(region_flux)
                max_flux = np.max(region_flux)

                # Get region centroid for labeling
                coords = np.where(region_mask)
                min_y, max_y = np.min(coords[0]), np.max(coords[0])
                min_x, max_x = np.min(coords[1]), np.max(coords[1])
                centroid_y, centroid_x = np.mean(coords[0]), np.mean(coords[1])

                # Position label above the region
                label_y = min_y - 2  # Place above the topmost patch
                label_x = centroid_x  # Center horizontally

                detected_regions.append({
                    'id': accepted_region_id,  # Use sequential ID for accepted regions
                    'size': region_size,
                    'sum_flux': sum_flux,
                    'max_flux': max_flux,
                    'centroid_y': centroid_y,
                    'centroid_x': centroid_x,
                    'label_y': label_y,
                    'label_x': label_x,
                    'mask': region_mask
                })

        # Plot flux contributions
        im1 = axes[0].imshow(flux_contrib_display, cmap='hot', interpolation='nearest', origin='lower')

        # Highlight detected regions with colored outlines
        for i, region in enumerate(detected_regions):
            # Create contour around the region
            axes[0].contour(region['mask'].astype(int), levels=[0.5],
                            colors=[region_colors[i % len(region_colors)]], linewidths=2)

            # Add region label with sum flux
            axes[0].text(region['label_x'], region['label_y'],
                         f"R{region['id']}\n{region['sum_flux']:.1e}",
                         ha='center', va='center', fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Build title with both prediction and ground truth
        title_text = f'Flux Contributions with Detected Regions\n{timestamp}\nPrediction: {pred_data["predictions"]:.2e}'
        
        # Add ground truth if available
        if 'groundtruth' in pred_data and not pd.isna(pred_data['groundtruth']):
            title_text += f'\nActual: {pred_data["groundtruth"]:.2e}'
        
        if detected_regions:
            total_region_flux = sum(r['sum_flux'] for r in detected_regions)
            title_text += f'\nTotal Region Flux: {total_region_flux:.2e} ({len(detected_regions)} regions)'

        axes[0].set_title(title_text)
        axes[0].set_xlabel('Patch X')
        axes[0].set_ylabel('Patch Y')

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Flux Contribution')

        # Add grid
        axes[0].set_xticks(np.arange(-0.5, self.grid_size[1], 1), minor=True)
        axes[0].set_yticks(np.arange(-0.5, self.grid_size[0], 1), minor=True)
        axes[0].grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

        if aia is not None:
            im2 = axes[1].imshow(aia, cmap='Blues', interpolation='nearest', origin='lower')
            axes[1].set_title(f'AIA Image 94 Å\n{timestamp}')
            axes[1].grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

        # Add legend for detected regions
        if detected_regions:
            legend_text = "Detected Regions:\n"
            for region in detected_regions:
                legend_text += f"R{region['id']}: {region['size']} patches, Sum: {region['sum_flux']:.1e}\n"

            # Add text box with region info
            axes[0].text(0.02, 0.98, legend_text.strip(), transform=axes[0].transAxes,
                         verticalalignment='top', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

        # Print region summary
        if detected_regions:
            print(f"\nDetected {len(detected_regions)} regions for timestamp {timestamp}:")
            for region in detected_regions:
                print(f"  Region {region['id']}: {region['size']} patches, "
                      f"Sum flux: {region['sum_flux']:.2e}, Max flux: {region['max_flux']:.2e}")



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

    def create_flare_movie(self, start_time, end_time, output_path, fps=2):
        """Create a movie showing flux contributions over time for a flare event"""
        # Filter timestamps in the time range
        mask = (self.predictions_df['datetime'] >= pd.to_datetime(start_time)) & \
               (self.predictions_df['datetime'] <= pd.to_datetime(end_time))
        event_data = self.predictions_df[mask].sort_values('datetime')

        if len(event_data) == 0:
            print(f"No data found in time range {start_time} to {end_time}")
            return

        print(f"Creating movie with {len(event_data)} frames...")

        # Create temporary directory for frames
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)

        # Determine global color scale
        all_flux = []
        for timestamp in event_data['timestamp']:
            flux_contrib = self.load_flux_contributions(timestamp)
            if flux_contrib is not None:
                all_flux.extend(flux_contrib.flatten())

        vmin, vmax = np.percentile(all_flux, [1, 99]) if all_flux else (0, 1)

        # Create frames
        frame_files = []
        for i, (_, row) in enumerate(event_data.iterrows()):
            timestamp = row['timestamp']
            flux_contrib = self.load_flux_contributions(timestamp)

            if flux_contrib is None:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))

            im = ax.imshow(flux_contrib, cmap='hot', vmin=vmin, vmax=vmax,
                           interpolation='nearest', origin='lower')

            ax.set_title(f'Flux Contributions\n{row["datetime"].strftime("%Y-%m-%d %H:%M:%S")}\n'
                         f'Prediction: {row["predictions"]:.2e}')
            ax.set_xlabel('Patch X')
            ax.set_ylabel('Patch Y')

            cbar = plt.colorbar(im)
            cbar.set_label('Flux Contribution')

            frame_file = temp_dir / f"frame_{i:04d}.png"
            plt.savefig(frame_file, dpi=100, bbox_inches='tight')
            plt.close()

            frame_files.append(frame_file)

        # Create video using matplotlib animation (simple approach)
        print(f"Saved {len(frame_files)} frames. Use external tool to create video:")
        print(f"ffmpeg -framerate {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_path}")

        return frame_files

    def generate_contour_frame_worker(self, timestamp):
        """Worker function to generate a single contour frame for movie creation"""
        try:
            print(f"Worker {os.getpid()}: Processing {timestamp}")

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

            # Create figure
            fig, axes = plt.subplots(1, 2 if aia is not None else 1,
                                   figsize=(15 if aia is not None else 8, 6))
            if not isinstance(axes, np.ndarray):
                axes = [axes]

            # Use config values for detection parameters
            threshold_percentile = self.flare_config.get('threshold_percentile', 97)
            min_patches = self.flare_config.get('min_patches', 2)
            max_patches = self.flare_config.get('max_patches', 50)

            # Calculate threshold and find connected regions (same logic as detect_flare_events)
            threshold = np.percentile(flux_contrib.flatten(), threshold_percentile)
            high_contrib_mask = flux_contrib > threshold

            # Use connected components to find flare regions
            structure_8 = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])

            labeled_regions, num_regions = nd.label(high_contrib_mask, structure=structure_8)

            # Create a copy for display and collect region info
            flux_contrib_display = flux_contrib.copy()

            detected_regions = []
            region_colors = plt.cm.Set3(np.linspace(0, 1, max(num_regions, 1)))
            accepted_region_id = 0

            for region_id in range(1, num_regions + 1):
                region_mask = labeled_regions == region_id
                region_size = np.sum(region_mask)

                if min_patches <= region_size <= max_patches:
                    accepted_region_id += 1
                    region_flux = flux_contrib[region_mask]
                    sum_flux = np.sum(region_flux)
                    max_flux = np.max(region_flux)

                    # Get region centroid for labeling
                    coords = np.where(region_mask)
                    min_y, max_y = np.min(coords[0]), np.max(coords[0])
                    min_x, max_x = np.min(coords[1]), np.max(coords[1])
                    centroid_y, centroid_x = np.mean(coords[0]), np.mean(coords[1])

                    # Position label above the region
                    label_y = min_y - 2  # Place above the topmost patch
                    label_x = centroid_x  # Center horizontally

                    detected_regions.append({
                        'id': accepted_region_id,  # Use sequential ID for accepted regions
                        'size': region_size,
                        'sum_flux': sum_flux,
                        'max_flux': max_flux,
                        'centroid_y': centroid_y,
                        'centroid_x': centroid_x,
                        'label_y': label_y,
                        'label_x': label_x,
                        'mask': region_mask
                    })

            # Plot flux contributions
            im1 = axes[0].imshow(flux_contrib_display, cmap='hot', interpolation='nearest', origin='lower')

            # Highlight detected regions with colored outlines
            for i, region in enumerate(detected_regions):
                # Create contour around the region
                axes[0].contour(region['mask'].astype(int), levels=[0.5],
                                colors=[region_colors[i % len(region_colors)]], linewidths=2)

                # Add region label with sum flux
                axes[0].text(region['label_x'], region['label_y'],
                             f"R{region['id']}\n{region['sum_flux']:.1e}",
                             ha='center', va='center', fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

            # Build title with both prediction and ground truth
            title_text = f'Flux Contributions Evolution\n{timestamp}\nPrediction: {pred_data["predictions"]:.2e}'
            
            # Add ground truth if available
            if 'groundtruth' in pred_data and not pd.isna(pred_data['groundtruth']):
                title_text += f'\nActual: {pred_data["groundtruth"]:.2e}'
            
            if detected_regions:
                total_region_flux = sum(r['sum_flux'] for r in detected_regions)
                title_text += f'\nTotal Region Flux: {total_region_flux:.2e} ({len(detected_regions)} regions)'

            axes[0].set_title(title_text)
            axes[0].set_xlabel('Patch X')
            axes[0].set_ylabel('Patch Y')

            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=axes[0])
            cbar1.set_label('Flux Contribution')

            # Add grid
            axes[0].set_xticks(np.arange(-0.5, self.grid_size[1], 1), minor=True)
            axes[0].set_yticks(np.arange(-0.5, self.grid_size[0], 1), minor=True)
            axes[0].grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

            if aia is not None:
                im2 = axes[1].imshow(aia, cmap='Blues', interpolation='nearest', origin='lower')
                axes[1].set_title(f'AIA Image 94 Å\n{timestamp}')
                axes[1].grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

            # Add legend for detected regions
            if detected_regions:
                legend_text = "Detected Regions:\n"
                for region in detected_regions:
                    legend_text += f"R{region['id']}: {region['size']} patches, Sum: {region['sum_flux']:.1e}\n"

                # Add text box with region info
                axes[0].text(0.02, 0.98, legend_text.strip(), transform=axes[0].transAxes,
                             verticalalignment='top', fontsize=8,
                             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

            print(f"Worker {os.getpid()}: Completed {timestamp}")
            return save_path

        except Exception as e:
            print(f"Worker {os.getpid()}: Error processing {timestamp}: {e}")
            plt.close('all')  # Clean up any open figures
            return None

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
            regions = self._detect_regions_for_timestamp(flux_contrib, timestamp, pred_data)
            return (timestamp, regions)
        except Exception as e:
            print(f"Error detecting regions for {timestamp}: {e}")
            return (timestamp, None)

    def track_regions_over_time(self, timestamps, max_distance=50):
        """
        Track regions across time using spatial proximity and temporal continuity.
        
        Args:
            timestamps: List of timestamps to analyze
            max_distance: Maximum distance (in pixels) to consider regions as the same
            
        Returns:
            Dictionary mapping region_id to list of (timestamp, region_data) tuples
        """
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
        
        # Optimization: Keep track of active tracks (those updated recently)
        active_tracks = set()  # Track IDs that were updated in recent timestamps
        max_time_gap = 30 * 60  # 30 minutes in seconds
        
        for i, timestamp in tqdm(enumerate(timestamps), desc="Tracking regions", unit="timestamp", total=len(timestamps)):
            if timestamp not in all_regions:
                continue
                
            current_regions = all_regions[timestamp]
            current_time = pd.to_datetime(timestamp)
            
            # Filter active tracks based on temporal proximity
            recently_active_tracks = set()
            for track_id in list(active_tracks):
                if track_id in region_tracks and region_tracks[track_id]:
                    last_timestamp, _ = region_tracks[track_id][-1]
                    time_diff = abs((current_time - pd.to_datetime(last_timestamp)).total_seconds())
                    if time_diff <= max_time_gap:
                        recently_active_tracks.add(track_id)
            
            for region in current_regions:
                # Create a copy to avoid mutating the original
                region_copy = region.copy()
                
                # Try to match with existing tracks (only check recently active ones first)
                best_track_id = None
                best_distance = float('inf')
                
                # First, check recently active tracks (much faster)
                for track_id in recently_active_tracks:
                    if not region_tracks[track_id]:
                        continue
                        
                    # Get the most recent region in this track
                    last_timestamp, last_region = region_tracks[track_id][-1]
                    
                    # Calculate distance between centroids
                    distance = np.sqrt(
                        (region_copy['centroid_img_x'] - last_region['centroid_img_x'])**2 + 
                        (region_copy['centroid_img_y'] - last_region['centroid_img_y'])**2
                    )
                    
                    if distance < max_distance and distance < best_distance:
                        best_distance = distance
                        best_track_id = track_id
                
                # Assign to existing track or create new one
                if best_track_id is not None:
                    # Update the region ID to match the track ID for consistency
                    region_copy['id'] = best_track_id
                    region_tracks[best_track_id].append((timestamp, region_copy))
                    active_tracks.add(best_track_id)
                else:
                    # Create new track with consistent ID
                    region_copy['id'] = next_track_id
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
        
        # Debug: Print track information
        for track_id, track_history in region_tracks.items():
            timestamps_in_track = [t for t, r in track_history]
            print(f"Track {track_id}: {len(track_history)} timestamps from {timestamps_in_track[0]} to {timestamps_in_track[-1]}")
        
        return region_tracks
    
    def _apply_temporal_smoothing(self, region_tracks, window_size=3):
        """
        Apply temporal smoothing to region flux values to reduce unrealistic jumps.
        
        Args:
            region_tracks: Dictionary of track_id -> list of (timestamp, region_data)
            window_size: Size of moving average window
            
        Returns:
            Updated region_tracks with smoothed flux values
        """
        print(f"Applying temporal smoothing with window size {window_size}...")
        
        for track_id, track_history in region_tracks.items():
            # Extract flux values
            flux_values = [r['sum_flux'] for t, r in track_history]
            
            # Apply moving average
            smoothed_flux = pd.Series(flux_values).rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).mean().tolist()
            
            # Update regions with smoothed values
            for i, (t, r) in enumerate(track_history):
                r['sum_flux_original'] = r['sum_flux']  # Keep original
                r['sum_flux'] = smoothed_flux[i]  # Use smoothed for display
                r['prediction'] = smoothed_flux[i]  # Update prediction too
        
        return region_tracks

    def _detect_regions_for_timestamp(self, flux_contrib, timestamp, pred_data):
        """
        Robust region detection with dual-threshold approach and morphological operations.
        
        This method uses:
        1. High-confidence cores (strict threshold) - ensures stable region centers
        2. Growth from cores (permissive threshold) - captures full region extent
        3. Morphological closing - fills small gaps for stability
        """
        # Get config values with new dual-threshold parameters
        core_percentile = self.flare_config.get('core_threshold_percentile', 98)
        growth_percentile = self.flare_config.get('growth_threshold_percentile', 95)
        min_core_patches = self.flare_config.get('min_core_patches', 2)
        min_patches = self.flare_config.get('min_patches', 3)
        max_patches = self.flare_config.get('max_patches', 50)
        closing_iterations = self.flare_config.get('closing_iterations', 1)
        dilation_iterations = self.flare_config.get('dilation_iterations', 3)
        prevent_overlap = self.flare_config.get('prevent_overlap', True)

        # Stage 1: Find high-confidence cores (strict threshold)
        core_threshold = np.percentile(flux_contrib.flatten(), core_percentile)
        core_mask = flux_contrib > core_threshold
        
        # Stage 2: Define growth region (more permissive threshold)
        growth_threshold = np.percentile(flux_contrib.flatten(), growth_percentile)
        growth_mask = flux_contrib > growth_threshold
        
        # Apply morphological closing to growth mask to fill small gaps
        structure_8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        if closing_iterations > 0:
            growth_mask = nd.binary_closing(growth_mask, structure=structure_8, iterations=closing_iterations)
        
        # Find connected core regions
        labeled_cores, num_cores = nd.label(core_mask, structure=structure_8)
        
        # Debug: Print threshold information
        if num_cores > 0:
            overlap_mode = "non-overlapping" if prevent_overlap else "overlapping"
            print(f"  {timestamp}: Found {num_cores} cores at {core_percentile}% threshold ({core_threshold:.2e}) - {overlap_mode} growth")
        
        regions = []
        accepted_region_id = 0
        skipped_due_to_overlap = 0
        
        # Track all claimed pixels to prevent overlap (if enabled)
        claimed_mask = np.zeros_like(growth_mask, dtype=bool) if prevent_overlap else None
        
        for core_id in range(1, num_cores + 1):
            core_region_mask = labeled_cores == core_id
            core_size = np.sum(core_region_mask)
            
            # Require minimum core size for stability
            if core_size < min_core_patches:
                continue
            
            # Check if core overlaps with already claimed regions (if overlap prevention enabled)
            if prevent_overlap and np.any(core_region_mask & claimed_mask):
                print(f"    Region {core_id}: Core overlaps with existing region, skipping")
                skipped_due_to_overlap += 1
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
                    
                    # Debug: Show if growth was limited by overlap prevention
                    final_size = np.sum(grown_mask)
                    if final_size < initial_size + dilation_iterations:
                        print(f"    Region {core_id}: Growth limited by overlap prevention ({initial_size} → {final_size})")
                    
                    # Debug: Show claimed mask state
                    claimed_pixels = np.sum(claimed_mask)
                    print(f"    Region {core_id}: {claimed_pixels} pixels claimed total")
                else:
                    # Original behavior: grow all at once (may overlap)
                    grown_mask = nd.binary_dilation(
                        core_region_mask,
                        structure=structure_8,
                        iterations=dilation_iterations,
                        mask=growth_mask
                    )
            
            region_size = np.sum(grown_mask)
            
            # Size filtering on grown region
            if min_patches <= region_size <= max_patches:
                accepted_region_id += 1
                
                # Mark ALL pixels as claimed to prevent future overlap (if enabled)
                if prevent_overlap:
                    claimed_mask = claimed_mask | grown_mask
                
                # Calculate flux for ENTIRE grown region
                region_flux = flux_contrib[grown_mask]
                sum_flux = np.sum(region_flux)
                max_flux = np.max(region_flux)
                
                # Debug: Print core vs grown region info
                growth_ratio = region_size / core_size if core_size > 0 else 1
                print(f"    Region {accepted_region_id}: core={core_size} patches, grown={region_size} patches (×{growth_ratio:.1f}), flux={sum_flux:.2e}")
                
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
        
        # Print summary
        if prevent_overlap and skipped_due_to_overlap > 0:
            print(f"    Skipped {skipped_due_to_overlap} regions due to overlap prevention")
        
        # Verify no overlaps if overlap prevention is enabled
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
                    print(f"    Region {region1['id']} size: {np.sum(region1['mask'])}")
                    print(f"    Region {region2['id']} size: {np.sum(region2['mask'])}")
        else:
            print(f"    Verified: No overlaps between {len(regions)} regions")

    def create_contour_movie(self, timestamps, auto_cleanup=True, fps=2, show_sxr_timeseries=True, 
                           all_timestamps_for_tracking=None):
        """
        Create a movie showing the evolution of contour plots over time with SXR time series
        
        Args:
            timestamps: Timestamps to generate frames for (subsampled for visualization)
            auto_cleanup: Whether to delete frame files after movie creation
            fps: Frames per second for the movie
            show_sxr_timeseries: Whether to show SXR time series plot
            all_timestamps_for_tracking: Full resolution timestamps for accurate region tracking
                                        (if None, uses timestamps parameter)
        """
        print(f"Creating contour movie with {len(timestamps)} frame timestamps...")

        # Track regions across time if SXR time series is requested
        region_tracks = {}
        if show_sxr_timeseries:
            # Use full resolution timestamps for tracking if provided, otherwise use frame timestamps
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
        
        # Calculate optimal chunk size for better load balancing
        chunksize = max(1, len(timestamps) // (num_processes * 4))
        print(f"Using chunksize={chunksize} for load balancing")

        # Process frames in parallel
        start_time = time.time()

        if show_sxr_timeseries and region_tracks:
            # Use SXR-enabled frame generation
            print("Using SXR time series frame generation...")
            with Pool(processes=num_processes) as pool:
                # Create a partial function with region_tracks
                from functools import partial
                frame_worker = partial(self.generate_contour_frame_with_sxr_worker, region_tracks=region_tracks)
                # Use imap with chunksize=1 for better progress bar updates
                results = []
                for result in tqdm(pool.imap(frame_worker, timestamps, chunksize=1), 
                                  desc="Generating frames", unit="frame", total=len(timestamps)):
                    results.append(result)
        else:
            # Use standard frame generation
            print("Using standard frame generation...")
            with Pool(processes=num_processes) as pool:
                # Use imap with chunksize=1 for better progress bar updates
                results = []
                for result in tqdm(pool.imap(self.generate_contour_frame_worker, timestamps, chunksize=1),
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

    def generate_contour_frame_with_sxr_worker(self, timestamp, region_tracks=None):
        """Worker function to generate a single contour frame with SXR time series for tracked regions"""
        try:
            # Removed verbose print for speed - progress bar shows this

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

            # Create figure with AIA+flux overlay on top and SXR plot on bottom
            # DPI=100 for faster rendering, fixed spacing for consistent frame size
            fig = plt.figure(figsize=(8, 13), dpi=100)
            # Overall grid: 2 rows x 1 column with fixed spacing
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], 
                                 hspace=0.3, left=0.1, right=0.95, top=0.95, bottom=0.05)

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
                                # Calculate label position from centroid (try multiple key names)
                                centroid_y = region_data_copy.get('centroid_patch_y', 
                                                                  region_data_copy.get('centroid_y', 0))
                                centroid_x = region_data_copy.get('centroid_patch_x', 
                                                                  region_data_copy.get('centroid_x', 0))
                                
                                # Position label above the region
                                region_data_copy['label_y'] = max(0, centroid_y - 2)
                                region_data_copy['label_x'] = centroid_x
                            
                            detected_regions.append(region_data_copy)
                            break
            else:
                # Fallback: detect regions independently if no tracks available
                threshold_percentile = self.flare_config.get('threshold_percentile', 97)
                min_patches = self.flare_config.get('min_patches', 2)
                max_patches = self.flare_config.get('max_patches', 50)

                # Calculate threshold and find connected regions
                threshold = np.percentile(flux_contrib.flatten(), threshold_percentile)
                high_contrib_mask = flux_contrib > threshold

                # Use connected components to find flare regions
                structure_8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
                labeled_regions, num_regions = nd.label(high_contrib_mask, structure=structure_8)

                accepted_region_id = 0
                for region_id in range(1, num_regions + 1):
                    region_mask = labeled_regions == region_id
                    region_size = np.sum(region_mask)

                    if min_patches <= region_size <= max_patches:
                        accepted_region_id += 1
                        region_flux = flux_contrib[region_mask]
                        sum_flux = np.sum(region_flux)
                        max_flux = np.max(region_flux)

                        # Get region centroid for labeling
                        coords = np.where(region_mask)
                        min_y, max_y = np.min(coords[0]), np.max(coords[0])
                        min_x, max_x = np.min(coords[1]), np.max(coords[1])
                        centroid_y, centroid_x = np.mean(coords[0]), np.mean(coords[1])

                        # Position label above the region
                        label_y = min_y - 2
                        label_x = centroid_x

                        # Convert to image coordinates
                        img_y = centroid_y * self.patch_size + self.patch_size // 2
                        img_x = centroid_x * self.patch_size + self.patch_size // 2

                        detected_regions.append({
                            'id': accepted_region_id,
                            'size': region_size,
                            'sum_flux': sum_flux,
                            'max_flux': max_flux,
                            'centroid_patch_y': centroid_y,
                            'centroid_patch_x': centroid_x,
                            'centroid_img_y': img_y,
                            'centroid_img_x': img_x,
                            'label_y': label_y,
                            'label_x': label_x,
                            'mask': region_mask
                        })

            # Create a copy for display
            flux_contrib_display = flux_contrib.copy()

            # Set up color mapping for regions (matching SXR plot colors)
            # Use the SAME color list as in the SXR plot below
            region_colors = ['#8dd3c7','#ffffb3','#bebada','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']
            region_to_color = {}
            
            if region_tracks and detected_regions:
                # Get all track IDs and sort them to ensure consistent color assignment
                track_ids = sorted(region_tracks.keys())
                for i, track_id in enumerate(track_ids):
                    region_to_color[track_id] = region_colors[i % len(region_colors)]
            else:
                # Fallback to original colors if no region tracks
                fallback_colors = plt.cm.Set3(np.linspace(0, 1, max(len(detected_regions), 1)))
                region_to_color = {region['id']: fallback_colors[i % len(fallback_colors)] 
                                 for i, region in enumerate(detected_regions)}

            # Plot AIA image with flux overlay (top row)
            if aia is not None:
                ax_aia = fig.add_subplot(gs[0, 0])
                
                # Display AIA image
                im2 = ax_aia.imshow(aia, cmap='Blues', interpolation='nearest', origin='lower')
                
                # Overlay flux contributions on AIA image
                # Resize flux contributions to match AIA image size
                flux_resized = cv2.resize(flux_contrib_display, (aia.shape[1], aia.shape[0]), 
                                        interpolation=cv2.INTER_CUBIC)
                
                # Overlay flux with transparency
                im3 = ax_aia.imshow(flux_resized, cmap='hot', interpolation='nearest', 
                                   origin='lower', alpha=0.6)
                
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
                        # Convert patch coordinates to image coordinates
                        if 'centroid_img_y' in region and 'centroid_img_x' in region:
                            label_y_img = int(region['centroid_img_y'] * aia.shape[0] / self.input_size)
                            label_x_img = int(region['centroid_img_x'] * aia.shape[1] / self.input_size)
                        else:
                            # Fallback to patch coordinates if image coordinates not available
                            label_y_img = int(region.get('centroid_y', 0) * aia.shape[0] / self.grid_size[0])
                            label_x_img = int(region.get('centroid_x', 0) * aia.shape[1] / self.grid_size[1])
                        
                        ax_aia.text(label_x_img, label_y_img - 40,
                                   f"R{region['id']}\n{region['sum_flux']:.1e}",
                                   ha='center', va='center', fontsize=8, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color),
                                   color='black')
                
                ax_aia.set_title(f'AIA 94 Å + Flux Overlay\n{timestamp}', fontsize=10)
                ax_aia.axis('off')
                
                # Add colorbar for flux overlay
                cbar2 = plt.colorbar(im3, ax=ax_aia, shrink=0.8)
                cbar2.set_label('Flux Contribution', fontsize=9)

            # Plot single SXR time series with integrated values
            if region_tracks:
                # Create single SXR plot in bottom row
                ax_sxr = fig.add_subplot(gs[1, 0])
                
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
                    time_window = pd.Timedelta(hours=4)
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
                time_window = pd.Timedelta(hours=4)
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
                region_colors_list = ['#8dd3c7','#ffffb3','#bebada','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']
                
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
                
                # Add current values text box
                # Calculate current sum of region fluxes (actual sum_flux values)
                current_region_sum = sum(r.get('sum_flux', 0) for r in detected_regions)
                
                info_text = f"Current Time: {current_time.strftime('%H:%M:%S')}\n"
                if pred_data.get('groundtruth') is not None:
                    info_text += f"Ground Truth: {pred_data['groundtruth']:.2e}\n"
                info_text += f"Model Prediction: {pred_data['predictions']:.2e}\n"
                info_text += f"Sum of All Regions: {current_region_sum:.2e}\n"
                info_text += f"Active Regions: {len(detected_regions)}"
                
                ax_sxr.text(0.02, 0.98, info_text, transform=ax_sxr.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.9),
                           verticalalignment='top')
                
                ax_sxr.set_title('SXR Time Series: Ground Truth, Model Prediction, and Region Fluxes', fontsize=12)
                ax_sxr.set_ylabel('SXR Flux (W/m²)', fontsize=11)
                ax_sxr.set_xlabel('Time', fontsize=11)
                
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
                ax_sxr.legend(fontsize=9, loc='upper right')
                
                # Set x-axis limits to show reasonable time window
                time_window = pd.Timedelta(hours=4)
                ax_sxr.set_xlim([current_time - time_window, current_time + time_window])
                
                # Rotate x-axis labels
                ax_sxr.tick_params(axis='x', rotation=45, labelsize=10)

            # Don't use tight_layout or bbox_inches='tight' - keeps frame size consistent
            plt.savefig(save_path, dpi=100)  # Reduced DPI for speed
            plt.close(fig)  # Close specific figure instead of all
            return save_path

        except Exception as e:
            print(f"Worker {os.getpid()}: Error processing {timestamp}: {e}")
            plt.close('all')
            return None


def main():
    parser = argparse.ArgumentParser(description='Analyze flux contributions for flare detection')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--flux_path', help='Path to flux contributions directory (overrides config)')
    parser.add_argument('--predictions_csv', help='Path to predictions CSV file (overrides config)')
    parser.add_argument('--attention_path', help='Path to attention weights directory (optional)')
    parser.add_argument('--output_dir', help='Output directory for results (overrides config)')
    parser.add_argument('--start_time', help='Start time for analysis (overrides config)')
    parser.add_argument('--end_time', help='End time for analysis (overrides config)')
    parser.add_argument('--viz_threshold', type=float, help='Visualization threshold (overrides config)')
    parser.add_argument('--create_contour_movie', action='store_true', help='Create contour evolution movie')
    parser.add_argument('--movie_fps', type=int, default=2, help='Frames per second for movie (default: 2)')
    parser.add_argument('--movie_interval_minutes', type=int, help='Minutes between frames for movie (overrides config)')
    parser.add_argument('--show_sxr_timeseries', action='store_true', help='Show SXR time series for tracked regions in movie')
    parser.add_argument('--max_tracking_distance', type=int, default=50, help='Maximum distance for region tracking (pixels)')

    args = parser.parse_args()

    # Initialize analyzer with config
    analyzer = FluxContributionAnalyzer(config_path=args.config)

    # Override config with command line arguments if provided
    if args.flux_path:
        analyzer.flux_path = Path(args.flux_path)
    if args.predictions_csv:
        analyzer.predictions_df = pd.read_csv(args.predictions_csv)
        analyzer.predictions_df['datetime'] = pd.to_datetime(analyzer.predictions_df['timestamp'])
        analyzer.predictions_df = analyzer.predictions_df.sort_values('datetime')
    if args.attention_path:
        analyzer.attention_path = Path(args.attention_path)
    if args.start_time and args.end_time:
        analyzer.time_period = {
            'start_time': args.start_time,
            'end_time': args.end_time
        }
        # Re-filter data
        start_time = pd.to_datetime(analyzer.time_period['start_time'])
        end_time = pd.to_datetime(analyzer.time_period['end_time'])
        mask = (analyzer.predictions_df['datetime'] >= start_time) & (analyzer.predictions_df['datetime'] <= end_time)
        analyzer.predictions_df = analyzer.predictions_df[mask].reset_index(drop=True)
        print(f"Filtered data to time period: {start_time} to {end_time}")

    # Get output directory from config or args
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir_str = analyzer.output_config.get('output_dir', 'flux_analysis_output')
        # Replace base_data_dir placeholder if present
        if '${base_data_dir}' in output_dir_str:
            base_dir = analyzer.config.get('base_data_dir', '/mnt/data/COMBINED')
            output_dir_str = output_dir_str.replace('${base_data_dir}', base_dir)
        output_dir = Path(output_dir_str)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Detect flare events using config parameters
    print("Detecting flare events...")
    flare_events = analyzer.detect_flare_events()

    # Detect simultaneous flares
    print("\nDetecting simultaneous flares...")
    simultaneous_threshold = analyzer.flare_config.get('simultaneous_flare_threshold', 5e-6)
    simultaneous_flares = analyzer.detect_simultaneous_flares(threshold=simultaneous_threshold)

    # Create flare event summary
    flare_summary_path = output_dir / 'flare_events_summary.csv'
    analyzer.create_flare_event_summary(flare_summary_path)

    # Save simultaneous flares summary
    if len(simultaneous_flares) > 0:
        simultaneous_summary_path = output_dir / 'simultaneous_flares_summary.csv'
        simultaneous_flares.to_csv(simultaneous_summary_path, index=False)
        print(f"Simultaneous flares summary saved to: {simultaneous_summary_path}")

    # Create visualizations if enabled in config
    if analyzer.output_config.get('create_visualizations', True) and len(flare_events) > 0:
        print("\nCreating visualizations for top flare events...")
        max_viz = analyzer.output_config.get('max_visualizations', 10)
        viz_threshold = args.viz_threshold if args.viz_threshold is not None else analyzer.output_config.get('visualization_threshold', 0.0)
        
        # Filter events by visualization threshold
        high_prediction_events = flare_events[flare_events['prediction'] >= viz_threshold]
        print(f"Found {len(high_prediction_events)} events above visualization threshold {viz_threshold:.2e}")
        
        if len(high_prediction_events) > 0:
            top_events = high_prediction_events.head(max_viz).sort_values('prediction', ascending=False)
            print(f"Creating visualizations for top {len(top_events)} events...")

            for i, (_, event) in enumerate(top_events.iterrows()):
                output_path = output_dir / f'flare_event_{i + 1}_{event["timestamp"]}.png'
                analyzer.plot_flux_contribution_heatmap(
                    event['timestamp'],
                    save_path=output_path,
                    show_attention=True,
                    threshold_percentile=analyzer.flare_config.get('threshold_percentile', 97)
                )
        else:
            print(f"No events found above visualization threshold {viz_threshold:.2e}")

    # Create visualizations for simultaneous flares in separate folder
    if len(simultaneous_flares) > 0 and analyzer.output_config.get('create_visualizations', True):
        print("\nCreating visualizations for simultaneous flare events...")
        simultaneous_output_dir = output_dir / 'simultaneous_flares'
        simultaneous_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get unique timestamps with simultaneous flares
        simultaneous_timestamps = simultaneous_flares['timestamp'].unique()
        print(f"Creating visualizations for {len(simultaneous_timestamps)} timestamps with simultaneous flares...")
        
        for i, timestamp in enumerate(simultaneous_timestamps):
            # Get all events for this timestamp
            timestamp_events = simultaneous_flares[simultaneous_flares['timestamp'] == timestamp]
            group_id = timestamp_events['group_id'].iloc[0]
            group_size = timestamp_events['group_size'].iloc[0]
            
            output_path = simultaneous_output_dir / f'simultaneous_group_{group_id}_{timestamp}.png'
            analyzer.plot_flux_contribution_heatmap(
                timestamp,
                save_path=output_path,
                show_attention=True,
                threshold_percentile=analyzer.flare_config.get('threshold_percentile', 97)
            )
            print(f"  Saved simultaneous flare visualization: {output_path} ({group_size} events)")

    # Create movie if enabled in config
    if analyzer.output_config.get('create_movie', False) and analyzer.time_period:
        print("\nCreating flare movie...")
        movie_fps = analyzer.output_config.get('movie_fps', 2)
        movie_path = output_dir / 'flare_event_movie.mp4'
        analyzer.create_flare_movie(
            start_time=analyzer.time_period['start_time'],
            end_time=analyzer.time_period['end_time'],
            output_path=movie_path,
            fps=movie_fps
        )

    # Create contour evolution movie if requested (command line or config)
    create_contour_movie = args.create_contour_movie or analyzer.output_config.get('create_contour_movie', False)
    if create_contour_movie and analyzer.time_period:
        print("\nCreating contour evolution movie...")
        # Generate timestamps for the time period
        start_time = pd.to_datetime(analyzer.time_period['start_time'])
        end_time = pd.to_datetime(analyzer.time_period['end_time'])
        
        # Get movie parameters
        movie_fps = args.movie_fps if args.movie_fps != 2 else analyzer.output_config.get('movie_fps', 2)
        movie_interval_minutes = args.movie_interval_minutes if args.movie_interval_minutes is not None else analyzer.output_config.get('movie_interval_minutes', 15)
        
        # Get available timestamps from the data within the time period
        available_timestamps = analyzer.predictions_df[
            (analyzer.predictions_df['datetime'] >= start_time) & 
            (analyzer.predictions_df['datetime'] <= end_time)
        ]['timestamp'].tolist()
        
        # Keep ALL timestamps for accurate region tracking
        all_timestamps_for_tracking = available_timestamps
        
        # Subsample timestamps for frame generation (visualization only)
        if movie_interval_minutes > 0:
            # Convert to datetime for easier filtering
            available_datetimes = pd.to_datetime(available_timestamps)
            
            # Filter to desired interval for frame generation
            filtered_timestamps = []
            last_time = None
            for i, dt in enumerate(available_datetimes):
                if last_time is None or (dt - last_time).total_seconds() >= movie_interval_minutes * 60:
                    filtered_timestamps.append(available_timestamps[i])
                    last_time = dt
        else:
            # Use all available timestamps
            filtered_timestamps = available_timestamps
        
        print(f"Found {len(available_timestamps)} total timestamps in time period")
        print(f"Tracking regions at full resolution: {len(all_timestamps_for_tracking)} timestamps")
        print(f"Generating frames at {movie_interval_minutes}-minute intervals: {len(filtered_timestamps)} frames")
        print(f"Movie will be {len(filtered_timestamps)/movie_fps:.1f} seconds long at {movie_fps} FPS")
        
        timestamps = filtered_timestamps
        
        # Set output directory for movie
        analyzer.output_dir = str(output_dir)
        
        # Create the movie
        show_sxr = args.show_sxr_timeseries or analyzer.output_config.get('show_sxr_timeseries', False)
        movie_path = analyzer.create_contour_movie(
            timestamps=timestamps,
            auto_cleanup=True,
            fps=movie_fps,
            show_sxr_timeseries=show_sxr,
            all_timestamps_for_tracking=all_timestamps_for_tracking  # Pass full resolution timestamps
        )
        
        if movie_path:
            print(f"Contour evolution movie created: {movie_path}")
    elif create_contour_movie and not analyzer.time_period:
        print("Warning: Cannot create contour movie without time period specified in config or command line")

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Found {len(flare_events)} potential flare events")
    if len(simultaneous_flares) > 0:
        print(f"Found {len(simultaneous_flares)} simultaneous flare events in {simultaneous_flares['group_id'].nunique()} groups")


if __name__ == "__main__":
    main()