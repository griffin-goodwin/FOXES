import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import cv2
from scipy import ndimage as nd
import warnings
import yaml
from datetime import datetime


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
            structure_4 = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]])

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

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Found {len(flare_events)} potential flare events")
    if len(simultaneous_flares) > 0:
        print(f"Found {len(simultaneous_flares)} simultaneous flare events in {simultaneous_flares['group_id'].nunique()} groups")


if __name__ == "__main__":
    main()