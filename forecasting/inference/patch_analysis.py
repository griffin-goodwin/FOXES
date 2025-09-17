import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import cv2
from scipy import ndimage as nd
import warnings


def create_flare_movie_with_aia(self, start_time, end_time, output_dir, fps=2):
    """Create a movie showing flux contributions overlaid on AIA images"""
    # Filter timestamps in the time range
    mask = (self.predictions_df['datetime'] >= pd.to_datetime(start_time)) & \
           (self.predictions_df['datetime'] <= pd.to_datetime(end_time))
    event_data = self.predictions_df[mask].sort_values('datetime')

    if len(event_data) == 0:
        print(f"No data found in time range {start_time} to {end_time}")
        return

    print(f"Creating AIA+flux movie with {len(event_data)} frames...")

    # Create output directory for frames
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Determine global color scales
    all_flux = []
    aia_intensities = []

    for timestamp in event_data['timestamp'].head(20):  # Sample first 20 for scaling
        flux_contrib = self.load_flux_contributions(timestamp)
        aia_image = self.load_aia_image(timestamp)

        if flux_contrib is not None:
            all_flux.extend(flux_contrib.flatten())
        if aia_image is not None:
            aia_clipped = np.clip(aia_image, np.percentile(aia_image, 1),
                                  np.percentile(aia_image, 99.5))
            aia_intensities.extend(aia_clipped.flatten())

    flux_vmin, flux_vmax = np.percentile(all_flux, [5, 95]) if all_flux else (0, 1)
    aia_vmin, aia_vmax = np.percentile(aia_intensities, [1, 99]) if aia_intensities else (1, 1000)

    print(f"Flux range: {flux_vmin:.2e} - {flux_vmax:.2e}")
    print(f"AIA range: {aia_vmin:.1f} - {aia_vmax:.1f}")

    # Create frames
    frame_files = []
    for i, (_, row) in enumerate(event_data.iterrows()):
        timestamp = row['timestamp']
        flux_contrib = self.load_flux_contributions(timestamp)
        aia_image = self.load_aia_image(timestamp)

        if flux_contrib is None:
            continue

        fig, axes = plt.subplots(1, 3 if aia_image is not None else 1, figsize=(18, 6))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # AIA background
        if aia_image is not None:
            aia_display = np.clip(aia_image, aia_vmin, aia_vmax)
            aia_display = np.log10(aia_display + 1)

            # Panel 1: AIA only
            axes[0].imshow(aia_display, cmap='viridis', vmin=np.log10(aia_vmin + 1),
                           vmax=np.log10(aia_vmax + 1))
            axes[0].set_title(f'AIA 94 Å\n{row["datetime"].strftime("%Y-%m-%d %H:%M:%S")}')
            axes[0].set_xlabel('Pixel X')
            axes[0].set_ylabel('Pixel Y')

            # Panel 2: Flux only
            axes[1].imshow(flux_contrib, cmap='hot', vmin=flux_vmin, vmax=flux_vmax)
            axes[1].set_title(f'Flux Contributions\nPred: {row["predictions"]:.2e}')
            axes[1].set_xlabel('Patch X')
            axes[1].set_ylabel('Patch Y')

            # Panel 3: Overlay
            axes[2].imshow(aia_display, cmap='gray', alpha=0.8,
                           vmin=np.log10(aia_vmin + 1), vmax=np.log10(aia_vmax + 1))

            # Resize and overlay flux
            flux_resized = self.resize_flux_to_image_size(flux_contrib)
            flux_threshold = np.percentile(flux_resized.flatten(), 60)
            flux_mask = flux_resized > flux_threshold
            flux_masked = np.ma.masked_where(~flux_mask, flux_resized)

            axes[2].imshow(flux_masked, cmap='hot', alpha=0.7,
                           vmin=flux_vmin, vmax=flux_vmax, interpolation='bilinear')
            axes[2].set_title('AIA + Flux Overlay')
            axes[2].set_xlabel('Pixel X')
            axes[2].set_ylabel('Pixel Y')

            # Add patch grid
            for j in range(0, 512, 16):
                axes[2].axvline(j, color='white', alpha=0.3, linewidth=0.3)
                axes[2].axhline(j, color='white', alpha=0.3, linewidth=0.3)


"""
Flux Contribution Analysis and Flare Detection Script

This script analyzes flux contributions from different patches to identify 
potential flaring events and visualize their spatial and temporal patterns.
"""



warnings.filterwarnings('ignore')


class FluxContributionAnalyzer:
    def __init__(self, flux_path, predictions_csv, aia_path=None, attention_path=None,
                 grid_size=(32, 32), patch_size=16, input_size=512):
        """
        Initialize the flux contribution analyzer

        Args:
            flux_path: Path to directory containing flux contribution files
            predictions_csv: Path to CSV file with predictions and timestamps
            aia_path: Path to directory containing AIA numpy files
            attention_path: Optional path to attention weights directory
            grid_size: Size of the flux contribution grid
            patch_size: Size of each patch in pixels
            input_size: Input image size
        """
        self.flux_path = Path(flux_path)
        self.aia_path = Path(aia_path) if aia_path else None
        self.attention_path = Path(attention_path) if attention_path else None
        self.predictions_df = pd.read_csv(predictions_csv)
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.input_size = input_size

        # Convert timestamps to datetime
        self.predictions_df['datetime'] = pd.to_datetime(self.predictions_df['timestamp'])
        self.predictions_df = self.predictions_df.sort_values('datetime')

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
                    if len(aia_data.shape) == 3:
                        aia_94 = aia_data[0]  # First channel is 94 Angstrom
                    else:
                        aia_94 = aia_data

                    # Ensure correct size
                    if aia_94.shape != (512, 512):
                        print(f"Warning: AIA image shape is {aia_94.shape}, expected (512, 512)")

                    return aia_94
                except Exception as e:
                    print(f"Error loading {aia_file}: {e}")
                    continue

        return None

    def resize_flux_to_image_size(self, flux_contrib):
        """Resize flux contribution map from patch grid to full image resolution"""
        # Use bicubic interpolation to smoothly upscale flux contributions
        return cv2.resize(flux_contrib, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_CUBIC)

    def detect_flare_events(self, threshold_percentile=95, min_patches=1, max_patches=15):
        """
        Detect potential flare events based on flux contribution patterns

        Args:
            threshold_percentile: Percentile threshold for high contribution patches
            min_patches: Minimum number of connected high-contribution patches
            max_patches: Maximum number of connected high-contribution patches
        """
        flare_events = []

        print("Analyzing flux contributions for flare detection...")

        for idx, row in self.predictions_df.iterrows():
            timestamp = row['timestamp']
            flux_contrib = self.load_flux_contributions(timestamp)

            if row['predictions']<5e-6:
                continue
            if flux_contrib is None:
                continue

            # Calculate threshold for this timestamp
            threshold = np.percentile(flux_contrib.flatten(), threshold_percentile)
            #threshold = 1e-5
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
            # Choose structure_4 for stricter connectivity or structure_8 for looser
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

    def load_attention_weights(self, timestamp):
        """Load attention weights for a specific timestamp if available"""
        if self.attention_path is None:
            return None
        attention_file = self.attention_path / f"{timestamp}"
        if attention_file.exists():
            return np.loadtxt(attention_file, delimiter=',')
        return None

    def plot_flux_overlay_on_aia(self, timestamp, save_path=None, flux_alpha=0.6,
                                 flux_threshold_percentile=50):
        """Plot flux contributions overlaid on AIA 94 Angstrom image"""
        flux_contrib = self.load_flux_contributions(timestamp)
        aia_image = self.load_aia_image(timestamp)

        if flux_contrib is None:
            print(f"No flux contributions found for {timestamp}")
            return

        # Get prediction data for this timestamp
        pred_data = self.predictions_df[self.predictions_df['timestamp'] == timestamp].iloc[0]

        # Create figure
        fig, axes = plt.subplots(1, 3 if aia_image is not None else 2, figsize=(18, 6))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Plot 1: Flux contributions only
        im1 = axes[0].imshow(flux_contrib, cmap='hot', interpolation='nearest')
        axes[0].set_title(f'Flux Contributions\n{timestamp}\nPrediction: {pred_data["predictions"]:.2e}')
        axes[0].set_xlabel('Patch X')
        axes[0].set_ylabel('Patch Y')
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
        cbar1.set_label('Flux Contribution')

        if aia_image is not None:
            # Plot 2: AIA image only
            # Use log scale for AIA display with clipping to avoid issues with zeros/negatives
            aia_display = np.clip(aia_image, np.percentile(aia_image, 1),
                                  np.percentile(aia_image, 99.5))
            aia_display = np.log10(aia_display + 1)  # Add 1 to avoid log(0)

            im2 = axes[1].imshow(aia_display, cmap='viridis', interpolation='nearest')
            axes[1].set_title(f'AIA 94 Å\n{timestamp}')
            axes[1].set_xlabel('Pixel X')
            axes[1].set_ylabel('Pixel Y')
            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
            cbar2.set_label('Log(AIA 94 Å Intensity)')

            # Plot 3: Overlay
            # Resize flux contributions to match image size
            flux_resized = self.resize_flux_to_image_size(flux_contrib)

            # Create mask for significant flux contributions
            flux_threshold = np.percentile(flux_resized.flatten(), flux_threshold_percentile)
            flux_mask = flux_resized > flux_threshold

            # Display AIA as background
            axes[2].imshow(aia_display, cmap='gray', alpha=0.8, interpolation='nearest')

            # Overlay flux contributions with transparency
            flux_masked = np.ma.masked_where(~flux_mask, flux_resized)
            im3 = axes[2].imshow(flux_masked, cmap='hot', alpha=flux_alpha,
                                 interpolation='bilinear')

            axes[2].set_title(f'AIA 94 Å + Flux Overlay\n{timestamp}\n'
                              f'Flux threshold: {flux_threshold_percentile}th percentile')
            axes[2].set_xlabel('Pixel X')
            axes[2].set_ylabel('Pixel Y')

            # Add colorbar for overlay
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
            cbar3.set_label('Flux Contribution')

            # Add patch grid overlay for reference
            for i in range(0, self.input_size, self.patch_size):
                axes[2].axvline(i, color='white', alpha=0.3, linewidth=0.5)
                axes[2].axhline(i, color='white', alpha=0.3, linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved AIA overlay plot to {save_path}")
        else:
            plt.show()

    def plot_flux_contribution_heatmap(self, timestamp, save_path=None, show_attention=True, threshold_percentile=99,
                                       min_patches=1, max_patches=15):
        """Plot flux contribution heatmap for a specific timestamp with detected regions highlighted"""
        flux_contrib = self.load_flux_contributions(timestamp)
        aia = self.load_aia_image(timestamp) if show_attention else None

        if flux_contrib is None:
            print(f"No flux contributions found for {timestamp}")
            return

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
        structure_4 = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]])

        # 8-connectivity: patches connected by edges or corners
        structure_8 = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

        # Use connected components to find flare regions
        # Choose structure_4 for stricter connectivity or structure_8 for looser
        labeled_regions, num_regions = nd.label(high_contrib_mask, structure=structure_8)

        # Create a copy for display and collect region info
        flux_contrib_display = flux_contrib.copy()
        flux_contrib_display[flux_contrib < threshold] = np.nan

        detected_regions = []
        region_colors = plt.cm.Set3(np.linspace(0, 1, max(num_regions, 1)))

        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            region_size = np.sum(region_mask)

            if min_patches <= region_size <= max_patches:
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
                    'id': region_id,
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
        im1 = axes[0].imshow(flux_contrib_display, cmap='hot', interpolation='nearest')

        # Highlight detected regions with colored outlines
        for i, region in enumerate(detected_regions):
            # Create contour around the region
            # Dilate the mask slightly to create an outline
            dilated = nd.binary_dilation(region['mask'])
            outline = dilated & ~region['mask']

            # Overlay the outline
            axes[0].contour(region['mask'].astype(int), levels=[0.5],
                            colors=[region_colors[i % len(region_colors)]], linewidths=2)

            # Add region label with sum flux
            axes[0].text(region['label_x'], region['label_y'],
                         f"R{region['id']}\n{region['sum_flux']:.1e}",
                         ha='center', va='center', fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        title_text = f'Flux Contributions with Detected Regions\n{timestamp}\nPrediction: {pred_data["predictions"]:.2e}'
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

        # Plot attention weights if available
        if aia is not None:
            im2 = axes[1].imshow(aia, cmap='Blues', interpolation='nearest')
            axes[1].set_title(f'Attention Weights\n{timestamp}')
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

    def create_comprehensive_flare_visualization(self, timestamp, save_path=None,
                                                 flux_alpha=0.7, attention_alpha=0.5):
        """Create a comprehensive 4-panel visualization showing AIA, flux, attention, and overlay"""
        flux_contrib = self.load_flux_contributions(timestamp)
        aia_image = self.load_aia_image(timestamp)
        attention_weights = self.load_attention_weights(timestamp)

        if flux_contrib is None:
            print(f"No flux contributions found for {timestamp}")
            return

        # Get prediction data
        pred_data = self.predictions_df[self.predictions_df['timestamp'] == timestamp].iloc[0]

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Panel 1: AIA image
        if aia_image is not None:
            aia_display = np.clip(aia_image, np.percentile(aia_image, 1),
                                  np.percentile(aia_image, 99.5))
            aia_display = np.log10(aia_display + 1)

            im1 = axes[0, 0].imshow(aia_display, cmap='viridis', interpolation='nearest')
            axes[0, 0].set_title(f'AIA 94 Å\n{timestamp}')
            axes[0, 0].set_xlabel('Pixel X')
            axes[0, 0].set_ylabel('Pixel Y')
            plt.colorbar(im1, ax=axes[0, 0], shrink=0.8).set_label('Log(Intensity)')
        else:
            axes[0, 0].text(0.5, 0.5, 'No AIA Data\nAvailable', ha='center', va='center',
                            transform=axes[0, 0].transAxes, fontsize=16)
            axes[0, 0].set_title(f'AIA 94 Å\n{timestamp}')

        # Panel 2: Flux contributions
        im2 = axes[0, 1].imshow(flux_contrib, cmap='hot', interpolation='nearest')
        axes[0, 1].set_title(f'Flux Contributions\nPrediction: {pred_data["predictions"]:.2e}')
        axes[0, 1].set_xlabel('Patch X')
        axes[0, 1].set_ylabel('Patch Y')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8).set_label('Flux Contribution')

        # Panel 3: Attention weights
        if attention_weights is not None:
            im3 = axes[1, 0].imshow(attention_weights, cmap='Blues', interpolation='nearest')
            axes[1, 0].set_title('Attention Weights')
            axes[1, 0].set_xlabel('Patch X')
            axes[1, 0].set_ylabel('Patch Y')
            plt.colorbar(im3, ax=axes[1, 0], shrink=0.8).set_label('Attention Weight')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Attention\nWeights Available', ha='center', va='center',
                            transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Attention Weights')

        # Panel 4: Combined overlay
        if aia_image is not None:
            # Background AIA
            axes[1, 1].imshow(aia_display, cmap='gray', alpha=0.7, interpolation='nearest')

            # Overlay flux contributions
            flux_resized = self.resize_flux_to_image_size(flux_contrib)
            flux_threshold = np.percentile(flux_resized.flatten(), 70)
            flux_mask = flux_resized > flux_threshold
            flux_masked = np.ma.masked_where(~flux_mask, flux_resized)

            im4 = axes[1, 1].imshow(flux_masked, cmap='hot', alpha=flux_alpha,
                                    interpolation='bilinear')

            # Overlay attention if available
            if attention_weights is not None:
                attention_resized = self.resize_flux_to_image_size(attention_weights)
                attention_threshold = np.percentile(attention_resized.flatten(), 80)
                attention_mask = attention_resized > attention_threshold
                attention_masked = np.ma.masked_where(~attention_mask, attention_resized)

                axes[1, 1].contour(attention_masked, levels=3, colors='cyan', alpha=attention_alpha,
                                   linewidths=1.5)

            axes[1, 1].set_title('Multi-modal Overlay\n(AIA + Flux + Attention)')
            axes[1, 1].set_xlabel('Pixel X')
            axes[1, 1].set_ylabel('Pixel Y')

            # Add patch grid
            for i in range(0, self.input_size, self.patch_size):
                axes[1, 1].axvline(i, color='white', alpha=0.2, linewidth=0.5)
                axes[1, 1].axhline(i, color='white', alpha=0.2, linewidth=0.5)
        else:
            axes[1, 1].text(0.5, 0.5, 'Overlay requires\nAIA data', ha='center', va='center',
                            transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Multi-modal Overlay')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comprehensive visualization to {save_path}")
        else:
            plt.show()
        """Analyze which patches contribute most across all timestamps"""


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
        print("-" * 80)
        print(f"{'Rank':<4} {'Timestamp':<20} {'Prediction':<12} {'Region Size':<12} {'Max Flux':<12} {'Location':<15}")
        print("-" * 80)

        for i, (_, event) in enumerate(flare_events.head(10).iterrows()):
            location = f"({event['centroid_img_y']:.0f},{event['centroid_img_x']:.0f})"
            print(f"{i + 1:<4} {event['datetime'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{event['prediction']:<12.2e} {event['region_size']:<12} "
                  f"{event['max_flux']:<12.2e} {location:<15}")

        # Statistics
        print(f"\nEvent Statistics:")
        print(f"  Mean region size: {flare_events['region_size'].mean():.1f} patches")
        print(f"  Mean prediction: {flare_events['prediction'].mean():.2e}")
        print(f"  Max prediction: {flare_events['prediction'].max():.2e}")
        print(f"  Min prediction: {flare_events['prediction'].min():.2e}")

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
                           interpolation='nearest')

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
    parser.add_argument('--flux_path', required=True, help='Path to flux contributions directory')
    parser.add_argument('--predictions_csv', required=True, help='Path to predictions CSV file')
    parser.add_argument('--attention_path', help='Path to attention weights directory (optional)')
    parser.add_argument('--output_dir', default='flux_analysis_output', help='Output directory for results')
    parser.add_argument('--grid_size', nargs=2, type=int, default=[32, 32], help='Grid size (height width)')
    parser.add_argument('--threshold_percentile', type=float, default=97,
                        help='Percentile threshold for flare detection')
    parser.add_argument('--min_patches', type=int, default=5, help='Minimum patches for flare detection')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize analyzer
    analyzer = FluxContributionAnalyzer(
        flux_path=args.flux_path,
        predictions_csv=args.predictions_csv,
        attention_path=args.attention_path,
        grid_size=tuple(args.grid_size),
        aia_path = "/mnt/data/ML-Ready-mixed/ML-Ready-mixed/AIA/test"
    )

    # Detect flare events
    print("Detecting flare events...")
    flare_events = analyzer.detect_flare_events(
        threshold_percentile=args.threshold_percentile,
        min_patches=args.min_patches
    )

    # Create flare event summary
    flare_summary_path = output_dir / 'flare_events_summary.csv'
    analyzer.create_flare_event_summary(flare_summary_path)

    # Analyze top contributing patches

    # Plot some examples
    if len(flare_events) > 0:
        # Plot top 5 flare events
        print("\nCreating visualizations for top flare events...")
        top_events = flare_events.head(630).sort_values('prediction', ascending=False)

        for i, (_, event) in enumerate(top_events.iterrows()):
            output_path = output_dir / f'flare_event_{i + 1}_{event["timestamp"]}.png'
            analyzer.plot_flux_contribution_heatmap(
                event['timestamp'],
                save_path=output_path,
                show_attention= True,
                threshold_percentile=args.threshold_percentile

            )
            # analyzer.plot_flux_overlay_on_aia(
            #     event['timestamp'],
            #     save_path=output_path,
            #     flux_threshold_percentile=args.threshold_percentile
            # )


    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Found {len(flare_events)} potential flare events")

    # analyzer.create_flare_movie(
    #     start_time='2023-08-05 00:00',
    #     end_time='2023-08-06 00:00',
    #     output_path=output_dir / 'flare_event_movie.mp4',
    #     fps=30
    # )


if __name__ == "__main__":
    main()