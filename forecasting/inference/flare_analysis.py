#!/usr/bin/env python3
"""
Unified Flare Analysis Script

This script combines region detection, tracking, flare catalog building,
and HEK matching into a single streamlined workflow.

Usage:
    python flare_analysis.py --config flare_analysis_config.yaml

The config file contains all essential parameters for:
- Data paths (predictions, AIA, flux contributions, HEK catalog)
- Time range for analysis
- Detection parameters (thresholds, clustering)
- Output options (plots)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import rcParams
from scipy import ndimage as nd
from scipy.ndimage import maximum_filter, gaussian_filter
from heapq import heappush, heappop
from scipy.signal import find_peaks
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Optional SunPy for HEK fetching
try:
    from sunpy.net import Fido
    from sunpy.net import attrs as a
except Exception:
    Fido = None
    a = None


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class FlareAnalysisConfig:
    """Configuration for flare analysis with sensible defaults."""
    
    # Paths
    data_dir: str = "/data/FOXES_Data"
    flux_path: Optional[str] = None
    aia_path: Optional[str] = None
    predictions_csv: Optional[str] = None
    hek_catalog: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Time range
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    auto_fetch_hek: bool = True
    
    # Detection parameters (user-tunable)
    min_goes_class: str = "B1.0"
    min_flux_threshold: float = 1e-7
    threshold_std_multiplier: float = 3.0
    peak_neighborhood_size: int = 15
    hek_match_window_minutes: float = 5.0
    simultaneous_flare_threshold: float = 5e-6
    sequence_window_hours: float = 1.0
    enable_simultaneous_flares: bool = True
    
    # Grid/patch parameters
    grid_size: Tuple[int, int] = (64, 64)
    patch_size: int = 8
    input_size: int = 512
    
    # Output options
    create_plots: bool = True
    plot_window_hours: float = 4.0
    create_movie: bool = False
    movie_fps: float = 2.0
    movie_frame_interval_minutes: float = 1.0
    movie_num_workers: int = 4  # Number of parallel workers for frame generation
    movie_dpi: float = 75.0  # DPI for movie frames (lower = faster, 75 is good for video)
    movie_frame_format: str = 'jpg'  # 'jpg' (faster) or 'png' (higher quality, slower)
    movie_jpeg_quality: int = 90  # JPEG quality 1-100 (90 is good balance)
    
    # HEK coverage filtering (for HEK-only events)
    hek_coverage_window_minutes: float = 30.0
    hek_max_gap_minutes: float = 10.0
    hek_min_points_around_peak: int = 6
    
    # Peak detection parameters (within tracks)
    peak_height_multiplier: float = 1.2
    peak_baseline_window: int = 30  # Rolling window size for adaptive baseline (number of points)
    min_peak_separation_minutes: float = 15.0
    peak_validation_window: int = 5  # Number of points to check on each side of peak
    peak_validation_min_lower: int = 1  # Minimum number of lower points required on each side
    
    # Spatial smoothing (reduces noise, stabilizes region boundaries)
    spatial_smoothing_sigma: float = 1.0  # Gaussian sigma (0 = disabled)
    
    # Radial expansion from peaks
    radial_expansion_threshold_percentile: float = 30.0  # Percentile for growth cutoff (0-100)
    
    # Minimum region size
    region_min_pixels: int = 4  # Minimum patches per region
    
    # Tracking similarity constraints (prevents track jumping)
    max_flux_ratio: float = 3.0  # Max flux change ratio between frames
    max_size_ratio: float = 2.0  # Max size change ratio between frames
    
    # Temporal smoothing (exponential moving average)
    flux_smoothing_alpha: float = 0.3  # EMA alpha (0-1, higher = more responsive)
    
    # Internal defaults (rarely need changing)
    cadence_seconds: float = 60.0
    max_gap_minutes: float = 60.0
    min_duration_minutes: float = 5.0
    min_samples: int = 2
    max_tracking_distance: int = 20
    max_time_gap_minutes: float = 180.0
    flare_distance_relax_factor: float = 4.5
    
    @classmethod
    def from_yaml(cls, path: str) -> "FlareAnalysisConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Flatten nested structure
        flat = {}
        
        # Paths section
        if 'paths' in data:
            for k, v in data['paths'].items():
                if v is not None:
                    flat[k] = v
        
        # Time range section
        if 'time_range' in data:
            for k, v in data['time_range'].items():
                if k == 'start':
                    flat['start_time'] = v
                elif k == 'end':
                    flat['end_time'] = v
                else:
                    flat[k] = v
        
        # Detection section
        if 'detection' in data:
            for k, v in data['detection'].items():
                flat[k] = v
        
        # Tracking section
        if 'tracking' in data:
            for k, v in data['tracking'].items():
                flat[k] = v
        
        # Quality section
        if 'quality' in data:
            for k, v in data['quality'].items():
                flat[k] = v
        
        # Grid section
        if 'grid' in data:
            for k, v in data['grid'].items():
                if k == 'grid_size' and isinstance(v, list):
                    flat[k] = tuple(v)
                else:
                    flat[k] = v
        
        # HEK coverage section
        if 'hek_coverage' in data:
            hek_cov = data['hek_coverage']
            if 'coverage_window_minutes' in hek_cov:
                flat['hek_coverage_window_minutes'] = hek_cov['coverage_window_minutes']
            if 'max_gap_minutes' in hek_cov:
                flat['hek_max_gap_minutes'] = hek_cov['max_gap_minutes']
            if 'min_points_around_peak' in hek_cov:
                flat['hek_min_points_around_peak'] = hek_cov['min_points_around_peak']
        
        # Peak detection section
        if 'peak_detection' in data:
            peak_det = data['peak_detection']
            if 'height_multiplier' in peak_det:
                flat['peak_height_multiplier'] = peak_det['height_multiplier']
            if 'baseline_window' in peak_det:
                flat['peak_baseline_window'] = peak_det['baseline_window']
            if 'min_peak_separation_minutes' in peak_det:
                flat['min_peak_separation_minutes'] = peak_det['min_peak_separation_minutes']
            if 'validation_window' in peak_det:
                flat['peak_validation_window'] = peak_det['validation_window']
            if 'validation_min_lower' in peak_det:
                flat['peak_validation_min_lower'] = peak_det['validation_min_lower']

        
        # Simultaneous flare detection (optional)
        if 'simultaneous_detection' in data:
            sim_det = data['simultaneous_detection']
            if 'enabled' in sim_det:
                flat['enable_simultaneous_flares'] = sim_det['enabled']
            if 'threshold' in sim_det:
                flat['simultaneous_flare_threshold'] = sim_det['threshold']
            if 'sequence_window_hours' in sim_det:
                flat['sequence_window_hours'] = sim_det['sequence_window_hours']
        
        # Output section
        if 'output' in data:
            for k, v in data['output'].items():
                flat[k] = v
        
        # Movie section (optional)
        if 'movie' in data:
            movie_data = data['movie']
            if 'create_movie' in movie_data:
                flat['create_movie'] = movie_data['create_movie']
            if 'fps' in movie_data:
                flat['movie_fps'] = movie_data['fps']
            if 'frame_interval_minutes' in movie_data:
                flat['movie_frame_interval_minutes'] = movie_data['frame_interval_minutes']
            if 'num_workers' in movie_data:
                flat['movie_num_workers'] = movie_data['num_workers']
            if 'dpi' in movie_data:
                flat['movie_dpi'] = movie_data['dpi']
            if 'frame_format' in movie_data:
                flat['movie_frame_format'] = movie_data['frame_format']
            if 'jpeg_quality' in movie_data:
                flat['movie_jpeg_quality'] = movie_data['jpeg_quality']
        
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid_fields and v is not None}
        
        return cls(**filtered)


# =============================================================================
# GOES Class Utilities
# =============================================================================

def goes_class_to_flux(goes_class: str) -> float:
    """Convert GOES class string (e.g., C5.0) to physical flux."""
    if not isinstance(goes_class, str):
        return np.nan
    goes_class = goes_class.strip().upper()
    if not goes_class:
        return np.nan
    prefix = goes_class[0]
    scale = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}.get(prefix)
    if scale is None:
        return np.nan
    try:
        magnitude = float(goes_class[1:])
    except ValueError:
        magnitude = np.nan
    return magnitude * scale


def flux_to_goes_class(flux: float) -> str:
    """Convert physical flux to GOES class string."""
    if not isinstance(flux, (int, float)) or np.isnan(flux) or flux <= 0:
        return "N/A"
    
    if flux >= 1e-4:
        prefix, scale = "X", 1e-4
    elif flux >= 1e-5:
        prefix, scale = "M", 1e-5
    elif flux >= 1e-6:
        prefix, scale = "C", 1e-6
    elif flux >= 1e-7:
        prefix, scale = "B", 1e-7
    else:
        prefix, scale = "A", 1e-8
    
    magnitude = flux / scale
    if magnitude >= 10:
        magnitude = min(magnitude, 9.9)
    
    if magnitude == int(magnitude):
        return f"{prefix}{int(magnitude)}.0"
    return f"{prefix}{magnitude:.1f}"


# =============================================================================
# Flux Contribution Analyzer (Core Detection Engine)
# =============================================================================

class FluxContributionAnalyzer:
    """
    Analyzes flux contributions from patches to identify and track flare events.
    
    This class handles:
    - Loading flux contribution data and AIA images
    - Detecting regions using peak-based clustering
    - Tracking regions across time
    - Detecting flare events from tracked regions
    """
    
    def __init__(self, config: FlareAnalysisConfig, output_dir: Optional[Path] = None):
        """Initialize the analyzer with configuration."""
        self.config = config
        
        # Set paths
        self.flux_path = Path(config.flux_path) if config.flux_path else None
        self.aia_path = Path(config.aia_path) if config.aia_path else None
        
        # Output directory
        self.output_dir = output_dir
        
        # In-memory cache for region labels (timestamp -> labels array)
        # Used for movie visualization without disk I/O
        self.region_labels_cache: Dict[str, np.ndarray] = {}
        
        # Grid parameters
        self.grid_size = config.grid_size
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        
        # Load predictions
        if config.predictions_csv:
            self.predictions_df = pd.read_csv(config.predictions_csv)
            self.predictions_df['datetime'] = pd.to_datetime(self.predictions_df['timestamp'])
            self.predictions_df = self.predictions_df.sort_values('datetime')
            
            # Filter by time period if specified
            if config.start_time and config.end_time:
                start = pd.to_datetime(config.start_time)
                end = pd.to_datetime(config.end_time)
                mask = (self.predictions_df['datetime'] >= start) & (self.predictions_df['datetime'] <= end)
                self.predictions_df = self.predictions_df[mask].reset_index(drop=True)
            
            print(f"Loaded {len(self.predictions_df)} predictions")
            print(f"Time range: {self.predictions_df['datetime'].min()} to {self.predictions_df['datetime'].max()}")
        else:
            self.predictions_df = pd.DataFrame()
        
        self.flare_events_df = None
    
    def load_flux_contributions(self, timestamp: str) -> Optional[np.ndarray]:
        """Load flux contributions for a specific timestamp."""
        if self.flux_path is None:
            return None
        flux_file = self.flux_path / f"{timestamp}"
        if flux_file.exists():
            return np.loadtxt(flux_file, delimiter=',')
        return None
    
    def load_aia_image(self, timestamp: str) -> Optional[np.ndarray]:
        """Load AIA image as RGB composite from 94, 131, 171 Angstrom channels."""
        if self.aia_path is None:
            return None
        
        # Search in subdirectories
        possible_dirs = [self.aia_path]
        for subdir in ['test', 'train', 'val']:
            subdir_path = self.aia_path / subdir
            if subdir_path.exists():
                possible_dirs.append(subdir_path)
        
        for aia_dir in possible_dirs:
            aia_file = aia_dir / f"{timestamp}.npy"
            if aia_file.exists():
                try:
                    aia_data = np.load(aia_file)
                    # Get 94, 131, 171 Angstrom channels
                    aia_94 = aia_data[1]
                    aia_131 = aia_data[2]
                    aia_171 = aia_data[5]
                    
                    # Normalize and stack
                    def normalize(arr):
                        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-10)
                    
                    return np.stack([normalize(aia_94), normalize(aia_131), normalize(aia_171)], axis=-1)
                except Exception as e:
                    print(f"Error loading {aia_file}: {e}")
                    continue
        return None
    
    def _find_flux_peaks(self, flux_contrib: np.ndarray, 
                         neighborhood_size: int = 10) -> Tuple[List, List]:
        """Identify local maxima in the flux contribution map.
        
        Excludes zeros and NaNs from being counted as maxima.
        """
        # Exclude 0s and nans from being counted as maxima
        flux_is_valid = (np.isfinite(flux_contrib)) & (flux_contrib > 0)
        
        # Use -inf for invalid pixels so maximum_filter never selects them
        valid_flux = np.where(flux_is_valid, flux_contrib, -np.inf)
        local_max = (maximum_filter(valid_flux, size=neighborhood_size) == valid_flux) & flux_is_valid
        
        peak_coords = np.where(local_max)
        peak_coords = list(zip(peak_coords[0], peak_coords[1]))
        peak_fluxes = [flux_contrib[y, x] for y, x in peak_coords]
        return peak_coords, peak_fluxes
    
    def _detect_regions_with_peak_clustering(
        self,
        flux_contrib: np.ndarray,
        timestamp: str,
        pred_data: pd.Series,
    ) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Detect regions using radial growth from peaks.
        
        Regions grow radially outward from each peak simultaneously using a priority queue.
        Each pixel is assigned to the nearest peak (by radial distance), ensuring 
        non-overlapping regions. Growth stops when flux falls below a threshold percentile.
        
        Returns:
            Tuple of (regions list, labels array). Labels array contains region IDs (1-indexed),
            0 for background. Can be saved and reused for visualization.
        """
        threshold_std = self.config.threshold_std_multiplier
        min_flux = self.config.min_flux_threshold
        peak_neighborhood = self.config.peak_neighborhood_size
        
        # Get valid flux values for threshold calculation
        valid_flux = flux_contrib[(np.isfinite(flux_contrib)) & (flux_contrib > 0)]
        if len(valid_flux) == 0:
            return [], None
        
        # Calculate high threshold for initial flux mask
        log_flux = np.log(valid_flux)
        high_threshold = np.exp(np.median(log_flux) + threshold_std * np.std(log_flux))
        
        # Apply initial flux mask
        flux_mask = flux_contrib > high_threshold
        flux_masked = np.where(flux_mask, flux_contrib, 0)
        
        # 1. SPATIAL SMOOTHING: Apply Gaussian filter to reduce noise
        sigma = getattr(self.config, 'spatial_smoothing_sigma', 1.0)
        if sigma > 0:
            flux_masked = gaussian_filter(flux_masked, sigma=sigma)
        
        # Find peaks in the masked/smoothed flux
        peak_coords, peak_fluxes = self._find_flux_peaks(
            flux_masked, neighborhood_size=peak_neighborhood
        )
        
        if len(peak_coords) == 0:
            return [], None
        
        # ------------------------------------------------------------------#
        # RADIAL GROWTH: Grow regions from peaks using priority queue
        # ------------------------------------------------------------------#
        
        # Initialize labels array (0 = unassigned)
        labels = np.zeros_like(flux_masked, dtype=np.int32)
        
        # Calculate growth threshold from percentile of valid flux values
        percentile = getattr(self.config, 'radial_expansion_threshold_percentile', 30.0)
        valid_vals = flux_masked[(flux_masked > 0) & np.isfinite(flux_masked)]
        growth_threshold = np.percentile(valid_vals, percentile) if valid_vals.size > 0 else 0
        
        # Initialize priority queue with all peaks (simultaneous start)
        # Format: (distance_from_peak, counter, y, x, label_id, peak_y, peak_x)
        pq = []
        counter = 0
        
        for peak_idx, ((peak_y, peak_x), peak_flux_value) in enumerate(zip(peak_coords, peak_fluxes)):
            label_id = peak_idx + 1
            labels[peak_y, peak_x] = label_id
            heappush(pq, (0.0, counter, peak_y, peak_x, label_id, peak_y, peak_x))
            counter += 1
        
        # Define neighbor offsets (8-connected for smoother radial growth)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 4-connected
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonals
        
        # Radial growth from all peaks simultaneously
        while pq:
            dist, _, y, x, label_id, peak_y, peak_x = heappop(pq)
            
            # Check neighbors
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                
                # Check bounds
                if (ny < 0 or ny >= flux_masked.shape[0] or 
                    nx < 0 or nx >= flux_masked.shape[1]):
                    continue
                
                # Skip if already assigned to a region
                if labels[ny, nx] > 0:
                    continue
                
                # Check if flux is above growth threshold
                if flux_masked[ny, nx] > growth_threshold:
                    # Calculate distance from peak to this neighbor
                    new_dist = np.sqrt((ny - peak_y)**2 + (nx - peak_x)**2)
                    
                    # Assign to this peak's region and continue growing
                    labels[ny, nx] = label_id
                    heappush(pq, (new_dist, counter, ny, nx, label_id, peak_y, peak_x))
                    counter += 1
        
        # ------------------------------------------------------------------#
        # Create regions from radial growth labels (guaranteed non-overlapping)
        # ------------------------------------------------------------------#
        regions = []
        min_pixels = getattr(self.config, 'region_min_pixels', 4)
        
        for label_id in range(1, len(peak_coords) + 1):
            region_mask = labels == label_id
            
            coords = np.where(region_mask)
            if len(coords[0]) == 0:
                continue
            
            region_size = len(coords[0])
            
            # Skip tiny regions
            if region_size < min_pixels:
                continue
            
            region_flux_values = flux_masked[region_mask]
            sum_flux = np.sum(region_flux_values)
            max_flux = np.max(region_flux_values)
            
            if min_flux is not None and sum_flux < min_flux:
                continue
            
            centroid_y, centroid_x = np.mean(coords[0]), np.mean(coords[1])
            img_y = centroid_y * self.patch_size + self.patch_size // 2
            img_x = centroid_x * self.patch_size + self.patch_size // 2
            
            brightest_idx = int(np.argmax(region_flux_values))
            bright_patch_y = int(coords[0][brightest_idx])
            bright_patch_x = int(coords[1][brightest_idx])
            
            # Get peak coordinates for this region (where radial growth started from)
            peak_y, peak_x = peak_coords[label_id - 1]  # 0-indexed, in patch coords
            peak_flux = peak_fluxes[label_id - 1]
            
            # Convert peak patch coords to image coords (center of patch)
            peak_img_y = peak_y * self.patch_size + self.patch_size // 2
            peak_img_x = peak_x * self.patch_size + self.patch_size // 2
            
            regions.append({
                "id": len(regions) + 1,
                "region_label": label_id,  # The actual label ID from radial growth (for direct lookup)
                "size": region_size,
                "sum_flux": sum_flux,
                "max_flux": max_flux,
                "centroid_patch_y": centroid_y,
                "centroid_patch_x": centroid_x,
                "centroid_img_y": img_y,
                "centroid_img_x": img_x,
                "mask": region_mask,
                "prediction": sum_flux,
                "groundtruth": pred_data.get("groundtruth", None),
                "bright_patch_y": bright_patch_y,
                "bright_patch_x": bright_patch_x,
                "peak_y": peak_y,
                "peak_x": peak_x,
                "peak_img_y": peak_img_y,
                "peak_img_x": peak_img_x,
                "peak_flux": peak_flux,
            })
        
        return regions, labels  # Radial growth guarantees non-overlapping regions
    
    def _detect_regions_worker(self, timestamp: str) -> Tuple[str, Optional[List], Optional[np.ndarray]]:
        """Worker function for parallel region detection.
        
        Returns:
            Tuple of (timestamp, regions list, labels array)
        """
        try:
            flux_contrib = self.load_flux_contributions(timestamp)
            if flux_contrib is None:
                return (timestamp, None, None)
            
            pred_data = self.predictions_df[self.predictions_df['timestamp'] == timestamp]
            if pred_data.empty:
                return (timestamp, None, None)
            pred_data = pred_data.iloc[0]
            
            regions, labels = self._detect_regions_with_peak_clustering(flux_contrib, timestamp, pred_data)
            
            # Return labels for caching in main process (avoid disk I/O)
            # Convert to int16 to save memory
            labels_compact = labels.astype(np.int16) if labels is not None else None
            
            return (timestamp, regions, labels_compact)
        except Exception as e:
            print(f"Error detecting regions for {timestamp}: {e}")
            return (timestamp, None, None)
    
    def _apply_temporal_smoothing(self, region_tracks: Dict, alpha: float) -> Dict:
        """Apply exponential moving average (EMA) smoothing to region flux values.
        
        EMA is more responsive than rolling mean and better at following genuine
        flux evolution while dampening noise.
        
        Args:
            region_tracks: Dictionary of track_id -> [(timestamp, region_dict), ...]
            alpha: EMA smoothing factor (0-1). Higher = more responsive to changes.
                   0 = no smoothing, 1 = no memory (use raw values).
        
        Returns:
            region_tracks with smoothed sum_flux and size values.
        """
        if alpha <= 0 or alpha >= 1:
            return region_tracks
        
        print(f"Applying exponential smoothing with alpha={alpha}...")
        
        for track_id, track_history in region_tracks.items():
            if len(track_history) < 2:
                continue
            
            # Initialize EMA with first value
            ema_flux = track_history[0][1]['sum_flux']
            ema_size = float(track_history[0][1].get('size', 0))
            
            for i, (t, r) in enumerate(track_history):
                # Save original values
                r['sum_flux_original'] = r['sum_flux']
                r['size_original'] = r.get('size', 0)
                
                if i == 0:
                    # First point: no smoothing yet
                    r['prediction'] = r['sum_flux']
                    continue
                
                # EMA: new_value = alpha * current + (1 - alpha) * previous_ema
                ema_flux = alpha * r['sum_flux'] + (1 - alpha) * ema_flux
                ema_size = alpha * float(r.get('size', 0)) + (1 - alpha) * ema_size
                
                r['sum_flux'] = ema_flux
                r['size'] = int(round(ema_size))
                r['prediction'] = ema_flux
        
        return region_tracks
    
    def track_regions_over_time(self, timestamps: List[str]) -> Dict:
        """Track regions across time using spatial proximity and temporal continuity."""
        max_distance = self.config.max_tracking_distance
        flare_relax = self.config.flare_distance_relax_factor
        flare_priority_flux = self.config.min_flux_threshold
        
        max_relaxed_distance = max_distance * flare_relax
        
        print("Tracking regions across time...")
        print(f"Phase 1/2: Detecting regions at each timestamp (parallel)...")
        
        # Parallel region detection
        num_processes = max(1, min(os.cpu_count() or 1, len(timestamps)) - 1)
        print(f"Using {num_processes} processes for region detection")
        
        all_regions = {}
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(self._detect_regions_worker, timestamps),
                               desc="Detecting regions", unit="timestamp", total=len(timestamps)))
        
        # Collect regions and cache labels for movie visualization
        for timestamp, regions, labels in results:
            if regions is not None:
                all_regions[timestamp] = regions
            if labels is not None:
                self.region_labels_cache[timestamp] = labels
        
        # Track regions across time
        print("Phase 2/2: Tracking regions across timestamps...")
        

        
        region_tracks = {}
        next_track_id = 1
        active_tracks = set()
        
        for i, timestamp in tqdm(enumerate(timestamps), desc="Tracking regions", 
                                  unit="timestamp", total=len(timestamps)):
            current_time = pd.to_datetime(timestamp)
            
            if timestamp not in all_regions:
                continue
            
            current_regions = all_regions[timestamp]
            
            for region in current_regions:
                region_copy = region.copy()
                current_flux = region_copy.get('sum_flux', 0.0)
                current_size = region_copy.get('size', 1)
                
                region_copy['is_flaring'] = bool(
                    flare_priority_flux is not None and current_flux >= flare_priority_flux
                )
                
                best_track_id = None
                best_score = float('inf')
                
                # Find matching track using combined distance + similarity score
                for track_id in active_tracks:
                    track_history = region_tracks[track_id]
                    last_timestamp, last_region = track_history[-1]
                    
                    # Calculate distance
                    distance = np.sqrt(
                        (region_copy['centroid_img_x'] - last_region['centroid_img_x'])**2 +
                        (region_copy['centroid_img_y'] - last_region['centroid_img_y'])**2
                    )
                    
                    last_flux = last_region.get('sum_flux', 1e-10)
                    last_size = last_region.get('size', 1)
                    last_is_flaring = bool(
                        flare_priority_flux is not None and last_flux >= flare_priority_flux
                    )
                    
                    # FLUX SIMILARITY: Check flux ratio constraint
                    flux_ratio = max(current_flux, last_flux) / max(min(current_flux, last_flux), 1e-10)

                    
                    # SIZE SIMILARITY: Check size ratio constraint
                    size_ratio = max(current_size, last_size) / max(min(current_size, last_size), 1)

                    
                    # Combined score: distance + weighted similarity penalties
                    score = distance + 0.1 * flux_ratio + 0.1 * size_ratio
                    
                    # Check distance threshold
                    if distance < max_distance:
                        if score < best_score:
                            best_score = score
                            best_track_id = track_id
                    elif (region_copy['is_flaring'] or last_is_flaring) and distance < max_relaxed_distance:
                        if score < best_score:
                            best_score = score
                            best_track_id = track_id
                
                if best_track_id is not None:
                    region_copy['id'] = best_track_id
                    region_copy['timestamp'] = timestamp
                    region_tracks[best_track_id].append((timestamp, region_copy))
                    active_tracks.add(best_track_id)
                else:
                    region_copy['id'] = next_track_id
                    region_copy['timestamp'] = timestamp
                    region_tracks[next_track_id] = [(timestamp, region_copy)]
                    active_tracks.add(next_track_id)
                    next_track_id += 1
        
        # Filter short tracks
        min_detections = 1
        original_count = len(region_tracks)
        region_tracks = {k: v for k, v in region_tracks.items() if len(v) >= min_detections}
        
        print(f"Found {len(region_tracks)} region tracks across {len(timestamps)} timestamps")
        
        # Apply smoothing
        # Apply EMA smoothing if alpha is set (0 < alpha < 1)
        alpha = getattr(self.config, 'flux_smoothing_alpha', 0.3)
        if 0 < alpha < 1:
            region_tracks = self._apply_temporal_smoothing(region_tracks, alpha)
        
        return region_tracks
    
    def detect_flare_events(self, timestamps: Optional[List[str]] = None) -> pd.DataFrame:
        """Detect potential flare events based on tracked regions."""
        if timestamps is None:
            timestamps = self.predictions_df['timestamp'].tolist()
        
        print("Detecting flare events using tracked regions...")
        region_tracks = self.track_regions_over_time(timestamps)
        
        flare_events = []
        for track_id, track_history in tqdm(region_tracks.items(), desc="Processing region tracks", total=len(region_tracks)):
            for timestamp, region_data in track_history:
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
                    'bright_patch_y': region_data.get('bright_patch_y', None),
                    'bright_patch_x': region_data.get('bright_patch_x', None),
                    'peak_img_y': region_data.get('peak_img_y', None),
                    'peak_img_x': region_data.get('peak_img_x', None),
                    'region_label': region_data.get('region_label', None),  # Direct label ID for contour lookup
                    'track_id': track_id
                })
        
        if flare_events:
            self.flare_events_df = pd.DataFrame(flare_events)
        else:
            # Create empty DataFrame with required columns
            self.flare_events_df = pd.DataFrame(columns=[
                'timestamp', 'datetime', 'prediction', 'groundtruth',
                'region_size', 'max_flux', 'mean_flux', 'sum_flux',
                'centroid_patch_y', 'centroid_patch_x',
                'centroid_img_y', 'centroid_img_x',
                'bright_patch_y', 'bright_patch_x',
                'peak_img_y', 'peak_img_x', 'region_label', 'track_id'
            ])
        
        print(f"Detected {len(flare_events)} potential flare events from {len(region_tracks)} tracked regions")
        return self.flare_events_df

    def detect_simultaneous_flares(
        self,
        threshold: Optional[float] = None,
        sequence_window_hours: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Detect simultaneous flaring events - multiple distinct regions within the same
        flux prediction where each region has a sum of flux above the threshold.
        Groups are then clustered into flare sequences if they occur within
        ±sequence_window_hours of each other.

        Args:
            threshold: Sum of flux threshold for considering a region as a flare.
                       Defaults to config.simultaneous_flare_threshold.
            sequence_window_hours: Time window in hours for clustering groups into
                                   sequences. Defaults to config.sequence_window_hours.

        Returns:
            DataFrame with simultaneous flare events, including group_id and sequence_id.
        """
        if self.flare_events_df is None or len(self.flare_events_df) == 0:
            print("Please run detect_flare_events() first")
            return pd.DataFrame()

        if threshold is None:
            threshold = getattr(self.config, "simultaneous_flare_threshold", 5e-6)
        if sequence_window_hours is None:
            sequence_window_hours = getattr(self.config, "sequence_window_hours", 1.0)

        # Filter regions by sum_flux threshold
        high_flux_regions = self.flare_events_df[
            self.flare_events_df["sum_flux"] >= float(threshold)
        ].copy()

        if len(high_flux_regions) == 0:
            print(f"No regions found with sum_flux above threshold {threshold}")
            return pd.DataFrame()

        # Step 1: Group regions by timestamp to find simultaneous flares
        print("Step 1/3: Grouping regions by timestamp for simultaneous flares...")
        simultaneous_groups: List[pd.DataFrame] = []
        unique_timestamps = high_flux_regions["timestamp"].unique()
        for timestamp in tqdm(unique_timestamps, desc="Grouping timestamps", unit="timestamp"):
            group = high_flux_regions[high_flux_regions["timestamp"] == timestamp]
            # Multiple distinct regions at the same timestamp
            if len(group) >= 2:
                simultaneous_groups.append(group)

        if len(simultaneous_groups) == 0:
            print("No simultaneous flare events detected")
            return pd.DataFrame()

        # Step 2: Cluster groups into sequences based on temporal proximity
        print(f"Step 2/3: Clustering {len(simultaneous_groups)} groups into sequences...")
        sequence_clusters: List[List[int]] = []
        used_group_indices: set[int] = set()

        for group_idx, group in tqdm(
            list(enumerate(simultaneous_groups)),
            desc="Clustering groups",
            total=len(simultaneous_groups),
            unit="group",
        ):
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

                    other_datetime = pd.to_datetime(other_group["datetime"].iloc[0])

                    # Check if this group is within sequence_window_hours of
                    # any group in the sequence
                    for seq_group_idx in sequence_groups:
                        seq_group = simultaneous_groups[seq_group_idx]
                        seq_datetime = pd.to_datetime(seq_group["datetime"].iloc[0])
                        time_diff_hours = abs(
                            (other_datetime - seq_datetime).total_seconds() / 3600.0
                        )

                        if time_diff_hours <= float(sequence_window_hours):
                            sequence_groups.append(other_idx)
                            used_group_indices.add(other_idx)
                            changed = True
                            break

            sequence_clusters.append(sequence_groups)

        # Step 3: Create results DataFrame with both group_id and sequence_id
        print(
            f"Step 3/3: Creating results DataFrame from "
            f"{len(sequence_clusters)} simultaneous flare sequences..."
        )
        simultaneous_events: List[Dict[str, Any]] = []
        for sequence_id, sequence_group_indices in tqdm(
            list(enumerate(sequence_clusters)),
            desc="Building simultaneous flare DataFrame",
            total=len(sequence_clusters),
            unit="sequence",
        ):
            for group_idx in sequence_group_indices:
                group = simultaneous_groups[group_idx]
                for _, event in group.iterrows():
                    simultaneous_events.append(
                        {
                            "sequence_id": sequence_id,
                            "group_id": group_idx,  # Original group ID (same timestamp)
                            "timestamp": event["timestamp"],
                            "datetime": event["datetime"],
                            "prediction": event.get("prediction"),
                            "groundtruth": event.get("groundtruth"),
                            "region_size": event.get("region_size"),
                            "max_flux": event.get("max_flux"),
                            "sum_flux": event.get("sum_flux"),
                            "centroid_img_y": event.get("centroid_img_y"),
                            "centroid_img_x": event.get("centroid_img_x"),
                            "bright_patch_y": event.get("bright_patch_y"),
                            "bright_patch_x": event.get("bright_patch_x"),
                            "track_id": event.get("track_id"),
                            "group_size": len(group),
                            "sequence_size": len(sequence_group_indices),
                        }
                    )

        simultaneous_df = pd.DataFrame(simultaneous_events)

        if len(simultaneous_df) > 0:
            print(f"Detected {len(simultaneous_groups)} timestamps with simultaneous flares")
            print(
                f"Clustered into {len(sequence_clusters)} flare sequences "
                f"(within ±{sequence_window_hours} hours)"
            )
            print(f"Total simultaneous events: {len(simultaneous_df)}")

            # Brief summary by sequence
            for sequence_id in sorted(simultaneous_df["sequence_id"].unique()):
                sequence_events = simultaneous_df[simultaneous_df["sequence_id"] == sequence_id]
                timestamps = sorted(sequence_events["datetime"].unique())
                print(f"\nSimultaneous Flare Sequence {sequence_id}:")
                print(f"  Number of groups: {sequence_events['sequence_size'].iloc[0]}")
                print(f"  Time span: {timestamps[0]} to {timestamps[-1]}")
                print(f"  Total events: {len(sequence_events)}")

        self.simultaneous_flares_df = simultaneous_df
        return simultaneous_df


# =============================================================================
# HEK Catalog Utilities
# =============================================================================

def load_hek_catalog(catalog_path: Optional[Path], start_time: Optional[str],
                     end_time: Optional[str], auto_fetch: bool = True,
                     auto_save_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load HEK catalog from CSV or fetch via SunPy."""
    if catalog_path and Path(catalog_path).exists():
        df = pd.read_csv(catalog_path)
        df.columns = df.columns.str.strip()
        print(f"Loaded HEK catalog with {len(df)} rows from {catalog_path}")
        return df
    
    if auto_fetch and start_time and end_time:
        return fetch_hek_flares(start_time, end_time, save_path=auto_save_path)
    
    return None


def fetch_hek_flares(start_time: str, end_time: str, 
                     save_path: Optional[Path] = None) -> pd.DataFrame:
    """Fetch HEK flare catalog via SunPy."""
    if Fido is None or a is None:
        raise ImportError("sunpy is required to fetch HEK flare catalog.")
    
    print(f"Fetching HEK flare catalog from {start_time} to {end_time}...")
    result = Fido.search(
        a.Time(start_time, end_time),
        a.hek.EventType("FL"),
    )
    hek_table = result["hek"]
    
    # Keep scalar columns
    scalar_cols = [name for name in hek_table.colnames if len(hek_table[name].shape) <= 1]
    hek_subset = hek_table[scalar_cols]
    
    df = hek_subset.to_pandas()
    df.columns = df.columns.str.strip()
    
    print(f"Fetched {len(df)} HEK flare entries")
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved HEK data to {save_path}")
    
    return df


def normalize_hek_catalog(df: pd.DataFrame, min_flux: float) -> pd.DataFrame:
    """Clean and filter HEK catalog entries."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Normalize time columns
    for col in ["event_starttime", "event_peaktime", "event_endtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Find GOES class column
    class_col = None
    for candidate in ["fl_goescls", "goes_class", "xray_class"]:
        if candidate in df.columns:
            class_col = candidate
            break
    
    if class_col is not None:
        df["goes_class"] = df[class_col].astype(str).str.upper()
        df["peak_flux_wm2"] = df["goes_class"].map(goes_class_to_flux)
        df = df[(df["peak_flux_wm2"] >= min_flux) | df["goes_class"].isna()].reset_index(drop=True)
    else:
        df["goes_class"] = pd.NA
        df["peak_flux_wm2"] = pd.NA
    
    return df


# =============================================================================
# Flare Event Detection
# =============================================================================

def merge_close_peaks(flux_series: pd.Series, peaks: np.ndarray,
                      min_separation_minutes: float = 15.0) -> np.ndarray:
    """Merge peaks that are too close together."""
    if len(peaks) <= 1:
        return peaks
    
    times = flux_series.index
    if not hasattr(times, 'to_pydatetime'):
        return peaks
    
    filtered_peaks = []
    i = 0
    
    while i < len(peaks):
        current_peak = peaks[i]
        current_time = times[current_peak]
        current_flux = flux_series.iloc[current_peak]
        
        close_peaks = [current_peak]
        close_fluxes = [current_flux]
        
        j = i + 1
        while j < len(peaks):
            next_peak = peaks[j]
            next_time = times[next_peak]
            time_diff = abs((next_time - current_time).total_seconds() / 60.0)
            
            if time_diff <= min_separation_minutes:
                close_peaks.append(next_peak)
                close_fluxes.append(flux_series.iloc[next_peak])
                j += 1
            else:
                break
        
        best_peak = close_peaks[np.argmax(close_fluxes)]
        filtered_peaks.append(best_peak)
        i = j
    
    return np.array(filtered_peaks)


def detect_flare_events_in_track(track_df: pd.DataFrame,
                                  height_multiplier: float = 1.2,
                                  baseline_window: int = 30,
                                  min_peak_separation_minutes: float = 15.0,
                                  min_flux_threshold: float = 1e-7,
                                  validation_window: int = 5,
                                  validation_min_lower: int = 1) -> List[Dict]:
    """Detect individual flare events within a single track using peak detection.
    
    Args:
        track_df: DataFrame with track data
        height_multiplier: Peak must be this multiple of baseline
        baseline_window: Rolling window size for adaptive baseline (number of points)
        min_peak_separation_minutes: Minimum time between peaks
        min_flux_threshold: Absolute flux threshold for peaks
        validation_window: Number of points to check on each side of peak for validation
        validation_min_lower: Minimum number of lower points required on each side
    """
    
    track_df = track_df.sort_values("datetime").reset_index(drop=True)
    flux_series = track_df["sum_flux"]
    track_id = int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else -1
    
    print(f"\n{'='*60}")
    print(f"PEAK DETECTION DEBUG - Track ID: {track_id}")
    print(f"{'='*60}")
    print(f"Track length: {len(track_df)} points")
    print(f"Time range: {track_df['datetime'].iloc[0]} to {track_df['datetime'].iloc[-1]}")
    print(f"Flux range: {flux_series.min():.2e} to {flux_series.max():.2e}")
    print(f"Flux median: {flux_series.median():.2e}")
    print(f"Parameters: height_multiplier={height_multiplier}, baseline_window={baseline_window}, "
          f"min_flux_threshold={min_flux_threshold:.2e}")
    
    # Use adaptive rolling median baseline instead of fixed quantile
    # This adapts to local flux levels over time
    if len(flux_series) > baseline_window:
        # Calculate rolling median baseline
        rolling_baseline = flux_series.rolling(window=baseline_window, center=True, min_periods=1).min()
        # For edges, use the nearest valid value
        rolling_baseline = rolling_baseline.bfill().ffill()
        print(f"Baseline: Rolling median (window={baseline_window})")
    else:
        # For short tracks, use a simple median
        rolling_baseline = pd.Series([flux_series.min()] * len(flux_series), index=flux_series.index)
        print(f"Baseline: Simple median (track too short for rolling window)")
    
    print(f"Baseline range: {rolling_baseline.min():.2e} to {rolling_baseline.max():.2e}")
    print(f"Baseline median: {rolling_baseline.median():.2e}")
    
    # Calculate adaptive height threshold: peak must exceed local baseline * height_multiplier
    # This creates a dynamic threshold that adapts to the local flux level
    height_threshold = rolling_baseline * height_multiplier
    
    # Ensure each point's threshold is at least the minimum absolute threshold
    # This prevents the threshold from being too low during quiet periods
    height_threshold = height_threshold.clip(lower=min_flux_threshold)
    
    print(f"Height threshold range: {height_threshold.min():.2e} to {height_threshold.max():.2e}")
    print(f"Height threshold median: {height_threshold.median():.2e}")
    print(f"Points above threshold: {(flux_series > height_threshold).sum()} / {len(flux_series)}")
    
    # Use dynamic height threshold array for find_peaks
    peaks, _ = find_peaks(
        flux_series.values,
        prominence=None,
        distance=None,
        height=height_threshold.values,
    )
    
    print(f"Initial peaks detected: {len(peaks)}")
    if len(peaks) > 0:
        peak_fluxes = flux_series.iloc[peaks].values
        peak_times = track_df.iloc[peaks]["datetime"].values
        print(f"  Peak fluxes: {peak_fluxes}")
        print(f"  Peak times: {peak_times}")
    
    
    # Filter peaks: ensure surrounding points are generally lower
    # Check a few points before and after each peak
    valid_peaks = []
    print(f"\nPeak validation (window={validation_window}, min_lower={validation_min_lower}):")
    for peak_idx in peaks:
        peak_flux = flux_series.iloc[peak_idx]
        peak_time = track_df.iloc[peak_idx]["datetime"]
        # Check points before and after (at least 1-2 points on each side should be lower)
        check_window = min(validation_window, len(flux_series) // 4)  # Check up to validation_window points on each side
        if check_window == 0:
            check_window = 1
        
        before_lower = 0
        after_lower = 0
        
        # Check points before peak
        for i in range(max(0, peak_idx - check_window), peak_idx):
            if flux_series.iloc[i] < peak_flux:
                before_lower += 1
        
        # Check points after peak
        for i in range(peak_idx + 1, min(len(flux_series), peak_idx + check_window + 1)):
            if flux_series.iloc[i] < peak_flux:
                after_lower += 1
        
        # Require minimum number of lower points on each side (or at boundaries)
        is_at_start = peak_idx < check_window
        is_at_end = peak_idx >= len(flux_series) - check_window
        
        # At boundaries, we're more lenient (allow if we have any lower points)
        # Otherwise require the minimum number
        before_ok = is_at_start or before_lower >= validation_min_lower
        after_ok = is_at_end or after_lower >= validation_min_lower
        
        status = "VALID" if (before_ok and after_ok) else "REJECTED"
        print(f"  Peak at {peak_time} (idx={peak_idx}, flux={peak_flux:.2e}): "
              f"before_lower={before_lower}/{check_window}, after_lower={after_lower}/{check_window}, "
              f"at_start={is_at_start}, at_end={is_at_end} -> {status}")
        
        if before_ok and after_ok:
            valid_peaks.append(peak_idx)
    
    peaks = np.array(valid_peaks) if valid_peaks else np.array([], dtype=int)
    print(f"Valid peaks after validation: {len(peaks)}")
    
    if len(peaks) > 1:
        print(f"\nMerging close peaks (min_separation={min_peak_separation_minutes} min)...")
        datetime_series = pd.Series(flux_series.values, index=track_df["datetime"])
        peaks_before_merge = len(peaks)
        peaks = merge_close_peaks(datetime_series, peaks, min_separation_minutes=min_peak_separation_minutes)
        print(f"  Peaks before merge: {peaks_before_merge}, after merge: {len(peaks)}")
    
    print(f"\nBuilding flare events from {len(peaks)} peaks:")
    flare_events = []
    for peak_idx in peaks:
        peak_time = track_df.iloc[peak_idx]["datetime"]
        peak_flux = flux_series.iloc[peak_idx]
        
        # Find start using adaptive baseline at that point
        start_idx = peak_idx
        at_data_start = False
        for i in range(peak_idx - 1, -1, -1):
            local_baseline = rolling_baseline.iloc[i] * height_multiplier
            if flux_series.iloc[i] < local_baseline:
                start_idx = i + 1
                break
        else:
            start_idx = 0
            at_data_start = True
        
        # Find end using adaptive baseline at that point
        end_idx = peak_idx
        at_data_end = False
        for i in range(peak_idx + 1, len(flux_series)):
            local_baseline = rolling_baseline.iloc[i] * height_multiplier
            if flux_series.iloc[i] < local_baseline:
                end_idx = i - 1
                break
        else:
            end_idx = len(track_df) - 1
            at_data_end = True
        
        start_idx = max(0, start_idx)
        end_idx = min(len(track_df) - 1, end_idx)
        
        if start_idx >= end_idx:
            continue
        
        event_df = track_df.iloc[start_idx:end_idx + 1]
        start_time = event_df["datetime"].iloc[0]
        end_time = event_df["datetime"].iloc[-1]
        duration_minutes = (end_time - start_time).total_seconds() / 60.0
                
        peak_row = track_df.iloc[peak_idx]
        median_flux = event_df["sum_flux"].median()
        prominence = (peak_flux - median_flux) / max(median_flux, 1e-8)
        
        flare_event = {
            "track_id": int(track_df["track_id"].iloc[0]),
            "start_time": start_time,
            "end_time": end_time,
            "peak_time": peak_time,
            "duration_minutes": duration_minutes,
            "num_samples": len(event_df),
            "peak_sum_flux": peak_flux,
            "peak_max_flux": peak_row.get("max_flux", np.nan),
            "peak_region_size": peak_row.get("region_size", np.nan),
            "median_sum_flux": median_flux,
            "prominence": prominence,
            "rise_time_minutes": (peak_time - start_time).total_seconds() / 60.0,
            "decay_time_minutes": (end_time - peak_time).total_seconds() / 60.0,
            "peak_centroid_img_x": peak_row.get("centroid_img_x", np.nan),
            "peak_centroid_img_y": peak_row.get("centroid_img_y", np.nan),
        }
        
        print(f"  Event {len(flare_events)+1}: {start_time} to {end_time} "
              f"(duration={duration_minutes:.1f} min, peak={peak_flux:.2e}, "
              f"prominence={prominence:.2f}, samples={len(event_df)})")
        
        flare_events.append(flare_event)
    
    print(f"\nTotal flare events detected: {len(flare_events)}")
    print(f"{'='*60}\n")
    
    return flare_events


def split_track_sequences(track_df: pd.DataFrame, max_gap_minutes: float) -> List[pd.DataFrame]:
    """Split a single track DataFrame into contiguous sequences."""
    if track_df.empty:
        return []
    
    sequences = []
    current = [track_df.iloc[0]]
    last_time = track_df.iloc[0]["datetime"]
    gap_limit = pd.Timedelta(minutes=max_gap_minutes)
    
    for _, row in track_df.iloc[1:].iterrows():
        dt = row["datetime"]
        if pd.isna(dt) or pd.isna(last_time) or (dt - last_time) > gap_limit:
            sequences.append(pd.DataFrame(current))
            current = [row]
        else:
            current.append(row)
        last_time = dt
    
    if current:
        sequences.append(pd.DataFrame(current))
    
    return sequences


def build_flare_events(flare_events_df: pd.DataFrame, config: FlareAnalysisConfig) -> List[Dict]:
    """Detect individual flare events within each track."""
    all_events = []
    
    # Check if flare_events_df is empty or missing required columns
    if flare_events_df.empty:
        print("Warning: flare_events_df is empty, no flare events to process")
        return all_events
    
    if 'track_id' not in flare_events_df.columns:
        print(f"Warning: flare_events_df missing 'track_id' column. Available columns: {flare_events_df.columns.tolist()}")
        return all_events
    
    for track_id, track_df in flare_events_df.groupby("track_id"):
        track_df = track_df.sort_values("datetime")
        sequences = split_track_sequences(track_df, config.max_gap_minutes)
        
        for seq_idx, seq_df in enumerate(sequences):
            if len(seq_df) < 2:
                continue
            
            events = detect_flare_events_in_track(
                seq_df,
                height_multiplier=config.peak_height_multiplier,
                baseline_window=config.peak_baseline_window,
                min_peak_separation_minutes=config.min_peak_separation_minutes,
                min_flux_threshold=config.min_flux_threshold,
                validation_window=config.peak_validation_window,
                validation_min_lower=config.peak_validation_min_lower,
            )
            
            for event in events:
                event["sequence_id"] = f"{track_id}_{seq_idx}"
                event["sequence_start_time"] = seq_df["datetime"].iloc[0]
                event["sequence_end_time"] = seq_df["datetime"].iloc[-1]
                event["sequence_samples"] = len(seq_df)
                
                duration_seconds = event["duration_minutes"] * 60.0
                expected_samples = max(duration_seconds / config.cadence_seconds, 1.0)
                event["expected_samples"] = expected_samples
                event["coverage_ratio"] = event["num_samples"] / expected_samples
                
                all_events.append(event)
    
    print(f"Detected {len(all_events)} flare events")
    return all_events


def filter_flare_events(events: List[Dict], config: FlareAnalysisConfig) -> pd.DataFrame:
    """Apply quality gates to individual flare events."""
    # Hardcoded quality thresholds (rarely need changing)
    min_coverage = 0.0
    min_prominence = None  # Disabled
    min_peak_flux = config.min_flux_threshold
    
    records = []
    for event in events:
        if event["duration_minutes"] < config.min_duration_minutes:
            continue
        if event["num_samples"] < config.min_samples:
            continue
        if event["coverage_ratio"] < min_coverage:
            continue
        if min_peak_flux is not None and event.get("peak_sum_flux", 0.0) < min_peak_flux:
            continue
        records.append(event)
    
    print(f"Filtered to {len(records)} events after quality checks")
    return pd.DataFrame(records)


# =============================================================================
# HEK Matching
# =============================================================================

def compare_flare_magnitude(foxes_flux: float, hek_goes_class: str,
                            tolerance: float = 1.0) -> Tuple[bool, str, str]:
    """Compare FOXES flux to HEK GOES class."""
    foxes_class = flux_to_goes_class(foxes_flux)
    hek_flux = goes_class_to_flux(hek_goes_class)
    
    if foxes_class == "N/A" or np.isnan(hek_flux) or hek_flux <= 0:
        return False, foxes_class, hek_goes_class if hek_goes_class else "N/A"
    
    foxes_flux_val = goes_class_to_flux(foxes_class)
    if np.isnan(foxes_flux_val) or foxes_flux_val <= 0:
        return False, foxes_class, hek_goes_class
    
    log_ratio = abs(np.log10(foxes_flux_val) - np.log10(hek_flux))
    return log_ratio <= tolerance, foxes_class, hek_goes_class


def match_events_to_hek(events_df: pd.DataFrame, hek_df: pd.DataFrame,
                        match_window_minutes: float,
                        magnitude_tolerance: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Associate FOXES events with HEK events."""
    if events_df.empty and (hek_df is None or hek_df.empty):
        events_df["matched_hek_entries"] = pd.NA
        events_df["match_status"] = "foxes_only"
        return events_df, pd.DataFrame()
    
    if events_df.empty:
        events_df["matched_hek_entries"] = pd.NA
        events_df["match_status"] = "foxes_only"
        return events_df, hek_df.copy() if hek_df is not None else pd.DataFrame()
    
    if hek_df is None or hek_df.empty:
        events_df = events_df.copy()
        events_df["matched_hek_entries"] = pd.NA
        events_df["match_status"] = "foxes_only"
        return events_df, pd.DataFrame()
    
    events_df = events_df.copy()
    hek_df = hek_df.copy()
    
    for col in ["event_starttime", "event_endtime", "event_peaktime"]:
        if col in hek_df.columns:
            hek_df[col] = pd.to_datetime(hek_df[col])
    
    tolerance = pd.Timedelta(minutes=match_window_minutes)
    matched_entries = []
    match_status = []
    foxes_goes_class = []
    hek_goes_class_list = []
    magnitude_match_list = []
    
    hek_df["__matched"] = False
    
    for idx, event in events_df.iterrows():
        event_start = event["start_time"]
        event_end = event["end_time"]
        event_peak = event["peak_time"]
        
        window_start = event_start - tolerance
        window_end = event_end + tolerance
        
        overlapping = hek_df[
            (hek_df["event_endtime"] >= window_start) &
            (hek_df["event_starttime"] <= window_end)
        ]
        
        # Also filter by peak-to-peak distance
        if not overlapping.empty:
            peak_tolerance_seconds = match_window_minutes * 60
            overlapping = overlapping[
                overlapping["event_peaktime"].apply(
                    lambda pt: abs((pt - event_peak).total_seconds()) <= peak_tolerance_seconds
                    if pd.notna(pt) else False
                )
            ]
        
        if overlapping.empty:
            matched_entries.append(pd.NA)
            match_status.append("foxes_only")
            foxes_goes_class.append(flux_to_goes_class(event.get("peak_sum_flux", 0)))
            hek_goes_class_list.append(pd.NA)
            magnitude_match_list.append(False)
            continue
        
        hek_df.loc[overlapping.index, "__matched"] = True
        
        # Find closest peak
        event_groups = overlapping.groupby("event_peaktime")
        peak_distances = []
        for peak_time, group in event_groups:
            distance = abs((peak_time - event_peak).total_seconds())
            peak_distances.append((distance, peak_time, group))
        
        if peak_distances:
            _, _, best_group = min(peak_distances, key=lambda x: x[0])
            entries = []
            best_hek_class = None
            
            for _, hek_event in best_group.iterrows():
                obs = hek_event.get("obs_observatory", "Unknown")
                goes_class = hek_event.get("fl_goescls", "")
                coords = hek_event.get("hpc_coord", "")
                
                # Get numeric coordinates directly (arcseconds)
                hpc_x = hek_event.get("hpc_x", None)
                hpc_y = hek_event.get("hpc_y", None)
                
                if goes_class and (best_hek_class is None or obs in ['GOES', 'SWPC']):
                    best_hek_class = goes_class
                
                entries.append({
                    "observatory": str(obs).strip() if obs else "Unknown",
                    "goes_class": str(goes_class).strip() if goes_class else "",
                    "coordinates": str(coords).strip() if coords else "",
                    "hpc_x": float(hpc_x) if hpc_x is not None and not pd.isna(hpc_x) else None,
                    "hpc_y": float(hpc_y) if hpc_y is not None and not pd.isna(hpc_y) else None,
                    "peak_time": str(hek_event.get("event_peaktime", "")).strip(),
                    "start_time": str(hek_event.get("event_starttime", "")).strip(),
                    "end_time": str(hek_event.get("event_endtime", "")).strip()
                })
            
            matched_entries.append(json.dumps(entries))
            
            foxes_flux = event.get("peak_sum_flux", 0)
            mag_matches, foxes_str, hek_str = compare_flare_magnitude(
                foxes_flux, best_hek_class or "", magnitude_tolerance
            )
            
            foxes_goes_class.append(foxes_str)
            hek_goes_class_list.append(hek_str)
            magnitude_match_list.append(mag_matches)
            
            if mag_matches:
                match_status.append("matched_time_magnitude")
            else:
                match_status.append("matched_time_only")
        else:
            matched_entries.append(pd.NA)
            foxes_goes_class.append(flux_to_goes_class(event.get("peak_sum_flux", 0)))
            hek_goes_class_list.append(pd.NA)
            magnitude_match_list.append(False)
            match_status.append("foxes_only")
    
    events_df["matched_hek_entries"] = matched_entries
    events_df["match_status"] = match_status
    events_df["foxes_goes_class"] = foxes_goes_class
    events_df["hek_goes_class"] = hek_goes_class_list
    events_df["magnitude_match"] = magnitude_match_list
    
    unmatched_hek_df = hek_df[~hek_df["__matched"]].copy()
    if "__matched" in unmatched_hek_df.columns:
        unmatched_hek_df.drop(columns="__matched", inplace=True)
    if not unmatched_hek_df.empty:
        unmatched_hek_df["match_status"] = "hek_only"
    
    # Summary
    n_time_mag = (events_df["match_status"] == "matched_time_magnitude").sum()
    n_time_only = (events_df["match_status"] == "matched_time_only").sum()
    n_foxes_only = (events_df["match_status"] == "foxes_only").sum()
    
    print(f"\n=== Matching Summary ===")
    print(f"  FOXES events: {len(events_df)}")
    print(f"    - Matched time + magnitude: {n_time_mag}")
    print(f"    - Matched time only: {n_time_only}")
    print(f"    - FOXES-only: {n_foxes_only}")
    print(f"  HEK-only (missed): {len(unmatched_hek_df)}")
    
    return events_df, unmatched_hek_df


# =============================================================================
# Coordinate Utilities
# =============================================================================

def parse_hpc_coordinates(hpc_coord_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse HEK helioprojective coordinate string in POINT(x y) format."""
    if not isinstance(hpc_coord_str, str) or pd.isna(hpc_coord_str):
        return None, None
    
    try:
        coord_str = hpc_coord_str.strip()
        
        # Handle POINT(x y) format from HEK
        if coord_str.startswith("POINT(") and coord_str.endswith(")"):
            coords = coord_str[6:-1].strip()
            parts = coords.split()
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        
        # Handle SkyCoord format (fallback)
        elif "SkyCoord" in coord_str and "Helioprojective" in coord_str:
            if "(Tx, Ty) in arcsec" in coord_str:
                arcsec_pos = coord_str.find("(Tx, Ty) in arcsec")
                if arcsec_pos != -1:
                    start_pos = coord_str.find("(", arcsec_pos + len("(Tx, Ty) in arcsec"))
                    if start_pos != -1:
                        end_pos = coord_str.find(")", start_pos)
                        if end_pos != -1:
                            coords_part = coord_str[start_pos+1:end_pos].strip()
                            parts = [p.strip() for p in coords_part.split(",")]
                            if len(parts) >= 2:
                                return float(parts[0]), float(parts[1])
        
        # Handle simple (x, y) format
        elif coord_str.startswith("(") and coord_str.endswith(")"):
            coords = coord_str[1:-1].strip()
            parts = [p.strip() for p in coords.split(",")]
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
                
    except (ValueError, IndexError):
        pass
    
    return None, None


def helioprojective_to_pixel(x_arcsec: float, y_arcsec: float,
                              image_size: int = 512,
                              fov_solar_radii: float = 1.1) -> Tuple[Optional[float], Optional[float]]:
    """Convert helioprojective coordinates (arcseconds) to pixel coordinates."""
    try:
        if not isinstance(x_arcsec, (int, float)) or not isinstance(y_arcsec, (int, float)):
            return None, None
        if np.isnan(x_arcsec) or np.isnan(y_arcsec):
            return None, None
        
        solar_radius_arcsec = 960.0
        image_center = image_size / 2.0
        total_fov_arcsec = fov_solar_radii * 2 * solar_radius_arcsec
        arcsec_per_pixel = total_fov_arcsec / image_size
        
        x_pixel = image_center + (x_arcsec / arcsec_per_pixel)
        y_pixel = image_center + (y_arcsec / arcsec_per_pixel)
        
        margin = 1.0
        if -margin <= x_pixel < image_size + margin and -margin <= y_pixel < image_size + margin:
            x_pixel = max(0, min(image_size - 1, x_pixel))
            y_pixel = max(0, min(image_size - 1, y_pixel))
            return x_pixel, y_pixel
            
    except Exception:
        pass
    
    return None, None


def load_aia_image_at_time(aia_path: Path, timestamp: str) -> Optional[np.ndarray]:
    """Load AIA image at specific timestamp for RGB composite."""
    if aia_path is None or not aia_path.exists():
        return None
    
    # Search directories
    possible_dirs = [aia_path]
    for subdir in ['test', 'train', 'val']:
        subdir_path = aia_path / subdir
        if subdir_path.exists():
            possible_dirs.append(subdir_path)
    
    for directory in possible_dirs:
        filepath = directory / f"{timestamp}.npy"
        if filepath.exists():
            try:
                data = np.load(filepath)
                if data.ndim == 3 and data.shape[0] >= 3:
                    rgb_image = np.zeros((data.shape[1], data.shape[2], 3))
                    for i in range(3):
                        channel_data = data[i]
                        if channel_data.max() > channel_data.min():
                            channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
                        rgb_image[:, :, i] = channel_data
                    return rgb_image
            except Exception:
                continue
    
    return None


# =============================================================================
# Plotting
# =============================================================================

def create_sxr_plots(catalog_df: pd.DataFrame, flare_events_df: pd.DataFrame,
                     output_dir: Path, config: FlareAnalysisConfig,
                     predictions_csv: Optional[str] = None) -> None:
    """Create SXR-style timeseries plots with AIA images for detected flare events."""
    if catalog_df.empty:
        print("No flare events to plot")
        return
    
    output_dir = Path(output_dir)
    aia_path = Path(config.aia_path) if config.aia_path else None
    
    # Create subfolders
    matched_dir = output_dir / "matched_time_magnitude"
    time_only_dir = output_dir / "matched_time_only"
    foxes_only_dir = output_dir / "foxes_only"
    for d in [matched_dir, time_only_dir, foxes_only_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    predictions_df = None
    if predictions_csv and Path(predictions_csv).exists():
        try:
            predictions_df = pd.read_csv(predictions_csv)
            if 'datetime' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])
            elif 'timestamp' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['timestamp'])
                predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        except Exception as e:
            print(f"Warning: Could not load predictions CSV: {e}")
    
    # Observatory styles for markers
    obs_styles = {
        'SDO': {'color': 'purple', 'marker': 's', 'size': 12, 'name': 'SDO'},
        'GOES': {'color': 'orange', 'marker': '^', 'size': 10, 'name': 'GOES'},
        'SWPC': {'color': 'orange', 'marker': 'D', 'size': 10, 'name': 'SWPC'},
        'Unknown': {'color': 'gray', 'marker': 'o', 'size': 6, 'name': 'Other'}
    }
    
    print(f"Creating plots for {len(catalog_df)} events...")
    
    for idx, event in catalog_df.iterrows():
        try:
            track_id = event["track_id"]
            track_data = flare_events_df[flare_events_df["track_id"] == track_id].copy()
            if track_data.empty:
                continue
            
            track_data = track_data.sort_values("datetime")
            peak_time = pd.to_datetime(event["peak_time"])
            
            # Try to load AIA image at peak time
            aia_image = None
            peak_timestamp = None
            if aia_path:
                peak_row = track_data.iloc[(track_data["datetime"] - peak_time).abs().argsort()[:1]]
                if not peak_row.empty:
                    peak_timestamp = peak_row["timestamp"].iloc[0]
                    aia_image = load_aia_image_at_time(aia_path, peak_timestamp)
            
            # Create figure with AIA image if available
            if aia_image is not None:
                fig = plt.figure(figsize=(16, 6))
                gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], hspace=0.3)
                ax_ts = fig.add_subplot(gs[0])
                ax_img = fig.add_subplot(gs[1])
            else:
                fig, ax_ts = plt.subplots(figsize=(12, 6))
            
            window_start = peak_time - pd.Timedelta(hours=config.plot_window_hours/2)
            window_end = peak_time + pd.Timedelta(hours=config.plot_window_hours/2)
            
            # Plot other tracks in faded grey for context
            other_tracks = flare_events_df[flare_events_df["track_id"] != track_id]
            if not other_tracks.empty:
                other_tracks_in_window = other_tracks[
                    (other_tracks["datetime"] >= window_start) & 
                    (other_tracks["datetime"] <= window_end)
                ]
                
                if not other_tracks_in_window.empty:
                    plotted_other_tracks = False
                    for other_track_id, other_track_data in other_tracks_in_window.groupby("track_id"):
                        other_track_data = other_track_data.sort_values("datetime")
                        label = 'Other Tracks' if not plotted_other_tracks else None
                        ax_ts.plot(other_track_data["datetime"], other_track_data["sum_flux"], 
                                 color='grey', linewidth=0.8, alpha=0.4, label=label)
                        ax_ts.scatter(other_track_data["datetime"], other_track_data["sum_flux"],
                                    color='grey', s=5, alpha=0.2, marker='o', zorder=3)
                        plotted_other_tracks = True
            
            # Plot predictions on timeseries
            if predictions_df is not None:
                preds_in_window = predictions_df[
                    (predictions_df["timestamp"] >= window_start) &
                    (predictions_df["timestamp"] <= window_end)
                ].copy()
                
                if not preds_in_window.empty:
                    if 'groundtruth' in preds_in_window.columns:
                        ax_ts.plot(preds_in_window["timestamp"], preds_in_window['groundtruth'],
                               'k-', linewidth=1.5, alpha=0.8, label='GOES Ground Truth')
                        ax_ts.scatter(preds_in_window["timestamp"], preds_in_window['groundtruth'],
                                    color='black', s=8, alpha=0.3, marker='o', zorder=4)
                    if 'predictions' in preds_in_window.columns:
                        ax_ts.plot(preds_in_window["timestamp"], preds_in_window['predictions'],
                               'r--', linewidth=1.5, alpha=0.8, label='FOXES Prediction')
                        ax_ts.scatter(preds_in_window["timestamp"], preds_in_window['predictions'],
                                    color='red', s=8, alpha=0.3, marker='s', zorder=4)
            
            # Plot track flux
            ax_ts.plot(track_data["datetime"], track_data["sum_flux"],
                   'b-', linewidth=2, alpha=0.9, label='Track Sum Flux')
            ax_ts.scatter(track_data["datetime"], track_data["sum_flux"],
                        color='blue', s=10, alpha=0.4, marker='o', zorder=5, edgecolors='white', linewidths=0.5)
            
            # Mark peak
            ax_ts.axvline(peak_time, color='blue', linestyle=':', linewidth=2, label='Event Peak')
            
            # Add FOXES class annotation on timeseries
            foxes_flux = event.get("peak_sum_flux", 0)
            foxes_class = flux_to_goes_class(foxes_flux)
            y_pos = ax_ts.get_ylim()[1] * 0.9
            ax_ts.annotate(f'FOXES: {foxes_class}', (peak_time, y_pos),
                       xytext=(0, -10), textcoords='offset points',
                       ha='center', va='top', fontsize=10, color='blue', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Add HEK peak times on timeseries
            if not pd.isna(event.get("matched_hek_entries")):
                try:
                    hek_entries = json.loads(event["matched_hek_entries"])
                    plotted_obs_peaks = set()
                    
                    for entry in hek_entries:
                        obs = entry.get("observatory", "Unknown")
                        peak_time_str = entry.get("peak_time", "")
                        goes_class = entry.get("goes_class", "")
                        
                        if peak_time_str:
                            try:
                                hek_peak_time = pd.to_datetime(peak_time_str)
                                color = obs_styles.get(obs, obs_styles['Unknown'])['color']
                                
                                label = f'{obs} Peak' if obs not in plotted_obs_peaks else None
                                if label:
                                    plotted_obs_peaks.add(obs)
                                
                                ax_ts.axvline(hek_peak_time, color=color, linestyle=':', 
                                            linewidth=2, alpha=0.8, label=label)
                                
                                if goes_class and obs in ['SDO', 'SWPC']:
                                    y_pos = ax_ts.get_ylim()[1] * 0.7
                                    ax_ts.annotate(goes_class, (hek_peak_time, y_pos),
                                                 xytext=(0, -10), textcoords='offset points',
                                                 ha='center', va='top', fontsize=8,
                                                 color=color, weight='bold',
                                                 bbox=dict(boxstyle='round,pad=0.2', 
                                                          facecolor='white', alpha=0.8, edgecolor=color))
                            except Exception:
                                continue
                except Exception:
                    pass
            
            ax_ts.set_xlim(window_start, window_end)
            ax_ts.set_yscale('log')
            ax_ts.set_ylabel('Flux', fontsize=12)
            ax_ts.set_xlabel('Time (UTC)', fontsize=12)
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_ts.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45)
            ax_ts.legend(loc='upper left', fontsize=9, framealpha=0.9)
            ax_ts.grid(True, alpha=0.3)
            
            # Plot AIA image with location markers
            if aia_image is not None:
                ax_img.imshow(aia_image, origin='lower', aspect='equal', alpha=0.8)
                
                ax_img.set_title(f'AIA Composite\n{peak_time.strftime("%H:%M:%S")}', fontsize=12)
                ax_img.set_xlabel('X (pixels)', fontsize=10)
                ax_img.set_ylabel('Y (pixels)', fontsize=10)
                
                # Mark FOXES centroid (blue star)
                if not pd.isna(event.get("peak_centroid_img_x")) and not pd.isna(event.get("peak_centroid_img_y")):
                    cx = event["peak_centroid_img_x"]
                    cy = event["peak_centroid_img_y"]
                    ax_img.plot(cx, cy, 'b*', markersize=15, markeredgecolor='white', 
                              markeredgewidth=2, label='FOXES Location')
                
                # Class annotations on image
                y_offset = 0.98
                
                # FOXES class (blue)
                ax_img.text(0.02, y_offset, f'FOXES: {foxes_class}', 
                          transform=ax_img.transAxes, fontsize=10, weight='bold',
                          horizontalalignment='left', verticalalignment='top', color='blue',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='blue'))
                
                # HEK classes and locations
                if not pd.isna(event.get("matched_hek_entries")):
                    try:
                        hek_entries = json.loads(event["matched_hek_entries"])
                        sdo_class = None
                        goes_class_hek = None
                        plotted_obs = set()
                        
                        for entry in hek_entries:
                            obs = entry.get("observatory", "Unknown").strip() if entry.get("observatory") else "Unknown"
                            entry_class = entry.get("goes_class", "").strip() if entry.get("goes_class") else ""
                            
                            # Track classes for annotation
                            if entry_class:
                                if obs == 'SDO':
                                    sdo_class = entry_class
                                elif obs in ['GOES', 'SWPC']:
                                    goes_class_hek = entry_class
                            
                            # Plot location marker using numeric coordinates
                            hpc_x = entry.get("hpc_x")
                            hpc_y = entry.get("hpc_y")
                            
                            # Try numeric coords first, fall back to parsing string
                            x_arcsec, y_arcsec = None, None
                            if hpc_x is not None and hpc_y is not None:
                                x_arcsec, y_arcsec = hpc_x, hpc_y
                            else:
                                coords = entry.get("coordinates", "").strip() if entry.get("coordinates") else ""
                                if coords:
                                    x_arcsec, y_arcsec = parse_hpc_coordinates(coords)
                            
                            if x_arcsec is not None and y_arcsec is not None:
                                hek_x, hek_y = helioprojective_to_pixel(x_arcsec, y_arcsec)
                                if hek_x is not None and hek_y is not None:
                                    style = obs_styles.get(obs, obs_styles['Unknown'])
                                    label = f'{style["name"]} Location' if obs not in plotted_obs else None
                                    if label:
                                        plotted_obs.add(obs)
                                    
                                    ax_img.scatter(hek_x, hek_y, marker=style['marker'], 
                                                 color=style['color'], s=style['size']**2,
                                                 edgecolors='white', linewidths=1.5, 
                                                 label=label, alpha=0.9, zorder=10)
                        
                        # Add SDO class (purple)
                        if sdo_class:
                            y_offset -= 0.08
                            ax_img.text(0.02, y_offset, f'SDO: {sdo_class}', 
                                      transform=ax_img.transAxes, fontsize=10, weight='bold',
                                      horizontalalignment='left', verticalalignment='top', color='purple',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='purple'))
                        
                        # Add GOES class (orange)
                        if goes_class_hek:
                            y_offset -= 0.08
                            ax_img.text(0.02, y_offset, f'GOES: {goes_class_hek}', 
                                      transform=ax_img.transAxes, fontsize=10, weight='bold',
                                      horizontalalignment='left', verticalalignment='top', color='orange',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='orange'))
                            
                    except Exception:
                        pass
                
                # Legend for image
                handles, labels = ax_img.get_legend_handles_labels()
                if handles:
                    ax_img.legend(loc='upper right', fontsize=8, framealpha=0.8)
            
            # Title
            match_status = event.get("match_status", "foxes_only")
            foxes_class_display = event.get("foxes_goes_class", foxes_class)
            hek_class_display = event.get("hek_goes_class", "N/A")
            
            status_label = {
                "matched_time_magnitude": "Matched (Time + Mag)",
                "matched_time_only": "Matched (Time Only)",
                "foxes_only": "FOXES Only"
            }.get(match_status, "Unknown")
            
            title_parts = [
                f"Track {track_id} ({status_label})",
                f"Peak: {peak_time.strftime('%Y-%m-%d %H:%M')}",

            ]
            if match_status in ["matched_time_magnitude", "matched_time_only"]:
                title_parts.append(f"FOXES: {foxes_class_display} vs HEK: {hek_class_display}")
            
            fig.suptitle(" | ".join(title_parts), fontsize=14, y=0.95)
            
            plt.tight_layout()
            
            # Save to appropriate folder
            filename = f"flare_event_{track_id}_{peak_time.strftime('%Y%m%d_%H%M%S')}.png"
            if match_status == "matched_time_magnitude":
                filepath = matched_dir / filename
            elif match_status == "matched_time_only":
                filepath = time_only_dir / filename
            else:
                filepath = foxes_only_dir / filename
            
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating plot for event {idx}: {e}")
            continue
    
    print(f"Saved plots to {output_dir}")


def filter_hek_by_foxes_coverage(hek_df: pd.DataFrame, foxes_timestamps: pd.DatetimeIndex,
                                  max_gap_minutes: float = 10.0, min_points_required: int = 5,
                                  coverage_window_minutes: float = 30.0,
                                  min_points_before_peak: int = 3,
                                  min_points_after_peak: int = 3) -> pd.DataFrame:
    """Filter HEK events to only those during adequate FOXES data coverage."""
    if hek_df.empty or len(foxes_timestamps) == 0:
        return hek_df
    
    hek_df = hek_df.copy()
    
    if "event_peaktime" in hek_df.columns:
        hek_df["event_peaktime"] = pd.to_datetime(hek_df["event_peaktime"])
    else:
        return hek_df
    
    foxes_timestamps = foxes_timestamps.sort_values()
    foxes_ts_array = foxes_timestamps.values
    
    def has_adequate_coverage(peak_time):
        if pd.isna(peak_time):
            return False
        
        peak_np = np.datetime64(peak_time)
        idx = np.searchsorted(foxes_ts_array, peak_np)
        
        # Check nearby point
        has_nearby = False
        if idx > 0:
            if abs(peak_np - foxes_ts_array[idx - 1]) <= np.timedelta64(int(max_gap_minutes * 60), 's'):
                has_nearby = True
        if idx < len(foxes_ts_array):
            if abs(peak_np - foxes_ts_array[idx]) <= np.timedelta64(int(max_gap_minutes * 60), 's'):
                has_nearby = True
        
        if not has_nearby:
            return False
        
        # Check coverage window
        window_start = peak_np - np.timedelta64(int(coverage_window_minutes * 60), 's')
        window_end = peak_np + np.timedelta64(int(coverage_window_minutes * 60), 's')
        
        start_idx = np.searchsorted(foxes_ts_array, window_start)
        end_idx = np.searchsorted(foxes_ts_array, window_end)
        peak_idx = np.searchsorted(foxes_ts_array, peak_np)
        
        points_in_window = end_idx - start_idx
        if points_in_window < min_points_required:
            return False
        
        points_before = peak_idx - start_idx
        if points_before < min_points_before_peak:
            return False
        
        points_after = end_idx - peak_idx
        if points_after < min_points_after_peak:
            return False
        
        return True
    
    coverage_mask = hek_df["event_peaktime"].apply(has_adequate_coverage)
    return hek_df[coverage_mask].copy()


def _generate_single_frame(args: Tuple) -> Optional[str]:
    """
    Generate a single frame for the movie. Designed for multiprocessing.
    
    Args:
        args: Tuple containing (frame_idx, timestamp, frame_data_dict)
        
    Returns:
        Path to saved frame, or None if failed
    """
    frame_idx, timestamp, frame_data = args
    
    try:
        # Unpack frame data
        catalog_df = frame_data['catalog_df']
        flare_events_df = frame_data['flare_events_df']
        predictions_df = frame_data['predictions_df']
        hek_df = frame_data['hek_df']
        frames_dir = Path(frame_data['frames_dir'])
        region_labels_cache = frame_data['region_labels_cache']  # Pre-computed region labels
        config = frame_data['config']
        obs_styles = frame_data['obs_styles']
        foxes_track_colors = frame_data['foxes_track_colors']
        foxes_track_color_map = frame_data['foxes_track_color_map']
        plot_window_hours = frame_data['plot_window_hours']
        aia_path = frame_data['aia_path']
        flux_path = frame_data['flux_path']
        
        current_time = pd.to_datetime(timestamp)
        
        # Create figure with 2 subplots: SXR timeseries (left) and AIA image (right)
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
        ax_sxr = fig.add_subplot(gs[0])
        ax_aia = fig.add_subplot(gs[1])
        
        # ========== SXR Timeseries Plot ==========
        window_start = current_time - pd.Timedelta(hours=plot_window_hours/2)
        window_end = current_time + pd.Timedelta(hours=plot_window_hours/2)
        
        # Plot predictions
        if predictions_df is not None and not predictions_df.empty:
            preds_in_window = predictions_df[
                (predictions_df["datetime"] >= window_start) &
                (predictions_df["datetime"] <= window_end)
            ]
            
            if not preds_in_window.empty:
                # Plot GOES ground truth if available (column name is 'groundtruth')
                if 'groundtruth' in preds_in_window.columns:
                    ax_sxr.plot(preds_in_window["datetime"], preds_in_window['groundtruth'],
                               'b-', linewidth=1.5, alpha=0.8, label='GOES XRSB (Ground Truth)')
                    ax_sxr.scatter(preds_in_window["datetime"], preds_in_window['groundtruth'],
                                 color='blue', s=8, alpha=0.3, marker='o', zorder=4)
                
                # Plot FOXES predictions if available
                if 'predictions' in preds_in_window.columns:
                    ax_sxr.plot(preds_in_window["datetime"], preds_in_window['predictions'],
                               'r--', linewidth=1.5, alpha=0.8, label='FOXES Prediction')
                    ax_sxr.scatter(preds_in_window["datetime"], preds_in_window['predictions'],
                                 color='red', s=8, alpha=0.3, marker='s', zorder=4)
        
        # ========== Mark FOXES active flares (from catalog) ==========
        # Get active FOXES flares first so we can highlight their tracks
        active_foxes_flares = pd.DataFrame()
        if not catalog_df.empty:
            active_foxes_flares = catalog_df[
                (catalog_df['start_time'] <= current_time) &
                (catalog_df['end_time'] >= current_time)
            ].copy()
        
        # Get set of active track IDs
        active_track_ids = set(active_foxes_flares['track_id'].values) if not active_foxes_flares.empty else set()
        
        # Plot all tracks (faded for context, highlighted for active flares)
        all_tracks_in_window = flare_events_df[
            (flare_events_df["datetime"] >= window_start) &
            (flare_events_df["datetime"] <= window_end)
        ]
        
        plotted_other_tracks = False
        for track_id, track_data in all_tracks_in_window.groupby("track_id"):
            track_data = track_data.sort_values("datetime")
            
            # Check if this track has an active flare
            if track_id in active_track_ids:
                # Highlight active flaring track with its color
                track_color = foxes_track_color_map.get(track_id, foxes_track_colors[0])
                ax_sxr.plot(track_data["datetime"], track_data["sum_flux"],
                           color=track_color, linewidth=3.0, alpha=0.9, 
                           label=f'Track {track_id} (Active)', zorder=6)
                ax_sxr.scatter(track_data["datetime"], track_data["sum_flux"],
                             color=track_color, s=25, alpha=0.8, marker='o', 
                             edgecolor='black', linewidth=0.5, zorder=7)
            else:
                # Non-active tracks shown faded
                label = 'Other Tracks' if not plotted_other_tracks else None
                ax_sxr.plot(track_data["datetime"], track_data["sum_flux"],
                           color='grey', linewidth=0.8, alpha=0.4, label=label)
                ax_sxr.scatter(track_data["datetime"], track_data["sum_flux"],
                             color='grey', s=5, alpha=0.2, marker='o', zorder=3)
                plotted_other_tracks = True
        
        # Mark current time with vertical line
        ax_sxr.axvline(current_time, color='red', linestyle='-', linewidth=3, 
                      alpha=0.8, label='Current Time', zorder=10)
        
        # Shade FOXES flare durations
        for i, (idx, flare) in enumerate(active_foxes_flares.iterrows()):
            track_id = flare['track_id']
            flare_color = foxes_track_color_map.get(track_id, foxes_track_colors[i % len(foxes_track_colors)])
            
            flare_start = flare['start_time']
            flare_end = flare['end_time']
            flare_peak = flare['peak_time']
            
            # Shade flare duration with track color
            ax_sxr.axvspan(flare_start, flare_end, alpha=0.15, color=flare_color, 
                         label=f'FOXES {track_id}' if i < 3 else None)
            
            # Mark peak
            ax_sxr.axvline(flare_peak, color=flare_color, linestyle=':', linewidth=2, alpha=0.8)
        
        # ========== Mark HEK active flares ==========
        active_hek_flares = pd.DataFrame()
        if not hek_df.empty and 'event_starttime' in hek_df.columns and 'event_endtime' in hek_df.columns:
            active_hek_flares = hek_df[
                (hek_df['event_starttime'] <= current_time) &
                (hek_df['event_endtime'] >= current_time)
            ].copy()
        
        # Shade HEK flare durations (purple for SDO, orange for GOES)
        hek_plotted = {'SDO': False, 'GOES': False}
        for idx, hek_event in active_hek_flares.iterrows():
            obs = hek_event.get('obs_observatory', 'Unknown')
            if pd.isna(obs):
                obs = 'Unknown'
            obs = str(obs).strip()
            
            style = obs_styles.get(obs, obs_styles['Unknown'])
            hek_start = hek_event.get('event_starttime')
            hek_end = hek_event.get('event_endtime')
            hek_peak = hek_event.get('event_peaktime')
            
            if pd.notna(hek_start) and pd.notna(hek_end):
                label = f'HEK {obs}' if not hek_plotted.get(obs, False) else None
                hek_plotted[obs] = True
                ax_sxr.axvspan(hek_start, hek_end, alpha=0.1, color=style['color'], 
                              label=label, hatch='//', edgecolor=style['color'], linewidth=0.5)
            
            if pd.notna(hek_peak):
                ax_sxr.axvline(hek_peak, color=style['color'], linestyle='--', linewidth=2, alpha=0.7)
        
        ax_sxr.set_xlim(window_start, window_end)
        ax_sxr.set_yscale('log')
        ax_sxr.set_ylabel('Flux', fontsize=12)
        ax_sxr.set_xlabel('Time (UTC)', fontsize=12)
        ax_sxr.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_sxr.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax_sxr.xaxis.get_majorticklabels(), rotation=45)
        ax_sxr.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax_sxr.grid(True, alpha=0.3)
        ax_sxr.set_title(f'SXR Timeseries\n{current_time.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=12)
        
        # ========== AIA Image with Locations and Contours ==========
        aia_image = None
        if aia_path:
            aia_image = load_aia_image_at_time(Path(aia_path), timestamp)
        
        if aia_image is not None:
            ax_aia.imshow(aia_image, origin='lower', aspect='equal', alpha=0.9)
        else:
            # Create blank image if AIA not available
            blank_img = np.zeros((512, 512, 3))
            ax_aia.imshow(blank_img, origin='lower', aspect='equal')
        
        ax_aia.set_title(f'AIA Composite\n{current_time.strftime("%H:%M:%S")}', fontsize=12)
        ax_aia.set_xlabel('X (pixels)', fontsize=10)
        ax_aia.set_ylabel('Y (pixels)', fontsize=10)
        
        # Draw AR contours from pre-computed region labels (cached in memory during detection)
        region_labels = region_labels_cache.get(timestamp)
        
        if region_labels is not None:
            # Draw contours for each region in cyan
            num_regions = region_labels.max()
            for label_id in range(1, num_regions + 1):
                region_mask = region_labels == label_id
                if np.any(region_mask):
                    try:
                        ax_aia.contour(region_mask.astype(float), levels=[0.5], colors='cyan',
                                      linewidths=1.5, alpha=0.6, extent=[0, 512, 0, 512])
                    except Exception:
                        pass
            #ax_aia.plot([], [], color='cyan', linewidth=1.5, alpha=0.6, label='AR Contours')
        
        # ========== FOXES Track Locations ==========
        # Show ALL FOXES track locations - find nearest data point to current timestamp for each track
        plotted_tracks = set()
        plotted_obs = set()
        
        # Get tracks that have data at the EXACT current timestamp
        # This ensures marker and contour are always in sync (both from same detection)
        if not flare_events_df.empty and 'timestamp' in flare_events_df.columns:
            current_events = flare_events_df[flare_events_df['timestamp'] == timestamp].copy()
        else:
            current_events = pd.DataFrame()

        
        for i, (_, track_event) in enumerate(current_events.iterrows()):
            track_id = track_event['track_id']
            if track_id in plotted_tracks:
                continue
            plotted_tracks.add(track_id)
            
            # Centroid coordinates
            cx = track_event.get('centroid_img_x')
            cy = track_event.get('centroid_img_y')
            # Peak coordinates (for star marker - where the brightest patch is)
            peak_x = track_event.get('peak_img_x')
            peak_y = track_event.get('peak_img_y')
            # Region label (direct association - no position lookup needed!)
            region_label = track_event.get('region_label')
            current_flux = track_event.get('sum_flux', 0)
            
            # Skip if coordinates are missing or invalid
            if pd.isna(cx) or pd.isna(cy) or cx < 0 or cx > 512 or cy < 0 or cy > 512:
                continue
            
            # Use peak coords if available, otherwise fall back to centroid
            marker_x = peak_x if pd.notna(peak_x) else cx
            marker_y = peak_y if pd.notna(peak_y) else cy
            
            # Get track color
            track_color = foxes_track_color_map.get(track_id, foxes_track_colors[i % len(foxes_track_colors)])
            
            # Check if this track has an active flare (from catalog)
            is_active_flare = False
            if not active_foxes_flares.empty:
                is_active_flare = track_id in active_foxes_flares['track_id'].values
            
            foxes_class = flux_to_goes_class(current_flux)
            
            # Draw highlighted contour using the track's directly-associated region_label
            # No position lookup needed - each track knows its own region!
            if region_labels is not None and pd.notna(region_label) and region_label > 0:
                region_mask = region_labels == int(region_label)
                if np.any(region_mask):
                    try:
                        # Thick contour for active flares, thinner for others
                        lw = 4.0 if is_active_flare else 2.5
                        ax_aia.contour(region_mask.astype(float), levels=[0.5], colors=track_color, 
                                      linewidths=lw, alpha=0.9, extent=[0, 512, 0, 512])
                    except Exception:
                        pass
            
            # Plot track location marker at peak brightness location
            if is_active_flare:
                # Large star for active flares (at peak brightness)
                ax_aia.plot(marker_x, marker_y, '*', markersize=30, color=track_color, 
                           markeredgecolor='black', markeredgewidth=2, 
                           label=f'FOXES {track_id}' if len(plotted_tracks) <= 5 else None, zorder=15)
                
                ax_aia.annotate(f'FOXES: {foxes_class}', (marker_x, marker_y),
                               xytext=(15, 15), textcoords='offset points',
                               fontsize=11, color='black', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=track_color, 
                                       alpha=0.95, edgecolor='black', linewidth=2))
            else:
                # Smaller colored circle for tracked regions (at peak brightness)
                ax_aia.plot(marker_x, marker_y, 'o', markersize=12, color=track_color, 
                           markeredgecolor='white', markeredgewidth=1.5, alpha=0.8, zorder=12)
        
        # ========== HEK Flare Locations (from HEK catalog) ==========
        # Show HEK events that are currently active (between start and end time)
        # This includes both matched HEK events and HEK-only events (missed by FOXES)
        for idx, hek_event in active_hek_flares.iterrows():
            obs = hek_event.get('obs_observatory', 'Unknown')
            if pd.isna(obs):
                obs = 'Unknown'
            obs = str(obs).strip()
            
            goes_class = hek_event.get('fl_goescls', '')
            if pd.isna(goes_class):
                goes_class = ''
            goes_class = str(goes_class).strip()
            
            # Check if this is a HEK-only event (not matched to FOXES)
            is_hek_only = hek_event.get('match_status', '') == 'hek_only'
            
            # Get HEK coordinates
            hpc_x = hek_event.get('hpc_x', hek_event.get('event_coord1'))
            hpc_y = hek_event.get('hpc_y', hek_event.get('event_coord2'))
            
            x_arcsec, y_arcsec = None, None
            if pd.notna(hpc_x) and pd.notna(hpc_y):
                try:
                    x_arcsec, y_arcsec = float(hpc_x), float(hpc_y)
                except:
                    pass
            
            if x_arcsec is not None and y_arcsec is not None:
                hek_x, hek_y = helioprojective_to_pixel(x_arcsec, y_arcsec)
                if hek_x is not None and hek_y is not None:
                    style = obs_styles.get(obs, obs_styles['Unknown'])
                    label = f'HEK {style["name"]}' if obs not in plotted_obs else None
                    if label:
                        plotted_obs.add(obs)
                    
                    # Visual distinction for HEK-only events: thicker edge and slightly different alpha
                    edge_width = 3.5 if is_hek_only else 2.5
                    marker_alpha = 0.85 if is_hek_only else 1.0
                    marker_style = style['marker']
                    
                    # Large, visible marker
                    ax_aia.scatter(hek_x, hek_y, marker=marker_style,
                                 color=style['color'], s=250,
                                 edgecolors='white', linewidths=edge_width,
                                 label=label, alpha=marker_alpha, zorder=16)
                    
                    # Add a dashed outer ring for HEK-only events to make them more distinct
                    if is_hek_only:
                        circle = plt.Circle((hek_x, hek_y), 15, fill=False,
                                           edgecolor=style['color'], linewidth=2.5,
                                           linestyle='--', alpha=0.8, zorder=15)
                        ax_aia.add_patch(circle)
                    
                    # Add class label (with "HEK-only" indicator if applicable)
                    label_text = f'{obs}: {goes_class}'
                    if is_hek_only:
                        label_text += ' (HEK-only)'
                    
                    if goes_class:
                        ax_aia.annotate(label_text, (hek_x, hek_y),
                                      xytext=(12, -18), textcoords='offset points',
                                      fontsize=10, color='white', weight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3', 
                                              facecolor=style['color'], alpha=0.95, 
                                              edgecolor='white', linewidth=2))
        
        # Also show HEK locations from matched FOXES flares (for completeness)
        for i, (idx, flare) in enumerate(active_foxes_flares.iterrows()):
            if not pd.isna(flare.get("matched_hek_entries")):
                try:
                    hek_entries = json.loads(flare["matched_hek_entries"])
                    for entry in hek_entries:
                        obs = entry.get("observatory", "Unknown").strip() if entry.get("observatory") else "Unknown"
                        goes_class = entry.get("goes_class", "").strip() if entry.get("goes_class") else ""
                        
                        hpc_x = entry.get("hpc_x")
                        hpc_y = entry.get("hpc_y")
                        
                        x_arcsec, y_arcsec = None, None
                        if hpc_x is not None and hpc_y is not None:
                            x_arcsec, y_arcsec = hpc_x, hpc_y
                        else:
                            coords = entry.get("coordinates", "").strip() if entry.get("coordinates") else ""
                            if coords:
                                x_arcsec, y_arcsec = parse_hpc_coordinates(coords)
                        
                        if x_arcsec is not None and y_arcsec is not None:
                            hek_x, hek_y = helioprojective_to_pixel(x_arcsec, y_arcsec)
                            if hek_x is not None and hek_y is not None:
                                style = obs_styles.get(obs, obs_styles['Unknown'])
                                label = f'HEK {style["name"]}' if obs not in plotted_obs else None
                                if label:
                                    plotted_obs.add(obs)
                                
                                ax_aia.scatter(hek_x, hek_y, marker=style['marker'],
                                             color=style['color'], s=200,
                                             edgecolors='white', linewidths=2,
                                             label=label, alpha=1.0, zorder=14)
                                
                                # Add class label
                                if goes_class:
                                    ax_aia.annotate(f'{obs}: {goes_class}', (hek_x, hek_y),
                                                  xytext=(12, -18), textcoords='offset points',
                                                  fontsize=10, color='white', weight='bold',
                                                  bbox=dict(boxstyle='round,pad=0.3', 
                                                          facecolor=style['color'], alpha=0.95, 
                                                          edgecolor='white', linewidth=2))
                except Exception as e:
                    pass
        
        # Legend
        handles, labels_legend = ax_aia.get_legend_handles_labels()
        if handles:
            ax_aia.legend(loc='upper right', fontsize=8, framealpha=0.8)
        
        # Overall title
        num_foxes_active = len(active_foxes_flares) if not active_foxes_flares.empty else 0
        num_hek_active = len(active_hek_flares) if not active_hek_flares.empty else 0
        title = f"Flare Analysis Movie - {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        if num_foxes_active > 0 or num_hek_active > 0:
            title += f" | FOXES: {num_foxes_active}, HEK: {num_hek_active} Active"
        fig.suptitle(title, fontsize=14, y=0.98)
        
        plt.tight_layout()
        
        # Save frame with unique counter
        # Use configurable format and DPI for faster encoding
        frame_format = getattr(config, 'movie_frame_format', 'jpg').lower()
        frame_dpi = getattr(config, 'movie_dpi', 75.0)
        
        if frame_format == 'jpg' or frame_format == 'jpeg':
            frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            # JPEG is much faster than PNG and good enough for video
            # Save as JPEG (matplotlib handles quality internally)
            plt.savefig(frame_path, dpi=frame_dpi, format='jpg')
        else:
            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            plt.savefig(frame_path, dpi=frame_dpi, format='png')
        
        plt.close()
        
        return str(frame_path)
        
    except Exception as e:
        plt.close('all')
        print(f"Error creating frame {frame_idx} for {timestamp}: {e}")
        return None


def _load_flux_contributions_standalone(flux_path: str, timestamp: str, config) -> Optional[np.ndarray]:
    """Load flux contributions for a given timestamp (standalone function for multiprocessing).
    
    Matches the analyzer's load_flux_contributions method: looks for file named exactly {timestamp}
    (no extension) and loads as CSV with np.loadtxt.
    """
    try:
        flux_dir = Path(flux_path)
        flux_file = flux_dir / f"{timestamp}"
        if flux_file.exists():
            return np.loadtxt(flux_file, delimiter=',')
        return None
    except Exception as e:
        # Silently fail - contours just won't be drawn for this frame
        return None


def create_flare_movie(catalog_df: pd.DataFrame, flare_events_df: pd.DataFrame,
                       output_dir: Path, config: FlareAnalysisConfig,
                       predictions_csv: Optional[str] = None,
                       analyzer: Optional[FluxContributionAnalyzer] = None,
                       hek_df: Optional[pd.DataFrame] = None,
                       fps: float = 2.0, frame_interval_minutes: float = 1.0,
                       num_workers: int = 4) -> Optional[str]:
    """
    Create a movie showing SXR data, AIA images with flare locations, and AR contours.
    
    Args:
        catalog_df: DataFrame with FOXES flare catalog events (has start_time, end_time, peak_time)
        flare_events_df: DataFrame with all FOXES flare event data points (track locations over time)
        output_dir: Output directory for movie
        config: Configuration object
        predictions_csv: Path to predictions CSV
        analyzer: FluxContributionAnalyzer instance (for loading flux/AIA data)
        hek_df: DataFrame with HEK flare catalog (event_starttime, event_endtime, event_peaktime)
        fps: Frames per second for movie
        frame_interval_minutes: Time between frames (minutes)
    
    Returns:
        Path to created movie file, or None if failed
    """
    if flare_events_df.empty:
        print("No flare data available for movie creation")
        return None
    
    output_dir = Path(output_dir)
    movie_dir = output_dir / "movies"
    movie_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions for SXR timeseries
    predictions_df = None
    if predictions_csv and Path(predictions_csv).exists():
        try:
            predictions_df = pd.read_csv(predictions_csv)
            if 'datetime' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])
            elif 'timestamp' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['timestamp'])
        except Exception as e:
            print(f"Warning: Could not load predictions CSV: {e}")
    
    # Get all timestamps from flare_events_df
    all_timestamps = sorted(flare_events_df['timestamp'].unique())
    if not all_timestamps:
        print("No timestamps found in flare_events_df")
        return None
    
    # Convert catalog times to datetime (FOXES catalog)
    if catalog_df is not None and not catalog_df.empty:
        catalog_df = catalog_df.copy()
        catalog_df['start_time'] = pd.to_datetime(catalog_df['start_time'])
        catalog_df['end_time'] = pd.to_datetime(catalog_df['end_time'])
        catalog_df['peak_time'] = pd.to_datetime(catalog_df['peak_time'])
    else:
        catalog_df = pd.DataFrame()
    
    # Convert HEK times to datetime
    if hek_df is not None and not hek_df.empty:
        hek_df = hek_df.copy()
        for col in ['event_starttime', 'event_endtime', 'event_peaktime']:
            if col in hek_df.columns:
                hek_df[col] = pd.to_datetime(hek_df[col])
    else:
        hek_df = pd.DataFrame()
    
    # Convert flare_events_df datetime
    flare_events_df = flare_events_df.copy()
    flare_events_df['datetime'] = pd.to_datetime(flare_events_df['datetime'])
    
    # Observatory styles for HEK
    obs_styles = {
        'SDO': {'color': '#9B59B6', 'marker': 's', 'size': 14, 'name': 'SDO'},
        'GOES': {'color': '#E67E22', 'marker': '^', 'size': 14, 'name': 'GOES'},
        'SWPC': {'color': '#E67E22', 'marker': 'D', 'size': 12, 'name': 'SWPC'},
        'Unknown': {'color': 'gray', 'marker': 'o', 'size': 10, 'name': 'Other'}
    }
    
    # Colors for different FOXES tracks (cycle through these)
    foxes_track_colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#1ABC9C', 
                          '#9B59B6', '#34495E', '#16A085', '#27AE60', '#2980B9']
    
    # Filter timestamps by interval
    if frame_interval_minutes > 0:
        timestamps_to_use = []
        last_time = None
        for ts in all_timestamps:
            ts_dt = pd.to_datetime(ts)
            if last_time is None or (ts_dt - last_time).total_seconds() >= frame_interval_minutes * 60:
                timestamps_to_use.append(ts)
                last_time = ts_dt
    else:
        timestamps_to_use = all_timestamps
    
    print(f"Creating movie with {len(timestamps_to_use)} frames...")
    
    # Create temporary directory for frames
    frames_dir = movie_dir / "frames_temp"
    frames_dir.mkdir(exist_ok=True)
    
    # Determine overall time range for SXR plot
    all_datetimes = [pd.to_datetime(ts) for ts in timestamps_to_use]
    overall_start = min(all_datetimes)
    overall_end = max(all_datetimes)
    plot_window_hours = config.plot_window_hours
    
    # Assign colors to FOXES tracks (pre-compute for all workers)
    foxes_track_color_map = {}
    unique_tracks = flare_events_df['track_id'].unique()
    for i, track_id in enumerate(unique_tracks):
        foxes_track_color_map[track_id] = foxes_track_colors[i % len(foxes_track_colors)]
    
    # Get region labels cache from analyzer (if available)
    region_labels_cache = analyzer.region_labels_cache if analyzer else {}
    
    # Prepare shared frame data for multiprocessing
    # Note: We serialize DataFrames to dicts for multiprocessing compatibility
    frame_data = {
        'catalog_df': catalog_df,
        'flare_events_df': flare_events_df,
        'predictions_df': predictions_df,
        'hek_df': hek_df,
        'frames_dir': str(frames_dir),
        'region_labels_cache': region_labels_cache,  # Pre-computed region labels (in-memory)
        'config': config,
        'obs_styles': obs_styles,
        'foxes_track_colors': foxes_track_colors,
        'foxes_track_color_map': foxes_track_color_map,
        'plot_window_hours': plot_window_hours,
        'aia_path': config.aia_path if hasattr(config, 'aia_path') else None,
        'flux_path': config.flux_path if hasattr(config, 'flux_path') else None,
    }
    
    # Create args for each frame
    frame_args = [(i, ts, frame_data) for i, ts in enumerate(timestamps_to_use)]
    
    # Generate frames using multiprocessing
    print(f"Generating {len(timestamps_to_use)} frames using {num_workers} workers...")
    
    frame_paths = []
    if num_workers > 1:
        # Use multiprocessing for parallel frame generation
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(_generate_single_frame, frame_args),
                total=len(frame_args),
                desc="Generating frames"
            ))
        
        # Filter out failed frames (None results)
        frame_paths = [Path(p) for p in results if p is not None]
    else:
        # Single-threaded fallback
        for args in tqdm(frame_args, desc="Generating frames"):
            result = _generate_single_frame(args)
            if result is not None:
                frame_paths.append(Path(result))
    
    # Sort frame paths to ensure correct order
    frame_paths = sorted(frame_paths, key=lambda p: p.name)
    
    if not frame_paths:
        print("No frames generated")
        return None
    
    # Create movie from frames
    print(f"Creating movie from {len(frame_paths)} frames...")
    movie_filename = f"flare_analysis_movie_{all_datetimes[0].strftime('%Y%m%d')}_{all_datetimes[-1].strftime('%Y%m%d')}.mp4"
    movie_path = movie_dir / movie_filename
    
    try:
        import time
        import subprocess
        video_start = time.time()
        
        # Use ffmpeg directly via subprocess - much faster than imageio for image sequences
        # This uses ffmpeg's native image sequence input which is highly optimized
        try:
            # Create a pattern for frame filenames (they should be frame_000000.jpg/png, frame_000001.jpg/png, etc.)
            # Get the first frame to determine the pattern
            first_frame = frame_paths[0]
            frame_ext = first_frame.suffix.lower()  # .jpg or .png
            frame_pattern = str(first_frame.parent / f"frame_%06d{frame_ext}")
            
            # Use ffmpeg directly - much faster for image sequences
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-framerate', str(fps),  # Input framerate
                '-i', frame_pattern,  # Input pattern
                '-c:v', 'libx264',  # Video codec
                '-preset', 'veryfast',  # Encoding speed (veryfast is faster than faster)
                '-crf', '25',  # Quality (23-28 is good, higher = smaller file)
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-threads', '0',  # Use all available CPU threads
                '-movflags', '+faststart',  # Enable fast start for web playback
                str(movie_path)
            ]
            
            print(f"Running ffmpeg to create video...")
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=False  # Don't raise on error, we'll check return code
            )
            
            if result.returncode != 0:
                print(f"Warning: ffmpeg returned code {result.returncode}")
                print(f"stderr: {result.stderr[:500]}")  # Print first 500 chars of error
                # Fall back to imageio method
                print("Falling back to imageio method...")
                raise subprocess.CalledProcessError(result.returncode, ffmpeg_cmd)
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Fallback: Try OpenCV VideoWriter first (faster than imageio), then imageio
            if isinstance(e, FileNotFoundError):
                print("ffmpeg not found, trying OpenCV VideoWriter (faster fallback)...")
            else:
                print("ffmpeg failed, trying OpenCV VideoWriter (faster fallback)...")
            
            # Try OpenCV VideoWriter (often faster than imageio)
            try:
                # Get frame dimensions from first frame
                first_image = cv2.imread(str(frame_paths[0]))
                if first_image is not None:
                    height, width = first_image.shape[:2]
                    
                    # OpenCV VideoWriter with optimized settings
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
                    video_writer = cv2.VideoWriter(
                        str(movie_path),
                        fourcc,
                        fps,
                        (width, height),
                        True  # isColor
                    )
                    
                    if not video_writer.isOpened():
                        raise RuntimeError("Could not open VideoWriter")
                    
                    print(f"Using OpenCV VideoWriter ({width}x{height} @ {fps} fps)...")
                    for frame_path in tqdm(frame_paths, desc="Writing movie"):
                        if frame_path.exists():
                            try:
                                # Read with OpenCV (BGR format)
                                frame = cv2.imread(str(frame_path))
                                if frame is not None:
                                    video_writer.write(frame)
                            except Exception as e:
                                print(f"Warning: Could not read frame {frame_path}: {e}")
                                continue
                    
                    video_writer.release()
                    print("OpenCV encoding complete")
                else:
                    raise RuntimeError("Could not read first frame")
                    
            except Exception as cv_error:
                # Final fallback to imageio
                print(f"OpenCV VideoWriter failed ({cv_error}), using imageio method...")
                
                # Use faster encoding settings: 'veryfast' preset for speed
                with imageio.get_writer(str(movie_path), fps=fps, codec='libx264', format='ffmpeg',
                                        pixelformat='yuv420p',
                                        output_params=['-preset', 'veryfast', '-crf', '25', '-threads', '0']) as writer:
                    for frame_path in tqdm(frame_paths, desc="Writing movie"):
                        if frame_path.exists():
                            try:
                                image = imageio.imread(str(frame_path))
                                writer.append_data(image)
                            except Exception as e:
                                print(f"Warning: Could not read frame {frame_path}: {e}")
                                continue
        
        video_time = time.time() - video_start
        print(f"Video encoding took {video_time:.2f} seconds ({video_time/len(frame_paths):.3f} sec/frame)")
        print(f"✅ Movie saved to: {movie_path}")
        
        # Clean up frames
        cleanup_start = time.time()
        for frame_path in frame_paths:
            if frame_path.exists():
                frame_path.unlink()
        frames_dir.rmdir()
        cleanup_time = time.time() - cleanup_start
        print(f"Temporary frames cleaned up ({cleanup_time:.2f} seconds)")
        
        return str(movie_path)
        
    except Exception as e:
        print(f"Error creating movie: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_hek_only_plots(hek_only_df: pd.DataFrame, output_dir: Path,
                          config: FlareAnalysisConfig,
                          predictions_csv: Optional[str] = None,
                          flare_events_df: Optional[pd.DataFrame] = None) -> None:
    """Create plots for HEK-only events (events in HEK but not detected by FOXES)."""
    if hek_only_df.empty:
        print("No HEK-only events to plot")
        return
    
    output_dir = Path(output_dir)
    hek_only_dir = output_dir / "hek_only"
    hek_only_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    predictions_df = None
    foxes_timestamps = None
    if predictions_csv and Path(predictions_csv).exists():
        try:
            predictions_df = pd.read_csv(predictions_csv)
            if 'datetime' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])
            elif 'timestamp' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['timestamp'])
                predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            
            if 'timestamp' in predictions_df.columns:
                foxes_timestamps = pd.DatetimeIndex(predictions_df['timestamp'])
            elif 'datetime' in predictions_df.columns:
                foxes_timestamps = pd.DatetimeIndex(predictions_df['datetime'])
        except Exception as e:
            print(f"Warning: Could not load predictions CSV: {e}")
    
    # Ensure datetime columns are parsed (filtering already done in main)
    hek_only_df = hek_only_df.copy()
    for col in ["event_starttime", "event_endtime", "event_peaktime"]:
        if col in hek_only_df.columns:
            hek_only_df[col] = pd.to_datetime(hek_only_df[col])
    
    hek_only_df = hek_only_df.sort_values("event_peaktime")
    
    print(f"Creating HEK-only plots for {len(hek_only_df)} events...")
    
    # Observatory styles
    obs_colors = {'SDO': 'purple', 'GOES': 'orange', 'SWPC': 'magenta', 'Unknown': 'gray'}
    
    plotted_peaks = []
    plots_created = 0
    
    for idx, hek_event in hek_only_df.iterrows():
        try:
            peak_time = hek_event.get("event_peaktime")
            if pd.isna(peak_time):
                continue
            
            # Skip duplicates within 5 minutes
            already_plotted = False
            for prev_peak in plotted_peaks:
                if abs((peak_time - prev_peak).total_seconds()) < 300:
                    already_plotted = True
                    break
            
            if already_plotted:
                continue
            
            plotted_peaks.append(peak_time)
            
            start_time = hek_event.get("event_starttime", peak_time - pd.Timedelta(minutes=30))
            end_time = hek_event.get("event_endtime", peak_time + pd.Timedelta(minutes=30))
            obs = hek_event.get("obs_observatory", "Unknown")
            goes_class = hek_event.get("fl_goescls", hek_event.get("goes_class", ""))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            window_start = peak_time - pd.Timedelta(hours=config.plot_window_hours/2)
            window_end = peak_time + pd.Timedelta(hours=config.plot_window_hours/2)
            
            has_data = False
            
            # Plot FOXES tracks in faded grey for context (even though HEK didn't match)
            if flare_events_df is not None and not flare_events_df.empty:
                tracks_in_window = flare_events_df[
                    (flare_events_df["datetime"] >= window_start) & 
                    (flare_events_df["datetime"] <= window_end)
                ]
                
                if not tracks_in_window.empty:
                    plotted_tracks = False
                    for track_id, track_data in tracks_in_window.groupby("track_id"):
                        track_data = track_data.sort_values("datetime")
                        label = 'FOXES Tracks' if not plotted_tracks else None
                        ax.plot(track_data["datetime"], track_data["sum_flux"], 
                                color='grey', linewidth=0.8, alpha=0.4, label=label)
                        plotted_tracks = True
            
            if predictions_df is not None:
                preds_in_window = predictions_df[
                    (predictions_df["timestamp"] >= window_start) &
                    (predictions_df["timestamp"] <= window_end)
                ].copy()
                
                if not preds_in_window.empty:
                    has_data = True
                    if 'groundtruth' in preds_in_window.columns:
                        ax.plot(preds_in_window["timestamp"], preds_in_window['groundtruth'],
                               'k-', linewidth=1.5, alpha=0.8, label='GOES Ground Truth')
                    if 'predictions' in preds_in_window.columns:
                        ax.plot(preds_in_window["timestamp"], preds_in_window['predictions'],
                               'r--', linewidth=1.5, alpha=0.8, label='FOXES Prediction')
            
            # Mark HEK event
            color = obs_colors.get(obs, 'purple')
            ax.axvline(peak_time, color=color, linestyle=':', linewidth=2, label=f'{obs} Peak')
            
            if not pd.isna(start_time) and not pd.isna(end_time):
                ax.axvspan(start_time, end_time, alpha=0.15, color=color, label='HEK Event Duration')
            
            # Add GOES class annotation
            if goes_class:
                y_pos = ax.get_ylim()[1] * 0.9 if has_data else 0.9
                ax.annotate(f'{goes_class}', (peak_time, y_pos),
                           xytext=(0, -10), textcoords='offset points',
                           ha='center', va='top', fontsize=12, color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color))
            
            ax.set_xlim(window_start, window_end)
            if has_data:
                ax.set_yscale('log')
            ax.set_ylabel('Flux', fontsize=12)
            ax.set_xlabel('Time (UTC)', fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            title_parts = ["HEK Only (Not detected by FOXES)",
                          f"Peak: {peak_time.strftime('%Y-%m-%d %H:%M')}"]
            if goes_class:
                title_parts.append(f"Class: {goes_class}")
            if obs:
                title_parts.append(f"Source: {obs}")
            
            fig.suptitle(" | ".join(title_parts), fontsize=14, y=0.95)
            plt.tight_layout()
            
            filename = f"hek_only_{peak_time.strftime('%Y%m%d_%H%M%S')}_{obs}.png"
            plt.savefig(hek_only_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            plots_created += 1
            
        except Exception as e:
            print(f"Error creating HEK-only plot: {e}")
            continue
    
    print(f"Created {plots_created} HEK-only plots in {hek_only_dir}")


# =============================================================================
# Main Workflow
# =============================================================================

def build_flare_catalog(config: FlareAnalysisConfig, output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, "FluxContributionAnalyzer"]:
    """Main orchestration function for building flare catalog.
    
    Args:
        config: Configuration object
        output_dir: Output directory for saving region labels (optional, but needed for movie visualization)
    
    Returns:
        catalog_df: FOXES flare catalog with HEK matching info
        hek_only_df: HEK events not matched to FOXES
        flare_events_df: All FOXES track data points
        simultaneous_flares_df: Simultaneous flare detections
        hek_df: Full HEK catalog (for movie generation)
        analyzer: FluxContributionAnalyzer instance (with region_labels_cache for movie visualization)
    """
    
    # Initialize analyzer (with output_dir for saving region labels during detection)
    analyzer = FluxContributionAnalyzer(config, output_dir=output_dir)
    
    # Step 1: Detect tracked regions
    print("\n=== Step 1: Detecting and tracking regions ===")
    flare_events_df = analyzer.detect_flare_events()
    
    # Ensure flare_events_df is a valid DataFrame with required columns
    if flare_events_df is None:
        flare_events_df = pd.DataFrame()
    if flare_events_df.empty:
        print("Warning: No flare events detected. Creating empty DataFrame with required columns.")
        flare_events_df = pd.DataFrame(columns=['track_id', 'timestamp', 'datetime', 'sum_flux', 
                                                'centroid_img_x', 'centroid_img_y', 'centroid_patch_x', 
                                                'centroid_patch_y', 'region_size', 'max_flux'])

    # Detect simultaneous flares from region-level events (optional)
    if config.enable_simultaneous_flares:
        print("\n=== Step 1b: Detecting simultaneous flares ===")
        simultaneous_flares_df = analyzer.detect_simultaneous_flares(
            threshold=config.simultaneous_flare_threshold,
            sequence_window_hours=config.sequence_window_hours,
        )
    else:
        print("\n=== Step 1b: Simultaneous flare detection disabled in config ===")
        simultaneous_flares_df = pd.DataFrame()
    
    # Step 2: Build individual flare events
    print("\n=== Step 2: Building flare events from tracks ===")
    flare_events = build_flare_events(flare_events_df, config)
    
    # Step 3: Apply quality filters
    print("\n=== Step 3: Applying quality filters ===")
    filtered_events = filter_flare_events(flare_events, config)
    
    # Step 4: Load HEK catalog
    print("\n=== Step 4: Loading HEK catalog ===")
    min_flux = goes_class_to_flux(config.min_goes_class)
    
    hek_save_path = None
    if config.auto_fetch_hek and not config.hek_catalog:
        output_path = Path(config.output_dir or config.data_dir)
        hek_save_path = output_path / f"hek_catalog_{config.start_time[:10]}_{config.end_time[:10]}.csv"
    
    hek_df = load_hek_catalog(
        Path(config.hek_catalog) if config.hek_catalog else None,
        config.start_time,
        config.end_time,
        auto_fetch=config.auto_fetch_hek,
        auto_save_path=hek_save_path,
    )
    
    if hek_df is not None and not hek_df.empty:
        hek_df = normalize_hek_catalog(hek_df, min_flux)
    
    # Step 5: Match to HEK
    print("\n=== Step 5: Matching to HEK catalog ===")
    catalog_df, hek_only_df = match_events_to_hek(
        filtered_events,
        hek_df,
        match_window_minutes=config.hek_match_window_minutes,
        magnitude_tolerance=1.0,  # Fixed internal default
    )
    
    return catalog_df, hek_only_df, flare_events_df, simultaneous_flares_df, hek_df, analyzer


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Flare Analysis")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to flare analysis YAML config file")
    args = parser.parse_args()
    
    # Load config
    config = FlareAnalysisConfig.from_yaml(args.config)
    
    # Create output directory FIRST (needed for saving region labels during detection)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir or config.data_dir) / f"run_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Output Directory ===")
    print(f"Created: {output_dir}")
    
    # Run analysis (pass output_dir for saving region labels)
    catalog_df, hek_only_df, flare_events_df, simultaneous_flares_df, hek_df, analyzer = build_flare_catalog(config, output_dir)
    
    # Save CSVs
    print(f"\n=== Saving CSV Catalogs ===")
    
    if 'match_status' in catalog_df.columns:
        matched_tm = catalog_df[catalog_df['match_status'] == 'matched_time_magnitude']
        matched_to = catalog_df[catalog_df['match_status'] == 'matched_time_only']
        foxes_only = catalog_df[catalog_df['match_status'] == 'foxes_only']
    else:
        matched_tm = pd.DataFrame()
        matched_to = pd.DataFrame()
        foxes_only = catalog_df
    
    # Filter HEK-only events to those with valid coverage BEFORE saving
    hek_only_filtered = hek_only_df.copy()
    if not hek_only_filtered.empty and config.predictions_csv:
        try:
            preds_df = pd.read_csv(config.predictions_csv)
            if 'timestamp' in preds_df.columns:
                foxes_ts = pd.DatetimeIndex(pd.to_datetime(preds_df['timestamp']))
            elif 'datetime' in preds_df.columns:
                foxes_ts = pd.DatetimeIndex(pd.to_datetime(preds_df['datetime']))
            else:
                foxes_ts = None
            
            if foxes_ts is not None and len(foxes_ts) > 0:
                for col in ["event_starttime", "event_endtime", "event_peaktime"]:
                    if col in hek_only_filtered.columns:
                        hek_only_filtered[col] = pd.to_datetime(hek_only_filtered[col])
                
                original_count = len(hek_only_filtered)
                hek_only_filtered = filter_hek_by_foxes_coverage(
                    hek_only_filtered, foxes_ts,
                    max_gap_minutes=config.hek_max_gap_minutes,
                    min_points_required=config.hek_min_points_around_peak,
                    coverage_window_minutes=config.hek_coverage_window_minutes,
                    min_points_before_peak=config.hek_min_points_around_peak,
                    min_points_after_peak=config.hek_min_points_around_peak,
                )
                print(f"  HEK-only filtered: {original_count} -> {len(hek_only_filtered)} (with valid coverage)")
        except Exception as e:
            print(f"  Warning: Could not filter HEK-only events: {e}")
    
    for name, df in [("matched_time_magnitude", matched_tm),
                     ("matched_time_only", matched_to),
                     ("foxes_only", foxes_only),
                     ("hek_only", hek_only_filtered)]:
        if not df.empty:
            path = output_dir / f"flare_events_{name}.csv"
            df.to_csv(path, index=False)
            print(f"  {name}: {len(df)} events -> {path}")
    
    # Save combined FOXES catalog
    if not catalog_df.empty:
        catalog_df.to_csv(output_dir / "flare_events_all_foxes.csv", index=False)

    # Save simultaneous flare summary (region-level) if enabled and present
    if config.enable_simultaneous_flares and simultaneous_flares_df is not None and not simultaneous_flares_df.empty:
        sim_path = output_dir / "simultaneous_flares_summary.csv"
        simultaneous_flares_df.to_csv(sim_path, index=False)
        num_sequences = simultaneous_flares_df["sequence_id"].nunique()
        num_groups = simultaneous_flares_df["group_id"].nunique()
        print(f"\n=== Simultaneous Flare Summary ===")
        print(f"  Timestamps with simultaneous flares: {num_groups}")
        print(f"  Simultaneous flare sequences: {num_sequences}")
        print(f"  Total simultaneous events: {len(simultaneous_flares_df)}")
        print(f"  Saved summary to: {sim_path}")
    
    # Summary
    print(f"\n=== Event Summary ===")
    print(f"  Total FOXES detections: {len(catalog_df)}")
    print(f"    - Matched time + magnitude: {len(matched_tm)}")
    print(f"    - Matched time only: {len(matched_to)}")
    print(f"    - FOXES-only: {len(foxes_only)}")
    print(f"  HEK-only (missed, with coverage): {len(hek_only_filtered)}")
    
    # Create plots
    if config.create_plots:
        print(f"\n=== Creating Plots ===")
        plots_dir = output_dir / "plots"
        create_sxr_plots(
            catalog_df,
            flare_events_df,
            plots_dir,
            config,
            predictions_csv=config.predictions_csv,
        )
        
        # Create HEK-only plots (already filtered)
        if not hek_only_filtered.empty:
            create_hek_only_plots(
                hek_only_filtered,
                plots_dir,
                config,
                predictions_csv=config.predictions_csv,
                flare_events_df=flare_events_df,
            )
    
    # Create movie
    if getattr(config, 'create_movie', False):
        print(f"\n=== Creating Movie ===")
        # Reuse analyzer from detection (has region_labels_cache populated)
        print(f"Using cached region labels ({len(analyzer.region_labels_cache)} timestamps)")
        
        # Combine full HEK catalog with HEK-only events to ensure all are shown
        # (hek_df already contains all events, but hek_only_df has match_status='hek_only' marked)
        # We'll use hek_df for the movie, which includes all events
        movie_hek_df = hek_df.copy() if hek_df is not None and not hek_df.empty else pd.DataFrame()
        
        # If we have HEK-only events, ensure they're in the movie dataframe
        # (they should already be in hek_df, but let's make sure by merging)
        if not hek_only_filtered.empty and not movie_hek_df.empty:
            # Mark HEK-only events in the movie dataframe for potential visual distinction
            if 'match_status' not in movie_hek_df.columns:
                movie_hek_df['match_status'] = 'matched'  # Default
            # Update match_status for HEK-only events
            for idx, hek_only_event in hek_only_filtered.iterrows():
                # Find matching event in movie_hek_df by peak time
                peak_time = hek_only_event.get('event_peaktime')
                if pd.notna(peak_time):
                    mask = movie_hek_df['event_peaktime'] == peak_time
                    if mask.any():
                        movie_hek_df.loc[mask, 'match_status'] = 'hek_only'
        
        print(f"Movie will include {len(movie_hek_df)} HEK events "
              f"({len(hek_only_filtered)} HEK-only, {len(movie_hek_df) - len(hek_only_filtered)} matched)")
        
        movie_path = create_flare_movie(
            catalog_df,
            flare_events_df,
            output_dir,
            config,
            predictions_csv=config.predictions_csv,
            analyzer=analyzer,
            hek_df=movie_hek_df,
            fps=getattr(config, 'movie_fps', 2.0),
            frame_interval_minutes=getattr(config, 'movie_frame_interval_minutes', 1.0),
            num_workers=getattr(config, 'movie_num_workers', 4),
        )
        if movie_path:
            print(f"Movie saved to: {movie_path}")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}")
    
    # Prompt user to delete this run's output directory
    try:
        response = input(f"\nDelete this run's output directory? ({output_dir}) [y/N]: ").strip().lower()
        if response in ('y', 'yes'):
            print(f"Deleting run directory: {output_dir}")
            shutil.rmtree(output_dir)
            print("Run directory deleted.")
        else:
            print("Keeping run directory.")
    except KeyboardInterrupt:
        print("\nCancelled. Keeping run directory.")
    except Exception as e:
        print(f"Warning: Failed to delete run directory {output_dir}: {e}")


if __name__ == "__main__":
    main()

