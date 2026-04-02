#!/usr/bin/env python3
"""
Flare Analysis — Frame & Movie Generator

Detects and tracks active regions from flux contribution maps,
then renders per-timestamp frames and stitches them into a movie.

Usage:
    python flux_map_analysis.py --config flux_map_config.yaml
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from heapq import heappush, heappop
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import rcParams
from scipy.ndimage import maximum_filter, gaussian_filter
from tqdm import tqdm

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FlareAnalysisConfig:
    """Configuration for flare analysis."""

    # Paths
    flux_path: Optional[str] = None
    aia_path: Optional[str] = None
    predictions_csv: Optional[str] = None
    output_dir: Optional[str] = None

    # Time range
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # Detection
    min_flux_threshold: float = 1e-7
    threshold_std_multiplier: float = 3.0
    spatial_smoothing_sigma: float = 1.0
    radial_expansion_threshold_percentile: float = 30.0
    peak_neighborhood_sizes: Tuple[int, ...] = (10, 15, 20, 25)
    peak_min_scale_agreement: int = 2
    peak_scale_tolerance: int = 2
    min_peak_distance: int = 10

    # Grid
    grid_size: Tuple[int, int] = (64, 64)
    patch_size: int = 8
    input_size: int = 512

    # Tracking
    max_tracking_distance: int = 8
    flux_ratio_weight: float = 0.1
    size_ratio_weight: float = 0.1
    distance_weight: float = 1.0
    age_bonus_weight: float = 1.0  # scales 1/(1+age) penalty on new tracks
    cadence_seconds: float = 60.0
    max_gap_frames: int = 1  # frames a track can persist without a detection before expiring

    # Movie / output
    create_movie: bool = False
    plot_window_hours: float = 4.0
    movie_fps: float = 2.0
    movie_frame_interval_minutes: float = 1.0
    movie_num_workers: int = 4
    movie_dpi: float = 75.0
    movie_frame_format: str = 'jpg'
    movie_jpeg_quality: int = 90

    @classmethod
    def from_yaml(cls, path: str) -> "FlareAnalysisConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Flatten one level of nesting
        # The 'movie' section uses short keys (fps, dpi, …) — prefix them with 'movie_'
        flat: Dict = {}
        for key, val in raw.items():
            if isinstance(val, dict):
                if key == 'movie':
                    valid = cls.__dataclass_fields__
                    for k, v in val.items():
                        if k in valid:
                            flat[k] = v
                        elif f'movie_{k}' in valid:
                            flat[f'movie_{k}'] = v
                        else:
                            flat[k] = v
                else:
                    flat.update(val)
            else:
                flat[key] = val

        # Renamed YAML keys
        if 'start' in flat:
            flat['start_time'] = flat.pop('start')
        if 'end' in flat:
            flat['end_time'] = flat.pop('end')

        # Lists → tuples for tuple-typed fields
        for k in ('grid_size', 'peak_neighborhood_sizes'):
            if k in flat and isinstance(flat[k], list):
                flat[k] = tuple(flat[k])

        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in flat.items() if k in valid and v is not None})


# =============================================================================
# Utilities
# =============================================================================

def flux_to_goes_class(flux: float) -> str:
    """Convert physical flux (W/m²) to GOES class string."""
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
    magnitude = min(flux / scale, 9.9)
    return f"{prefix}{magnitude:.1f}" if magnitude != int(magnitude) else f"{prefix}{int(magnitude)}.0"


def setup_barlow_font() -> None:
    """Register and activate the Barlow font if available."""
    try:
        barlow_fonts = [
            (f.name, f.fname) for f in fm.fontManager.ttflist
            if 'barlow' in f.name.lower()
        ]
        if barlow_fonts:
            preferred = next((n for n, _ in barlow_fonts if n.lower() in ('barlow', 'barlow regular')), barlow_fonts[0][0])
            rcParams['font.family'] = preferred
            return

        search_paths = [
            os.path.expanduser('~/Library/Fonts/Barlow-Regular.otf'),
            os.path.expanduser('~/Library/Fonts/Barlow-Regular.ttf'),
            '/Library/Fonts/Barlow-Regular.otf',
            '/usr/share/fonts/truetype/barlow/Barlow-Regular.ttf',
        ]
        for path in search_paths:
            if os.path.exists(path):
                fm.fontManager.addfont(path)
                from matplotlib.font_manager import FontProperties
                rcParams['font.family'] = FontProperties(fname=path).get_name()
                return
    except Exception:
        pass
    rcParams['font.family'] = 'sans-serif'


def load_aia_image_at_time(aia_path: Path, timestamp: str) -> Optional[np.ndarray]:
    """Load AIA image as normalised RGB composite (channels 0, 1, 2 → 94, 131, 171 Å)."""
    if aia_path is None or not aia_path.exists():
        return None

    search_dirs = [aia_path] + [aia_path / s for s in ('test', 'train', 'val') if (aia_path / s).exists()]
    for d in search_dirs:
        fp = d / f"{timestamp}.npy"
        if fp.exists():
            try:
                data = np.load(fp)           # (7, H, W)
                if data.ndim == 3 and data.shape[0] >= 3:
                    rgb = np.zeros((data.shape[1], data.shape[2], 3))
                    for i in range(3):
                        ch = data[i]
                        r = ch.max() - ch.min()
                        rgb[..., i] = (ch - ch.min()) / r if r > 0 else ch
                    return rgb
            except Exception:
                continue
    return None


# =============================================================================
# Region Detection & Tracking
# =============================================================================

class FluxContributionAnalyzer:
    """Detects and tracks active regions from per-patch flux contribution maps."""

    def __init__(self, config: FlareAnalysisConfig, output_dir: Optional[Path] = None):
        self.config = config
        self.flux_path = Path(config.flux_path) if config.flux_path else None
        self.aia_path  = Path(config.aia_path)  if config.aia_path  else None
        self.output_dir = output_dir
        self.grid_size  = config.grid_size
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.region_labels_cache: Dict[str, np.ndarray] = {}

        if config.predictions_csv:
            self.predictions_df = pd.read_csv(config.predictions_csv)
            self.predictions_df['datetime'] = pd.to_datetime(self.predictions_df['timestamp'])
            self.predictions_df = self.predictions_df.sort_values('datetime')
            if config.start_time and config.end_time:
                start, end = pd.to_datetime(config.start_time), pd.to_datetime(config.end_time)
                mask = (self.predictions_df['datetime'] >= start) & (self.predictions_df['datetime'] <= end)
                self.predictions_df = self.predictions_df[mask].reset_index(drop=True)
            print(f"Loaded {len(self.predictions_df)} predictions "
                  f"({self.predictions_df['datetime'].min()} → {self.predictions_df['datetime'].max()})")
        else:
            self.predictions_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_flux_contributions(self, timestamp: str) -> Optional[np.ndarray]:
        if self.flux_path is None:
            return None
        fp = self.flux_path / f"{timestamp}.npy"
        return np.load(fp) if fp.exists() else None

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------

    def _find_flux_peaks_single_scale(self, flux: np.ndarray, size: int) -> Tuple[List, List]:
        valid = np.isfinite(flux) & (flux > 0)
        masked = np.where(valid, flux, -np.inf)
        local_max = (maximum_filter(masked, size=size) == masked) & valid
        ys, xs = np.where(local_max)
        coords = list(zip(ys.tolist(), xs.tolist()))
        fluxes = [float(flux[y, x]) for y, x in coords]
        return coords, fluxes

    def _find_flux_peaks_multiscale(self, flux: np.ndarray) -> Tuple[List, List]:
        cfg = self.config
        registry: Dict[Tuple, dict] = {}

        for size in cfg.peak_neighborhood_sizes:
            coords, fluxes = self._find_flux_peaks_single_scale(flux, size)
            for (y, x), fv in zip(coords, fluxes):
                matched = next(
                    ((py, px) for (py, px) in registry
                     if abs(y - py) <= cfg.peak_scale_tolerance and abs(x - px) <= cfg.peak_scale_tolerance),
                    None
                )
                if matched:
                    e = registry[matched]
                    e['count'] += 1
                    if fv > e['best_flux']:
                        e['best_flux'] = fv
                        e['best_coord'] = (y, x)
                else:
                    registry[(y, x)] = {'count': 1, 'best_flux': fv, 'best_coord': (y, x)}

        stable = [(e['best_coord'], e['best_flux'])
                  for e in registry.values() if e['count'] >= cfg.peak_min_scale_agreement]
        if not stable:
            return [], []

        stable.sort(key=lambda p: p[1], reverse=True)

        coords  = [p[0] for p in stable]
        fluxes  = [p[1] for p in stable]

        if cfg.min_peak_distance > 0 and len(coords) > 1:
            coords, fluxes = self._merge_close_peaks(coords, fluxes, cfg.min_peak_distance)

        return coords, fluxes

    def _merge_close_peaks(self, coords, fluxes, min_dist):
        order = np.argsort(fluxes)[::-1]
        kept = []
        for i in order:
            if all(np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) >= min_dist
                   for j in kept):
                kept.append(i)
        kept = sorted(kept)
        return [coords[i] for i in kept], [fluxes[i] for i in kept]

    # ------------------------------------------------------------------
    # Region segmentation (radial flood-fill from peaks)
    # ------------------------------------------------------------------

    def _detect_regions_with_peak_clustering(
        self, flux_contrib: np.ndarray, pred_data: pd.Series
    ) -> Tuple[List[Dict], Optional[np.ndarray], str]:

        cfg = self.config
        valid = flux_contrib[np.isfinite(flux_contrib) & (flux_contrib > 0)]
        if valid.size == 0:
            return [], None, "no_valid_flux"

        total_flux = float(flux_contrib[flux_contrib > 0].sum())
        log_flux = np.log(valid)
        threshold = max(
            np.exp(np.median(log_flux) + cfg.threshold_std_multiplier * np.std(log_flux)),
            cfg.min_flux_threshold,
        )
        above = int((flux_contrib > threshold).sum())
        masked = np.where(flux_contrib > threshold, flux_contrib, 0.0)

        if above == 0:
            return [], None, f"all_below_threshold(thr={threshold:.3e} total={total_flux:.3e})"

        if cfg.spatial_smoothing_sigma > 0:
            masked = gaussian_filter(masked, sigma=cfg.spatial_smoothing_sigma)

        peak_coords, peak_fluxes = self._find_flux_peaks_multiscale(masked)
        if not peak_coords:
            return [], None, f"no_peaks(thr={threshold:.3e} above={above} total={total_flux:.3e})"

        # Radial flood-fill from all peaks simultaneously (Dijkstra-style)
        labels = np.zeros_like(masked, dtype=np.int32)
        valid_vals = masked[(masked > 0) & np.isfinite(masked)]
        growth_threshold = np.percentile(valid_vals, cfg.radial_expansion_threshold_percentile) if valid_vals.size else 0

        pq, counter = [], 0
        for idx, ((py, px), _) in enumerate(zip(peak_coords, peak_fluxes)):
            labels[py, px] = idx + 1
            heappush(pq, (0.0, counter, py, px, idx + 1, py, px))
            counter += 1

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        H, W = masked.shape
        while pq:
            dist, _, y, x, label, py, px = heappop(pq)
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and labels[ny, nx] == 0 and masked[ny, nx] > growth_threshold:
                    labels[ny, nx] = label
                    new_dist = np.hypot(ny - py, nx - px)
                    heappush(pq, (new_dist, counter, ny, nx, label, py, px))
                    counter += 1

        regions = []
        skipped_below_min = 0
        for lid in range(1, len(peak_coords) + 1):
            mask = labels == lid
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            fv = masked[mask]
            total = float(fv.sum())
            if total < cfg.min_flux_threshold:
                skipped_below_min += 1
                continue
            cy, cx = float(ys.mean()), float(xs.mean())
            peak_y, peak_x = peak_coords[lid - 1]
            regions.append({
                'id': len(regions) + 1,
                'region_label': lid,
                'size': int(ys.size),
                'sum_flux': total,
                'max_flux': float(fv.max()),
                'centroid_patch_y': cy,
                'centroid_patch_x': cx,
                'centroid_img_y': cy * self.patch_size + self.patch_size // 2,
                'centroid_img_x': cx * self.patch_size + self.patch_size // 2,
                'peak_y': peak_y,
                'peak_x': peak_x,
                'peak_img_y': peak_y * self.patch_size + self.patch_size // 2,
                'peak_img_x': peak_x * self.patch_size + self.patch_size // 2,
                'peak_flux': peak_fluxes[lid - 1],
                'mask': mask,
            })

        n_peaks = len(peak_coords)
        reason = (f"ok: {len(regions)} regions from {n_peaks} peaks"
                  f"  thr={threshold:.3e}  above={above}  total={total_flux:.3e}"
                  + (f"  skipped={skipped_below_min}_below_min_flux" if skipped_below_min else ""))
        return regions, labels, reason

    def _detect_regions_worker(self, timestamp: str) -> Tuple[str, Optional[List], Optional[np.ndarray], str]:
        try:
            flux = self.load_flux_contributions(timestamp)
            if flux is None:
                return timestamp, None, None, "no_flux_file"
            pred = self.predictions_df[self.predictions_df['timestamp'] == timestamp]
            if pred.empty:
                return timestamp, None, None, "no_prediction_row"
            regions, labels, reason = self._detect_regions_with_peak_clustering(flux, pred.iloc[0])
            return timestamp, regions, (labels.astype(np.int16) if labels is not None else None), reason
        except Exception as e:
            return timestamp, None, None, f"exception: {e}"

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def track_regions_over_time(self, timestamps: List[str]) -> Dict:
        cfg = self.config
        print("Detecting regions (parallel)…")
        n_workers = max(1, min((os.cpu_count() or 1) - 1, len(timestamps)))

        all_regions: Dict[str, List] = {}
        detection_reasons: Dict[str, str] = {}
        with Pool(processes=n_workers) as pool:
            for ts, regions, labels, reason in tqdm(
                pool.imap(self._detect_regions_worker, timestamps),
                total=len(timestamps), desc="Detecting regions"
            ):
                detection_reasons[ts] = reason
                if regions is not None:
                    all_regions[ts] = regions
                if labels is not None:
                    self.region_labels_cache[ts] = labels

        print("Tracking regions across time…")
        print(f"  max_tracking_distance={cfg.max_tracking_distance}  "
              f"max_gap_frames={cfg.max_gap_frames}  "
              f"age_bonus_weight={cfg.age_bonus_weight}  "
              f"distance_weight={cfg.distance_weight}")
        tracks: Dict[int, List] = {}
        next_id = 1
        last_seen: Dict[int, int] = {}  # track_id → frame index when last matched
        _debug_log: List[str] = []  # per-frame tracking log

        for frame_idx, ts in enumerate(tqdm(timestamps, desc="Tracking")):
            # Expire tracks that haven't been seen within max_gap_frames
            active = {tid for tid, fi in last_seen.items()
                      if frame_idx - fi <= cfg.max_gap_frames}

            if ts not in all_regions:
                det_reason = detection_reasons.get(ts, "unknown")
                _debug_log.append(f"{ts}  SKIP    {det_reason}")
                continue

            current_regions = all_regions[ts]

            # Build all valid (score, region_idx, track_id) candidates
            candidates = []
            for ri, region in enumerate(current_regions):
                cur_flux = region.get('sum_flux', 0.0)
                cur_size = region.get('size', 1)
                for tid in active:
                    history = tracks[tid]
                    # Smooth position over last few frames to reduce centroid jitter.
                    # Use PATCH coordinates so max_tracking_distance is in patch units (matching config).
                    n_smooth = min(5, len(history))
                    avg_x = np.mean([h[1]['centroid_patch_x'] for h in history[-n_smooth:]])
                    avg_y = np.mean([h[1]['centroid_patch_y'] for h in history[-n_smooth:]])
                    dist = np.hypot(
                        region['centroid_patch_x'] - avg_x,
                        region['centroid_patch_y'] - avg_y,
                    )
                    _, last = history[-1]
                    if dist >= cfg.max_tracking_distance:
                        continue
                    lf = last.get('sum_flux', 1e-15)
                    ls = last.get('size', 1)
                    flux_ratio = max(cur_flux, lf) / max(min(cur_flux, lf), 1e-15)
                    size_ratio = max(cur_size, ls) / max(min(cur_size, ls), 1)
                    track_age = len(tracks[tid])
                    # Discount grows with age: 0 (new) → age_bonus_weight (very old)
                    # Makes established tracks harder to beat at equal distance
                    age_discount = cfg.age_bonus_weight * track_age / (1.0 + track_age)
                    score = (cfg.distance_weight * dist
                             + cfg.flux_ratio_weight * flux_ratio
                             + cfg.size_ratio_weight * size_ratio
                             - age_discount)
                    candidates.append((score, ri, tid))

            # Greedy one-to-one assignment: best scores first, each region/track used once
            candidates.sort()
            assigned_regions: set = set()
            assigned_tracks:  set = set()
            assignments: Dict[int, int] = {}   # region_idx → track_id
            for score, ri, tid in candidates:
                if ri in assigned_regions or tid in assigned_tracks:
                    continue
                assignments[ri] = tid
                assigned_regions.add(ri)
                assigned_tracks.add(tid)

            # Log detection outcome for this frame
            det_reason = detection_reasons.get(ts, "unknown")
            _debug_log.append(f"{ts}  DETECT  {det_reason}")

            # Log active-but-unmatched tracks (gaps)
            for tid in active:
                if tid not in assigned_tracks:
                    gap = frame_idx - last_seen.get(tid, frame_idx)
                    cx = tracks[tid][-1][1]['centroid_patch_x']
                    cy = tracks[tid][-1][1]['centroid_patch_y']
                    _debug_log.append(
                        f"{ts}  GAP     track={tid:3d}  age={len(tracks[tid]):4d}  "
                        f"gap_frames={gap:2d}  last_patch=({cx:.1f},{cy:.1f})"
                    )

            # Apply assignments; spawn new track for unmatched regions
            for ri, region in enumerate(current_regions):
                r = region.copy()
                r['timestamp'] = ts
                if ri in assignments:
                    tid = assignments[ri]
                    r['id'] = tid
                    tracks[tid].append((ts, r))
                    cx, cy = r['centroid_patch_x'], r['centroid_patch_y']
                    _debug_log.append(
                        f"{ts}  MATCH   track={tid:3d}  age={len(tracks[tid]):4d}  "
                        f"patch=({cx:.1f},{cy:.1f})  flux={r.get('sum_flux', 0):.3e}"
                    )
                else:
                    r['id'] = next_id
                    tracks[next_id] = [(ts, r)]
                    cx, cy = r['centroid_patch_x'], r['centroid_patch_y']
                    _debug_log.append(
                        f"{ts}  NEW     track={next_id:3d}  age=   1  "
                        f"patch=({cx:.1f},{cy:.1f})  flux={r.get('sum_flux', 0):.3e}"
                    )
                    next_id += 1
                    tid = r['id']
                last_seen[tid] = frame_idx

        tracks = {k: v for k, v in tracks.items() if v}
        print(f"Found {len(tracks)} region tracks across {len(timestamps)} timestamps")

        if self.output_dir and _debug_log:
            log_path = Path(self.output_dir) / "tracking_debug.log"
            with open(log_path, 'w') as f:
                f.write(f"# Tracking log — {len(tracks)} tracks, {len(timestamps)} timestamps\n")
                f.write(f"# max_tracking_distance={cfg.max_tracking_distance}  "
                        f"max_gap_frames={cfg.max_gap_frames}  "
                        f"age_bonus_weight={cfg.age_bonus_weight}\n")
                f.write("#\n# timestamp               event   track  age   detail\n")
                f.write('\n'.join(_debug_log))
            print(f"Tracking debug log → {log_path}")

        return tracks

    def detect_flare_events(self, timestamps: Optional[List[str]] = None) -> pd.DataFrame:
        """Run detection + tracking and return a per-timestamp events DataFrame."""
        if timestamps is None:
            timestamps = self.predictions_df['timestamp'].tolist()

        tracks = self.track_regions_over_time(timestamps)
        rows = []
        for track_id, history in tracks.items():
            for ts, r in history:
                pred = self.predictions_df[self.predictions_df['timestamp'] == ts]
                if pred.empty:
                    continue
                pred = pred.iloc[0]
                rows.append({
                    'timestamp': ts,
                    'datetime': pred['datetime'],
                    'prediction': pred['predictions'],
                    'groundtruth': pred.get('groundtruth', None),
                    'region_size': r.get('size', 0),
                    'sum_flux': r.get('sum_flux', 0.0),
                    'max_flux': r.get('max_flux', 0.0),
                    'mean_flux': r.get('sum_flux', 0.0) / max(r.get('size', 1), 1),
                    'centroid_patch_y': r.get('centroid_patch_y', 0.0),
                    'centroid_patch_x': r.get('centroid_patch_x', 0.0),
                    'centroid_img_y': r.get('centroid_img_y', 0.0),
                    'centroid_img_x': r.get('centroid_img_x', 0.0),
                    'peak_img_y': r.get('peak_img_y', None),
                    'peak_img_x': r.get('peak_img_x', None),
                    'region_label': r.get('region_label', None),
                    'track_id': track_id,
                })

        print(f"Recorded {len(rows)} events from {len(tracks)} tracks")
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# Frame Generation
# =============================================================================

# Colours cycled across FOXES tracks
_TRACK_COLORS = [
    '#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231',
    '#911EB4', '#42D4F4', '#F032E6', '#BFEF45', '#FABED4',
    '#469990', '#DCBEFF', '#9A6324', '#FFFAC8', '#800000',
    '#AAFFC3', '#808000', '#FFD8B1', '#000075', '#A9A9A9',
]


def _generate_single_frame(args: Tuple) -> Optional[str]:
    """Render one frame. Designed for multiprocessing."""
    setup_barlow_font()

    frame_idx, timestamp, fd = args
    try:
        flare_events_df    = fd['flare_events_df']
        predictions_df     = fd['predictions_df']
        region_labels_cache = fd['region_labels_cache']
        config             = fd['config']
        track_color_map    = fd['track_color_map']
        plot_window_hours  = fd['plot_window_hours']
        aia_path           = fd['aia_path']
        frames_dir         = Path(fd['frames_dir'])

        current_time  = pd.to_datetime(timestamp)
        window_start  = current_time - pd.Timedelta(hours=plot_window_hours / 2)
        window_end    = current_time + pd.Timedelta(hours=plot_window_hours / 2)

        # ── Figure: AIA (left) + SXR timeseries (right) ─────────────────────
        fig = plt.figure(figsize=(14, 7))
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3,
                               left=0.07, right=0.97, top=0.93, bottom=0.10)
        ax_aia = fig.add_subplot(gs[0])
        ax_sxr = fig.add_subplot(gs[1])

        # ── AIA image ────────────────────────────────────────────────────────
        aia_image = load_aia_image_at_time(Path(aia_path), timestamp) if aia_path else None
        if aia_image is not None:
            ax_aia.imshow(aia_image, origin='lower', aspect='equal', alpha=0.9)
        else:
            ax_aia.imshow(np.zeros((512, 512, 3)), origin='lower', aspect='equal')

        ax_aia.set_title(f'{current_time.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=11)
        ax_aia.set_xlabel('X (pixels)', fontsize=9)
        ax_aia.set_ylabel('Y (pixels)', fontsize=9)

        # ── Region contours + FOXES markers ──────────────────────────────────
        region_labels = region_labels_cache.get(timestamp)
        current_events = (
            flare_events_df[flare_events_df['timestamp'] == timestamp].copy()
            if not flare_events_df.empty and 'timestamp' in flare_events_df.columns
            else pd.DataFrame()
        )
        plotted_tracks: set = set()

        for _, ev in current_events.iterrows():
            tid = ev['track_id']
            if tid in plotted_tracks:
                continue
            plotted_tracks.add(tid)

            cx, cy = ev.get('centroid_img_x'), ev.get('centroid_img_y')
            if pd.isna(cx) or pd.isna(cy) or not (0 <= cx <= 512) or not (0 <= cy <= 512):
                continue

            px = ev.get('peak_img_x') if pd.notna(ev.get('peak_img_x')) else cx
            py = ev.get('peak_img_y') if pd.notna(ev.get('peak_img_y')) else cy
            color = track_color_map.get(tid, _TRACK_COLORS[0])
            cur_flux  = ev.get('sum_flux', 0.0)
            is_active = cur_flux >= config.min_flux_threshold

            # Contour
            rl = ev.get('region_label')
            if region_labels is not None and pd.notna(rl) and int(rl) > 0:
                region_mask = region_labels == int(rl)
                if np.any(region_mask):
                    try:
                        # Upsample 64×64 mask to 512×512 for crisp contours on AIA image
                        scale = 512 // region_labels.shape[0]
                        mask_up = region_mask.repeat(scale, axis=0).repeat(scale, axis=1).astype(float)
                        ax_aia.contour(mask_up, levels=[0.5],
                                       colors=color, linewidths=4.0 if is_active else 2.5,
                                       alpha=0.9, extent=[0, 512, 0, 512])
                    except Exception:
                        pass

            # Marker
            if is_active:
                ax_aia.plot(px, py, '*', markersize=15, color=color,
                            markeredgecolor='black', markeredgewidth=2, alpha=0.7, zorder=15)
                ax_aia.annotate(f'FOXES: {flux_to_goes_class(cur_flux)}', (px, py),
                                xytext=(15, 15), textcoords='offset points', fontsize=11,
                                color='black', weight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                                          alpha=0.95, edgecolor='black', linewidth=2))
            else:
                ax_aia.plot(px, py, 'o', markersize=10, color=color,
                            markeredgecolor='white', markeredgewidth=1.5, alpha=0.8, zorder=12)

        # ── SXR timeseries ───────────────────────────────────────────────────
        if predictions_df is not None and not predictions_df.empty:
            in_win = predictions_df[
                (predictions_df['datetime'] >= window_start) &
                (predictions_df['datetime'] <= window_end)
            ]
            if not in_win.empty:
                if 'groundtruth' in in_win.columns:
                    ax_sxr.plot(in_win['datetime'], in_win['groundtruth'],
                                'b-', linewidth=1.5, alpha=0.8, label='GOES (Truth)')
                if 'predictions' in in_win.columns:
                    ax_sxr.plot(in_win['datetime'], in_win['predictions'],
                                'r--', linewidth=1.5, alpha=0.8, label='FOXES')

        # Track fluxes
        all_tracks_in_win = (
            flare_events_df[
                (flare_events_df['datetime'] >= window_start) &
                (flare_events_df['datetime'] <= window_end)
            ] if not flare_events_df.empty else pd.DataFrame()
        )
        first_other = True
        for tid, tdata in (all_tracks_in_win.groupby('track_id') if not all_tracks_in_win.empty else []):
            tdata = tdata.sort_values('datetime')
            color = track_color_map.get(tid, _TRACK_COLORS[0])
            is_active = tdata['sum_flux'].max() >= config.min_flux_threshold
            if is_active:
                ax_sxr.plot(tdata['datetime'], tdata['sum_flux'],
                            color=color, linewidth=2.5, alpha=0.9, label=f'Track {tid}', zorder=4)
            else:
                label = 'Other tracks' if first_other else None
                ax_sxr.plot(tdata['datetime'], tdata['sum_flux'],
                            color=color, linewidth=0.9, alpha=0.35, label=label, zorder=3)
                first_other = False

        ax_sxr.axvline(current_time, color='#E5446D', linewidth=2, alpha=0.8, zorder=10)
        ax_sxr.set_xlim(window_start, window_end)
        ax_sxr.set_yscale('log')
        ax_sxr.set_ylabel('Flux (W/m²)', fontsize=9)
        ax_sxr.set_xlabel('Time (UTC)', fontsize=9)
        ax_sxr.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_sxr.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax_sxr.xaxis.get_majorticklabels(), rotation=0)
        ax_sxr.legend(loc='lower right', fontsize=8, framealpha=1)
        ax_sxr.grid(True, alpha=0.3)

        plt.tight_layout()

        fmt = getattr(config, 'movie_frame_format', 'jpg').lower()
        dpi = getattr(config, 'movie_dpi', 75.0)
        ext = 'jpg' if fmt in ('jpg', 'jpeg') else 'png'
        frame_path = frames_dir / f"frame_{frame_idx:06d}.{ext}"
        plt.savefig(frame_path, dpi=dpi, format=ext)
        plt.close()
        return str(frame_path)

    except Exception as e:
        plt.close('all')
        print(f"Error creating frame {frame_idx} ({timestamp}): {e}")
        return None


# =============================================================================
# Movie Assembly
# =============================================================================

def create_flare_movie(
    flare_events_df: pd.DataFrame,
    output_dir: Path,
    config: FlareAnalysisConfig,
    predictions_csv: Optional[str] = None,
    analyzer: Optional[FluxContributionAnalyzer] = None,
    fps: float = 2.0,
    frame_interval_minutes: float = 1.0,
    num_workers: int = 4,
) -> Optional[str]:
    """Generate per-timestamp frames and stitch into an MP4."""
    setup_barlow_font()

    if flare_events_df.empty:
        print("No flare data — skipping movie.")
        return None

    output_dir = Path(output_dir)
    movie_dir  = output_dir / "movies"
    movie_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions for timeseries
    predictions_df = None
    if predictions_csv and Path(predictions_csv).exists():
        predictions_df = pd.read_csv(predictions_csv)
        dt_col = 'datetime' if 'datetime' in predictions_df.columns else 'timestamp'
        predictions_df['datetime'] = pd.to_datetime(predictions_df[dt_col])

    flare_events_df = flare_events_df.copy()
    flare_events_df['datetime'] = pd.to_datetime(flare_events_df['datetime'])

    all_timestamps = sorted(flare_events_df['timestamp'].unique())

    # Subsample by frame_interval_minutes
    timestamps_to_use, last_dt = [], None
    for ts in all_timestamps:
        dt = pd.to_datetime(ts)
        if last_dt is None or (dt - last_dt).total_seconds() >= frame_interval_minutes * 60:
            timestamps_to_use.append(ts)
            last_dt = dt

    print(f"Creating movie: {len(timestamps_to_use)} frames @ {fps} fps")

    # Assign consistent colours to tracks
    unique_tracks = flare_events_df['track_id'].unique()
    track_color_map = {tid: _TRACK_COLORS[i % len(_TRACK_COLORS)] for i, tid in enumerate(unique_tracks)}

    frames_dir = movie_dir / "frames_temp"
    frames_dir.mkdir(exist_ok=True)

    frame_data = {
        'flare_events_df':     flare_events_df,
        'predictions_df':      predictions_df,
        'frames_dir':          str(frames_dir),
        'region_labels_cache': analyzer.region_labels_cache if analyzer else {},
        'config':              config,
        'track_color_map':     track_color_map,
        'plot_window_hours':   config.plot_window_hours,
        'aia_path':            config.aia_path,
    }

    frame_args = [(i, ts, frame_data) for i, ts in enumerate(timestamps_to_use)]

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(_generate_single_frame, frame_args),
                                total=len(frame_args), desc="Generating frames"))
    else:
        results = [_generate_single_frame(a) for a in tqdm(frame_args, desc="Generating frames")]

    frame_paths = sorted(
        (Path(p) for p in results if p is not None),
        key=lambda p: p.name
    )

    if not frame_paths:
        print("No frames generated.")
        return None

    # Stitch into MP4 via imageio (reads frames as RGB → passes to ffmpeg correctly)
    datetimes  = [pd.to_datetime(ts) for ts in timestamps_to_use]
    movie_name = (f"flare_movie_{datetimes[0].strftime('%Y%m%d')}"
                  f"_{datetimes[-1].strftime('%Y%m%d')}.mp4")
    movie_path = movie_dir / movie_name

    # Read first frame to get dimensions
    first_frame = imageio.imread(str(frame_paths[0]))
    h, w = first_frame.shape[:2]
    # yuv420p requires even dimensions
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1

    t0 = time.time()
    writer = imageio_ffmpeg.write_frames(
        str(movie_path),
        size=(w, h),
        fps=fps,
        codec='libx264',
        pix_fmt_in='rgb24',
        pix_fmt_out='yuv420p',
        output_params=['-preset', 'veryfast', '-crf', '25', '-movflags', '+faststart'],
    )
    writer.send(None)  # initialise
    for fp in tqdm(frame_paths, desc="Writing movie"):
        if fp.exists():
            frame = imageio.imread(str(fp))
            writer.send(frame[:h, :w].tobytes())
    writer.close()

    print(f"Movie saved → {movie_path}  ({time.time() - t0:.1f}s)")
    print(f"Frames kept → {frames_dir}")

    return str(movie_path)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Flare Analysis — Frame & Movie Generator")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = FlareAnalysisConfig.from_yaml(args.config)

    run_ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(config.output_dir or '.') / f"run_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    analyzer        = FluxContributionAnalyzer(config, output_dir=out_dir)
    flare_events_df = analyzer.detect_flare_events()

    if not flare_events_df.empty:
        flare_events_df.to_csv(out_dir / "flare_events.csv", index=False)
        print(f"Saved {len(flare_events_df)} events → {out_dir / 'flare_events.csv'}")

    if config.create_movie:
        create_flare_movie(
            flare_events_df  = flare_events_df,
            output_dir       = out_dir,
            config           = config,
            predictions_csv  = config.predictions_csv,
            analyzer         = analyzer,
            fps              = config.movie_fps,
            frame_interval_minutes = config.movie_frame_interval_minutes,
            num_workers      = config.movie_num_workers,
        )

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
