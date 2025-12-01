#!/usr/bin/env python3
"""
Flare catalog construction utilities.

This script runs the FluxContributionAnalyzer on a full prediction archive to
detect flare events, aggregates them into track-based sequences, applies quality
filters (duration, cadence coverage, prominence, data continuity), and optionally
matches the resulting sequences against a NOAA/HEK flare catalog (C5.0+).

Usage example:

python3 flare_catalog_builder.py \
    --config patch_analysis_config_v6.yaml \
    --output-csv /data/FOXES_Data/flare_catalog_v6.csv \
    --min-class C5.0 \
    --start-time 2023-08-01T00:00:00 \
    --end-time 2023-08-15T23:59:59 \
    --auto-fetch-hek
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Ensure repository root is on sys.path when running as a script
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting.inference.patch_analysis_v6 import FluxContributionAnalyzer

try:
    from sunpy.net import Fido
    from sunpy.net import attrs as a
except Exception:  # pragma: no cover - sunpy is optional at runtime
    Fido = None
    a = None


def goes_class_to_flux(goes_class: str) -> float:
    """Convert GOES class string (e.g., C5.0) to physical flux."""
    if not isinstance(goes_class, str):
        return np.nan
    goes_class = goes_class.strip().upper()
    if not goes_class:
        return np.nan
    prefix = goes_class[0]
    scale = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4,
    }.get(prefix)
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
    
    # Determine the class prefix and scale
    if flux >= 1e-4:
        prefix = "X"
        scale = 1e-4
    elif flux >= 1e-5:
        prefix = "M"
        scale = 1e-5
    elif flux >= 1e-6:
        prefix = "C"
        scale = 1e-6
    elif flux >= 1e-7:
        prefix = "B"
        scale = 1e-7
    else:
        prefix = "A"
        scale = 1e-8
    
    # Calculate magnitude
    magnitude = flux / scale
    
    # Format to 1 decimal place, but avoid unnecessary .0
    if magnitude >= 10:
        magnitude = min(magnitude, 9.9)  # Cap at 9.9 to stay within class
    
    if magnitude == int(magnitude):
        return f"{prefix}{int(magnitude)}.0"
    else:
        return f"{prefix}{magnitude:.1f}"


def flux_threshold_from_class(min_class: str) -> float:
    """Translate a minimum GOES class string into a flux threshold."""
    flux = goes_class_to_flux(min_class)
    if math.isnan(flux):
        raise ValueError(f"Invalid GOES class '{min_class}'")
    return flux


def load_hek_catalog(
    catalog_path: Optional[Path],
    start_time: Optional[str],
    end_time: Optional[str],
    auto_fetch: bool = True,
    auto_save_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Load HEK catalog from CSV or fetch via SunPy if requested.
    
    Args:
        catalog_path: Path to pre-downloaded HEK catalog CSV
        start_time: Start time for HEK query (if auto-fetching)
        end_time: End time for HEK query (if auto-fetching)
        auto_fetch: Whether to fetch from HEK if no catalog provided
        auto_save_path: Path to save auto-fetched HEK data
        
    Returns:
        DataFrame with HEK flare entries or None
    """
    if catalog_path:
        df = pd.read_csv(catalog_path)
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        print(f"Loaded HEK catalog with {len(df)} rows from {catalog_path}")
        print(f"Cleaned column names. Sample columns: {list(df.columns)[:5]}")
        return df
    if auto_fetch and start_time and end_time:
        return fetch_hek_flares(start_time, end_time, save_path=auto_save_path)
    return None


def fetch_hek_flares(start_time: str, end_time: str, save_path: Optional[Path] = None) -> pd.DataFrame:
    """Fetch HEK flare catalog via SunPy with all observatories and coordinates.
    
    Args:
        start_time: Start time for HEK query
        end_time: End time for HEK query
        save_path: Optional path to save fetched HEK data as CSV
        
    Returns:
        DataFrame with HEK flare entries
    """
    if Fido is None or a is None:
        raise ImportError("sunpy is required to fetch HEK flare catalog.")
    print(f"Fetching HEK flare catalog from {start_time} to {end_time}...")
    result = Fido.search(
        a.Time(start_time, end_time),
        a.hek.EventType("FL"),
    )
    hek_table = result["hek"]
    
    # Keep scalar columns plus coordinate columns we need
    scalar_cols = [name for name in hek_table.colnames if len(hek_table[name].shape) <= 1]
    coord_cols = [name for name in hek_table.colnames if any(coord_type in name for coord_type in 
                  ['hpc_coord', 'hgc_coord', 'hgs_coord', 'hpc_x', 'hpc_y'])]
    
    # Combine scalar and coordinate columns
    keep_cols = list(set(scalar_cols + coord_cols))
    
    dropped = set(hek_table.colnames) - set(keep_cols)
    if dropped:
        print(f"  Dropping HEK columns with multidimensional entries: {sorted(dropped)}")
    
    # Convert coordinate columns to string format before pandas conversion
    hek_subset = hek_table[keep_cols]
    for coord_col in coord_cols:
        if coord_col in hek_subset.colnames:
            # Convert coordinate objects to string representation
            coord_data = hek_subset[coord_col]
            hek_subset[coord_col] = [str(coord) for coord in coord_data]
    
    df = hek_subset.to_pandas()
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    print(f"Fetched {len(df)} HEK flare entries from all observatories")
    print(f"Available coordinate columns: {[col for col in df.columns if any(coord_type in col for coord_type in ['hpc', 'hgc', 'hgs'])]}")
    
    # Show observatory breakdown
    if 'obs_observatory' in df.columns:
        obs_counts = df['obs_observatory'].value_counts()
        print(f"Observatory breakdown: {dict(obs_counts)}")
    
    # Save fetched HEK data if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved fetched HEK data to {save_path}")
    
    return df


def normalize_hek_catalog(df: pd.DataFrame, min_flux: float) -> pd.DataFrame:
    """Clean and filter HEK catalog entries."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    
    # Normalize time columns
    time_cols = ["event_starttime", "event_peaktime", "event_endtime"]
    for col in time_cols:
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
        # Filter by flux, but keep entries without GOES class for coordinate info
        df = df[(df["peak_flux_wm2"] >= min_flux) | df["goes_class"].isna()].reset_index(drop=True)
    else:
        print("Warning: No GOES class column found in HEK catalog")
        df["goes_class"] = pd.NA
        df["peak_flux_wm2"] = pd.NA
    
    # Handle active region numbers
    if "ar_noaanum" in df.columns:
        df["ar_noaanum"] = df["ar_noaanum"].fillna(-1).astype(int)
    
    return df


def merge_close_peaks(
    flux_series: pd.Series,
    peaks: np.ndarray,
    min_separation_minutes: float = 30.0,
) -> np.ndarray:
    """
    Merge peaks that are too close together, keeping only the highest peak in each group.
    
    Args:
        flux_series: Flux timeseries with datetime index
        peaks: Array of peak indices
        min_separation_minutes: Minimum time separation between peaks
        
    Returns:
        Filtered array of peak indices
    """
    if len(peaks) <= 1:
        return peaks
    
    # Get the datetime index from the flux series
    times = flux_series.index
    if not hasattr(times, 'to_pydatetime'):
        # If index is not datetime, assume it's positional and use track_df times
        return peaks  # Fall back to original peaks if we can't get times
    
    filtered_peaks = []
    i = 0
    
    while i < len(peaks):
        current_peak = peaks[i]
        current_time = times[current_peak]
        current_flux = flux_series.iloc[current_peak]
        
        # Find all peaks within min_separation_minutes
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
        
        # Keep the peak with the highest flux
        max_flux_idx = np.argmax(close_fluxes)
        best_peak = close_peaks[max_flux_idx]
        filtered_peaks.append(best_peak)
        
        # Move to the next group
        i = j
    
    return np.array(filtered_peaks)


def detect_flare_events_in_track(
    track_df: pd.DataFrame,
    min_prominence: float = 2.0,
    min_duration_minutes: float = 10.0,
    smoothing_window: int = 3,
) -> List[Dict[str, object]]:
    """
    Detect individual flare events within a single track using peak detection.
    
    Args:
        track_df: DataFrame with flux timeseries for one track
        min_prominence: Minimum prominence for peak detection
        min_duration_minutes: Minimum duration for a valid flare event
        smoothing_window: Window size for flux smoothing
    
    Returns:
        List of flare event dictionaries
    """
    if track_df.empty or len(track_df) < 3:
        return []
    
    track_df = track_df.sort_values("datetime").reset_index(drop=True)
    
    # Smooth the flux timeseries
    flux_series = track_df["sum_flux"].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    
    # Find peaks using scipy's peak detection
    from scipy.signal import find_peaks
    
    # Calculate baseline (rolling minimum or percentile)
    baseline_window = max(len(flux_series)//4, 1)  # Ensure window is at least 1
    baseline = flux_series.rolling(window=baseline_window, center=True, min_periods=1).quantile(0.2)
    
    # Find peaks with minimum prominence and distance
    min_distance = max(len(flux_series) // 20, 5)  # At least 5% of track length or 5 samples
    
    # Handle None/null prominence - use a very small default if disabled
    effective_prominence = None
    if min_prominence is not None:
        effective_prominence = min_prominence * baseline.median()
    
    peaks, properties = find_peaks(
        flux_series.values,
        prominence=effective_prominence,  # None disables prominence check in find_peaks
        distance=min_distance,  # Minimum distance between peaks
        height=baseline.median() * 1.5,  # Must be above 1.5x baseline
    )
    
    # Additional filtering: merge peaks that are too close in time
    if len(peaks) > 1:
        # Create a temporary series with datetime index for peak merging
        datetime_series = pd.Series(flux_series.values, index=track_df["datetime"])
        peaks = merge_close_peaks(datetime_series, peaks, min_separation_minutes=30)
    
    flare_events = []
    
    for peak_idx in peaks:
        peak_time = track_df.iloc[peak_idx]["datetime"]
        peak_flux = flux_series.iloc[peak_idx]
        
        # Find flare start and end by looking for where flux drops to baseline level
        baseline_threshold = baseline.iloc[peak_idx] + 0.5 * (peak_flux - baseline.iloc[peak_idx])
        
        # Find start (going backwards from peak)
        start_idx = peak_idx
        at_data_start = False  # NEW: Track if we hit data boundary
        for i in range(peak_idx - 1, -1, -1):
            if flux_series.iloc[i] < baseline_threshold:
                start_idx = i + 1
                break
        else:
            # NEW: Reached beginning of data without finding start - this is a boundary event
            start_idx = 0
            at_data_start = True
        
        # Find end (going forwards from peak)
        end_idx = peak_idx
        at_data_end = False  # NEW: Track if we hit data boundary
        for i in range(peak_idx + 1, len(flux_series)):
            if flux_series.iloc[i] < baseline_threshold:
                end_idx = i - 1
                break
        else:
            # NEW: Reached end of data without finding end - this is a boundary event
            end_idx = len(track_df) - 1
            at_data_end = True
        
        # Ensure we have valid indices
        start_idx = max(0, start_idx)
        end_idx = min(len(track_df) - 1, end_idx)
        
        if start_idx >= end_idx:
            continue
            
        # Extract flare event data
        event_df = track_df.iloc[start_idx:end_idx + 1]
        
        start_time = event_df["datetime"].iloc[0]
        end_time = event_df["datetime"].iloc[-1]
        duration_minutes = (end_time - start_time).total_seconds() / 60.0
        
        # NEW: Be more lenient with duration for boundary events
        is_boundary_event = at_data_start or at_data_end
        effective_min_duration = min_duration_minutes / 2.0 if is_boundary_event else min_duration_minutes
        
        # Skip events that are too short (with lenient threshold for boundary events)
        if duration_minutes < effective_min_duration:
            continue
        
        # Calculate event metrics
        peak_row = track_df.iloc[peak_idx]
        median_flux = event_df["sum_flux"].median()
        prominence = (peak_flux - median_flux) / max(median_flux, 1e-8)
        
        rise_time_minutes = (peak_time - start_time).total_seconds() / 60.0
        decay_time_minutes = (end_time - peak_time).total_seconds() / 60.0
        
        flare_events.append({
            "track_id": int(track_df["track_id"].iloc[0]),
            "event_start_idx": start_idx,
            "event_end_idx": end_idx,
            "event_peak_idx": peak_idx,
            "start_time": start_time,
            "end_time": end_time,
            "peak_time": peak_time,
            "duration_minutes": duration_minutes,
            "num_samples": len(event_df),
            "peak_sum_flux": peak_flux,
            "peak_max_flux": peak_row.get("max_flux", np.nan),
            "peak_region_size": peak_row.get("region_size", np.nan),
            "median_sum_flux": median_flux,
            "baseline_flux": baseline.iloc[peak_idx],
            "prominence": prominence,
            "rise_time_minutes": rise_time_minutes,
            "decay_time_minutes": decay_time_minutes,
            "peak_centroid_img_x": peak_row.get("centroid_img_x", np.nan),
            "peak_centroid_img_y": peak_row.get("centroid_img_y", np.nan),
            "is_boundary_event": is_boundary_event,  # NEW: Flag boundary events
            "at_data_start": at_data_start,          # NEW
            "at_data_end": at_data_end,              # NEW
        })
    
    return flare_events


def split_track_sequences(
    track_df: pd.DataFrame,
    max_gap_minutes: float,
) -> List[pd.DataFrame]:
    """Split a single track DataFrame into contiguous sequences."""
    if track_df.empty:
        return []
    sequences: List[pd.DataFrame] = []
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


def summarize_sequence(
    sequence_df: pd.DataFrame,
    cadence_seconds: float,
) -> Dict[str, object]:
    """Compute summary metrics for a contiguous flare sequence."""
    sequence_df = sequence_df.sort_values("datetime").reset_index(drop=True)
    start_time = sequence_df["datetime"].iloc[0]
    end_time = sequence_df["datetime"].iloc[-1]
    duration_seconds = (end_time - start_time).total_seconds()
    duration_minutes = max(duration_seconds / 60.0, 0.0)
    num_samples = len(sequence_df)
    expected_samples = max(duration_seconds / cadence_seconds, 1.0)
    coverage_ratio = num_samples / expected_samples
    time_diffs = (
        sequence_df["datetime"]
        .diff()
        .dt.total_seconds()
        .dropna()
    )
    max_gap_minutes = (time_diffs.max() if not time_diffs.empty else 0.0) / 60.0
    peak_idx = sequence_df["sum_flux"].idxmax()
    peak_row = sequence_df.loc[peak_idx]
    median_flux = sequence_df["sum_flux"].median()
    prominence = (
        (peak_row["sum_flux"] - median_flux)
        / max(median_flux, 1e-8)
        if not np.isnan(median_flux)
        else np.nan
    )
    rise_time_minutes = (
        (peak_row["datetime"] - start_time).total_seconds() / 60.0
    )
    decay_time_minutes = (
        (end_time - peak_row["datetime"]).total_seconds() / 60.0
    )
    return {
        "track_id": int(sequence_df["track_id"].iloc[0]),
        "start_time": start_time,
        "end_time": end_time,
        "peak_time": peak_row["datetime"],
        "duration_minutes": duration_minutes,
        "num_samples": num_samples,
        "expected_samples": expected_samples,
        "coverage_ratio": coverage_ratio,
        "max_gap_minutes": max_gap_minutes,
        "peak_sum_flux": peak_row["sum_flux"],
        "peak_max_flux": peak_row.get("max_flux", np.nan),
        "peak_region_size": peak_row.get("region_size", np.nan),
        "median_sum_flux": median_flux,
        "prominence": prominence,
        "rise_time_minutes": rise_time_minutes,
        "decay_time_minutes": decay_time_minutes,
    }


def build_flare_events(
    flare_events_df: pd.DataFrame,
    cadence_seconds: float,
    max_gap_minutes: float,
    min_prominence: float = 2.0,
    min_duration_minutes: float = 10.0,
) -> List[Dict[str, object]]:
    """Detect individual flare events within each track."""
    all_events: List[Dict[str, object]] = []
    
    # Track filtering statistics for debugging
    filter_stats = {
        'total_tracks': 0,
        'total_sequences': 0,
        'skipped_too_few_samples': 0,
        'skipped_no_peaks': 0,
        'tracks_with_events': 0,
    }
    
    for track_id, track_df in flare_events_df.groupby("track_id"):
        filter_stats['total_tracks'] += 1
        track_df = track_df.sort_values("datetime")
        
        # Split track into contiguous sequences first (handle data gaps)
        sequences = split_track_sequences(track_df, max_gap_minutes)
        
        track_had_events = False
        for seq_idx, seq_df in enumerate(sequences):
            filter_stats['total_sequences'] += 1
            
            # Lowered threshold from 3 to 2 to catch more short-lived regions
            if len(seq_df) < 2:
                filter_stats['skipped_too_few_samples'] += 1
                print(f"  [FILTER] Track {track_id} seq {seq_idx}: skipped - only {len(seq_df)} sample(s) (need >= 2)")
                continue
                
            # Detect flare events within this sequence
            events = detect_flare_events_in_track(
                seq_df,
                min_prominence=min_prominence,
                min_duration_minutes=min_duration_minutes,
            )
            
            if len(events) == 0:
                filter_stats['skipped_no_peaks'] += 1
                # Log why no peaks were found
                flux_range = seq_df["sum_flux"].max() - seq_df["sum_flux"].min()
                flux_median = seq_df["sum_flux"].median()
                time_span = (seq_df["datetime"].max() - seq_df["datetime"].min()).total_seconds() / 60.0
                print(f"  [FILTER] Track {track_id} seq {seq_idx}: no peaks detected - "
                      f"{len(seq_df)} samples, {time_span:.1f} min span, "
                      f"flux range={flux_range:.2e}, median={flux_median:.2e}")
                continue
            
            track_had_events = True
            
            # Add sequence metadata to each event
            for event in events:
                event["sequence_id"] = f"{track_id}_{seq_idx}"
                event["sequence_start_time"] = seq_df["datetime"].iloc[0]
                event["sequence_end_time"] = seq_df["datetime"].iloc[-1]
                event["sequence_samples"] = len(seq_df)
                event["sequence_start_timestamp"] = seq_df["timestamp"].iloc[0]
                event["sequence_end_timestamp"] = seq_df["timestamp"].iloc[-1]
                
                # Calculate coverage metrics for the event
                duration_seconds = event["duration_minutes"] * 60.0
                expected_samples = max(duration_seconds / cadence_seconds, 1.0)
                event["expected_samples"] = expected_samples
                event["coverage_ratio"] = event["num_samples"] / expected_samples
                
                all_events.append(event)
        
        if track_had_events:
            filter_stats['tracks_with_events'] += 1
    
    # Print summary statistics
    print(f"\n=== Flare Event Detection Summary ===")
    print(f"  Total tracks analyzed: {filter_stats['total_tracks']}")
    print(f"  Total sequences (after gap splitting): {filter_stats['total_sequences']}")
    print(f"  Sequences skipped (< 2 samples): {filter_stats['skipped_too_few_samples']}")
    print(f"  Sequences skipped (no peaks detected): {filter_stats['skipped_no_peaks']}")
    print(f"  Tracks with at least one event: {filter_stats['tracks_with_events']}")
    print(f"  Total flare events detected: {len(all_events)}")
    print(f"=====================================\n")
    
    return all_events


def filter_flare_events(
    events: List[Dict[str, object]],
    min_duration_minutes: float,
    min_samples: int,
    min_coverage: float,
    min_prominence: float,
    min_peak_flux: float,
) -> pd.DataFrame:
    """Apply quality gates to individual flare events."""
    records = []
    
    # Track rejection reasons for debugging
    rejection_stats = {
        'duration': 0,
        'samples': 0,
        'coverage': 0,
        'prominence': 0,
        'peak_flux': 0,
    }
    
    print(f"\n=== Quality Filter Debug (thresholds) ===")
    print(f"  min_duration_minutes: {min_duration_minutes}")
    print(f"  min_samples: {min_samples}")
    print(f"  min_coverage: {min_coverage}")
    print(f"  min_prominence: {min_prominence}")
    print(f"  min_peak_flux: {min_peak_flux}")
    print(f"=========================================\n")
    
    for event in events:
        track_id = event.get("track_id", "?")
        peak_time = event.get("peak_time", "?")
        
        if event["duration_minutes"] < min_duration_minutes:
            print(f"  [QUALITY] Track {track_id} @ {peak_time}: REJECTED - duration {event['duration_minutes']:.1f} min < {min_duration_minutes} min")
            rejection_stats['duration'] += 1
            continue
        if event["num_samples"] < min_samples:
            print(f"  [QUALITY] Track {track_id} @ {peak_time}: REJECTED - samples {event['num_samples']} < {min_samples}")
            rejection_stats['samples'] += 1
            continue
        if event["coverage_ratio"] < min_coverage:
            print(f"  [QUALITY] Track {track_id} @ {peak_time}: REJECTED - coverage {event['coverage_ratio']:.2f} < {min_coverage}")
            rejection_stats['coverage'] += 1
            continue
        if min_prominence is not None and not math.isnan(min_prominence) and event.get("prominence", 0.0) < min_prominence:
            print(f"  [QUALITY] Track {track_id} @ {peak_time}: REJECTED - prominence {event.get('prominence', 0.0):.4f} < {min_prominence}")
            rejection_stats['prominence'] += 1
            continue
        if min_peak_flux is not None and not math.isnan(min_peak_flux) and event.get("peak_sum_flux", 0.0) < min_peak_flux:
            print(f"  [QUALITY] Track {track_id} @ {peak_time}: REJECTED - peak_flux {event.get('peak_sum_flux', 0.0):.2e} < {min_peak_flux:.2e}")
            rejection_stats['peak_flux'] += 1
            continue
        
        print(f"  [QUALITY] Track {track_id} @ {peak_time}: PASSED - duration={event['duration_minutes']:.1f}min, samples={event['num_samples']}, prominence={event.get('prominence', 0.0):.4f}, peak_flux={event.get('peak_sum_flux', 0.0):.2e}")
        records.append(event)
    
    df = pd.DataFrame(records)
    
    print(f"\n=== Quality Filter Summary ===")
    print(f"  Input events: {len(events)}")
    print(f"  Rejected - duration too short: {rejection_stats['duration']}")
    print(f"  Rejected - too few samples: {rejection_stats['samples']}")
    print(f"  Rejected - low coverage: {rejection_stats['coverage']}")
    print(f"  Rejected - low prominence: {rejection_stats['prominence']}")
    print(f"  Rejected - low peak flux: {rejection_stats['peak_flux']}")
    print(f"  PASSED quality filter: {len(df)}")
    print(f"==============================\n")
    
    return df




def compare_flare_magnitude(
    foxes_flux: float,
    hek_goes_class: str,
    tolerance_class_levels: float = 1.0,
) -> Tuple[bool, str, str]:
    """
    Compare FOXES flux to HEK GOES class to determine if magnitudes match.
    
    Args:
        foxes_flux: FOXES peak_sum_flux value
        hek_goes_class: HEK GOES class string (e.g., "C5.0", "M1.2")
        tolerance_class_levels: How many class levels difference is acceptable
                               (1.0 = within same order of magnitude, e.g., C5 vs C8)
                               
    Returns:
        Tuple of (magnitude_matches, foxes_class, hek_class)
    """
    # Convert FOXES flux to GOES class
    foxes_class = flux_to_goes_class(foxes_flux)
    
    # Get HEK flux
    hek_flux = goes_class_to_flux(hek_goes_class)
    
    # If either is invalid, can't compare
    if foxes_class == "N/A" or np.isnan(hek_flux) or hek_flux <= 0:
        return False, foxes_class, hek_goes_class if hek_goes_class else "N/A"
    
    # Get FOXES flux value
    foxes_flux_val = goes_class_to_flux(foxes_class)
    if np.isnan(foxes_flux_val) or foxes_flux_val <= 0:
        return False, foxes_class, hek_goes_class
    
    # Compare using log ratio - within tolerance_class_levels orders of magnitude
    # One GOES class level = factor of 10 (e.g., C1 to M1)
    # Within same letter class, numbers differ by factor up to 10 (e.g., C1 to C9.9)
    log_ratio = abs(np.log10(foxes_flux_val) - np.log10(hek_flux))
    
    # tolerance_class_levels of 1.0 means within same major class (e.g., both C-class)
    # tolerance_class_levels of 0.5 means within half a class (e.g., C5 vs C1.5 to C5*3)
    magnitude_matches = log_ratio <= tolerance_class_levels
    
    return magnitude_matches, foxes_class, hek_goes_class


def match_events_to_hek_multi_obs(
    events_df: pd.DataFrame,
    hek_df: pd.DataFrame,
    match_window_minutes: float,
    magnitude_tolerance: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Associate each FOXES event with HEK events from multiple observatories,
    and also return unmatched HEK events.
    
    Match status categories:
    - "matched_time_magnitude": Time AND magnitude match (true detection)
    - "matched_time_only": Time matches but magnitude differs (possible simultaneous flare separation)
    - "foxes_only": No HEK event at this time
    
    Args:
        events_df: DataFrame of FOXES-detected events
        hek_df: DataFrame of HEK catalog events
        match_window_minutes: Time tolerance for matching
        magnitude_tolerance: GOES class levels tolerance for magnitude matching
                            (1.0 = within same major class, e.g., both C-class)
        
    Returns:
        Tuple of (matched_events_df, hek_only_events_df)
        - matched_events_df: FOXES events with matched_hek_entries column and match_status
        - hek_only_events_df: HEK events not matched to any FOXES detection
    """

    import json

    print(f"\n=== HEK Matching Debug ===")
    print(f"Events DF: {len(events_df)} rows, empty={events_df.empty}")
    print(f"HEK DF is None: {hek_df is None}")
    if hek_df is not None:
        print(f"HEK DF: {len(hek_df)} rows, empty={hek_df.empty}")
        print(f"HEK columns (first 10): {list(hek_df.columns)[:10]}")
    
    # Handle edge cases
    if events_df.empty and (hek_df is None or hek_df.empty):
        print("Early exit: both events and HEK df are empty/None")
        events_df["matched_hek_entries"] = pd.NA
        events_df["match_status"] = "foxes_only"
        return events_df, pd.DataFrame()
    
    if events_df.empty:
        print("No FOXES events - all HEK events are unmatched")
        events_df["matched_hek_entries"] = pd.NA
        events_df["match_status"] = "foxes_only"
        return events_df, hek_df.copy() if hek_df is not None else pd.DataFrame()
    
    if hek_df is None or hek_df.empty:
        print("No HEK events - all FOXES events are unmatched")
        events_df = events_df.copy()
        events_df["matched_hek_entries"] = pd.NA
        events_df["match_status"] = "foxes_only"
        return events_df, pd.DataFrame()

    events_df = events_df.copy()
    hek_df = hek_df.copy()
    
    # Ensure HEK time columns are datetime
    time_cols = ["event_starttime", "event_endtime", "event_peaktime"]
    for col in time_cols:
        if col in hek_df.columns:
            print(f"Converting {col} to datetime...")
            hek_df[col] = pd.to_datetime(hek_df[col])

    tolerance = pd.Timedelta(minutes=match_window_minutes)
    matched_entries: List[Optional[str]] = []
    match_status: List[str] = []
    foxes_goes_class: List[str] = []
    hek_goes_class_list: List[str] = []
    magnitude_match_list: List[bool] = []

    # Track which HEK entries have been matched
    hek_df["__matched"] = False
    
    # Counters for different match types
    num_time_mag_matched = 0
    num_time_only_matched = 0

    print(f"\nStarting matching with tolerance: ±{match_window_minutes} minutes")
    print(f"Magnitude tolerance: {magnitude_tolerance} class levels")
    if len(events_df) > 0:
        print(f"Sample FOXES event: {events_df[['start_time', 'peak_time', 'end_time']].iloc[0].to_dict()}")
    if len(hek_df) > 0:
        time_cols_available = [col for col in ['event_starttime', 'event_peaktime', 'event_endtime'] if col in hek_df.columns]
        if time_cols_available:
            print(f"Sample HEK event: {hek_df[time_cols_available].iloc[0].to_dict()}")
        else:
            print(f"Sample HEK event (first 5 columns): {hek_df.iloc[0, :5].to_dict()}")

    num_matched = 0
    # Keep track of matched HEK row indices to find unmatched HEK at the end
    matched_hek_indices = set()

    for idx, event in events_df.iterrows():
        event_start = event["start_time"]
        event_end = event["end_time"]
        event_peak = event["peak_time"]
        
        window_start = event_start - tolerance
        window_end = event_end + tolerance

        # Find overlapping HEK events
        overlapping = hek_df[
            (hek_df["event_endtime"] >= window_start)
            & (hek_df["event_starttime"] <= window_end)
        ]

        if overlapping.empty:
            matched_entries.append(pd.NA)
            match_status.append("foxes_only")
            foxes_goes_class.append(flux_to_goes_class(event.get("peak_sum_flux", 0)))
            hek_goes_class_list.append(pd.NA)
            magnitude_match_list.append(False)
            continue

        num_matched += 1
        # Mark these HEK entries as matched
        hek_df.loc[overlapping.index, "__matched"] = True
        matched_hek_indices.update(overlapping.index)

        # Group by peak time to find events from different observatories
        event_groups = overlapping.groupby("event_peaktime")

        # Find the group with peak time closest to FOXES event
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
                ar_num = hek_event.get("ar_noaanum", "")
                obs = obs.strip() if isinstance(obs, str) else str(obs)
                goes_class = goes_class.strip() if isinstance(goes_class, str) else str(goes_class)
                coords = coords.strip() if isinstance(coords, str) else str(coords)
                
                # Track the HEK GOES class (prefer GOES/SWPC source)
                if goes_class and (best_hek_class is None or obs in ['GOES', 'SWPC']):
                    best_hek_class = goes_class
                
                entry_info = {
                    "observatory": obs,
                    "goes_class": goes_class,
                    "coordinates": coords,
                    "ar_number": ar_num,
                    "peak_time": str(hek_event.get("event_peaktime", "")).strip(),
                    "start_time": str(hek_event.get("event_starttime", "")).strip(),
                    "end_time": str(hek_event.get("event_endtime", "")).strip()
                }
                entries.append(entry_info)
            
            matched_entries.append(json.dumps(entries))
            
            # Compare magnitudes
            foxes_flux = event.get("peak_sum_flux", 0)
            mag_matches, foxes_class_str, hek_class_str = compare_flare_magnitude(
                foxes_flux, best_hek_class or "", magnitude_tolerance
            )
            
            foxes_goes_class.append(foxes_class_str)
            hek_goes_class_list.append(hek_class_str)
            magnitude_match_list.append(mag_matches)
            
            if mag_matches:
                match_status.append("matched_time_magnitude")
                num_time_mag_matched += 1
            else:
                match_status.append("matched_time_only")
                num_time_only_matched += 1
            
            if num_matched <= 3:  # Show details for first 3 matches
                print(f"\nMatch #{num_matched}: FOXES event at {event_peak}")
                print(f"  FOXES class: {foxes_class_str}, HEK class: {hek_class_str}")
                print(f"  Magnitude match: {mag_matches}")
                print(f"  Found {len(overlapping)} overlapping HEK events")
                if len(overlapping) > 0:
                    obs_list = overlapping['obs_observatory'].unique() if 'obs_observatory' in overlapping.columns else ['Unknown']
                    print(f"  Observatories: {list(obs_list)}")
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

    # Get unmatched HEK events (HEK-only events not associated with any FOXES detection)
    unmatched_hek_df = hek_df[~hek_df["__matched"]].copy()
    
    # Remove temporary column
    if "__matched" in unmatched_hek_df.columns:
        unmatched_hek_df.drop(columns="__matched", inplace=True)
    if "__matched" in hek_df.columns:
        hek_df.drop(columns="__matched", inplace=True)
    
    # Add match_status to unmatched HEK events
    if not unmatched_hek_df.empty:
        unmatched_hek_df["match_status"] = "hek_only"

    print(f"\n=== Matching Summary ===")
    print(f"Total FOXES events: {len(events_df)}")
    print(f"  - Matched time + magnitude: {num_time_mag_matched}")
    print(f"  - Matched time only: {num_time_only_matched}")
    print(f"  - FOXES-only (no HEK match): {len(events_df) - num_matched}")
    print(f"Total HEK events: {len(hek_df)}")
    print(f"  - HEK-only (no FOXES match): {len(unmatched_hek_df)}")
    if len(events_df) > 0:
        print(f"Match rate (FOXES): {num_matched / len(events_df) * 100:.1f}%")
    if len(hek_df) > 0:
        print(f"Match rate (HEK): {(len(hek_df) - len(unmatched_hek_df)) / len(hek_df) * 100:.1f}%")

    if not unmatched_hek_df.empty:
        print(f"\nFirst 3 HEK-only events:")
        sample = unmatched_hek_df.head(3)
        for _, row in sample.iterrows():
            obs = row.get("obs_observatory", "Unknown")
            cls = row.get("fl_goescls", "")
            coords = row.get("hpc_coord", "")
            print(f'  [{row.get("event_peaktime", "")}] {obs} {cls}  {coords}')
    
    return events_df, unmatched_hek_df


def load_catalog_config(config_path: str) -> Dict:
    """Load flare catalog configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_flare_catalog(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """Main orchestration function.
    
    Returns:
        Tuple of (catalog_df, hek_only_df, flare_events_df, analyzer)
        - catalog_df: FOXES events with HEK matches and match_status column
        - hek_only_df: HEK events not matched to any FOXES detection
        - flare_events_df: Raw track data from analyzer
        - analyzer: The FluxContributionAnalyzer instance
    """
    config_time_period = None
    analyzer = FluxContributionAnalyzer(
        config_path=config['paths']['config'],
        flux_path=config['paths']['flux_path'],
        predictions_csv=config['paths']['predictions_csv'],
        aia_path=config['paths']['aia_path'],
    )
    if analyzer.time_period:
        config_time_period = (
            analyzer.time_period["start_time"],
            analyzer.time_period["end_time"],
        )
    start_time = config['time_range']['start_time'] or (config_time_period[0] if config_time_period else None)
    end_time = config['time_range']['end_time'] or (config_time_period[1] if config_time_period else None)
    
    # Step 1: Get tracked regions from FOXES detector
    flare_events_df = analyzer.detect_flare_events()
    
    # Step 2: Detect individual flare events within each track
    flare_events = build_flare_events(
        flare_events_df,
        cadence_seconds=config['detection']['cadence_seconds'],
        max_gap_minutes=config['detection']['max_gap_minutes'],
        min_prominence=config['detection']['min_prominence'],
        min_duration_minutes=config['detection']['min_duration_minutes'],
    )
    
    # Step 3: Apply quality filters to individual events
    filtered_events = filter_flare_events(
        flare_events,
        min_duration_minutes=config['detection']['min_duration_minutes'],
        min_samples=config['detection']['min_samples'],
        min_coverage=config['detection']['min_coverage'],
        min_prominence=config['detection']['min_prominence'],
        min_peak_flux=config['detection']['min_peak_sum_flux'],
    )
    
    # Step 4: Load and normalize HEK catalog (with auto-save if fetching)
    min_flux = flux_threshold_from_class(config['detection']['min_goes_class'])
    
    # Determine save path for auto-fetched HEK data
    hek_auto_save_path = None
    if config['time_range'].get('auto_fetch_hek') and not config['paths'].get('hek_catalog'):
        # Save to same directory as output_csv with descriptive name
        output_path = Path(config['paths']['output_csv'])
        hek_auto_save_path = output_path.parent / f"hek_catalog_{start_time[:10]}_{end_time[:10]}.csv"
    
    hek_df = load_hek_catalog(
        Path(config['paths']['hek_catalog']) if config['paths']['hek_catalog'] else None,
        start_time,
        end_time,
        auto_fetch=config['time_range']['auto_fetch_hek'],
        auto_save_path=hek_auto_save_path,
    )
    if hek_df is not None and not hek_df.empty:
        print(f"HEK columns before normalization: {list(hek_df.columns)[:10]}...")
        hek_df = normalize_hek_catalog(hek_df, min_flux)
        print(f"HEK columns after normalization: {list(hek_df.columns)[:10]}...")
        print(f"HEK shape after normalization: {hek_df.shape}")
    
    # Step 5: Match individual FOXES events to HEK events (multi-observatory)
    catalog_df, hek_only_df = match_events_to_hek_multi_obs(
        filtered_events,
        hek_df,
        match_window_minutes=config['matching']['match_window_minutes'],
        magnitude_tolerance=config['matching'].get('magnitude_tolerance', 1.0),
    )
    
    # Step 6: Add metadata to FOXES events
    if not catalog_df.empty:
        catalog_df["min_goes_class_requirement"] = config['detection']['min_goes_class']
        catalog_df["cadence_seconds"] = config['detection']['cadence_seconds']
        catalog_df["quality_config"] = [
            {
                "min_duration_minutes": config['detection']['min_duration_minutes'],
                "min_samples": config['detection']['min_samples'],
                "min_coverage": config['detection']['min_coverage'],
                "max_gap_minutes": config['detection']['max_gap_minutes'],
                "min_prominence": config['detection']['min_prominence'],
                "min_peak_sum_flux": config['detection']['min_peak_sum_flux'],
            }
        ] * len(catalog_df)
    
    return catalog_df, hek_only_df, flare_events_df, analyzer




def parse_hpc_coordinates(hpc_coord_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse HEK helioprojective coordinate string in POINT(x y) format.
    
    Args:
        hpc_coord_str: Coordinate string like "POINT(601.83486 -219.350382)"
        
    Returns:
        Tuple of (x_arcsec, y_arcsec) from disk center, or (None, None) if parsing fails
    """
    if not isinstance(hpc_coord_str, str) or pd.isna(hpc_coord_str):
        return None, None
    
    try:
        coord_str = hpc_coord_str.strip()
        
        # Handle POINT(x y) format from HEK
        if coord_str.startswith("POINT(") and coord_str.endswith(")"):
            coords = coord_str[6:-1].strip()  # Remove "POINT(" and ")"
            parts = coords.split()
            if len(parts) >= 2:
                x_arcsec = float(parts[0])
                y_arcsec = float(parts[1])
                return x_arcsec, y_arcsec
        
        # Handle SkyCoord format with helioprojective coordinates (fallback)
        elif "SkyCoord" in coord_str and "Helioprojective" in coord_str:
            # Look for coordinates in arcsec format
            if "(Tx, Ty) in arcsec" in coord_str:
                # Find the coordinates in parentheses after "arcsec"
                arcsec_pos = coord_str.find("(Tx, Ty) in arcsec")
                if arcsec_pos != -1:
                    # Look for the next opening parenthesis after "arcsec"
                    start_pos = coord_str.find("(", arcsec_pos + len("(Tx, Ty) in arcsec"))
                    if start_pos != -1:
                        # Find the matching closing parenthesis
                        end_pos = coord_str.find(")", start_pos)
                        if end_pos != -1:
                            coords_part = coord_str[start_pos+1:end_pos].strip()
                            # Split by comma and parse
                            parts = [p.strip() for p in coords_part.split(",")]
                            if len(parts) >= 2:
                                x_arcsec = float(parts[0])
                                y_arcsec = float(parts[1])
                                return x_arcsec, y_arcsec
        
        # Handle simple coordinate formats
        elif coord_str.startswith("(") and coord_str.endswith(")"):
            coords = coord_str[1:-1].strip()  # Remove parentheses
            parts = [p.strip() for p in coords.split(",")]
            if len(parts) >= 2:
                x_arcsec = float(parts[0])
                y_arcsec = float(parts[1])
                return x_arcsec, y_arcsec
        
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse HPC coordinate string '{hpc_coord_str}': {e}")
    
    return None, None


def helioprojective_to_pixel(
    x_arcsec: float,
    y_arcsec: float,
    image_size: int = 512,
    fov_solar_radii: float = 1.1
) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert helioprojective coordinates (arcseconds from disk center) to pixel coordinates.
    
    Args:
        x_arcsec: X coordinate in arcseconds from disk center
        y_arcsec: Y coordinate in arcseconds from disk center
        image_size: Image size in pixels (assumed square)
        fov_solar_radii: Field of view in solar radii
        
    Returns:
        Tuple of (x_pixel, y_pixel) or (None, None) if outside FOV
    """
    try:
        # Validate inputs
        if not isinstance(x_arcsec, (int, float)) or not isinstance(y_arcsec, (int, float)):
            return None, None
        
        if np.isnan(x_arcsec) or np.isnan(y_arcsec) or np.isinf(x_arcsec) or np.isinf(y_arcsec):
            return None, None
        
        # Solar parameters
        solar_radius_arcsec = 960.0  # Solar radius in arcseconds
        image_center = image_size / 2.0  # Center pixel (256 for 512x512)
        
        # Calculate pixel scale: FOV covers fov_solar_radii * 2 * solar_radius_arcsec
        total_fov_arcsec = fov_solar_radii * 2 * solar_radius_arcsec
        arcsec_per_pixel = total_fov_arcsec / image_size
        
        # Convert arcseconds to pixels (relative to image center)
        # Note: Using origin='lower' in imshow, so Y increases upward (no flip needed)
        x_pixel = image_center + (x_arcsec / arcsec_per_pixel)
        y_pixel = image_center + (y_arcsec / arcsec_per_pixel)  # No flip for origin='lower'
        
        # Check if coordinates are within the image bounds (with small margin for edge cases)
        margin = 1.0  # Allow 1 pixel margin
        if -margin <= x_pixel < image_size + margin and -margin <= y_pixel < image_size + margin:
            # Clamp to image bounds
            x_pixel = max(0, min(image_size - 1, x_pixel))
            y_pixel = max(0, min(image_size - 1, y_pixel))
            return x_pixel, y_pixel
        else:
            return None, None
            
    except Exception as e:
        print(f"Warning: Error in HPC coordinate transformation for x={x_arcsec}, y={y_arcsec}: {e}")
        return None, None

def load_flux_overlay_at_time(
    timestamp: str,
    flux_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Load FOXES flux contribution data at specific timestamp.
    
    Args:
        timestamp: Timestamp string to find
        flux_path: Path to flux data directory
        
    Returns:
        Flux contribution array or None if not found
    """
    if flux_path is None:
        return None
    
    flux_path = Path(flux_path)
    if not flux_path.exists():
        return None
    
    # Try different filename patterns for flux data
    patterns = [
        f"flux_{timestamp}.npy",
        f"{timestamp}_flux.npy",
        f"flux_contribution_{timestamp}.npy",
    ]
    
    for pattern in patterns:
        filepath = flux_path / pattern
        if filepath.exists():
            try:
                flux_data = np.load(filepath)
                return flux_data
            except Exception as e:
                print(f"Error loading flux data {filepath}: {e}")
                continue
    
    return None


def load_aia_image_at_time(
    aia_path: Path,
    timestamp: str,
    channels: List[str] = ["94", "131", "171"],
) -> Optional[np.ndarray]:
    """
    Load AIA image at specific timestamp for RGB composite.
    
    Args:
        aia_path: Path to AIA data directory
        timestamp: Timestamp string to find
        channels: AIA channels to load for RGB composite
        
    Returns:
        RGB image array or None if not found
    """
    if aia_path is None or not aia_path.exists():
        return None
    
    # Try different possible filename formats and subdirectories
    possible_dirs = [aia_path]
    for subdir in ['test', 'train', 'val']:
        subdir_path = aia_path / subdir
        if subdir_path.exists():
            possible_dirs.append(subdir_path)
    
    # Try different filename patterns
    patterns = [
        f"aia_{timestamp}.npy",
        f"{timestamp}.npy",
        f"aia_{timestamp}_composite.npy",
    ]
    
    for directory in possible_dirs:
        for pattern in patterns:
            filepath = directory / pattern
            if filepath.exists():
                try:
                    # Load the numpy file
                    data = np.load(filepath)
                    
                    # Handle different data formats
                    if data.ndim == 3 and data.shape[0] >= 3:
                        # Multi-channel data, create RGB composite
                        rgb_image = np.zeros((data.shape[1], data.shape[2], 3))
                        
                        # Use first 3 channels for RGB (typically 94, 131, 171 Å)
                        for i in range(3):
                            channel_data = data[i]
                            # Normalize to 0-1 range
                            if channel_data.max() > channel_data.min():
                                channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
                            rgb_image[:, :, i] = channel_data
                        
                        return rgb_image
                    
                    elif data.ndim == 2:
                        # Single channel, convert to grayscale RGB
                        normalized = (data - data.min()) / (data.max() - data.min()) if data.max() > data.min() else data
                        return np.stack([normalized] * 3, axis=-1)
                        
                except Exception as e:
                    print(f"Error loading AIA image {filepath}: {e}")
                    continue
    
    return None


def create_sxr_timeseries_plots_from_tracks(
    catalog_df: pd.DataFrame,
    flare_events_df: pd.DataFrame,
    output_dir: Path,
    plot_window_hours: float = 6.0,
    aia_path: Optional[Path] = None,
    flux_path: Optional[Path] = None,
    predictions_csv: Optional[str] = None,
) -> None:
    """
    Create SXR-style timeseries plots using actual track data with optional AIA images.
    
    Plots are organized into subfolders based on match_status:
    - matched_time_magnitude/: Events matched by BOTH time and magnitude (true detections)
    - matched_time_only/: Events matched by time but different magnitude (possible simultaneous flare separation)
    - foxes_only/: Events detected by FOXES but not in HEK
    
    Args:
        catalog_df: DataFrame with flare events (must have match_status column)
        flare_events_df: Original track data from analyzer
        output_dir: Base directory to save plots
        plot_window_hours: Hours before/after event peak to show
        aia_path: Path to AIA data directory (optional)
        flux_path: Path to flux data directory (optional)
        predictions_csv: Path to predictions CSV with ground truth and predictions
    """
    if catalog_df.empty:
        print("No flare events to plot")
        return
    
    output_dir = Path(output_dir)
    
    # Create subfolders for each match status
    matched_time_mag_dir = output_dir / "matched_time_magnitude"
    matched_time_only_dir = output_dir / "matched_time_only"
    foxes_only_dir = output_dir / "foxes_only"
    matched_time_mag_dir.mkdir(parents=True, exist_ok=True)
    matched_time_only_dir.mkdir(parents=True, exist_ok=True)
    foxes_only_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions data if available
    predictions_df = None
    if predictions_csv and Path(predictions_csv).exists():
        try:
            print(f"Loading predictions data from {predictions_csv}...")
            predictions_df = pd.read_csv(predictions_csv)
            
            # Ensure datetime column
            if 'datetime' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])
            elif 'timestamp' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['timestamp'])
                
            print(f"Loaded predictions data: {len(predictions_df)} samples")
            print(f"Available columns: {list(predictions_df.columns)}")
            
        except Exception as e:
            print(f"Warning: Could not load predictions CSV: {e}")
            predictions_df = None
    
    # Count events by status
    time_mag_count = (catalog_df['match_status'] == 'matched_time_magnitude').sum() if 'match_status' in catalog_df.columns else 0
    time_only_count = (catalog_df['match_status'] == 'matched_time_only').sum() if 'match_status' in catalog_df.columns else 0
    foxes_only_count = (catalog_df['match_status'] == 'foxes_only').sum() if 'match_status' in catalog_df.columns else len(catalog_df)
    print(f"Creating SXR timeseries plots for {len(catalog_df)} events...")
    print(f"  - Matched time + magnitude: {time_mag_count} -> {matched_time_mag_dir}")
    print(f"  - Matched time only: {time_only_count} -> {matched_time_only_dir}")
    print(f"  - FOXES-only: {foxes_only_count} -> {foxes_only_dir}")
    
    for idx, event in catalog_df.iterrows():
        try:
            track_id = event["track_id"]
            
            # Get the full track data
            track_data = flare_events_df[flare_events_df["track_id"] == track_id].copy()
            if track_data.empty:
                continue
                
            track_data = track_data.sort_values("datetime")
            
            # Get event timing
            peak_time = pd.to_datetime(event["peak_time"])
            start_time = pd.to_datetime(event["start_time"])
            end_time = pd.to_datetime(event["end_time"])
            
            # Try to load AIA image at peak time
            aia_image = None
            flux_overlay = None
            peak_row = None
            if aia_path:
                # Find the closest timestamp in track data to peak time
                peak_row = track_data.iloc[(track_data["datetime"] - peak_time).abs().argsort()[:1]]
                if not peak_row.empty:
                    peak_timestamp = peak_row["timestamp"].iloc[0]
                    aia_image = load_aia_image_at_time(aia_path, peak_timestamp)
                    
                    # Try to load flux contribution data for overlay
                    flux_overlay = load_flux_overlay_at_time(peak_timestamp, str(flux_path) if flux_path else None)
            
            # Create figure with subplots
            if aia_image is not None:
                fig = plt.figure(figsize=(16, 6))
                gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], hspace=0.3)
                ax_ts = fig.add_subplot(gs[0])
                ax_img = fig.add_subplot(gs[1])
            else:
                fig, ax_ts = plt.subplots(figsize=(12, 6))
            
            # Define plot window for context
            window_start = peak_time - pd.Timedelta(hours=plot_window_hours/2)
            window_end = peak_time + pd.Timedelta(hours=plot_window_hours/2)
            
            # Plot other tracks in faded grey for context
            other_tracks = flare_events_df[flare_events_df["track_id"] != track_id]
            if not other_tracks.empty:
                # Filter to tracks that have data within the time window
                other_tracks_in_window = other_tracks[
                    (other_tracks["datetime"] >= window_start) & 
                    (other_tracks["datetime"] <= window_end)
                ]
                
                if not other_tracks_in_window.empty:
                    # Group by track and plot each in grey
                    plotted_other_tracks = False
                    plotted_other_peaks = False
                    
                    for other_track_id, other_track_data in other_tracks_in_window.groupby("track_id"):
                        other_track_data = other_track_data.sort_values("datetime")
                        label = 'Other Tracks' if not plotted_other_tracks else None
                        ax_ts.plot(other_track_data["datetime"], other_track_data["sum_flux"], 
                                 color='grey', linewidth=0.5, alpha=0.3, label=label)
                        plotted_other_tracks = True
                        
                        # Find and mark peak times for other tracks
                        if len(other_track_data) > 0:
                            peak_idx = other_track_data["sum_flux"].idxmax()
                            peak_time_other = other_track_data.loc[peak_idx, "datetime"]
                            
                            # Only show peak if it's within the window
                            if window_start <= peak_time_other <= window_end:
                                peak_label = 'Other Track Peaks' if not plotted_other_peaks else None
                                #ax_ts.axvline(peak_time_other, color='grey', linestyle=':', 
                                #            linewidth=1, alpha=0.5, label=peak_label)
                                plotted_other_peaks = True
            
            # Plot ground truth and predictions if available
            if predictions_df is not None:
                # Filter predictions to the time window
                predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
                preds_in_window = predictions_df[
                    (predictions_df["timestamp"] >= window_start) & 
                    (predictions_df["timestamp"] <= window_end)
                ].copy()
                
                if not preds_in_window.empty:
                    # Plot ground truth GOES X-ray flux
                    if 'groundtruth' in preds_in_window.columns:
                        ax_ts.plot(preds_in_window["timestamp"], preds_in_window['groundtruth'], 
                                   'k-', linewidth=1.5, alpha=0.8, label='GOES Ground Truth')
                    
                    # Plot FOXES prediction
                    if 'predictions' in preds_in_window.columns:
                        ax_ts.plot(preds_in_window["timestamp"], preds_in_window['predictions'], 
                                   'r--', linewidth=1.5, alpha=0.8, label='FOXES Prediction')
            
            # Plot the main event track (sum of flux contributions)
            ax_ts.plot(track_data["datetime"], track_data["sum_flux"], 'b-', linewidth=2, alpha=0.9, label='Track Sum Flux')
            
            # Mark the detected event boundaries
            #ax_ts.axvline(start_time, color='green', linestyle='--', alpha=0.7, label='Event Start')
            ax_ts.axvline(peak_time, color='blue', linestyle=':', linewidth=2, label='Event Peak')
            #ax_ts.axvline(end_time, color='orange', linestyle='--', alpha=0.7, label='Event End')
            
            # Add FOXES class annotation at peak
            foxes_flux = event.get("peak_sum_flux", 0)
            foxes_class = flux_to_goes_class(foxes_flux)
            y_pos = ax_ts.get_ylim()[1] * 0.9
            ax_ts.annotate(f'FOXES: {foxes_class}', (peak_time, y_pos),
                         xytext=(0, -10), textcoords='offset points',
                         ha='center', va='top', fontsize=8,
                         color='blue', weight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', 
                                  facecolor='white', alpha=0.8, edgecolor='blue'))
            
            # Shade the event region
            #ax_ts.axvspan(start_time, end_time, alpha=0.2, color='yellow', label='Event Duration')
            
            # Add HEK event peak times if available
            if not pd.isna(event.get("matched_hek_entries")):
                try:
                    import json
                    hek_entries = json.loads(event["matched_hek_entries"])
                    
                    # Colors for different observatories
                    obs_colors = {'SDO': 'purple', 'GOES': 'orange', 'SWPC': 'magenta', 'Unknown': 'gray'}
                    plotted_obs_peaks = set()
                    
                    for entry in hek_entries:
                        obs = entry.get("observatory", "Unknown")
                        peak_time_str = entry.get("peak_time", "")
                        goes_class = entry.get("goes_class", "")
                        
                        if peak_time_str:
                            try:
                                hek_peak_time = pd.to_datetime(peak_time_str)
                                color = obs_colors.get(obs, 'gray')
                                
                                # Only add label for first occurrence of each observatory
                                label = f'{obs} Peak' if obs not in plotted_obs_peaks else None
                                if label:
                                    plotted_obs_peaks.add(obs)
                                
                                # Plot peak time as vertical line
                                ax_ts.axvline(hek_peak_time, color=color, linestyle=':', 
                                            linewidth=2, alpha=0.8, label=label)
                                
                                # Add text annotation with GOES class if available
                                if goes_class and obs in ['SDO', 'SWPC']:  # Only annotate reliable sources
                                    # Get y-position for annotation (top of plot)
                                    y_pos = ax_ts.get_ylim()[1] * 0.7
                                    ax_ts.annotate(goes_class, (hek_peak_time, y_pos),
                                                 xytext=(0, -10), textcoords='offset points',
                                                 ha='center', va='top', fontsize=8,
                                                 color=color, weight='bold',
                                                 bbox=dict(boxstyle='round,pad=0.2', 
                                                          facecolor='white', alpha=0.8, edgecolor=color))
                                
                            except Exception as e:
                                print(f"Warning: Could not parse HEK peak time '{peak_time_str}': {e}")
                                continue
                                
                except Exception as e:
                    print(f"Warning: Could not parse HEK entries for event {idx}: {e}")
            
            # Legacy NOAA event info (if still present)
            if not pd.isna(event.get("matched_noaa_peak")):
                noaa_peak = pd.to_datetime(event["matched_noaa_peak"])
                ax_ts.axvline(noaa_peak, color='purple', linestyle=':', linewidth=2, label='NOAA Peak')
                
                if not pd.isna(event.get("matched_noaa_start")) and not pd.isna(event.get("matched_noaa_end")):
                    noaa_start = pd.to_datetime(event["matched_noaa_start"])
                    noaa_end = pd.to_datetime(event["matched_noaa_end"])
                    ax_ts.axvspan(noaa_start, noaa_end, alpha=0.1, color='purple', label='NOAA Event')
            
            # Set plot window (already defined above for filtering data)
            ax_ts.set_xlim(window_start, window_end)
            
            # Formatting timeseries plot
            ax_ts.set_yscale('log')
            ax_ts.set_ylabel('Sum Flux', fontsize=12)
            ax_ts.set_xlabel('Time (UTC)', fontsize=12)
            
            # Format x-axis
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_ts.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45)
            
            # Legend and grid for timeseries
            ax_ts.legend(loc='upper left', fontsize=9, framealpha=0.9)
            ax_ts.grid(True, alpha=0.3)
            
            # Plot AIA image if available
            print(f"Debug: AIA image available: {aia_image is not None}")
            if aia_image is not None:
                print(f"Debug: AIA image shape: {aia_image.shape}")
                # Display base AIA image
                ax_img.imshow(aia_image, origin='lower', aspect='equal', alpha=0.8)
                
                # Overlay FOXES flux contribution if available
                if flux_overlay is not None:
                    # Normalize flux overlay for visualization
                    flux_normalized = flux_overlay / np.max(flux_overlay) if np.max(flux_overlay) > 0 else flux_overlay
                    
                    # Create a colormap for flux overlay (hot colors for high flux)
                    from matplotlib.colors import LinearSegmentedColormap
                    colors = ['transparent', 'yellow', 'orange', 'red', 'white']
                    n_bins = 100
                    cmap = LinearSegmentedColormap.from_list('flux', colors, N=n_bins)
                    
                    # Only show significant flux contributions (threshold at 10% of max)
                    flux_mask = flux_normalized > 0.1
                    flux_display = np.where(flux_mask, flux_normalized, np.nan)
                    
                    im = ax_img.imshow(flux_display, origin='lower', aspect='equal', 
                                     cmap=cmap, alpha=0.7, vmin=0.1, vmax=1.0)
                    
                    # Add colorbar for flux overlay
                    cbar = plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
                    cbar.set_label('FOXES Flux\n(Normalized)', fontsize=8)
                    cbar.ax.tick_params(labelsize=8)
                
                ax_img.set_title(f'AIA + FOXES Flux\n{peak_time.strftime("%H:%M:%S")}', fontsize=12)
                ax_img.set_xlabel('X (pixels)', fontsize=10)
                ax_img.set_ylabel('Y (pixels)', fontsize=10)
                
                # Mark the flare centroid if we have coordinates
                if not pd.isna(event.get("peak_centroid_img_x")) and not pd.isna(event.get("peak_centroid_img_y")):
                    cx = event["peak_centroid_img_x"]
                    cy = event["peak_centroid_img_y"]
                    # Add a marker at the flare location
                    ax_img.plot(cx, cy, 'b*', markersize=15, markeredgecolor='white', 
                              markeredgewidth=2, label='Flare Centroid')
                
                # Mark the brightest patch if available
                if (not pd.isna(event.get("bright_patch_img_x")) and 
                    not pd.isna(event.get("bright_patch_img_y")) and
                    peak_row is not None and not peak_row.empty):
                    
                    bx = peak_row.get("bright_patch_img_x", event.get("bright_patch_img_x"))
                    by = peak_row.get("bright_patch_img_y", event.get("bright_patch_img_y"))
                    
                    if not pd.isna(bx) and not pd.isna(by):
                        ax_img.plot(bx, by, 'w+', markersize=12, markeredgewidth=3, 
                                  label='Brightest Patch')
                
                # Calculate FOXES flare class from event peak flux
                foxes_flux = event.get("peak_sum_flux", 0)
                foxes_class = flux_to_goes_class(foxes_flux)
                print(f"Debug: FOXES class calculated: {foxes_class} from flux {foxes_flux}")
                
                # Collect HEK classes for display
                sdo_class = None
                goes_class = None
                
                if not pd.isna(event.get("matched_hek_entries")):
                    try:
                        import json
                        hek_entries = json.loads(event["matched_hek_entries"])
                        
                        for entry in hek_entries:
                            obs = entry.get("observatory", "Unknown").strip() if entry.get("observatory") else "Unknown"
                            hek_goes_class = entry.get("goes_class", "").strip() if entry.get("goes_class") else ""
                            
                            if hek_goes_class:
                                if obs == 'SDO':
                                    sdo_class = hek_goes_class
                                elif obs in ['GOES', 'SWPC']:
                                    goes_class = hek_goes_class
                    except:
                        pass
                
                # Create class annotation text in top-left corner
                y_offset = 0.98
                
                # Add FOXES class (blue)
                print(f"Debug: Adding FOXES class text annotation: {foxes_class}")
                ax_img.text(0.02, y_offset, f'FOXES: {foxes_class}', 
                          transform=ax_img.transAxes, fontsize=10, weight='bold',
                          horizontalalignment='left', verticalalignment='top', color='blue',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='blue'))
                print(f"Debug: FOXES text added successfully")
                
                # Add SDO class (purple) if available
                if sdo_class:
                    y_offset -= 0.08
                    ax_img.text(0.02, y_offset, f'SDO: {sdo_class}', 
                              transform=ax_img.transAxes, fontsize=10, weight='bold',
                              horizontalalignment='left', verticalalignment='top', color='purple',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='purple'))
                
                # Add GOES class (orange) if available
                if goes_class:
                    y_offset -= 0.08
                    ax_img.text(0.02, y_offset, f'GOES: {goes_class}', 
                              transform=ax_img.transAxes, fontsize=10, weight='bold',
                              horizontalalignment='left', verticalalignment='top', color='orange',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='orange'))
                
                # Mark HEK flare locations if available (multiple observatories)
                if not pd.isna(event.get("matched_hek_entries")):
                    try:
                        import json
                        hek_entries = json.loads(event["matched_hek_entries"])
                        
                        # Enhanced colors and markers for different observatories
                        obs_styles = {
                            'SDO': {'color': 'purple', 'marker': 's', 'size': 12, 'name': 'SDO'},  # Square
                            'GOES': {'color': 'orange', 'marker': '^', 'size': 10, 'name': 'GOES'},  # Triangle
                            'SWPC': {'color': 'orange', 'marker': 'D', 'size': 10, 'name': 'SWPC'},  # Diamond (same as GOES)
                            'Unknown': {'color': 'gray', 'marker': 'o', 'size': 6, 'name': 'Other'}  # Circle
                        }
                        plotted_obs = set()
                        
                        print(f"Debug: Processing {len(hek_entries)} HEK entries for coordinate plotting")
                        
                        for entry in hek_entries:
                            obs = entry.get("observatory", "Unknown").strip() if entry.get("observatory") else "Unknown"
                            coords = entry.get("coordinates", "").strip() if entry.get("coordinates") else ""
                            entry_goes_class = entry.get("goes_class", "").strip() if entry.get("goes_class") else ""
                            
                            print(f"Debug: {obs} entry - coords: '{coords[:50] if coords else 'None'}...' goes_class: '{entry_goes_class}'")
                            
                            if coords:
                                x_arcsec, y_arcsec = parse_hpc_coordinates(coords)
                                print(f"Debug: Parsed coordinates: x={x_arcsec}, y={y_arcsec}")
                                
                                if x_arcsec is not None and y_arcsec is not None:
                                    hek_x, hek_y = helioprojective_to_pixel(x_arcsec, y_arcsec)
                                    print(f"Debug: Pixel coordinates: x={hek_x}, y={hek_y}")
                                    
                                    if hek_x is not None and hek_y is not None:
                                        style = obs_styles.get(obs, obs_styles['Unknown'])
                                        
                                        # Only add label for first occurrence of each observatory
                                        label = f'{style["name"]} Location' if obs not in plotted_obs else None
                                        if label:
                                            plotted_obs.add(obs)
                                        
                                        # Plot marker with observatory-specific style
                                        ax_img.scatter(hek_x, hek_y, marker=style['marker'], 
                                                     color=style['color'], s=style['size']**2,
                                                     edgecolors='white', linewidths=1.5, 
                                                     label=label, alpha=0.9, zorder=10)
                                        
                                        print(f"Debug: Plotted {obs} marker at ({hek_x:.1f}, {hek_y:.1f})")
                        
                    except Exception as e:
                        print(f"Warning: Could not plot HEK coordinates for event {idx}: {e}")
                
                # Add legend if we have markers
                handles, labels = ax_img.get_legend_handles_labels()
                if handles:
                    ax_img.legend(loc='upper right', fontsize=8, framealpha=0.8)
            
            # Determine match status for title and folder
            match_status = event.get("match_status", "foxes_only")
            foxes_class = event.get("foxes_goes_class", "N/A")
            hek_class = event.get("hek_goes_class", "N/A")
            
            # Create status label for title
            if match_status == "matched_time_magnitude":
                status_label = "Matched (Time + Mag)"
            elif match_status == "matched_time_only":
                status_label = "Matched (Time Only)"
            else:
                status_label = "FOXES Only"
            
            # Title with event info including class comparison
            title_parts = [
                f"Track {track_id} ({status_label})",
                f"Peak: {peak_time.strftime('%Y-%m-%d %H:%M')}",
            ]
            
            # Add class comparison for matched events
            if match_status in ["matched_time_magnitude", "matched_time_only"]:
                title_parts.append(f"FOXES: {foxes_class} vs HEK: {hek_class}")
            elif foxes_class != "N/A":
                title_parts.append(f"FOXES: {foxes_class}")
            
            fig.suptitle(" | ".join(title_parts), fontsize=14, y=0.95)
            
            # Tight layout
            plt.tight_layout()
            
            # Save plot to appropriate subfolder based on match_status
            filename = f"flare_event_{track_id}_{peak_time.strftime('%Y%m%d_%H%M%S')}.png"
            if match_status == "matched_time_magnitude":
                filepath = matched_time_mag_dir / filename
            elif match_status == "matched_time_only":
                filepath = matched_time_only_dir / filename
            else:
                filepath = foxes_only_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating plot for event {idx}: {e}")
            continue
    
    print(f"Saved timeseries plots to {output_dir}")
    print(f"  - Matched time + magnitude: {matched_time_mag_dir}")
    print(f"  - Matched time only: {matched_time_only_dir}")
    print(f"  - FOXES-only plots: {foxes_only_dir}")


def filter_hek_by_foxes_coverage(
    hek_df: pd.DataFrame,
    foxes_timestamps: pd.DatetimeIndex,
    max_gap_minutes: float = 10.0,
    min_points_required: int = 5,
    coverage_window_minutes: float = 30.0,
    min_points_before_peak: int = 3,
    min_points_after_peak: int = 3,
) -> pd.DataFrame:
    """
    Filter HEK events to only include those where FOXES had sufficient data coverage.
    
    This ensures a fair comparison - we only count HEK events as "missed" by FOXES
    if FOXES actually had enough data during that time period to detect it.
    
    NEW: Now requires BALANCED coverage - points both before AND after the peak.
    This is essential for flare detection which needs to see rise AND decay.
    
    Args:
        hek_df: DataFrame of HEK events
        foxes_timestamps: DatetimeIndex of all timestamps where FOXES has data
        max_gap_minutes: Maximum time gap to nearest FOXES point
        min_points_required: Minimum number of FOXES data points required in the window
        coverage_window_minutes: Time window around HEK peak to check for FOXES points
        min_points_before_peak: Minimum points required BEFORE the peak (for rise detection)
        min_points_after_peak: Minimum points required AFTER the peak (for decay detection)
        
    Returns:
        Filtered DataFrame with only HEK events during adequate FOXES coverage
    """
    if hek_df.empty or len(foxes_timestamps) == 0:
        return hek_df
    
    hek_df = hek_df.copy()
    
    # Ensure peak time is datetime
    if "event_peaktime" in hek_df.columns:
        hek_df["event_peaktime"] = pd.to_datetime(hek_df["event_peaktime"])
    else:
        print("Warning: No event_peaktime column in HEK data")
        return hek_df
    
    # Sort FOXES timestamps for efficient searching
    foxes_timestamps = foxes_timestamps.sort_values()
    foxes_ts_array = foxes_timestamps.values  # numpy array for faster operations
    
    def has_adequate_foxes_coverage(peak_time):
        """
        Check if FOXES has adequate BALANCED data coverage around this peak time.
        
        Requires:
        1. At least one FOXES point within max_gap_minutes of the peak
        2. At least min_points_required FOXES points total within coverage_window_minutes
        3. At least min_points_before_peak points BEFORE the peak (for rise detection)
        4. At least min_points_after_peak points AFTER the peak (for decay detection)
        """
        if pd.isna(peak_time):
            return False
        
        peak_np = np.datetime64(peak_time)
        
        # Find the index where peak_time would be inserted
        idx = np.searchsorted(foxes_ts_array, peak_np)
        
        # Check 1: Is there at least one point within max_gap_minutes?
        has_nearby_point = False
        if idx > 0:
            time_diff = abs(peak_np - foxes_ts_array[idx - 1])
            if time_diff <= np.timedelta64(int(max_gap_minutes * 60), 's'):
                has_nearby_point = True
        if idx < len(foxes_ts_array):
            time_diff = abs(peak_np - foxes_ts_array[idx])
            if time_diff <= np.timedelta64(int(max_gap_minutes * 60), 's'):
                has_nearby_point = True
        
        if not has_nearby_point:
            return False
        
        # Define time windows
        window_start = peak_np - np.timedelta64(int(coverage_window_minutes * 60), 's')
        window_end = peak_np + np.timedelta64(int(coverage_window_minutes * 60), 's')
        
        # Find indices of points within the window
        start_idx = np.searchsorted(foxes_ts_array, window_start)
        end_idx = np.searchsorted(foxes_ts_array, window_end)
        peak_idx = np.searchsorted(foxes_ts_array, peak_np)
        
        # Check 2: Total points in window
        points_in_window = end_idx - start_idx
        if points_in_window < min_points_required:
            return False
        
        # Check 3: Points BEFORE peak (for rise detection)
        points_before = peak_idx - start_idx
        if points_before < min_points_before_peak:
            return False
        
        # Check 4: Points AFTER peak (for decay detection)
        points_after = end_idx - peak_idx
        if points_after < min_points_after_peak:
            return False
        
        return True
    
    # Apply the coverage filter
    coverage_mask = hek_df["event_peaktime"].apply(has_adequate_foxes_coverage)
    filtered_df = hek_df[coverage_mask].copy()
    
    return filtered_df


def create_hek_only_plots(
    hek_only_df: pd.DataFrame,
    output_dir: Path,
    plot_window_hours: float = 6.0,
    predictions_csv: Optional[str] = None,
    coverage_gap_minutes: float = 10.0,
    min_coverage_points: int = 5,
    coverage_window_minutes: float = 30.0,
    min_points_before_peak: int = 3,
    min_points_after_peak: int = 3,
) -> None:
    """
    Create timeseries plots for HEK-only events (events in HEK but not detected by FOXES).
    
    IMPORTANT: Only plots HEK events that occurred during times when FOXES had 
    SUFFICIENT and BALANCED data coverage. This ensures a fair comparison - we only 
    show events FOXES "missed" if it had adequate data to detect them.
    
    Coverage requirements:
    1. At least one FOXES data point within coverage_gap_minutes of the HEK peak
    2. At least min_coverage_points FOXES data points total within ±coverage_window_minutes
    3. At least min_points_before_peak points BEFORE the peak (to see rise)
    4. At least min_points_after_peak points AFTER the peak (to see decay)
    
    These plots show the GOES ground truth and FOXES predictions (if available) along with
    HEK event markers to visualize what FOXES might have missed.
    
    Args:
        hek_only_df: DataFrame of HEK events not matched to FOXES detections
        output_dir: Directory to save plots (will create 'hek_only' subfolder)
        plot_window_hours: Hours before/after event peak to show
        predictions_csv: Path to predictions CSV with ground truth and predictions
        coverage_gap_minutes: Max gap in minutes to nearest FOXES point
        min_coverage_points: Minimum FOXES data points required in coverage window
        coverage_window_minutes: Time window (±minutes) to check for FOXES data density
        min_points_before_peak: Minimum points required BEFORE peak (for rise detection)
        min_points_after_peak: Minimum points required AFTER peak (for decay detection)
    """
    if hek_only_df.empty:
        print("No HEK-only events to plot")
        return
    
    output_dir = Path(output_dir)
    hek_only_dir = output_dir / "hek_only"
    hek_only_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions data if available
    predictions_df = None
    foxes_timestamps = None
    if predictions_csv and Path(predictions_csv).exists():
        try:
            print(f"Loading predictions data from {predictions_csv}...")
            predictions_df = pd.read_csv(predictions_csv)
            
            # Ensure datetime column
            if 'datetime' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])
            elif 'timestamp' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['timestamp'])
                predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
                
            print(f"Loaded predictions data: {len(predictions_df)} samples")
            
            # Extract FOXES coverage timestamps
            if 'timestamp' in predictions_df.columns:
                foxes_timestamps = pd.DatetimeIndex(predictions_df['timestamp'])
            elif 'datetime' in predictions_df.columns:
                foxes_timestamps = pd.DatetimeIndex(predictions_df['datetime'])
                
            if foxes_timestamps is not None:
                print(f"FOXES data coverage: {foxes_timestamps.min()} to {foxes_timestamps.max()}")
            
        except Exception as e:
            print(f"Warning: Could not load predictions CSV: {e}")
            predictions_df = None
    
    # Ensure HEK time columns are datetime
    hek_only_df = hek_only_df.copy()
    for col in ["event_starttime", "event_endtime", "event_peaktime"]:
        if col in hek_only_df.columns:
            hek_only_df[col] = pd.to_datetime(hek_only_df[col])
    
    original_count = len(hek_only_df)
    
    # Filter HEK events to only those during adequate FOXES coverage
    if foxes_timestamps is not None and len(foxes_timestamps) > 0:
        print(f"\nFiltering HEK-only events to those with adequate BALANCED FOXES data coverage...")
        print(f"  Requirements:")
        print(f"    - At least one FOXES point within {coverage_gap_minutes} minutes of HEK peak")
        print(f"    - At least {min_coverage_points} FOXES points total within ±{coverage_window_minutes} minutes")
        print(f"    - At least {min_points_before_peak} points BEFORE peak (rise detection)")
        print(f"    - At least {min_points_after_peak} points AFTER peak (decay detection)")
        
        hek_only_df = filter_hek_by_foxes_coverage(
            hek_only_df, 
            foxes_timestamps, 
            max_gap_minutes=coverage_gap_minutes,
            min_points_required=min_coverage_points,
            coverage_window_minutes=coverage_window_minutes,
            min_points_before_peak=min_points_before_peak,
            min_points_after_peak=min_points_after_peak,
        )
        filtered_count = len(hek_only_df)
        print(f"  Original HEK-only events: {original_count}")
        print(f"  Events with adequate BALANCED coverage: {filtered_count}")
        print(f"  Events with insufficient/unbalanced coverage (excluded): {original_count - filtered_count}")
        
        if hek_only_df.empty:
            print("No HEK-only events remain after filtering by FOXES coverage")
            return
    else:
        print("Warning: No FOXES timestamps available - cannot filter by coverage")
        print("  Showing ALL HEK-only events (may include times without FOXES data)")
    
    # Group HEK events by peak time to avoid duplicate plots for same event from different observatories
    # Use a 5-minute window to group same events
    hek_only_df = hek_only_df.sort_values("event_peaktime")
    
    print(f"\nCreating HEK-only plots for {len(hek_only_df)} HEK events with FOXES coverage...")
    print(f"  -> Saving to {hek_only_dir}")
    
    # Track which peak times we've already plotted (within 5 min tolerance)
    plotted_peaks = []
    plots_created = 0
    
    for idx, hek_event in hek_only_df.iterrows():
        try:
            peak_time = hek_event.get("event_peaktime")
            if pd.isna(peak_time):
                continue
            
            # Check if we've already plotted an event within 5 minutes of this one
            already_plotted = False
            for prev_peak in plotted_peaks:
                if abs((peak_time - prev_peak).total_seconds()) < 300:  # 5 minutes
                    already_plotted = True
                    break
            
            if already_plotted:
                continue
            
            plotted_peaks.append(peak_time)
            
            # Get event info
            start_time = hek_event.get("event_starttime", peak_time - pd.Timedelta(minutes=30))
            end_time = hek_event.get("event_endtime", peak_time + pd.Timedelta(minutes=30))
            obs = hek_event.get("obs_observatory", "Unknown")
            goes_class = hek_event.get("fl_goescls", hek_event.get("goes_class", ""))
            coords = hek_event.get("hpc_coord", "")
            ar_num = hek_event.get("ar_noaanum", "")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Define plot window
            window_start = peak_time - pd.Timedelta(hours=plot_window_hours/2)
            window_end = peak_time + pd.Timedelta(hours=plot_window_hours/2)
            
            has_data = False
            
            # Plot ground truth and predictions if available
            if predictions_df is not None:
                # Filter predictions to the time window
                preds_in_window = predictions_df[
                    (predictions_df["timestamp"] >= window_start) & 
                    (predictions_df["timestamp"] <= window_end)
                ].copy()
                
                if not preds_in_window.empty:
                    has_data = True
                    
                    # Plot ground truth GOES X-ray flux
                    if 'groundtruth' in preds_in_window.columns:
                        ax.plot(preds_in_window["timestamp"], preds_in_window['groundtruth'], 
                                'k-', linewidth=1.5, alpha=0.8, label='GOES Ground Truth')
                    
                    # Plot FOXES prediction
                    if 'predictions' in preds_in_window.columns:
                        ax.plot(preds_in_window["timestamp"], preds_in_window['predictions'], 
                                'r--', linewidth=1.5, alpha=0.8, label='FOXES Prediction')
            
            # Mark HEK event
            ax.axvline(peak_time, color='purple', linestyle=':', linewidth=2, label=f'{obs} Peak')
            
            # Shade HEK event region
            if not pd.isna(start_time) and not pd.isna(end_time):
                ax.axvspan(start_time, end_time, alpha=0.15, color='purple', label='HEK Event Duration')
            
            # Add GOES class annotation
            if goes_class:
                y_pos = ax.get_ylim()[1] * 0.9 if has_data else 0.9
                ax.annotate(f'{goes_class}', (peak_time, y_pos),
                           xytext=(0, -10), textcoords='offset points',
                           ha='center', va='top', fontsize=12,
                           color='purple', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', alpha=0.9, edgecolor='purple'))
            
            # Set plot window
            ax.set_xlim(window_start, window_end)
            
            # Formatting
            if has_data:
                ax.set_yscale('log')
            ax.set_ylabel('Flux', fontsize=12)
            ax.set_xlabel('Time (UTC)', fontsize=12)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Legend and grid
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            # Title
            title_parts = [
                "HEK Only (Not detected by FOXES)",
                f"Peak: {peak_time.strftime('%Y-%m-%d %H:%M')}",
            ]
            if goes_class:
                title_parts.append(f"Class: {goes_class}")
            if obs:
                title_parts.append(f"Source: {obs}")
            
            fig.suptitle(" | ".join(title_parts), fontsize=14, y=0.95)
            
            # Tight layout
            plt.tight_layout()
            
            # Save plot
            filename = f"hek_only_{peak_time.strftime('%Y%m%d_%H%M%S')}_{obs}.png"
            filepath = hek_only_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            plots_created += 1
            
        except Exception as e:
            print(f"Error creating HEK-only plot for event {idx}: {e}")
            continue
    
    print(f"Created {plots_created} HEK-only plots in {hek_only_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a flare catalog from FOXES predictions using config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to flare catalog builder YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_catalog_config(args.config)
    
    catalog_df, hek_only_df, flare_events_df, analyzer = build_flare_catalog(config)
    
    # Create timestamped output directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(config['paths']['output_csv'])
    base_output_dir = output_path.parent
    output_dir = base_output_dir / f"run_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Output Directory ===")
    print(f"Created timestamped output folder: {output_dir}")
    
    print(f"\n=== Saving CSV Catalogs ===")
    
    # Split catalog by match status
    matched_time_mag_df = catalog_df[catalog_df['match_status'] == 'matched_time_magnitude'] if 'match_status' in catalog_df.columns else pd.DataFrame()
    matched_time_only_df = catalog_df[catalog_df['match_status'] == 'matched_time_only'] if 'match_status' in catalog_df.columns else pd.DataFrame()
    foxes_only_events_df = catalog_df[catalog_df['match_status'] == 'foxes_only'] if 'match_status' in catalog_df.columns else catalog_df
    
    # 1. Save matched time + magnitude events (true detections)
    matched_time_mag_path = output_dir / "flare_events_matched_time_magnitude.csv"
    if not matched_time_mag_df.empty:
        matched_time_mag_df.to_csv(matched_time_mag_path, index=False)
        print(f"  [1] Matched time + magnitude: {len(matched_time_mag_df)} events -> {matched_time_mag_path}")
    else:
        print(f"  [1] Matched time + magnitude: 0 events (no file created)")
    
    # 2. Save matched time only events (possible simultaneous flare separation)
    matched_time_only_path = output_dir / "flare_events_matched_time_only.csv"
    if not matched_time_only_df.empty:
        matched_time_only_df.to_csv(matched_time_only_path, index=False)
        print(f"  [2] Matched time only (diff magnitude): {len(matched_time_only_df)} events -> {matched_time_only_path}")
    else:
        print(f"  [2] Matched time only (diff magnitude): 0 events (no file created)")
    
    # 3. Save FOXES-only events (detected by FOXES but not in HEK)
    foxes_only_path = output_dir / "flare_events_foxes_only.csv"
    if not foxes_only_events_df.empty:
        foxes_only_events_df.to_csv(foxes_only_path, index=False)
        print(f"  [3] FOXES-only (no HEK match): {len(foxes_only_events_df)} events -> {foxes_only_path}")
    else:
        print(f"  [3] FOXES-only (no HEK match): 0 events (no file created)")
    
    # 4. Save HEK-only events (in HEK but not detected by FOXES)
    hek_only_path = output_dir / "flare_events_hek_only.csv"
    if not hek_only_df.empty:
        hek_only_df.to_csv(hek_only_path, index=False)
        print(f"  [4] HEK-only (FOXES missed): {len(hek_only_df)} events -> {hek_only_path}")
    else:
        print(f"  [4] HEK-only (FOXES missed): 0 events (no file created)")
    
    # 5. Also save combined FOXES catalog (for backward compatibility)
    combined_path = output_dir / "flare_events_all_foxes.csv"
    if not catalog_df.empty:
        catalog_df.to_csv(combined_path, index=False)
        print(f"  [*] Combined FOXES catalog: {len(catalog_df)} events -> {combined_path}")
    
    # Summary
    print(f"\n=== Event Summary ===")
    print(f"  Total FOXES detections: {len(catalog_df)}")
    print(f"    - Matched time + magnitude: {len(matched_time_mag_df)}")
    print(f"    - Matched time only: {len(matched_time_only_df)}")
    print(f"    - FOXES-only: {len(foxes_only_events_df)}")
    print(f"  Total HEK-only (missed by FOXES): {len(hek_only_df)}")
    
    # Create SXR plots if requested
    if config['plotting']['create_sxr_plots']:
        # Use plots subfolder within the timestamped output directory
        plots_dir = output_dir / "plots"
        
        # Get paths from analyzer config or config overrides
        aia_path = Path(config['paths']['aia_path']) if config['paths']['aia_path'] else (
            Path(analyzer.aia_path) if hasattr(analyzer, 'aia_path') and analyzer.aia_path else None
        )
        flux_path = Path(config['paths']['flux_path']) if config['paths']['flux_path'] else (
            Path(analyzer.flux_path) if hasattr(analyzer, 'flux_path') and analyzer.flux_path else None
        )
        
        # Get predictions CSV path from config or analyzer
        predictions_csv = config['paths']['predictions_csv'] or (
            analyzer.predictions_csv if hasattr(analyzer, 'predictions_csv') and analyzer.predictions_csv else None
        )
        
        print(f"\n=== Creating Plots ===")
        print(f"Output directory: {plots_dir}")
        print(f"  Subfolders: matched_time_magnitude/, matched_time_only/, foxes_only/, hek_only/")
        
        # Create plots for FOXES events (matched + foxes_only)
        create_sxr_timeseries_plots_from_tracks(
            catalog_df,
            flare_events_df,
            plots_dir,
            plot_window_hours=config['plotting']['plot_window_hours'],
            aia_path=aia_path,
            flux_path=flux_path,
            predictions_csv=predictions_csv,
        )
        
        # Create plots for HEK-only events
        if not hek_only_df.empty:
            # Get HEK coverage config (with defaults for backward compatibility)
            hek_cov = config.get('hek_coverage', {})
            create_hek_only_plots(
                hek_only_df,
                plots_dir,
                plot_window_hours=config['plotting']['plot_window_hours'],
                predictions_csv=predictions_csv,
                coverage_gap_minutes=hek_cov.get('max_gap_minutes', 10.0),
                min_coverage_points=hek_cov.get('min_points_total', 5),
                coverage_window_minutes=hek_cov.get('coverage_window_minutes', 30.0),
                min_points_before_peak=hek_cov.get('min_points_before_peak', 3),
                min_points_after_peak=hek_cov.get('min_points_after_peak', 3),
            )
        else:
            print("No HEK-only events to plot")


if __name__ == "__main__":  # pragma: no cover
    main()

