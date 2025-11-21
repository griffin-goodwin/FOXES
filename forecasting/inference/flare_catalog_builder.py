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
) -> Optional[pd.DataFrame]:
    """Load HEK catalog from CSV or fetch via SunPy if requested."""
    if catalog_path:
        df = pd.read_csv(catalog_path)
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        print(f"Loaded HEK catalog with {len(df)} rows from {catalog_path}")
        print(f"Cleaned column names. Sample columns: {list(df.columns)[:5]}")
        return df
    if auto_fetch and start_time and end_time:
        return fetch_hek_flares(start_time, end_time)
    return None


def fetch_hek_flares(start_time: str, end_time: str) -> pd.DataFrame:
    """Fetch HEK flare catalog via SunPy with all observatories and coordinates."""
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
    peaks, properties = find_peaks(
        flux_series.values,
        prominence=min_prominence * baseline.median(),
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
        for i in range(peak_idx - 1, -1, -1):
            if flux_series.iloc[i] < baseline_threshold:
                start_idx = i + 1
                break
        
        # Find end (going forwards from peak)
        end_idx = peak_idx
        for i in range(peak_idx + 1, len(flux_series)):
            if flux_series.iloc[i] < baseline_threshold:
                end_idx = i - 1
                break
        
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
        
        # Skip events that are too short
        if duration_minutes < min_duration_minutes:
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
    
    for track_id, track_df in flare_events_df.groupby("track_id"):
        track_df = track_df.sort_values("datetime")
        
        # Split track into contiguous sequences first (handle data gaps)
        sequences = split_track_sequences(track_df, max_gap_minutes)
        
        for seq_idx, seq_df in enumerate(sequences):
            if len(seq_df) < 3:  # Need minimum points for peak detection
                continue
                
            # Detect flare events within this sequence
            events = detect_flare_events_in_track(
                seq_df,
                min_prominence=min_prominence,
                min_duration_minutes=min_duration_minutes,
            )
            
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
    
    print(f"Detected {len(all_events)} individual flare events from {flare_events_df['track_id'].nunique()} tracks")
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
    for event in events:
        if event["duration_minutes"] < min_duration_minutes:
            continue
        if event["num_samples"] < min_samples:
            continue
        if event["coverage_ratio"] < min_coverage:
            continue
        if min_prominence is not None and not math.isnan(min_prominence) and event.get("prominence", 0.0) < min_prominence:
            continue
        if min_peak_flux is not None and not math.isnan(min_peak_flux) and event.get("peak_sum_flux", 0.0) < min_peak_flux:
            continue
        records.append(event)
    df = pd.DataFrame(records)
    print(f"Quality filter retained {len(df)} individual flare events")
    return df




def match_events_to_hek_multi_obs(
    events_df: pd.DataFrame,
    hek_df: pd.DataFrame,
    match_window_minutes: float,
) -> pd.DataFrame:
    """Associate each FOXES event with HEK events from multiple observatories."""
    print(f"\n=== HEK Matching Debug ===")
    print(f"Events DF: {len(events_df)} rows, empty={events_df.empty}")
    print(f"HEK DF is None: {hek_df is None}")
    if hek_df is not None:
        print(f"HEK DF: {len(hek_df)} rows, empty={hek_df.empty}")
        print(f"HEK columns (first 10): {list(hek_df.columns)[:10]}")
    
    if events_df.empty or hek_df is None or hek_df.empty:
        print("Early exit: events or HEK df is empty/None")
        events_df["matched_hek_entries"] = pd.NA
        return events_df
    
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
    
    print(f"\nStarting matching with tolerance: ±{match_window_minutes} minutes")
    if len(events_df) > 0:
        print(f"Sample FOXES event: {events_df[['start_time', 'peak_time', 'end_time']].iloc[0].to_dict()}")
    if len(hek_df) > 0:
        # Check which time columns actually exist
        time_cols_available = [col for col in ['event_starttime', 'event_peaktime', 'event_endtime'] if col in hek_df.columns]
        if time_cols_available:
            print(f"Sample HEK event: {hek_df[time_cols_available].iloc[0].to_dict()}")
        else:
            print(f"Sample HEK event (first 5 columns): {hek_df.iloc[0, :5].to_dict()}")
    
    num_matched = 0
    for idx, event in events_df.iterrows():
        event_start = event["start_time"]
        event_end = event["end_time"]
        event_peak = event["peak_time"]
        
        # Create matching window around the FOXES event
        window_start = event_start - tolerance
        window_end = event_end + tolerance
        
        # Find overlapping HEK events
        overlapping = hek_df[
            (hek_df["event_endtime"] >= window_start)
            & (hek_df["event_starttime"] <= window_end)
        ]
        
        if overlapping.empty:
            matched_entries.append(pd.NA)
            continue
        
        num_matched += 1
        if num_matched <= 3:  # Show details for first 3 matches
            print(f"\nMatch #{num_matched}: FOXES event at {event_peak}")
            print(f"  Window: {window_start} to {window_end}")
            print(f"  Found {len(overlapping)} overlapping HEK events")
            if len(overlapping) > 0:
                obs_list = overlapping['obs_observatory'].unique() if 'obs_observatory' in overlapping.columns else ['Unknown']
                print(f"  Observatories: {list(obs_list)}")
        
        # Group by peak time to find events from different observatories
        # that are reporting the same physical flare
        event_groups = overlapping.groupby("event_peaktime")
        
        # Find the group with peak time closest to FOXES event
        peak_distances = []
        for peak_time, group in event_groups:
            distance = abs((peak_time - event_peak).total_seconds())
            peak_distances.append((distance, peak_time, group))
        
        if peak_distances:
            # Get the closest group
            _, _, best_group = min(peak_distances, key=lambda x: x[0])
            
            # Create a summary of all observatory entries for this flare
            entries = []
            for _, hek_event in best_group.iterrows():
                obs = hek_event.get("obs_observatory", "Unknown")
                goes_class = hek_event.get("fl_goescls", "")
                coords = hek_event.get("hpc_coord", "")
                ar_num = hek_event.get("ar_noaanum", "")
                
                # Strip whitespace from string values
                obs = obs.strip() if isinstance(obs, str) else str(obs)
                goes_class = goes_class.strip() if isinstance(goes_class, str) else str(goes_class)
                coords = coords.strip() if isinstance(coords, str) else str(coords)
                
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
            
            # Store as JSON string for easy parsing later
            import json
            matched_entries.append(json.dumps(entries))
        else:
            matched_entries.append(pd.NA)
    
    events_df["matched_hek_entries"] = matched_entries
    
    # Report matching statistics
    print(f"\n=== Matching Complete ===")
    print(f"Total FOXES events: {len(events_df)}")
    print(f"Events with HEK matches: {num_matched}")
    print(f"Match rate: {num_matched / len(events_df) * 100:.1f}%")
    print(f"Events without matches: {len(events_df) - num_matched}")
    
    return events_df


def load_catalog_config(config_path: str) -> Dict:
    """Load flare catalog configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_flare_catalog(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Main orchestration function."""
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
    
    # Step 4: Load and normalize HEK catalog
    min_flux = flux_threshold_from_class(config['detection']['min_goes_class'])
    hek_df = load_hek_catalog(
        Path(config['paths']['hek_catalog']) if config['paths']['hek_catalog'] else None,
        start_time,
        end_time,
        auto_fetch=config['time_range']['auto_fetch_hek'],
    )
    if hek_df is not None and not hek_df.empty:
        print(f"HEK columns before normalization: {list(hek_df.columns)[:10]}...")
        hek_df = normalize_hek_catalog(hek_df, min_flux)
        print(f"HEK columns after normalization: {list(hek_df.columns)[:10]}...")
        print(f"HEK shape after normalization: {hek_df.shape}")
    
    # Step 5: Match individual FOXES events to HEK events (multi-observatory)
    catalog_df = match_events_to_hek_multi_obs(
        filtered_events,
        hek_df,
        match_window_minutes=config['matching']['match_window_minutes'],
    )
    
    # Step 6: Add metadata
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
    
    return catalog_df, flare_events_df, analyzer




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
    
    Args:
        catalog_df: DataFrame with flare events
        flare_events_df: Original track data from analyzer
        output_dir: Directory to save plots
        plot_window_hours: Hours before/after event peak to show
        aia_path: Path to AIA data directory (optional)
        flux_path: Path to flux data directory (optional)
        predictions_csv: Path to predictions CSV with ground truth and predictions
    """
    if catalog_df.empty:
        print("No flare events to plot")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    print(f"Creating SXR timeseries plots with real data for {len(catalog_df)} events...")
    
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
                                    y_pos = ax_ts.get_ylim()[1] * 0.8
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
            
            # Title with event info
            title_parts = [
                f"Track {track_id} Event",
                f"Peak: {peak_time.strftime('%Y-%m-%d %H:%M')}",
            ]
            
            if not pd.isna(event.get("matched_goes_class")):
                title_parts.append(f"NOAA: {event['matched_goes_class']}")
            
            fig.suptitle(" | ".join(title_parts), fontsize=14, y=0.95)
            
            # Tight layout
            plt.tight_layout()
            
            # Save plot
            filename = f"flare_event_{track_id}_{peak_time.strftime('%Y%m%d_%H%M%S')}.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating plot for event {idx}: {e}")
            continue
    
    print(f"Saved timeseries plots to {output_dir}")


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
    
    catalog_df, flare_events_df, analyzer = build_flare_catalog(config)
    
    # Save catalog
    output_path = Path(config['paths']['output_csv'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_df.to_csv(output_path, index=False)
    print(f"Wrote flare catalog with {len(catalog_df)} rows to {output_path}")
    
    # Create SXR plots if requested
    if config['plotting']['create_sxr_plots']:
        plots_dir = Path(config['paths']['sxr_plots_dir']) if config['paths']['sxr_plots_dir'] else output_path.parent / "sxr_plots"
        
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
        
        create_sxr_timeseries_plots_from_tracks(
            catalog_df,
            flare_events_df,
            plots_dir,
            plot_window_hours=config['plotting']['plot_window_hours'],
            aia_path=aia_path,
            flux_path=flux_path,
            predictions_csv=predictions_csv,
        )


if __name__ == "__main__":  # pragma: no cover
    main()

