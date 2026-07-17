"""
Build an index of AIA samples matched across all 7 wavelength channels in
the SDOML v2 Zarr store on S3, plus a fast per-index pixel reader used by
data/download_streaming_sdoml.py to materialize them as .npy files.

Each row of the index is one sample: a timestamp and the per-wavelength
zarr index into that year's SDOML arrays. AIA wavelength arrays are NOT
index-aligned with each other (each channel has independent data gaps --
see build_streaming_index_config.yaml for max_gap_minutes), so this does a
nearest-timestamp match per channel.

SXR matching and train/val/test splitting are NOT done here -- once
data/download_streaming_sdoml.py has written .npy files, use the existing
data/align_aia_sxr.py and data/split_train_val_test.py against them
unmodified, same as the raw-FITS pipeline (data/build_dataset.py) does.

Usage (standalone -- inspect match coverage / dump an index to parquet
without downloading any pixel data):
    python data/build_streaming_index.py --config data/build_streaming_index_config.yaml
"""
import argparse
import bisect
import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import s3fs
import yaml
import zarr

DEFAULT_S3_URL = "nasa-radiant-data/helioai-datasets/us-fdlx-ard/sdomlv2a/AIA.zarr"
WAVELENGTH_MAP = {94: "94A", 131: "131A", 171: "171A", 193: "193A",
                   211: "211A", 304: "304A", 335: "335A"}


def open_sdoml_store(s3_url=DEFAULT_S3_URL):
    """Open the SDOML AIA Zarr root as a plain zarr group (not xarray --
    there's no consolidated .zmetadata and no _ARRAY_DIMENSIONS, so xarray
    can't interpret it as a normal labeled dataset)."""
    fs = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=s3_url, s3=fs, check=False)
    return zarr.open_group(store, mode="r")


def _parse_tobs(t_obs):
    return [(datetime.strptime(t.rstrip("Z").split(".")[0], "%Y-%m-%dT%H:%M:%S"), i)
            for i, t in enumerate(t_obs)]


def get_channel_timestamps(root, year, wavelength, cache_dir):
    """
    Sorted [(timestamp, zarr_index), ...] for one (year, wavelength).

    Cached to disk: fetching T_OBS means downloading and parsing that
    channel's entire .zattrs blob -- one JSON object per channel covering
    every FITS header keyword for every timestep in the year, ~250MB and
    20-30s in testing, with no way to fetch a date-range subset of it.
    Every later call for the same (year, wavelength) is a local parquet
    read instead. (Pixel reads never pay this cost at all -- see
    get_zarray_meta/read_channel_slice below, which only fetch .zarray.)
    """
    cache_path = Path(cache_dir) / f"tobs_{year}_{wavelength}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return list(zip(df["timestamp"], df["zarr_index"]))

    arr = root[str(year)][WAVELENGTH_MAP[wavelength]]
    parsed = sorted(_parse_tobs(arr.attrs["T_OBS"]))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(parsed, columns=["timestamp", "zarr_index"]).to_parquet(cache_path)
    return parsed


def _nearest(sorted_pairs, times, target):
    pos = bisect.bisect_left(times, target)
    cands = [sorted_pairs[p] for p in (pos - 1, pos) if 0 <= p < len(sorted_pairs)]
    return min(cands, key=lambda p: abs(p[0] - target))


def get_zarray_meta(fs, base_path, cache_dir=None):
    """
    Fetch a channel's .zarray metadata (shape/dtype/chunks/compressor --
    a few hundred bytes) directly, WITHOUT going through zarr's own
    Group/Array opening path. zarr unconditionally fetches .zarray and
    .zattrs together when opening an array (zarr.core.array.get_array_metadata),
    even though pixel reads never touch attributes -- and .zattrs is the
    same ~250MB-per-channel blob get_channel_timestamps caches for T_OBS.
    Fetching it a second time here just to read pixels was the actual
    cause of the ~20s-per-channel-per-sample slowdown found while
    building this out; this bypasses it entirely.
    """
    cache_path = (Path(cache_dir) / (base_path.replace("/", "_") + ".zarray.json")
                  if cache_dir is not None else None)
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text())
    meta = json.loads(fs.cat_file(f"{base_path}/.zarray"))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(meta))
    return meta


def make_chunk_cache(fs, maxsize=64):
    """
    A bounded cache of decoded (chunk_len, H, W) chunks, keyed by
    (base_path, chunk_idx). Each SDOML chunk covers 15 consecutive
    timesteps (~90 min at native cadence), and build_matched_index walks
    at a similar cadence by default -- so nearby samples in a matched
    index usually land in the SAME chunk. Without this, downloading N
    samples close together in time re-fetches and re-decodes that same
    ~12MB chunk N times; this makes repeat reads free. Thread-safe
    (lru_cache uses an internal lock) -- share one instance across a
    thread pool's workers.

    maxsize=64 chunks is a rough memory/hit-rate tradeoff -- each decoded
    chunk is ~15MB, so worst case this holds ~1GB.
    """
    @lru_cache(maxsize=maxsize)
    def _fetch(base_path, chunk_idx, compressor_json, dtype_str, chunk_shape):
        try:
            raw = fs.cat_file(f"{base_path}/{chunk_idx}.0.0")
        except FileNotFoundError:
            return None
        decoded = numcodecs.get_codec(json.loads(compressor_json)).decode(raw)
        assert decoded is not None
        return np.frombuffer(bytes(decoded), dtype=np.dtype(dtype_str)).reshape(chunk_shape)
    return _fetch


def read_channel_slice(fs, base_path, zarray_meta, index, chunk_cache=None):
    """
    Fetch the (H, W) slice at `index` from the chunk that covers it,
    decoding with the codec .zarray declares. Verified to produce output
    byte-for-byte identical to zarr's own Array.__getitem__. Missing
    chunks (sparse/never-written) fall back to fill_value, same as zarr's
    own behavior.

    Pass a cache from make_chunk_cache() to reuse decoded chunks across
    calls -- worth it whenever nearby indices get read repeatedly (e.g.
    data/download_streaming_sdoml.py downloading many samples close
    together in time).
    """
    chunks = zarray_meta["chunks"]
    dtype = np.dtype(zarray_meta["dtype"])
    chunk_idx = index // chunks[0]
    within = index % chunks[0]

    if chunk_cache is not None:
        compressor_json = json.dumps(zarray_meta["compressor"], sort_keys=True)
        arr = chunk_cache(base_path, chunk_idx, compressor_json, zarray_meta["dtype"], tuple(chunks))
    else:
        try:
            raw = fs.cat_file(f"{base_path}/{chunk_idx}.0.0")
        except FileNotFoundError:
            return np.full(chunks[1:], zarray_meta["fill_value"], dtype=dtype)
        decoded = numcodecs.get_codec(zarray_meta["compressor"]).decode(raw)
        assert decoded is not None
        arr = np.frombuffer(bytes(decoded), dtype=dtype).reshape(chunks)

    if arr is None:
        return np.full(chunks[1:], zarray_meta["fill_value"], dtype=dtype)
    return arr[within]


def build_matched_index(root, years, wavelengths, cache_dir, cadence_minutes=6, max_gap_minutes=3):
    """
    Step through the first wavelength's own timestamps at ~cadence_minutes,
    and for each one, look up the nearest sample in every other channel.
    Keep the timestamp only if every channel is within max_gap_minutes
    (channels have independent calibration/eclipse gaps that can run hours
    long -- see the notebook exploration this was built from).

    Returns a DataFrame: timestamp, year, idx_<wavelength> for each channel.
    """
    max_gap = timedelta(minutes=max_gap_minutes)
    rows = []
    for year in years:
        print(f"Indexing {year}...")
        channel_pairs = {wl: get_channel_timestamps(root, year, wl, cache_dir) for wl in wavelengths}
        channel_times = {wl: [p[0] for p in pairs] for wl, pairs in channel_pairs.items()}

        ref_wl = wavelengths[0]
        last_kept = None
        year_rows = 0
        for ts, idx in channel_pairs[ref_wl]:
            if last_kept is not None and (ts - last_kept) < timedelta(minutes=cadence_minutes):
                continue
            matches = {ref_wl: (ts, idx)}
            ok = True
            for wl in wavelengths[1:]:
                m = _nearest(channel_pairs[wl], channel_times[wl], ts)
                if abs(m[0] - ts) > max_gap:
                    ok = False
                    break
                matches[wl] = m
            last_kept = ts  # advance regardless of match so gaps don't stall the cadence walk
            if not ok:
                continue
            # Round only the label to the nearest minute -- the zarr index
            # fetched above is still the nearest REAL sample (ts, unrounded).
            # SDOML's native T_OBS has a few seconds of jitter (e.g.
            # 00:06:07), but data/align_aia_sxr.py matches AIA filenames
            # against GOES's avg1m CSVs by EXACT timestamp, and GOES avg1m
            # rows sit on the minute -- without rounding, virtually no
            # downloaded sample would ever match downstream.
            rounded_ts = pd.Timestamp(ts).round("min").to_pydatetime()
            row = {"timestamp": rounded_ts, "year": year}
            row.update({f"idx_{wl}": matches[wl][1] for wl in wavelengths})
            rows.append(row)
            year_rows += 1
        print(f"  {year_rows} matched samples")

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build an AIA cross-channel matched index against the SDOML Zarr store on S3.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                         help="Optional parquet path to save the index to (default: just print coverage).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    wavelengths = config["wavelengths"]
    years = config["years"]
    cache_dir = config["cache_dir"]
    s3_url = config.get("s3_url", DEFAULT_S3_URL)
    matching = config.get("matching", {})

    root = open_sdoml_store(s3_url)
    index_df = build_matched_index(
        root, years, wavelengths, cache_dir,
        cadence_minutes=matching.get("cadence_minutes", 6),
        max_gap_minutes=matching.get("max_gap_minutes", 3),
    )
    print(f"Total AIA-matched samples across {years}: {len(index_df)}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        index_df.to_parquet(args.output)
        print(f"Saved index to {args.output}")


if __name__ == "__main__":
    main()
