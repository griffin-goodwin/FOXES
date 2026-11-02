"""Download extra AIA/GOES training pairs around GOES flares and between them.

Run from the repository root (settings in expand_training_data.yaml):

    python -m download.expand_training_data --mode plan
    python -m download.expand_training_data --mode download --limit 2
    python -m download.expand_training_data --mode download

`plan` queries the HEK GOES flare list (cached per month), picks timestamps that
are not already in the existing dataset and not near a validation/test day,
looks up their GOES XRS-B flux, and writes <output_dir>/samples.csv.
`download` fetches and processes AIA for every planned timestamp and saves
AIA_processed/train/<t>.npy and SXR_processed/train/<t>.npy. Rerunning it
skips pairs that are already saved.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import itertools
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading

import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WAVELENGTHS = [94, 131, 171, 193, 211, 304, 335]
HEK_COLUMNS = {"event_starttime": "start_time", "event_peaktime": "peak_time",
               "event_endtime": "end_time", "fl_goescls": "goes_class"}


def file_times(folders):
    """Timestamps of every *.npy under the folders, named like 2014-04-06T12:00:00.npy."""
    stems = []
    for folder in folders:
        if not Path(folder).is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        stems += [path.stem.replace("_", ":") for path in Path(folder).rglob("*.npy")]
    times = pd.to_datetime(pd.Series(stems, dtype=str), format="ISO8601", errors="coerce")
    return pd.DatetimeIndex(times.dropna())


def fetch_month(month, cache_dir):
    """One month of the HEK GOES flare list, cached as CSV."""
    path = Path(cache_dir) / f"goes_{month}.csv"
    if not path.exists():
        from sunpy.net import attrs as a
        from sunpy.net.hek import HEKClient

        period = pd.Period(month, freq="M")
        print(f"Querying HEK GOES flares for {month}")
        result = HEKClient().search(a.Time(period.start_time, (period + 1).start_time),
                                    a.hek.EventType("FL"), a.hek.OBS.Observatory == "GOES")
        rows = [{new: str(row[old]) for old, new in HEK_COLUMNS.items()} for row in result]
        pd.DataFrame(rows, columns=list(HEK_COLUMNS.values())).to_csv(path, index=False)
    return pd.read_csv(path, usecols=list(HEK_COLUMNS.values()))


def get_catalog(config, cache_dir):
    """GOES flares covering date_ranges, plus neighbouring months for flares that straddle the edges."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    months = sorted({str(month) for start, end in config["date_ranges"]
                     for month in pd.period_range(pd.Timestamp(start) - pd.Timedelta(days=1),
                                                  pd.Timestamp(end) + pd.Timedelta(days=2), freq="M")})
    # Separate processes because sunpy's HEK client isn't thread safe.
    with ProcessPoolExecutor(config["catalog_workers"]) as pool:
        catalog = pd.concat(pool.map(fetch_month, months, [cache_dir] * len(months)), ignore_index=True)
    for column in ("start_time", "peak_time", "end_time"):
        times = pd.to_datetime(catalog[column], utc=True, format="ISO8601", errors="coerce")
        catalog[column] = times.dt.tz_localize(None)
    catalog = catalog.dropna(subset=["peak_time"]).drop_duplicates("peak_time").reset_index(drop=True)
    # Some HEK records put start/end on the wrong side of the peak; widen them to cover all three.
    times = catalog[["start_time", "peak_time", "end_time"]]
    catalog["start_time"], catalog["end_time"] = times.min(axis=1), times.max(axis=1)
    catalog["goes_class"] = catalog.goes_class.fillna("").astype(str).str.strip().str.upper()
    return catalog


def make_plan(config, catalog, existing, heldout_days):
    """Candidate timestamps: windows around flare peaks plus a jittered grid between flares."""
    minutes = lambda value: pd.to_timedelta(value, unit="min")
    half = minutes(config["event_half_window_minutes"])

    # Near-event: every event_cadence_minutes within ±half of each selected peak.
    flares = catalog[catalog.goes_class.str[:1].isin(list(config["event_classes"]))]
    offsets = np.arange(-config["event_half_window_minutes"], config["event_half_window_minutes"] + 1,
                        config["event_cadence_minutes"])
    near = pd.DataFrame({"kind": "near_event",
                         "peak_time": np.repeat(flares.peak_time.to_numpy(), len(offsets)),
                         "goes_class": np.repeat(flares.goes_class.to_numpy(), len(offsets)),
                         "offset_minutes": np.tile(offsets, len(flares))})
    near["timestamp"] = near.peak_time.dt.floor("min") + minutes(near.offset_minutes)
    # Where peak windows overlap, keep each timestamp's nearest peak.
    near = near.sort_values("offset_minutes", key=abs, kind="stable").drop_duplicates("timestamp")

    # Inter-event: a jittered grid that avoids every flare of any class, padded by
    # inter_event_guard_minutes, and every peak window.
    ranges = [(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)) for start, end in config["date_ranges"]]
    grid = pd.DatetimeIndex(np.concatenate([
        pd.date_range(start, end, freq=f"{config['inter_event_cadence_minutes']}min", inclusive="left")
        for start, end in ranges]))
    jitter = config["inter_event_jitter_minutes"]
    grid = (grid + minutes(np.random.default_rng(config["seed"]).integers(-jitter, jitter + 1, len(grid)))).to_numpy()
    guard = minutes(config["inter_event_guard_minutes"])
    busy_start = np.minimum(catalog.start_time - guard, catalog.peak_time - half).to_numpy()
    busy_end = np.maximum(catalog.end_time + guard, catalog.peak_time + half).to_numpy()
    order = np.argsort(busy_start)
    busy_start, latest_end = busy_start[order], np.maximum.accumulate(busy_end[order])
    # A grid time is during a flare if some interval that started before it hasn't ended yet.
    i = np.searchsorted(busy_start, grid, side="right") - 1
    between = pd.DataFrame({"timestamp": grid[(i < 0) | (grid > latest_end[i])], "kind": "inter_event"})

    plan = pd.concat([near, between], ignore_index=True).drop_duplicates("timestamp")
    t = plan.timestamp
    in_range = np.logical_or.reduce([(t >= start) & (t < end) for start, end in ranges])
    heldout_guard = minutes(config["heldout_guard_minutes"])
    near_heldout = np.logical_or.reduce([(t + shift).dt.normalize().isin(heldout_days)
                                         for shift in (-heldout_guard, pd.Timedelta(0), heldout_guard)])
    plan = plan[in_range & ~near_heldout & ~t.isin(existing)]
    if config["max_samples"] and len(plan) > config["max_samples"]:
        plan = plan.sample(config["max_samples"], random_state=config["seed"])
    return plan.sort_values("timestamp")[["timestamp", "kind", "peak_time", "goes_class", "offset_minutes"]]


def add_goes_flux(plan, goes_dir):
    """Attach the training pipeline's GOES XRS-B flux at each timestamp; drop timestamps without one."""
    from data.align_aia_sxr import create_combined_lookup_table, load_and_prepare_goes_data

    records = create_combined_lookup_table(load_and_prepare_goes_data(goes_dir), plan.timestamp)
    flux = pd.DataFrame(records, columns=["timestamp", "sxr_b", "instrument"]).rename(
        columns={"sxr_b": "xrsb_flux", "instrument": "satellites"})
    plan = plan.assign(timestamp=plan.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S")).merge(flux, on="timestamp")
    return plan[plan.xrsb_flux > 0].reset_index(drop=True)


def plan_samples(config):
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    existing = file_times(config["existing_dirs"])
    heldout_days = file_times(config["heldout_dirs"]).normalize().unique()
    print(f"{len(existing)} existing timestamps; {len(heldout_days)} held-out days")
    catalog = get_catalog(config, root / "catalog_cache")
    candidates = make_plan(config, catalog, existing, heldout_days)
    plan = add_goes_flux(candidates, config["goes_dir"])
    plan.to_csv(root / "samples.csv", index=False)
    print(f"Planned {len(plan)} samples {plan.kind.value_counts().to_dict()} "
          f"({len(candidates) - len(plan)} dropped for missing GOES data), "
          f"about {len(plan) * 7 * config['resolution'] ** 2 * 4 / 1024 ** 3:.0f} GiB")


def save_npy(path, array):
    """Write through a temporary name so an interrupted run never leaves a truncated .npy."""
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as stream:
        np.save(stream, array)
    os.replace(part, path)


def download(config, limit=None):
    """Fetch, process, and save every planned pair that isn't saved yet. Returns the failure count."""
    from tqdm import tqdm
    from data.convert_aia import AIAStackDataset
    from download.download_sdo import SDODownloader
    from itipy.data.editor import get_auto_calibration_table

    if not config.get("jsoc_emails"):
        raise SystemExit("Set jsoc_emails in the config")
    root = Path(config["output_dir"])
    plan = pd.read_csv(root / "samples.csv")
    aia_dir, sxr_dir = root / "AIA_processed" / "train", root / "SXR_processed" / "train"
    todo = [row for row in plan.itertuples()
            if not ((aia_dir / f"{row.timestamp}.npy").exists() and (sxr_dir / f"{row.timestamp}.npy").exists())]
    print(f"{len(plan) - len(todo)} of {len(plan)} planned pairs already saved")
    todo = todo[:limit]
    if not todo:
        return 0
    aia_dir.mkdir(parents=True, exist_ok=True)
    sxr_dir.mkdir(parents=True, exist_ok=True)
    staging = root / "staging"
    shutil.rmtree(staging, ignore_errors=True)  # FITS left behind by an interrupted run
    staging.mkdir()
    # ITI downloads this table lazily without a lock; fetch it once before the threads race for it.
    get_auto_calibration_table()

    worker = threading.local()
    worker_numbers = itertools.count()

    def start_worker():
        # Each thread keeps one JSOC email (assigned round-robin) and its own FITS staging folder.
        emails = config["jsoc_emails"]
        worker.dir = Path(tempfile.mkdtemp(dir=staging))
        worker.downloader = SDODownloader(str(worker.dir), email=emails[next(worker_numbers) % len(emails)],
                                          wavelengths=WAVELENGTHS, n_workers=config["wavelength_workers"],
                                          tolerance=config["aia_tolerance_seconds"])

    def fetch(row):
        fits_files = [worker.dir / str(wl) / f"{row.timestamp}.fits" for wl in WAVELENGTHS]
        try:
            worker.downloader.downloadDate(pd.Timestamp(row.timestamp).to_pydatetime())
            missing = [wl for wl, path in zip(WAVELENGTHS, fits_files) if not path.exists()]
            if missing:
                raise FileNotFoundError(f"no AIA frame within {config['aia_tolerance_seconds']} s for {missing}")
            stack = AIAStackDataset(data=[[str(path)] for path in fits_files], wavelengths=WAVELENGTHS,
                                    resolution=config["resolution"], ext=".fits", allow_errors=False)[0]
            save_npy(aia_dir / f"{row.timestamp}.npy", np.asarray(stack, dtype=np.float32))
            save_npy(sxr_dir / f"{row.timestamp}.npy", np.array([row.xrsb_flux], dtype=np.float32))
        finally:
            for path in fits_files:
                path.unlink(missing_ok=True)

    failures = []
    pool = ThreadPoolExecutor(config["parallel_timestamps"], initializer=start_worker)
    try:
        futures = {pool.submit(fetch, row): row.timestamp for row in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc="AIA/GOES pairs"):
            if future.exception() is not None:
                failures.append({"timestamp": futures[future], "error": repr(future.exception())})
                tqdm.write(f"{futures[future]}: {future.exception()!r}")
    finally:
        # On Ctrl-C, drop queued timestamps; downloads already in progress still finish.
        pool.shutdown(cancel_futures=True)
        pd.DataFrame(failures, columns=["timestamp", "error"]).to_csv(root / "failed.csv", index=False)
    print(f"Saved {len(todo) - len(failures)} pairs; {len(failures)} failed (see failed.csv, rerun to retry)")
    return len(failures)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="download/expand_training_data.yaml")
    parser.add_argument("--mode", choices=["plan", "download"], default="plan")
    parser.add_argument("--limit", type=int, help="Download at most this many pairs in this run")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.mode == "plan":
        plan_samples(config)
        return 0
    return 1 if download(config, args.limit) else 0


if __name__ == "__main__":
    raise SystemExit(main())
