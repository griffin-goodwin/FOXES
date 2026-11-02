Additional training data
========================

`download/expand_training_data.py` downloads new AIA/GOES training pairs that
don't overlap the existing dataset. Run it from the repository root in the
`foxes` environment. All settings are in `download/expand_training_data.yaml`.

```bash
conda activate foxes
python -m download.expand_training_data --mode plan
python -m download.expand_training_data --mode download --limit 2   # check these two pairs first
python -m download.expand_training_data --mode download
```

**plan** writes `samples.csv` in `output_dir`:

1. Reads the timestamps in `existing_dirs` (skipped) and the days in
   `heldout_dirs` (no sample on those days or within `heldout_guard_minutes` of them).
   A missing folder is an error, so an unmounted drive can't leak validation days.
2. Queries the HEK GOES flare list one month at a time, cached in `catalog_cache/`.
3. Picks `near_event` samples every `event_cadence_minutes` within
   ±`event_half_window_minutes` of each peak in `event_classes`, keeping the
   nearest peak where windows overlap. It then picks `inter_event` samples on a
   jittered `inter_event_cadence_minutes` grid that avoids every flare (any
   class, padded by `inter_event_guard_minutes`) and every peak window.
4. Randomly keeps `max_samples` of them (seeded), then looks up the GOES XRS-B
   flux with `data/align_aia_sxr.py`, the code that built the existing
   training set. It drops times without a value, so the final count can be
   slightly under `max_samples`.

Rerunning `plan` rebuilds `samples.csv` from the current config.

**download** fetches every planned timestamp that isn't saved yet. It uses
`SDODownloader` from `download_sdo.py` (nearest frame within
`aia_tolerance_seconds`, preferring quality 0) and `AIAStackDataset` from `data/convert_aia.py`, the
same ITI processing as the existing data. `parallel_timestamps` threads run at
once, each with its own JSOC email from `jsoc_emails` (round-robin). Raw FITS are
deleted after each timestamp. Failures are listed in `failed.csv`; rerun to
retry them. Ctrl-C stops queued work, and saved pairs are kept.

Output layout:

```text
AIA_processed/train/2014-04-06T12:00:00.npy  # float32 (7, 512, 512)
SXR_processed/train/2014-04-06T12:00:00.npy  # float32 (1,), XRS-B in W/m²
samples.csv                                  # planned times, provenance, GOES flux
catalog_cache/goes_YYYY-MM.csv               # cached HEK queries
failed.csv                                   # failures from the last download run
```

Each pair is about 7 MiB, so 100,000 samples need about 700 GiB. The new
folder only has a `train` split. Point training at it when you're ready to
combine datasets.
