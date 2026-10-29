# Python API guide

[Back to the README](../README.md)

Use this guide for observation inputs, time ranges, saved data, plotting, and
measured GOES comparisons. Code examples assume an installed checkout and a
compatible checkpoint on your machine.

- [One observation and flux maps](#use-foxes-from-a-python-ui)
- [Timestamp formats](#flexible-timestamps)
- [Time ranges and HARP light curves](#time-ranges-and-harp-light-curves)
- [Save and reload observations](#save-processed-observations-and-reload-them-later)
- [Bring your own arrays or FITS files](#predicting-your-own-observations)
- [Measured GOES SXR](#measured-goes-sxr)

## Use FOXES from a Python UI

Install this checkout in the UI environment:

```bash
python -m pip install -e .
```

Load a checkpoint once when the UI starts, then reuse it for each image:

```python
import helio_foxes

model = helio_foxes.load_model("/path/to/model.ckpt", device="auto")
processor = helio_foxes.ObservationProcessor(email="you@example.com")
observation = processor.fetch_observation("2023-08-01T00:00:00Z")
print(observation.data.shape)  # (7, 512, 512)
result = helio_foxes.predict(observation, model=model)

print(result.flux)       # full-disk SXR flux, W/m²
print(result.flux_map)   # patch contributions, W/m²
print(result.harp_regions)  # HARP numbers, locations, and flux sums, W/m²
print(result.sxr)        # measured GOES XRS-A/XRS-B fluxes and sample time
ax = helio_foxes.plot_flux_map(result)  # optional: vmin=1e-10, vmax=1e-6
ax.figure.savefig("flux_map.png")
```

For a notebook walkthrough, see [heliofoxes.ipynb](../tutorials/heliofoxes_demo/heliofoxes.ipynb).
The package name for installation is `helio-foxes`; the Python import is `helio_foxes`.

When plotting a prediction, the title automatically shows the predicted FOXES
flux and the observed SXR-B mean when available, in W/m². For Gaussian models,
it also shows the model's full-disk uncertainty beside the predicted flux as
`± value`, computed as the square root of the summed patch variances. This is
a model estimate, not a calibrated error bar.

`ObservationProcessor` calls `AIAProcessor`, `HARPProcessor`, and `SXRProcessor`
for the same requested timestamp. All three sources are fetched by default.
AIA and HARPs share the crop geometry and JSOC email; GOES SXR needs no email.
Prediction adds a `flux_sum` in W/m² and a canonical lowercase `harpnum` to
each region in `result.harp_regions`. Each result includes `top_pct`,
`left_pct`, `width_pct`, and `height_pct` for its location in the image.
Original JSOC fields, including `LON_MIN`, `LON_MAX`, `LAT_MIN`, and `LAT_MAX`,
are preserved. The observation's queried rows are not modified.

```python
for harp in result.harp_regions or []:
    print(harp["harpnum"], harp["flux_sum"], "W/m²")
```

Plotting reuses these regions and the same flux summation rule:

```python
ax = helio_foxes.plot_flux_map(result, show_harp_boxes=True)
```

For Gaussian models, each HARP result also contains `flux_sum_uncertainty`
in W/m²: square the patch uncertainties, sum within the same box as the flux,
then take the square root. Empty boxes have zero uncertainty; deterministic
models return `None`. These values follow the model's additive variance
attribution; they have not been calibrated as regional error bars.

Uncertainty appears on a third label line as `± value` when available.
Hide it with `show_harp_uncertainty=False`:

```python
ax = helio_foxes.plot_flux_map(result, show_harp_boxes=True, show_harp_uncertainty=False)
for harp in result.harp_regions or []:
    print(harp["harpnum"], harp["flux_sum"], harp["flux_sum_uncertainty"])
```

Each HARP box is labeled with its number and flux sum in W/m². Labels use a
fixed font size; set `harp_label_fontsize` to change it. A standard
Matplotlib annotation places each label diagonally outside a box corner with
a three-point gap in both x and y. The entire label extends outward from
the corner. The side
toward the image center is preferred. A simple overlap pass tries the other
corners before four-point nudges, keeping automatic positions outside their
HARP box and inside the map. Shifted
labels get connector lines. `harp_label_max_shift=80` limits movement to 80
points from a corner; lower it to keep labels closer, or set it
to zero for labels fixed at their preferred corner. If no free position is found within the limit, the
label stays at its starting position. Crowded cases may need manual offsets
or a higher `min_harp_flux`. Labels are not automatically hidden. This runs
once when plotting; replot after substantial size changes.

For crowded maps, set individual labels' starting positions. Offsets are in points
from the chosen corner: positive x moves right, positive y moves up. Manually
offset labels get matching connector lines:

```python
ax = helio_foxes.plot_flux_map(
    result, show_harp_boxes=True,
    harp_label_offsets={1234: (15, 30), 5678: (-20, -30)},
)
```

Box outlines and labels share a fixed
color by flux rank (strongest: sky blue, second: orange, third: green).

Expand HARP boxes when fetching observations so prediction sums and plot
outlines use the same expanded region:

```python
processor = helio_foxes.ObservationProcessor(email=email, harp_bbox_expansion=0.1)
observation = processor.fetch_observation(timestamp)
result = helio_foxes.predict(observation, model=model)
```

`harp_bbox_expansion` defaults to zero. `0.1` adds 10% of each box's width and
height on each side (20% more total width and height), clipped to the image.
Original JSOC heliographic bounds are preserved. For direct HARP queries, use
`helio_foxes.HARPProcessor(email=email, bbox_expansion=0.1)`. The expansion is carried
in observation metadata for HARP queries deferred until plotting.

Set `min_harp_flux` to hide regions below an absolute flux sum in W/m²:

```python
ax = helio_foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=1e-8)
```

The default threshold is `1e-7` W/m²; use `min_harp_flux=0` to show all regions.
Filtering only affects the plot;
`result.harp_regions` retains all predicted regional sums.

Set `include_aia=False`, `include_harps=False`, or `include_sxr=False` on
`ObservationProcessor` to disable a source. Override any setting for one
observation with `processor.fetch_observation(timestamp, include_sxr=False)`
(and similarly for AIA/HARPs). With AIA disabled, `observation.data` is `None`,
and prediction requires fetching AIA first. Each component processor handles
only its own source.
If the observation did not fetch HARPs, plotting with `show_harp_boxes=True`
queries them using the prediction metadata and its `jsoc_email` argument.
Pass `harp_regions=regions` to the plot to use an explicit set of regions.

`harp_regions=None` means no query was requested; `[]` means no usable HARP
regions were returned, including query failures. Plotting reuses an empty
result without repeating the query. HARP data is supplementary, so an empty
HARP result does not discard the AIA observation.

`observation.metadata` is preserved as `result.metadata`. It records the
requested UTC timestamp, channel order, original processed image shape, and
crop padding. The timestamp is the requested time, not the individual AIA
exposure times. The JSOC email stays on the processor or plotting call.

For array-based workflows, `helio_foxes.AIAProcessor(email=email).fetch_date(timestamp)`
returns a NumPy stack. Supply its time once to `helio_foxes.predict(image, model=model,
timestamp=timestamp)` to enable automatic HARP queries from the result. This
assumes the standard AIA channel order and crop padding of 0.1. For a different
crop, wrap the array in `helio_foxes.Observation` with explicit
`helio_foxes.ObservationMetadata`. Plotting a plain flux-map array also continues to
accept an explicit `timestamp`.

Each sum includes positive patch contributions whose centers fall inside the
HARP box. Boxes can overlap, so their sums need not add to the full-disk flux.

## Flexible timestamps

All public timestamp arguments accept `datetime`, `date`, `numpy.datetime64`,
ISO strings, year-first slash/dot dates, and named-month dates. For example,
`"2023-08-01T12:00:00Z"`, `"2023/08/01 12:00"`,
`"2023.08.01_12:00:00_UTC"`, and `"August 1, 2023 12:00 UTC"` work.
`helio_foxes.parse_timestamp(value)` lets you validate a value before fetching.
It returns an aware UTC datetime. Date-only inputs mean midnight; missing
timezones mean UTC; explicit offsets are converted to UTC.

Incomplete/invalid dates raise `ValueError` with examples. Ambiguous numeric
dates such as `"01/08/2023"` and timezone abbreviations such as `EST` are
rejected: use year-first dates and numeric offsets. Unix timestamp numbers
are rejected with `TypeError`, avoiding accidental unit assumptions.

## Time ranges and HARP light curves

```python
processor = helio_foxes.ObservationProcessor(email="you@example.com")
model = helio_foxes.load_model("/path/to/model.ckpt")  # load once

# Download and preprocess at a 12-minute cadence, then predict each stack.
# The iterator avoids retaining all processed AIA arrays in memory.
observations = processor.iter_observations(
    "2023-08-01 00:00", "2023-08-01 06:00", cadence="12min",
    on_error="skip",  # log missing/bad AIA samples and continue
    progress=True,  # tqdm during the bulk query and observation processing
)
series = model.predict_series(observations)
totals = series.to_dataframe()  # UTC index; flux, observed_xrsb_flux (W/m²)
curves = series.harp_light_curves()  # {integer HARP number: DataFrame}
for harpnum, curve in curves.items():
    curve.to_csv(f"harp_{harpnum}.csv")

ax = series.plot()  # one line per available HARP
ax.figure.savefig("harp_light_curves.png")
# Optional: series.plot(harpnums=[1234], include_full_disk=True, include_sxr=True)

# To keep the processed observations for later use instead:
observations = list(processor.iter_observations(
    "2023-08-01", "2023-08-01 01:00", cadence="12min",
))
```

Ranges include the start and any cadence-aligned timestamp at or before the
end. Cadence accepts duration strings (`"1h"`, `"12min"`), `timedelta`, or
numeric seconds, in positive whole seconds. The default cadence is one hour.
The default error policy is `"raise"`; `"skip"` logs individual AIA download
or preprocessing failures. Initial bulk query failures and prediction errors
raise. Source toggles such as `include_sxr=False` work on range methods too.

AIA resolves records and image segments with **one bulk JSOC query** using a
`aia.lev1_euv_12s[startZ/durations@12s][wavelengths]{image}` selector over the
range padded by the tolerance. A separate series-key lookup is cached by the
downloader. Querying the native 12-second records allows nearest-frame
selection within tolerance even when the requested cadence is not aligned
with JSOC records. Only the selected frames are downloaded, then each complete
seven-channel set is preprocessed with the single-observation pipeline. HARP
and GOES queries currently run per requested timestamp. Large ranges can
return large metadata tables; split long jobs into smaller ranges. Raw FITS
files are temporary; `list(processor.iter_observations(...))` retains processed arrays in memory.

Curve indices are requested UTC times, not channel exposure times. Different
requests may select the same exposure. HARP identities use HARPNUM, not NOAA
numbers. Each curve includes `flux_sum` and `flux_sum_uncertainty` in W/m².
An absent HARP at a successful timestamp produces `NaN` and a plotted gap;
skipped AIA timestamps are absent entirely. There is no interpolation or zero
filling. Existing positive patch-center box sums and overlapping-box behavior
apply. Gaussian regional uncertainty is an additive variance attribution,
not a calibrated regional error bar. HARP query failures also produce gaps.

To place the selected flux map and matching AIA image to the right of the curves:

```python
start = helio_foxes.parse_timestamp("2023-08-01 00:00")
end = helio_foxes.parse_timestamp("2023-08-01 06:00")
series = model.predict_series(
    processor.iter_observations(start, end, cadence="12min", progress=True),
    snapshot_time=start + (end - start) / 2,  # Or an explicit timestamp string.
    snapshot_wavelength=193,
)
ax = series.plot(show_snapshot=True)
# Toggle off without predicting again:
# ax = series.plot(show_snapshot=False)
```

Supply `snapshot_time` to retain the nearest successful observation; ties favor
the earlier time. The example above targets the query midpoint. During prediction,
the API keeps one channel image and replaces it whenever a closer observation
arrives. No temporary image files are written. Omit `snapshot_time` to retain no image.
Change the target time or wavelength by running prediction again; only the chosen
image is retained, so it cannot be reselected at plot time. Saved observations can
be reused to avoid another download and preprocessing pass.

The three panels are light curves, the selected prediction rendered directly
by `plot_flux_map` (including its logarithmic flux scale, colorbar, and HARP
overlays), and a plain AIA image of the chosen wavelength. HARP colors match
the curves. A dashed vertical line marks their common requested UTC time.
Only selected HARPs visible at that time have boxes on the flux map; all
selected curves remain plotted. The AIA image has no overlays and uses a
display-only percentile stretch. The snapshot is available as `series.snapshot`,
a one-channel observation for visualization rather than model input. No
additional download is needed. For an existing figure, supply `ax` (curves),
`snapshot_ax` (flux map), and `aia_ax`; otherwise `series.plot` creates all three automatically
and returns the curve Axes.

Regional uncertainty bands and snapshot `±` labels are enabled by default
when the prediction supplies them. `series.plot(show_uncertainty=False)` hides
both. Bands show regional flux ± one `flux_sum_uncertainty`;
these remain Gaussian variance attributions, not calibrated regional confidence
intervals. Deterministic models produce plain curves/labels automatically.
The flux axis is logarithmic: zero, negative, and nonfinite flux samples are
displayed as gaps, and uncertainty bands with nonpositive lower limits are
omitted. Unknown uncertainties and missing regions are not filled or bridged.

Use `min_peak_flux` in W/m² to show only HARPs that reach a chosen peak anywhere
in the series (not just at the selected snapshot):

```python
ax = series.plot(show_snapshot=True, min_peak_flux=1e-7, legend="compact")
```

This filters the curves and HARP overlays without changing the prediction
tables. HARPs with no positive samples are always omitted. Colors stay stable
when the threshold changes. The default `legend="auto"` uses a compact legend
below the SXR axes when the flux map is visible, with HARP IDs on that map;
without snapshots it places the full HARP legend below the axes. Choose
`legend="full"` to list every displayed HARP, `"compact"` for a region count
and comparison/marker keys, or `"none"` to hide the legend.

## Save processed observations and reload them later

Pass `save_dir` to `iter_observations` to save each successful
observation **before it is yielded**. Files are compressed NumPy `.npz` archives
containing the processed array, UTC timestamp, wavelengths, crop geometry,
HARP regions, and measured SXR data. They use JSON metadata and no pickle.

```python
for observation in processor.iter_observations(
    "2023-08-01", "2023-08-01 06:00", cadence="12min",
    save_dir="saved_observations", progress=True,
):
    print(observation.metadata.timestamp)  # already saved; prediction is optional

# Later, without downloading or preprocessing again:
saved = helio_foxes.load_observations("saved_observations", progress=True)
series = model.predict_series(saved)
series.plot()

# Save or load just one complete observation:
path = observation.save("one_observation.npz")
restored = helio_foxes.Observation.load(path)
result = model.predict(restored)
```

Generated filenames include the requested UTC time and sort chronologically.
`load_observations` reads one archive at a time, in filename order. Use a
dedicated directory for these archives; missing/empty directories and corrupt
archives raise errors. Optional AIA/HARP/SXR values, including `None` versus an
empty HARP list, are preserved. Reloading applies no additional normalization:
use the same predictor normalization settings as for the original observation.

Saving is opt-in. When `save_dir` is set, `iter_observations` automatically loads and
yield existing archives, skipping their downloads, preprocessing, and HARP/SXR
queries. Only missing timestamps are downloaded and processed. A fully saved
range needs no data queries, so rerunning an interrupted job resumes the missing
work while still returning the complete available range in time order.
Use `overwrite=True` to refresh the range deliberately, including existing
files. Cached observations reuse their saved metadata and enabled-source contents;
refresh after changing preprocessing/normalization or query settings. Disabled
sources are omitted from the returned observation without changing the archive.
Corrupt archives, mismatched timestamps, or missing required AIA data raise
errors instead of being silently replaced.
Each new file is published after a complete write, so successfully saved
observations survive an interruption. Load/save errors always raise, including
with `on_error="skip"`. Standalone `observation.save(path)` still raises
`FileExistsError` unless `overwrite=True`. For a plain `.npy` array without metadata, use NumPy's
`np.save` and the existing `Observation.from_npy` constructor instead.

`progress=True` enables tqdm in terminals and notebooks. The AIA bar initially
shows `Querying JSOC` at zero, then advances once per requested observation,
including skipped attempts. Its ETA includes time spent processing and consuming
each observation (including prediction when streaming into `predict_series`).
`load_observations(..., progress=True)` also shows loading progress. Progress is
disabled by default.

## Predicting your own observations

For an **already preprocessed** array, use `Observation.from_array`:

```python
import numpy as np
import helio_foxes

stack = np.load("my_processed_stack.npy", allow_pickle=False)
observation = helio_foxes.Observation.from_array(
    stack, timestamp="2023/08/01 12:00",
    wavelengths=(94, 131, 171, 193, 211, 304, 335),
    padding_factor=0.1,
    harp_regions=[],  # or your own boxes; None means HARPs were not queried
)
# Equivalent shortcut for one .npy stack:
observation = helio_foxes.Observation.from_npy(
    "my_processed_stack.npy", timestamp="2023-08-01 12:00", harp_regions=[],
)
model = helio_foxes.load_model("/path/to/model.ckpt", aia_norm_path="aia_norm.npz")
result = model.predict(observation)
# A list or generator of these observations also works with model.predict_series.
```

The array must be finite, real, channel-first `(7, H, W)` in the wavelength
order shown. Model patch/grid settings determine H and W (typically 512).
The constructor copies it to float32 and infers the metadata shape; prediction
checks channel order and model compatibility. It does **not** calibrate,
resize, crop, or normalize your array. Match the training preprocessing.
Use the checkpoint's normalization artifact exactly once; omit `aia_norm_path`
from model loading if that transform has already been applied to your stack.
Optional HARP boxes need `HARPNUM` (or `harpnum`) plus `top_pct`, `left_pct`,
`width_pct`, and `height_pct` in percentages of this exact crop, measured from
the top-left display corner. Predictions do not automatically query HARPs.

For **local raw AIA FITS**, run the same ITI preprocessing used by downloads:

```python
aia = helio_foxes.AIAProcessor(resolution=512)
paths = [f"my_fits/{w}.fits" for w in aia.WAVELENGTHS]  # one file per channel
stack = aia.process_files(paths)  # preserves your local FITS files
observation = helio_foxes.Observation.from_array(
    stack, timestamp="2023-08-01 12:00", harp_regions=[],
)
result = model.predict(observation)
```

Supply all seven files in wavelength order and use a mutually consistent
observation time and crop. `process_files` handles preprocessing only; it
does not fetch HARP or SXR data. Optional `harp_regions` and `sxr` can be passed
to `from_array` for an entirely local workflow.

## Measured GOES SXR

The observation's `sxr` dict is also carried into `result.sxr`; it is a measured
comparison value and is not used as model input. XRS-B is the 1–8 Å band to
compare with the predicted `result.flux`; XRS-A is 0.5–4 Å. Fluxes are in W/m².

```python
processor = helio_foxes.ObservationProcessor(
    email="you@example.com", include_sxr=True, sxr_tolerance=120,
    sxr_satellite=None,  # optionally restrict to e.g. 18
)
observation = processor.fetch_observation("2023-08-01T00:00:00Z")
result = helio_foxes.predict(observation, model=model)
if result.sxr is not None:
    print("Predicted:", result.flux, "Observed:", result.sxr["xrsb_flux"])
    print(result.sxr["satellites"], result.sxr["timestamp"])
    print(result.sxr["measurements"])  # individual contributing satellite values
    print(result.sxr["time_offset_seconds"])  # sample minus requested UTC time

# Fetch just SXR, using the same datetime/ISO string format:
sxr = helio_foxes.SXRProcessor(tolerance=120, satellite=18).fetch_date(timestamp)
# Or through the observation fetcher:
sxr_only = processor.fetch_observation(timestamp, include_aia=False, include_harps=False)
```

`SXRProcessor` downloads the GOES science `avg1m` product and finds the nearest
usable timestamp within `tolerance` seconds (120 by default; time ties favor
the earlier sample). At that timestamp, `xrsb_flux` is the arithmetic mean of
available satellites' finite positive XRS-B values with zero XRS-B quality
flags. This follows `data/align_aia_sxr.py`: average in physical W/m² before
the training transform applies log normalization. `result.flux` also uses
physical W/m², so no normalization is needed for comparison.

`satellites` names the contributors and `measurements` retains their individual
values. Each satellite counts once; measurements at different timestamps are
never averaged together. Restricting `satellite` yields a single-satellite
result. XRS-A is averaged over valid short-band measurements from the XRS-B
contributors, or `None` if unavailable. Bad XRS-A values do not exclude good
XRS-B measurements from the model comparison. The returned `timestamp` is the
actual sample time, while `observation.metadata.timestamp` remains the requested
time. Daily files are filtered to the requested window and deleted after use.
No interpolation or additional legacy scaling is applied to the reprocessed
science fluxes. `sxr=None` means fetching was disabled or no usable measurement
was available, including network or processing failures; these failures log a
warning and do not discard the AIA observation. The science archive may lag
recent observations. Check the calibration used for older training datasets
before interpreting measured-versus-predicted differences. Unlike the bulk
dataset converter, this fetcher does not fill missing fluxes using log-space
interpolation; only valid measured samples contribute.

GOES netCDF reading requires `h5netcdf` and `h5py`, both declared package
dependencies. If an existing environment reports either missing, run
`python -m pip install h5netcdf h5py` using that environment's Python. In a
notebook, run `%pip install h5netcdf h5py` and restart the kernel.

`AIAProcessor.fetch_date` writes the seven raw AIA FITS frames to a temporary
directory, processes them with ITI, and returns the stack in memory.
`ObservationProcessor.fetch_observation` uses this same processing path. The raw files are
removed when the call finishes, including when a download is incomplete.
The requested timestamp is interpreted as UTC. An ISO string with an offset
is converted to UTC; a timestamp without an offset is assumed to be UTC.
If you provide `aia_norm_path` to `AIAProcessor`, do not also pass that
artifact to `load_model`, since that would normalize the stack twice.

The input uses channel order `[94, 131, 171, 193, 211, 304, 335]` and must
already have the same preprocessing as the checkpoint's training data. If that
training used an AIA normalization artifact, pass `aia_norm_path` to
`helio_foxes.load_model`. `result.intervals` and `result.patch_uncertainty_map` are
available for the coupled Gaussian model. Checkpoints are passed by path and
are not bundled into the Python package. Pass a path to a **full checkpoint
file** available on the UI machine.
