# Complete Python API reference

[README](../README.md#api-overview) · [Usage guide](python_api.md)

Generated from the public API's source signatures and docstrings. All explicit keyword arguments and defaults are included. For methods accepting `**kwargs`, the docstring lists the forwarded options.

Regenerate after API edits with `python tools/generate_api_reference.py`.


- [Observation](#observation)
- [ObservationMetadata](#observationmetadata)
- [ObservationProcessor](#observationprocessor)
- [AIAProcessor](#aiaprocessor)
- [HARPProcessor](#harpprocessor)
- [SXRProcessor](#sxrprocessor)
- [Prediction](#prediction)
- [Predictor](#predictor)
- [load_model](#load_model)
- [predict](#predict)
- [plot_flux_map](#plot_flux_map)
- [plot_uncertainty_map](#plot_uncertainty_map)
- [parse_timestamp](#parse_timestamp)
- [PredictionSeries](#predictionseries)
- [load_observations](#load_observations)

## Observation

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L183-L382)


A processed observation, its metadata, and optional HARP/SXR data.

`data=None` means AIA fetching was disabled.
`harp_regions=None` means no HARP query was requested. An empty list
means a query returned no usable regions, including query failures.
`sxr` is the measured GOES result from `SXRProcessor`, or
`None` when disabled or unavailable.


#### Parameters


**metadata : ObservationMetadata**

Timestamp, channel order, and geometry describing this exact array.

**data : array-like or None**

Channel-first (C, H, W) data matching metadata.image_shape, or None
when AIA is disabled. Use from_array for finite float32 validation.

**harp_regions : list of dict or None, optional**

Each region needs HARPNUM or harpnum (integer-like) and numeric
top_pct, left_pct, width_pct, height_pct, in percent of the exact crop.
Origin is the top-left display corner; dimensions must be positive.
For example: {"HARPNUM": 1234, "top_pct": 10, "left_pct": 20,
"width_pct": 15, "height_pct": 12}. None means unqueried; [] means
explicitly no usable regions. Prediction does not query HARPs itself.

**sxr : dict or None, optional**

Measured flux/provenance from SXRProcessor.fetch_date; xrsb_flux and
xrsa_flux are in W/m². This is comparison data, not a model input.

### Observation.from_array

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L224-L269)

```python
helio_foxes.Observation.from_array(data, *, timestamp, wavelengths=(94, 131, 171, 193, 211, 304, 335), padding_factor=0.1, harp_regions=None, sxr=None)
```

Wrap an already preprocessed channel-first AIA stack, without downloading.

This does not calibrate, crop, resize, or normalize the input. Match
training preprocessing and apply the checkpoint's normalization once.
Optional HARP boxes use percentages of this exact crop, measured from
its top-left display corner. `None` leaves HARPs unqueried; `[]`
explicitly supplies no regions. The predictor validates model shape
and channel order. The returned observation owns a float32 copy.


#### Parameters


**data : array-like**

Finite real numeric (C, H, W) stack, usually (7, 512, 512), in channel
order. Nested numeric lists or a NumPy array work. Channels-last arrays
are not transposed automatically. The returned array is a float32 copy.

**timestamp : datetime.datetime, datetime.date, numpy.datetime64, or str**

Required keyword. Formats accepted by parse_timestamp, e.g.
"2023/08/01 12:00" or "2023-08-01T12:00:00Z". Naive values mean UTC.

**wavelengths : sequence of int, optional**

One Angstrom wavelength per channel. Prediction requires the default
(94, 131, 171, 193, 211, 304, 335); one-channel snapshots cannot be predicted.

**padding_factor : float, default 0.1**

Nonnegative padding fraction matching the preprocessing crop.

**harp_regions : list of dict or None, optional**

HARP identifiers and top/left/width/height percentages; see Observation.

**sxr : dict or None, optional**

Optional measured SXR flux/provenance; see Observation.


#### Returns

Observation
Validated input container, ready for a shape-compatible predictor.

### Observation.from_npy

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L272-L292)

```python
helio_foxes.Observation.from_npy(path, *, timestamp, **kwargs)
```

Load one processed `.npy` stack; see `from_array`.


#### Parameters


**path : str or pathlib.Path**

Local .npy file containing exactly one numeric (C, H, W) stack, e.g.
"processed_aia.npy". Pickled/object arrays are disabled. Complete
.npz observation archives must instead use Observation.load.

**timestamp : datetime.datetime, datetime.date, numpy.datetime64, or str**

Required keyword; e.g. "2023-08-01 12:00". Naive values mean UTC.

**Additional keyword arguments (`**kwargs`)**

wavelengths, padding_factor, harp_regions, and sxr, with the same
types and meanings as from_array. Timestamp is not read from filename.


#### Returns

Observation
Processed array plus the explicitly supplied metadata.

### Observation.save

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L294-L353)

```python
helio_foxes.Observation.save(path, *, overwrite=False)
```

Save a complete observation as a compressed NumPy `.npz` archive.

Stores `data` (if present) and scalar JSON `metadata`, including
the UTC time, crop geometry, channel order, HARP rows, and SXR data.
No pickle is used. Arrays retain their dtype and values. A temporary
file in the destination directory is published only after a complete
write. Existing files raise FileExistsError unless overwrite=True.
Returns the saved Path; load with `load`, without preprocessing
or normalizing again. `from_npy` remains for plain array-only files.


#### Parameters


**path : str or pathlib.Path**

Destination ending in .npz, e.g. "saved/observation.npz". Parent
directories are created and a leading ~ is expanded. This is a complete
archive, unlike a plain .npy array.

**overwrite : bool, default False**

True atomically replaces the file; False raises FileExistsError if it
exists. Range iterators separately reuse existing saved observations.


#### Returns

pathlib.Path
Destination of the completed archive.

### Observation.load

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L356-L382)

```python
helio_foxes.Observation.load(path)
```

Reload a complete archive made by `save`, with pickle disabled.


#### Parameters


**path : str or pathlib.Path**

Existing complete .npz archive produced by Observation.save, e.g.
"saved/observation.npz". Not a raw FITS file or array-only .npy file.


#### Returns

Observation
Restored numeric array, datetime metadata, and optional HARP/SXR data.
No preprocessing, normalization, network access, or pickle execution.

## ObservationMetadata

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L133-L179)


Requested UTC time, channel order, and processed AIA crop geometry.

`timestamp` is the requested time, not the actual exposure time of every
channel. `image_shape` is the channel-first crop shape before model
patching. If AIA fetching is disabled, it is the configured target shape.
`padding_factor` describes the crop's padding beyond the solar radius.
`harp_bbox_expansion` records the per-side HARP box expansion, including
for HARP queries deferred until plotting.


#### Parameters


**timestamp : datetime.datetime, datetime.date, numpy.datetime64, or str**

Complete observation date/time accepted by parse_timestamp, e.g.
"2023/08/01 12:00" or "2023-08-01T12:00:00Z". Naive inputs mean UTC.

**image_shape : sequence of three positive int**

Channel-first (C, H, W), e.g. (7, 512, 512); never (H, W, C).

**wavelengths : sequence of int, optional**

Channel wavelengths in Angstroms, one per channel, in array order.
FOXES prediction requires (94, 131, 171, 193, 211, 304, 335).

**padding_factor : float, default 0.1**

Nonnegative crop padding beyond the solar radius; not a pixel count.

**harp_bbox_expansion : float, default 0.0**

Nonnegative fraction of box width/height added on each side; 0.1 means
10 percent per side, not 0.1 pixels.

## ObservationProcessor

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L987-L1254)


Fetch AIA, HARP, and measured GOES SXR data for one UTC time.

All sources are enabled by default. `include_aia`, `include_harps`, and
`include_sxr` can be set here or overridden on `fetch_observation`. Disabled AIA
fetching produces `Observation.data=None`; disabled HARP fetching
produces `Observation.harp_regions=None`.

`email`, `resolution`, `tolerance`, `n_workers`, and `aia_norm_path`
configure the component processors. AIA and HARP share the same crop
geometry and email. HARP failures return an empty region list; AIA
downloading or processing failures raise as in `AIAProcessor`.
`harp_bbox_expansion` adds that fraction on each side of projected HARP
boxes, clipped to the image. Prediction sums use the expanded boxes.
`sxr_tolerance` (seconds) and `sxr_satellite` configure the independent
`SXRProcessor`. Disabled or unavailable SXR produces `sxr=None`.

### ObservationProcessor

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1005-L1045)

```python
helio_foxes.ObservationProcessor(*, email=None, resolution=512, tolerance=120, n_workers=4, aia_norm_path=None, include_aia=True, include_harps=True, harp_bbox_expansion=0.0, include_sxr=True, sxr_tolerance=120, sxr_satellite=None)
```

Configure observation sources and shared crop geometry.


#### Parameters


**email : str or None, optional**

Registered JSOC email shared by AIA and HARP queries.

**resolution : int, default 512**

Positive processed AIA height/width in pixels; match the checkpoint.

**tolerance : int, default 120**

AIA search half-width in seconds; independent of sxr_tolerance.

**n_workers : int, default 4**

AIA download thread count.

**aia_norm_path : str or pathlib.Path or None, optional**

Matching .npz/.csv AIA normalization artifact. Do not also normalize
in the predictor when this is supplied.

**include_aia, include_harps, include_sxr : bool, default True**

Default source toggles, overridable per fetch/range call.

**harp_bbox_expansion : float, default 0.0**

Nonnegative box expansion fraction per side; 0.1 means 10% per side.

**sxr_tolerance : float, default 120**

Finite positive GOES lookup half-width in seconds.

**sxr_satellite : positive int or None, optional**

Restrict GOES to a number such as 18, or use all valid satellites.

### ObservationProcessor.fetch_observation

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1047-L1093)

```python
helio_foxes.ObservationProcessor.fetch_observation(date, *, include_aia=None, include_harps=None, include_sxr=None) -> Observation
```

Fetch enabled sources for any supported complete date/time format.

Per-call toggles override the processor's defaults for this call only.
Metadata and any fetched HARP/SXR data travel with the returned
observation through prediction and plotting. AIA data is required
for prediction. Raw downloads are removed before returning.


#### Parameters


**date : datetime.datetime, datetime.date, numpy.datetime64, or str**

Complete time accepted by parse_timestamp, e.g. "2023/08/01 12:00",
"2023-08-01T12:00:00Z", or "August 1, 2023". Date-only values mean
midnight and naive values mean UTC; numeric epoch times are not accepted.

**include_aia, include_harps, include_sxr : bool or None, optional**

True enables the source, False disables it, and None uses the constructor
default. The constructor settings are not modified.


#### Returns

Observation
Requested UTC time and processed array, plus enabled HARP/SXR data.

### ObservationProcessor.iter_observations

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1095-L1254)

```python
helio_foxes.ObservationProcessor.iter_observations(start, end, *, cadence='1h', on_error='raise', save_dir=None, overwrite=False, progress=False, **kwargs)
```

Download/preprocess a UTC range one observation at a time.

Includes start and any cadence-aligned time <= end. Cadence accepts
a pandas duration string (`'12min'`), timedelta, or seconds as a
number. Source toggles are passed to `fetch_observation`.
AIA uses one bulk JSOC record/segment query across the padded range,
followed by downloads of the selected frames. HARP and SXR queries
currently run separately for each timestamp. Large spans can have
large metadata responses; split long jobs into smaller ranges.
Defaults to raising AIA errors. `on_error='skip'` logs failures
with their requested timestamps and continues. HARP/SXR retain
their single-observation best-effort behavior. No interpolation is
performed; nearby requests can select the same exposure. Raw files
are temporary. Use this iterator to avoid retaining all AIA arrays.
`save_dir` writes each successful observation as a complete .npz
archive before yielding it; see `Observation.save` and
`load_observations`. Saving is opt-in and never changes the data.
Existing archives are loaded and yielded without downloading,
preprocessing, or HARP/SXR queries. Only missing times enter the
bulk AIA request. `overwrite=True` explicitly refreshes all times.
Cache load/save errors raise even with `on_error='skip'`. Cached
metadata and enabled-source contents are reused as saved; use
overwrite=True after changing preprocessing or query settings.
`progress=True` shows tqdm during the bulk query and subsequent
AIA downloading/preprocessing (including skipped attempts). Without
AIA it tracks fetching the supplemental observations instead.


#### Parameters


**start, end : datetime.datetime, datetime.date, numpy.datetime64, or str**

Inclusive range bounds accepted by parse_timestamp, e.g.
start="2023/08/01 00:00", end="2023-08-01T06:00:00Z". Both use UTC
when timezone-free. End must be at or after start. End is only yielded
when it lands on the cadence grid anchored at start.

**cadence : str, datetime.timedelta, pandas.Timedelta, int, or float, default "1h"**

Positive duration in whole seconds. Examples: "12min", "1h",
timedelta(minutes=12), or 720 (numeric values mean seconds, not minutes).

**on_error : {"raise", "skip"}, default "raise"**

Raise individual download/preprocessing failures or log and skip them.
Initial bulk-query errors and archive load/save failures always raise.

**save_dir : str or pathlib.Path or None, optional**

Directory for complete timestamped .npz observations, e.g. "saved_aia".
Existing files are reused without data queries or preprocessing unless
overwrite=True. None disables persistent saving/reuse.

**overwrite : bool, default False**

True downloads/processes again and atomically replaces saved observations.

**progress : bool, default False**

Show tqdm for remaining work or for loading a completely saved range.

**Additional keyword arguments (`**kwargs`)**

include_aia, include_harps, include_sxr: bool or None, each with the
same source-toggle rules as fetch_observation.


#### Yields

Observation
Successful saved/new observations in requested timestamp order. Merely
creating this generator does no work: consume it with a for loop, list,
or model.predict_series to start downloading/saving.


#### Examples

    >>> observations = processor.iter_observations(
    ...     "2023/08/01", "2023-08-01 01:00", cadence="12min",
    ...     save_dir="saved_aia", progress=True,
    ... )
    >>> # Consume observations with model.predict_series(observations).

## AIAProcessor

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L639-L822)


Download AIA frames and assemble one stack for FOXES inference.

The seven channels are ordered 94, 131, 171, 193, 211, 304, and 335 Å.
Raw FITS downloads are kept only for the duration of `fetch_date`.
ITI processing produces a channel-first `float32` array.


#### Parameters


**email : str or None, optional**

Registered JSOC email address used by the SDO downloader.

**resolution : int, default 512**

Height and width in pixels of each processed channel.

**tolerance : int, default 120**

Maximum time difference in seconds between the requested time and
each selected AIA frame.

**n_workers : int, default 4**

Number of download threads for the seven wavelengths.

**aia_norm_path : str or pathlib.Path or None, optional**

AIA normalization artifact for the checkpoint. If supplied here,
do not also supply it to `load_model`.

### AIAProcessor

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L666-L695)

```python
helio_foxes.AIAProcessor(*, email=None, resolution=512, tolerance=120, n_workers=4, aia_norm_path=None)
```

Configure JSOC downloading and optional AIA normalization.


#### Parameters


**email : str or None, optional**

Registered JSOC email, e.g. "you@example.com".

**resolution : int, default 512**

Positive processed height and width in pixels.

**tolerance : int, default 120**

Search half-width in seconds around the requested timestamp.

**n_workers : int, default 4**

Number of concurrent download threads.

**aia_norm_path : str or pathlib.Path or None, optional**

Checkpoint-matched .npz or .csv AIA normalization artifact. Apply it
here or in Predictor, never both; None leaves this transform disabled.

### AIAProcessor.fetch_date

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L697-L754)

```python
helio_foxes.AIAProcessor.fetch_date(date)
```

Download and process all seven channels near a requested UTC time.


#### Parameters


**date : datetime.datetime, datetime.date, numpy.datetime64, or str**

Complete date/time accepted by parse_timestamp, e.g.
"2023-08-01T12:00:00Z", "2023/08/01 12:00",
"2023.08.01_12:00:00_UTC", or "August 1, 2023".
Date-only values mean midnight; timezone-free values mean UTC.
Numeric Unix timestamps and ambiguous dates are not accepted.


#### Returns

numpy.ndarray
Finite `float32` AIA stack with shape
`(7, resolution, resolution)` in `WAVELENGTHS` order.
Optional AIA normalization is applied before returning.


#### Raises

TypeError
If `date` is not a supported datetime/date/string value.
FileNotFoundError
If any wavelength has no downloaded frame in the search window.
ValueError
If ITI returns an unexpected shape or nonfinite values.


#### Notes

The raw FITS directory is deleted before the method returns, including
when downloading or processing fails. The returned array is in memory.

### AIAProcessor.process_files

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L756-L796)

```python
helio_foxes.AIAProcessor.process_files(paths)
```

Preprocess seven local FITS files in `WAVELENGTHS` order.

Uses the same ITI calibration/crop/resize pipeline as downloads.
Caller-owned files are never removed. Returns a processed array;
wrap it with `Observation.from_array(stack, timestamp=...)`.


#### Parameters


**paths : iterable of str or pathlib.Path**

Exactly seven existing local FITS file paths in wavelength order
(94, 131, 171, 193, 211, 304, 335). Supply a list/tuple of paths, not a
directory name or a single string. Example:
[Path("fits") / f"{w}.fits" for w in AIAProcessor.WAVELENGTHS].


#### Returns

numpy.ndarray
Finite float32 array shaped (7, resolution, resolution). Optional
processor normalization is applied once; local FITS files are preserved.

## HARPProcessor

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L419-L637)


Query HARP regions and project their boxes onto an AIA image crop.

`bbox_expansion` adds this fraction of each projected box's width and
height on each side, before clipping to the image. It defaults to zero.

### HARPProcessor

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L428-L454)

```python
helio_foxes.HARPProcessor(*, email=None, resolution=512, padding_factor=0.1, bbox_expansion=0.0)
```

Configure JSOC access and the AIA crop geometry.


#### Parameters


**email : str or None, optional**

Registered JSOC email; e.g. "you@example.com".

**resolution : int, default 512**

Positive image side length in pixels, matching the processed AIA crop.

**padding_factor : float, default 0.1**

Crop padding fraction beyond the solar radius, not pixels.

**bbox_expansion : float, default 0.0**

Finite nonnegative fractional expansion per box side; 0.1 means 10%.

### HARPProcessor.fetch_date

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L456-L506)

```python
helio_foxes.HARPProcessor.fetch_date(timestamp: datetime | str, jsoc_email: str | None=None) -> list[dict]
```

Return raw JSOC rows from the nearest good-quality HARP sweep.

`timestamp` accepts all parse_timestamp formats, as in
`AIAProcessor.fetch_date`. A timezone-free value is treated as
UTC; an aware value is converted to UTC. For example,
`"2023-08-01T00:00:00Z"` is accepted.

Rows retain JSOC column names and values, including raw `NOAA_ARS`.
Visible regions also include `top_pct`, `left_pct`, `width_pct`,
and `height_pct` for the AIA crop. Query failures and absent usable
regions return an empty list.


#### Parameters


**timestamp : datetime.datetime, datetime.date, numpy.datetime64, or str**

Complete time accepted by parse_timestamp, e.g. "2023/08/01 12:00",
"August 1, 2023 12:00 UTC", or "2023-08-01T08:00:00-04:00".
Date-only values mean midnight; naive values mean UTC. JSOC's internal
TAI record syntax is not a supported public timestamp format.

**jsoc_email : str or None, optional**

Per-call registered JSOC email; None uses the processor's email.


#### Returns

list of dict
Good-quality HARP rows and projected crop percentages; [] on absent
usable regions or query failure. Invalid input timestamps still raise.

## SXRProcessor

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L825-L984)


Fetch GOES XRS and average satellites as in the training targets.

`tolerance` is the maximum distance from the requested time, in seconds
(default 120). `satellite` optionally restricts the GOES satellite
number. Otherwise average all available satellites at the nearest usable
sample timestamp, with equidistant timestamps favoring the earlier one.
The arithmetic mean is taken in physical flux space, before any model
log normalization, as in `data.align_aia_sxr`.

### SXRProcessor

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L836-L855)

```python
helio_foxes.SXRProcessor(*, tolerance=120, satellite=None)
```

Configure the GOES XRS lookup.


#### Parameters


**tolerance : float, default 120**

Finite positive search half-width in seconds.

**satellite : positive int or None, optional**

GOES satellite number, e.g. 18 (not "GOES-18"). None combines valid
satellites at the nearest usable timestamp as described by the class.

### SXRProcessor.fetch_date

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L857-L951)

```python
helio_foxes.SXRProcessor.fetch_date(date) -> dict | None
```

Return measured XRS-A (0.5–4 Å) and XRS-B (1–8 Å) flux in W/m².

Accepts all date/time formats described by `parse_timestamp`.
XRS-B must have a zero quality flag and finite positive flux, matching
the training target's band selection. At the nearest valid timestamp,
return the arithmetic mean `xrsb_flux` and the mean of available
good-quality `xrsa_flux` values (or None if none are usable).
`timestamp` is the actual UTC sample time; `time_offset_seconds`
is sample minus requested time. `satellites` lists the contributors
and `measurements` retains their individual fluxes. Samples from
different timestamps are never averaged together. Fluxes retain the
science product's calibration; no interpolation or additional legacy
scaling is applied. Missing data or query/processing failures
return `None`. Temporary downloads are removed before returning.


#### Parameters


**date : datetime.datetime, datetime.date, numpy.datetime64, or str**

Complete date/time accepted by parse_timestamp. For example,
"2023-08-01T12:00:00Z", "2023/08/01 12:00", or "August 1, 2023".
Date-only values mean midnight; naive inputs mean UTC.


#### Returns

dict or None
Physical xrsa_flux/xrsb_flux in W/m², an actual UTC sample timestamp,
offset in seconds, satellites, and individual measurements; None on
unavailable data/query failure. Invalid input timestamps still raise.

## Prediction

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1258-L1309)


FOXES output for one AIA stack.


#### Attributes


**metadata : ObservationMetadata or None**

Input observation metadata, preserved for automatic HARP overlays.
The original input array is not retained.

**flux : float**

Predicted full-disk soft X-ray flux in W/m².

**flux_map : numpy.ndarray**

Two-dimensional patch contributions in W/m². Its dimensions are the
checkpoint's patch grid, which is smaller than the input image.

**intervals : dict or None**

For Gaussian checkpoints, `"68"` and `"95"` map to lower and
upper flux bounds in W/m². `None` for deterministic checkpoints.

**patch_uncertainty_map : numpy.ndarray or None**

For Gaussian checkpoints, the square root of each patch's contribution
to global flux variance, in W/m². These are variance attributions,
not calibrated patch-level error bars.

**harp_regions : list[dict] or None**

Per-HARP prediction results. Each dict contains `harpnum`,
`flux_sum` in W/m², and `top_pct`, `left_pct`, `width_pct`,
`height_pct` locations. Original JSOC fields, including heliographic
longitude/latitude bounds, are preserved. Flux sums include positive
patch contributions whose centers fall inside each box. Overlapping
boxes can count the same patch. `None` means HARPs were not fetched;
an empty list means no usable regions were returned.
`flux_sum_uncertainty` is sqrt(sum(patch_uncertainty_map**2))
inside the same box, in W/m², or None for deterministic predictions.
This follows the model's additive variance attribution; regional
uncertainty calibration is not established.

**sxr : dict or None**

Measured GOES fluxes and sample provenance, preserved from Observation.
Compare `sxr["xrsb_flux"]` with the predicted `flux`. This measured
value is not an input to the model. None when disabled or unavailable.

## Predictor

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1630-L1818)


Keep a FOXES checkpoint loaded for repeated predictions.


#### Parameters


**checkpoint_path : str or pathlib.Path**

Path to a compatible Lightning `.ckpt` or serialized model file.

**device : {"auto", "cpu", "cuda", "mps"}, default "auto"**

Inference device. `"auto"` selects an available accelerator.

**aia_norm_path : str or pathlib.Path or None, optional**

Matching AIA normalization artifact, if the checkpoint was trained
with one. Do not apply that artifact in `AIAProcessor` too.


#### Notes

Create the predictor once when the UI starts and reuse it for requests.

### Predictor

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1648-L1676)

```python
helio_foxes.Predictor(checkpoint_path, *, device='auto', aia_norm_path=None)
```

Load the checkpoint and its optional input normalization.


#### Parameters


**checkpoint_path : str or pathlib.Path**

Existing full .ckpt/model file, not a Git LFS pointer; ~ is expanded.

**device : {"auto", "cpu", "cuda", "mps"}, default "auto"**

Compute device; auto selects an available accelerator.

**aia_norm_path : str or pathlib.Path or None, optional**

Matching .npz/.csv AIA normalization artifact. Supply only if input
arrays have not already received that normalization.

### Predictor.predict_series

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1678-L1735)

```python
helio_foxes.Predictor.predict_series(observations, *, snapshot_time=None, snapshot_wavelength=193)
```

Predict observations, optionally keeping the AIA image nearest a target time.

The iterable is consumed once, reusing this model. With a snapshot target,
each observation replaces the retained image only if it is closer to that
time (earlier wins ties). Only one channel image is kept in memory.
No temporary image files are written. Prediction failures raise.


#### Parameters


**observations : iterable of Observation**

List, tuple, or generator such as processor.iter_observations(...) or
load_observations("saved_aia"). Each item needs a finite seven-channel
processed array matching the checkpoint. Wrap plain arrays in an
Observation first.

**snapshot_time : datetime.datetime, datetime.date, numpy.datetime64, str, or None**

Target time for the optional snapshot, e.g. "2024-03-23 01:40".
Accepts all parse_timestamp formats; naive values mean UTC.
None retains no image. To display the query midpoint, pass
start + (end - start) / 2 using parsed datetime values.

**snapshot_wavelength : int, default 193**

Angstrom channel to retain, e.g. 171 or 193, not a channel index.
Must occur in each observation's metadata.wavelengths when a
snapshot_time is supplied.


#### Returns

PredictionSeries
Timestamped predictions and an optional one-channel snapshot for
series.plot(show_snapshot=True). Full input stacks are not retained.

### Predictor.predict

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1737-L1818)

```python
helio_foxes.Predictor.predict(image, *, timestamp=None)
```

Predict global SXR flux and patch contributions from one AIA stack.


#### Parameters


**image : Observation or array-like**

Finite channel-first processed AIA stack in 94, 131, 171, 193,
211, 304, 335 Å order. Shape is `(7, H, W)` with `H` and
`W` determined by the checkpoint's patch size and grid.
Observation metadata is copied into the returned prediction.

**timestamp : datetime.datetime, datetime.date, numpy.datetime64, str, or None**

Time for a plain array; all parse_timestamp formats work, e.g.
"2023/08/01 12:00" or "2023-08-01T12:00:00Z". Naive values
mean UTC; None leaves an array prediction without time metadata. Its crop is assumed to use
the standard AIA channel order and 0.1 padding. For other crops,
supply an Observation with explicit ObservationMetadata.


#### Returns

Prediction
Full-disk flux, patch flux map, and optional Gaussian uncertainty.


#### Raises

ValueError
If the image shape is incompatible or values are nonfinite.

## load_model

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1821-L1840)


```python
helio_foxes.load_model(checkpoint_path, *, device='auto', aia_norm_path=None)
```


Load a checkpoint for repeated calls to `predict`.


#### Parameters


**checkpoint_path : str or pathlib.Path**

Existing full checkpoint/model file, e.g. "models/ComplexModel.ckpt".
Git LFS pointer text files are not model checkpoints. ~ is expanded.

**device : {"auto", "cpu", "cuda", "mps"}, default "auto"**

Inference device; auto selects an available accelerator.

**aia_norm_path : str or pathlib.Path or None, optional**

Checkpoint-matched .npz/.csv AIA normalization artifact. Apply it once:
omit here if preprocessing already applied the same normalization.


#### Returns

Predictor
Loaded model and optional AIA transform.

## predict

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1843-L1864)


```python
helio_foxes.predict(image, *, model, timestamp=None)
```


Run one processed AIA stack through a loaded FOXES model.


#### Parameters


**image : Observation or array-like**

Channel-first AIA stack; see `Predictor.predict` for its shape
and channel order.

**model : Predictor**

Predictor returned by `load_model`.

**timestamp : datetime.datetime, datetime.date, numpy.datetime64, str, or None**

Array input time accepted by parse_timestamp, e.g.
"2023/08/01 12:00"; naive values mean UTC. None omits array
time metadata. Observation inputs carry their own
timestamp and crop metadata. See `Predictor.predict`.


#### Returns

Prediction
Full-disk flux, flux map, and optional uncertainty in W/m².

## plot_flux_map

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1940-L2203)


```python
helio_foxes.plot_flux_map(prediction, *, ax=None, cmap='inferno', vmin=None, vmax=None, show_harp_boxes=False, harp_regions=None, timestamp=None, jsoc_email=None, min_harp_flux=1e-07, harp_label_fontsize=6, show_harp_uncertainty=True, harp_label_offsets=None, harp_label_max_shift=80, aia_image=None, harp_colors=None)
```


Plot patch flux contributions with a logarithmic color scale.

For a Prediction without an AIA background, the title shows its predicted
FOXES flux with the model's full-disk uncertainty when available, and the measured SXR-B
satellite mean when available, all in W/m². The uncertainty is the
square root of summed patch variances, not a calibrated error bar.
Missing values are omitted. Plain map arrays do not add a flux title.


#### Parameters


**prediction : Prediction or array-like**

A FOXES result or a two-dimensional patch flux map in W/m².

**ax : matplotlib.axes.Axes or None, optional**

Axes to draw on. A new figure and axes are created if omitted.

**cmap : str or matplotlib.colors.Colormap, default "inferno"**

Colormap name such as "inferno" or "gray", or a Colormap instance.

**vmin, vmax : float or None, optional**

Positive limits for the logarithmic color scale in W/m². By default,
use the smallest and largest positive values in the map.

**show_harp_boxes : bool, default False**

Draw HARP outlines labeled with their numbers and flux sums, with
fixed labels anchored to box corners. Box and label colors are fixed
by descending flux rank, independent of the number of HARPs. Sums include
positive patch contributions whose centers fall inside each box.

**harp_regions : list[dict] or None, optional**

Pre-fetched regions from `HARPProcessor.fetch_date`.
For manually supplied boxes, each dict needs HARPNUM or harpnum,
plus top_pct, left_pct, width_pct, and height_pct, in percent of
the input crop measured from the top-left display corner.
Defaults to the prediction's regions. If neither has been fetched
and `show_harp_boxes` is true, regions are queried from JSOC. An
empty list suppresses the query, including one carried by a prediction.

**timestamp : datetime.datetime, datetime.date, numpy.datetime64, str, or None**

Complete AIA time, e.g. "2023/08/01 12:00" or an aware
datetime; naive values mean UTC. Accepts the same formats as
`AIAProcessor.fetch_date`. Defaults to the prediction's metadata.
Required for an automatic HARP query when metadata is absent.

**jsoc_email : str or None, optional**

JSOC email for the automatic HARP query.

**min_harp_flux : float, default 1e-7**

Minimum HARP flux sum in W/m² to show in the map. Regions
below this value are hidden; prediction results remain complete.

**harp_label_fontsize : float, default 6**

Fixed HARP label size in points, independent of box size.

**show_harp_uncertainty : bool, default True**

Add a third label line, `± uncertainty`, when patch uncertainties
are available. This is sqrt(sum(sigma_patch**2)) within the HARP box,
in W/m², using the model's additive variance attribution. Disabling
this affects labels only; prediction results retain uncertainties.

**harp_label_offsets : dict[int, tuple[float, float]] or None, optional**

Manual (dx, dy) label offsets in points, keyed by HARP number. Positive
dx moves right; positive dy moves up, relative to the chosen box corner.
Custom offsets add a matching connector. Defaults to a three-point
diagonal gap outside a box corner, preferring the side toward the image center.
The overlap pass tries all four corners before small four-point nudges.
Offsets are starting positions; shifted labels get connectors.
Placement is calculated once; replot after substantial size changes.

**harp_label_max_shift : float, default 80**

Maximum automatic nudge from a corner, in points. Set to zero to fix
labels at their preferred corner. Automatic positions stay outside
their HARP box and inside the map, unless a manual offset was supplied.
If no clear position is found within this limit, retain the starting position; crowded maps
may still need manual offsets or a higher min_harp_flux.

**aia_image : array-like or None, optional**

Display a processed 2D AIA channel instead of the patch flux map,
reusing the same HARP box/label placement. The image must cover the
exact crop used for prediction. Flux labels still sum predicted
patch contributions, not AIA intensity. Uses a linear 1–99 percentile
stretch (or vmin/vmax), pixel axes, and no flux colorbar. Pass
`cmap='gray'` for a neutral background. No downloading occurs.

**harp_colors : dict[int, color] or None, optional**

Override outline/label colors by HARP number, for matching other
plots. Unspecified regions retain descending-flux-rank colors.
Examples: {1234: "tab:blue"}, {1234: "#56B4E9"}, or
{1234: (0.2, 0.5, 0.8)} with RGB components in [0, 1].


#### Returns

matplotlib.axes.Axes
Axes containing the map. Flux maps also have a flux colorbar.


#### Raises

ValueError
If the map is not finite and two-dimensional, has no positive values,
or the color limits are invalid. Nonpositive cells are masked.

## plot_uncertainty_map

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L2205-L2252)


```python
helio_foxes.plot_uncertainty_map(prediction, *, ax=None, cmap='magma', vmin=None, vmax=None)
```


Plot patch uncertainty contributions with a logarithmic color scale.


#### Parameters


**prediction : Prediction or array-like**

A FOXES result or a two-dimensional patch uncertainty map in W/m².

**ax : matplotlib.axes.Axes or None, optional**

Axes to draw on. A new figure and axes are created if omitted.

**cmap : str or matplotlib.colors.Colormap, default "magma"**

Colormap name such as "magma", or a Colormap instance.

**vmin, vmax : float or None, optional**

Positive limits for the logarithmic color scale in W/m². By default,
use the smallest and largest positive values in the map.


#### Returns

matplotlib.axes.Axes
Axes containing the map. Its figure also has an uncertainty colorbar.


#### Raises

ValueError
If the map is not finite and two-dimensional, has no positive values,
or the color limits are invalid. Nonpositive cells are masked.

## parse_timestamp

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L28-L99)


```python
helio_foxes.parse_timestamp(value)
```


Parse a complete date/time and return an aware UTC datetime.

Accept datetime/date, numpy.datetime64, ISO 8601, year-first slash/dot
dates (including `2023.08.01_12:00:00_UTC`), and named-month dates.
Date-only values mean midnight; naive values mean UTC. Numeric dates
such as `01/08/2023` are rejected rather than guessing day/month order.
Unix numbers, partial dates, relative dates, and timezone abbreviations
other than UTC/GMT are not supported. Use an explicit offset for zones.


#### Parameters


**value : datetime.datetime, datetime.date, numpy.datetime64, or str**

A complete date, optionally with a time. Examples: "2023-08-01",
"2023-08-01T12:00:00Z", "2023/08/01 12:00",
"2023.08.01_12:00:00_UTC", "August 1, 2023 12:00 UTC", or
"2023-08-01T08:00:00-04:00". Date-only inputs mean midnight; naive
inputs mean UTC. Numeric Unix timestamps, NaT, incomplete dates such
as "August 2023", ambiguous "01/08/2023", and non-UTC/GMT timezone
abbreviations such as EST are rejected.


#### Returns

datetime.datetime
Timezone-aware UTC time.

## PredictionSeries

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1335-L1626)


Timestamped predictions with full-disk and per-HARP light curves.

Times are requested UTC times, not individual channel exposure times.
Duplicate requested timestamps are rejected. Missing HARP detections
become NaN, never zero; skipped AIA samples are absent from the series.
Regional fluxes follow Prediction's positive patch-center box sums,
including overlapping boxes and additive variance attribution.


#### Parameters


**predictions : iterable of Prediction**

Outputs with non-None ObservationMetadata and unique timestamps.
Stored as a tuple sorted by UTC time.

**snapshot : Observation or None, optional**

Default one-channel snapshot shaped (1, H, W), with timestamp matching
a prediction. This visualization container is not a seven-channel model
input. Usually populated by Predictor.predict_series.

### PredictionSeries.to_dataframe

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1379-L1387)

```python
helio_foxes.PredictionSeries.to_dataframe()
```

Return full-disk predicted and measured XRS-B flux (W/m²).

### PredictionSeries.harp_light_curves

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1389-L1411)

```python
helio_foxes.PredictionSeries.harp_light_curves()
```

Return `{harpnum: DataFrame}` indexed by UTC requested time.

Columns `flux_sum` and `flux_sum_uncertainty` use W/m². Each
frame covers every successful prediction; an absent HARP or unknown
uncertainty is NaN. No query, interpolation, or zero filling occurs.

### PredictionSeries.plot

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1458-L1546)

```python
helio_foxes.PredictionSeries.plot(*, harpnums=None, ax=None, include_full_disk=False, include_sxr=False, show_snapshot=False, snapshot_ax=None, aia_ax=None, show_uncertainty=True, min_peak_flux=0.0, legend='auto')
```

Plot selected HARP curves (all by default), with optional totals.

Missing HARP rows break lines. `show_uncertainty=True` shades
regional flux +/- one flux_sum_uncertainty and enables snapshot
uncertainty labels, only where available. Deterministic predictions
have no bands or +/- labels. These Gaussian variance attributions
are not calibrated regional errors. The flux axis is logarithmic:
nonpositive/nonfinite flux samples break lines, and bands whose lower
bound is nonpositive are omitted rather than clipped onto the axis.
`min_peak_flux` selects HARPs whose maximum finite positive flux
over the entire series is at least this threshold, in W/m². Regions
with no positive samples are hidden. It filters snapshot HARP labels
too, without changing the returned tables or predictions.
`legend='auto'` shows a compact legend below the curves when a
snapshot is visible (HARP IDs remain on the flux map), otherwise a
full legend below the curves. Use 'full', 'compact', or 'none' to
override. HARP colors remain stable when filtering.
Returns a matplotlib Axes. Use `harpnums=[]` for totals only.
`show_snapshot=True` adds the selected prediction's standard
`plot_flux_map` panel, then its retained AIA image on the far right.
Flux-map HARP overlays share the curve colors; AIA has no overlays.
Create it with `predict_series(..., snapshot_time=...)` first.
Only selected HARPs present at the snapshot time are outlined. A
dashed line marks that requested UTC time in the curves. If ax is
supplied, snapshot_ax (flux map) and aia_ax must be distinct Axes in
the same figure; otherwise all three are created automatically.
The returned Axes
is always the light-curve panel. Disabling shows only the curves.
The snapshot is the observation nearest the target supplied to predict_series,
with earlier observations winning ties.


#### Parameters


**harpnums : iterable of int or None, optional**

HARP identifiers to display, e.g. [1234, 5678]. None considers all
regions; [] displays only any enabled comparison curves. A single
integer must be wrapped in a list. Unknown IDs raise ValueError.

**ax : matplotlib.axes.Axes or None, optional**

Curve axes. None creates a figure; with snapshots, custom axes require
all three of ax, snapshot_ax, and aia_ax in the same figure.

**include_full_disk, include_sxr : bool, default False**

Add the full-disk prediction and measured GOES XRS-B comparison,
respectively. Neither comparison is affected by min_peak_flux.

**show_snapshot : bool, default False**

Add the selected flux map and matching plain AIA image panels.

**snapshot_ax, aia_ax : matplotlib.axes.Axes or None, optional**

Flux-map and AIA axes, distinct from ax and each other. Omit all three
axes for automatic layout.

**show_uncertainty : bool, default True**

Show available regional uncertainty shading and flux-map +/- labels.
Deterministic outputs never receive artificial uncertainty values.

**min_peak_flux : float, default 0.0**

Finite nonnegative HARP peak threshold in W/m², e.g. 1e-7. Tests the
maximum positive predicted regional flux across the whole series,
not just the snapshot. Filters visuals only; returned tables stay complete.

**legend : {"auto", "full", "compact", "none"}, default "auto"**

Legend below the curve axes. Auto uses compact with snapshots, full
otherwise. Compact shows a region count and comparison/marker keys.


#### Returns

matplotlib.axes.Axes
Light-curve axes. Its figure contains the optional snapshot panels.


#### Examples

    >>> ax = series.plot(show_snapshot=True,
    ...                  min_peak_flux=1e-7, legend="compact")

## load_observations

[Source](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L385-L416)


```python
helio_foxes.load_observations(directory, *, progress=False)
```


Yield saved `.npz` observations in filename order, without network access.

Iterator-generated filenames sort by requested UTC time. Reads one file
at a time so it can be passed directly to `model.predict_series`.
A missing directory or one with no .npz files raises FileNotFoundError.
Invalid archives raise instead of silently dropping observations.


#### Parameters


**directory : str or pathlib.Path**

Directory of complete observation .npz archives, e.g. "saved_observations".
A leading ~ is expanded. Files are sorted lexicographically; the range
iterator's timestamp filenames therefore yield chronological order.
Do not mix normalization artifacts or other .npz files in this directory.

**progress : bool, default False**

Enable a tqdm loading bar in a terminal or notebook.


#### Yields

Observation
One restored observation at a time; accepts direct use as input to
model.predict_series(load_observations("saved_observations")).
