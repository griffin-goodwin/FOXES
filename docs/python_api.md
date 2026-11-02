# Getting Started

This page takes gives you a nice overview on how to get started with the FOXES package. Every method and their default kwargs is listed in the
[API Reference](api_reference.md).

## How the Pieces Fit

Every workflow follows the same four steps:

```text
ObservationFetcher().fetch(time)   ──►  Observation
                                             │
FOXESModel().predict(observation)  ◄─────────┘
                                             │
                                             ▼
                                         Prediction  ──►  .total_flux, .ground_truth, .flux_map, .harp_contributions
                                                          .plot_flux_map(), .plot_uncertainty_map()
```

1. **Get data.** An {py:class}`~helio_foxes.Observation` is one processed AIA
   stack plus any HARP and GOES data. Get it from
   {py:class}`~helio_foxes.ObservationFetcher`, load saved ones with
   {py:func}`~helio_foxes.load_observations`, or wrap your own array with
   {py:meth}`Observation.from_array <helio_foxes.Observation.from_array>`.
2. **Load a model.** Create a {py:class}`~helio_foxes.FOXESModel`. It
   downloads the current checkpoint the first time. Load it once and reuse it.
3. **Predict.** {py:meth}`FOXESModel.predict <helio_foxes.FOXESModel.predict>`
   turns an `Observation` into a {py:class}`~helio_foxes.Prediction`. For a
   time range, {py:meth}`~helio_foxes.FOXESModel.predict_time_series` turns
   many observations into a {py:class}`~helio_foxes.PredictionTimeSeries`.
4. **Look at the results.** Read the numbers from the `Prediction` and call
   `plot_flux_map` or `plot_uncertainty_map`. For a `PredictionTimeSeries`,
   get the light curves as DataFrames and call `plot_light_curves`.

## Install

FOXES needs Python 3.10 or newer.

```bash
python -m pip install helio-foxes
```

The model weights aren't bundled with the package. `FOXESModel()` downloads
the current checkpoint the first time you use it and caches it in
`~/.cache/helio_foxes` (set `HELIO_FOXES_CACHE` to use another folder). The
older checkpoints are listed under [Models and Checkpoints](models.md).

You'll also need an email address registered with JSOC to download AIA data.
If you don't have one yet, you can
[register here](http://jsoc.stanford.edu/ajax/register_email.html); it only
takes a minute. Set it once in your environment so you don't have to pass it
every time:

```bash
export JSOC_EMAIL="you@example.com"
```

Or pass `email="you@example.com"` to `ObservationFetcher`.

## Predict One Observation

Loading the model is probably the slowest part, so do it once and reuse it.

```python
import matplotlib.pyplot as plt
import helio_foxes

model = helio_foxes.FOXESModel()
fetcher = helio_foxes.ObservationFetcher()
observation = fetcher.fetch("2023-08-01T00:00:00Z")
print("AIA stack shape:", observation.aia_data.shape)
result = model.predict(observation)

print(f"Predicted GOES XRS-B flux: {result.total_flux:.3e} W/m²")
if result.ground_truth is not None:
    print(f"Measured GOES XRS-B flux: {result.ground_truth:.3e} W/m²")
print("68% interval:", result.prediction_intervals["68"])
result.plot_flux_map()
plt.show()
```

The prediction fields are `metadata`, `total_flux`, `flux_map`, `prediction_intervals`,
`flux_std_map`, `harp_contributions`, `ground_truth`, `legacy_correction_enabled`, and
`flux_scale_factor`.

`observation.aia_data` holds the processed AIA stack; it is `None` when AIA
fetching is disabled. The observation fields are `metadata`, `aia_data`,
`sxr_data`, and `harp_regions`, in that order. `sxr_data` contains the measured
GOES data and its provenance.

`result.flux_map` holds the per-patch contribution to the soft x-ray flux... they add up to
`result.total_flux`. The boxes drawn on the map are HARP active regions; the flux
inside each HARP is saved in `result.harp_contributions`. `result.ground_truth` is
the matched measured GOES XRS-B flux in W/m², for comparison (`None` if no
sample is available). GOES XRS-B covers the 1–8 Å band.

Each HARP row contains `total_flux` and `flux_std` in W/m², alongside its
region ID and bounding box. `flux_std` is the square root of the summed
patch variance contributions inside the box. These are model-assigned
uncertainty attributions; regional calibration hasn't been established.
`flux_std` is `None` for deterministic checkpoints. HARP light-curve tables
use these same column names.

The current checkpoint also gives you nominal 68% and 95% full-disk
`prediction_intervals`. They are asymmetric in W/m² because the Gaussian
is defined in transformed log-flux space. `result.flux_std_map` contains the
per-patch standard-deviation contributions: their squares add to the
full-disk variance. Plot them with `result.plot_uncertainty_map()`.

## Predict a Time Range

Using the same `model` and `fetcher`:

```python
observations = fetcher.fetch_range(
    "2023-08-01 00:00", "2023-08-01 01:00",
    cadence="12min", save_dir="saved_observations", progress=True,
)
series = model.predict_time_series(observations)
series.plot_light_curves(show_full_disk=True, show_observed=True)
plt.show()

series.full_disk_light_curve().to_csv("predictions.csv")
harp_curves = series.harp_light_curves()
```

Full-disk light-curve tables use `total_flux` and `ground_truth` columns in W/m².

Observations are downloaded and processed one at a time, so long ranges won't
fill up your memory. Anything already in `save_dir` is reused instead of
downloaded again.

`fetch_range` returns a generator, so it can only be looped over once. After
`predict_time_series` has used it, `observations` is empty. Call `fetch_range`
again (it reuses `save_dir`), or use `list(observations)` if you need them more
than once and they fit in memory.

## Reuse Saved Observations

Once you've saved a range, you can run the model on it again without
downloading anything:

```python
saved = helio_foxes.load_observations("saved_observations")
series = model.predict_time_series(saved)
```

## Use Your Own Data

If you already have a processed AIA stack:

```python
import numpy as np

observation = helio_foxes.Observation.from_array(
    np.load("my_processed_stack.npy"), timestamp="2023-08-01 12:00",
)
result = model.predict(observation)
```

The current checkpoint expects an array of shape `(7, 512, 512)` with channels
in the order `[94, 131, 171, 193, 211, 304, 335]` Å. `from_array` uses the array
as is, with no resizing, cropping, or normalization, so it needs to match the
training preprocessing already.

For raw FITS files on disk, run them through
{py:meth}`helio_foxes.AIAProcessor.process_files` first. The
[notebook walkthrough](https://github.com/griffin-goodwin/FOXES/blob/main/tutorials/heliofoxes_demo/heliofoxes.ipynb)
has more plotting and data-loading examples.

## Older Checkpoints

The older deterministic checkpoints need a correction for their outdated
training targets. Loading one by name, e.g.
`FOXESModel("inverted-attention-mask")`, turns it on for you; if you load one
from a file path, pass `legacy_correction_enabled=True`. [Legacy Models](models.md#legacy-models)
explains what the correction does.
