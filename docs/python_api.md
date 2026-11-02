# Getting started

Make one prediction, plot its flux map, then try a time range.
For every parameter and default, see the [API reference](api_reference.md).

## Install and get a checkpoint

```bash
python -m pip install helio-foxes
```

Download [UncertaintyModel.ckpt](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/model_assets/UncertaintyModel.ckpt)
and save it as `model_assets/UncertaintyModel.ckpt` in your working directory.
Checkpoints are separate from the Python package. For a repository clone,
`git lfs pull` downloads the weights. Other weights are listed under
[Models and checkpoints](models.md).

## Predict one observation

Use your registered JSOC email to download AIA data. Load the model once and
reuse it for later observations.

```python
import matplotlib.pyplot as plt
import helio_foxes

model = helio_foxes.load_model("model_assets/UncertaintyModel.ckpt")
processor = helio_foxes.ObservationProcessor(email="you@example.com")
observation = processor.fetch_observation("2023-08-01T00:00:00Z")
result = model.predict(observation)

print(f"Predicted SXR flux: {result.flux:.3e} W/m²")
print("68% interval:", result.intervals["68"])
helio_foxes.plot_flux_map(result, show_harp_boxes=True)
plt.show()
```

`result.flux_map` contains patch contributions that sum to `result.flux`.
HARP boxes identify active regions; their flux sums are in `result.harp_regions`.
When measured GOES data is available, compare `result.flux` with
`result.sxr["xrsb_flux"]`. GOES measurements are not model inputs.

The current checkpoint also returns Gaussian intervals and
`result.patch_uncertainty_map`. Patch and HARP uncertainties attribute the
model's global variance; they are not validated regional error bars.
See [Results](results.md) for measured interval coverage.

## Predict a time range

Continue with the same `model` and `processor`:

```python
observations = processor.iter_observations(
    "2023-08-01 00:00", "2023-08-01 01:00",
    cadence="12min", save_dir="saved_observations", progress=True,
)
series = model.predict_series(observations)
series.plot(include_full_disk=True, include_sxr=True)
plt.show()

series.to_dataframe().to_csv("predictions.csv")
harp_curves = series.harp_light_curves()
```

The iterator processes one observation at a time and reuses existing files in
`save_dir`. Times without an explicit timezone are interpreted as UTC.

## Reuse saved observations

```python
saved = helio_foxes.load_observations("saved_observations")
series = model.predict_series(saved)
```

This avoids downloading and preprocessing the same data again. Refresh the
saved files if your preprocessing or source settings change.

## Predicting your own observations

For an already processed AIA stack:

```python
observation = helio_foxes.Observation.from_npy(
    "my_processed_stack.npy", timestamp="2023-08-01 12:00",
)
result = model.predict(observation)
```

For the current checkpoint, use shape `(7, 512, 512)` and wavelength order
`[94, 131, 171, 193, 211, 304, 335]` Å. Match the training preprocessing:
`from_npy` does not resize, crop, or normalize the array. If a checkpoint needs
an AIA normalization artifact, apply it once, either during preprocessing or
with `load_model(..., aia_norm_path=...)`.

For local raw FITS files, use
{py:meth}`helio_foxes.AIAProcessor.process_files` before prediction.
The [notebook walkthrough](https://github.com/griffin-goodwin/FOXES/blob/main/tutorials/heliofoxes_demo/heliofoxes.ipynb)
contains more plotting and observation examples.

## Legacy checkpoint correction

For older deterministic weights, use `legacy_model=True` when loading the
model. See [Legacy models](models.md#legacy-models) for the correction rule.
Leave the default `legacy_model=False` for `UncertaintyModel.ckpt`.
