---
license: mit
task_categories:
  - image-to-image
  - time-series-forecasting
tags:
  - solar
  - heliophysics
  - astronomy
  - AIA
  - SDO
  - EUV
  - SXR
  - GOES
  - flare
  - space-weather
pretty_name: "FOXES: Framework for Operational X-ray Emission Synthesis"
size_categories:
  - 100K<n<1M
language:
  - en
datasets:
  - griffingoodwin04/FOXES-Data
---

# FOXES: Framework for Operational X-ray Emission Synthesis

<p align="center">
  <img src="https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/foxes_logo.png" alt="FOXES Logo" width="300"/>
</p>

FOXES predicts full-disk solar soft X-ray (SXR) flux and spatial patch contributions
from seven-channel SDO/AIA extreme-ultraviolet images. This repository contains
the Python API, data tools, training code, and evaluation workflows developed
through Frontier Development Lab's Heliolab 2025.

[Model Demo](https://huggingface.co/spaces/griffingoodwin04/FOXES-model) ·
[Dataset](https://huggingface.co/datasets/griffingoodwin04/FOXES-Data) ·
[Notebook Walkthrough](https://github.com/griffin-goodwin/FOXES/blob/main/tutorials/heliofoxes_demo/heliofoxes.ipynb)

## API overview

Import the public API with `import helio_foxes`. The examples below use
`processor` for an `ObservationProcessor`, `model` for a `Predictor`,
`result` for a `Prediction`, and `series` for a `PredictionSeries`.
See the [Python API guide](https://github.com/griffin-goodwin/FOXES/blob/main/docs/python_api.md)
for usage examples, or the **[complete API reference](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md)**
for parameter descriptions, return values, and every keyword default, rendered
from the source docstrings. `*` marks keyword-only arguments. The source links below open each definition
in [helio_foxes/api.py](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py).

| Function or method | What it does | Source |
| --- | --- | --- |
| `load_model(checkpoint_path, *, device='auto', aia_norm_path=None)` | Load a checkpoint and return a reusable `Predictor`. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#load_model) | [load_model](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1821-L1840) |
| `predict(image, *, model, timestamp=None)` / `Predictor.predict(image, *, timestamp=None)` | Predict from an `Observation` or a processed AIA array. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#predict) · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#predictorpredict) | [predict](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1843-L1864), [Predictor.predict](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1737-L1818) |
| `ObservationProcessor.fetch_observation(date, *, include_aia=None, include_harps=None, include_sxr=None) -> Observation` | Fetch and preprocess AIA with optional HARP regions and measured GOES flux. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#observationprocessorfetch_observation) | [ObservationProcessor.fetch_observation](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1047-L1093) |
| `ObservationProcessor.iter_observations(start, end, *, cadence='1h', on_error='raise', save_dir=None, overwrite=False, progress=False, **kwargs)` | Yield observations over a time range; use `save_dir` to save and reuse processed observations. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#observationprocessoriter_observations) | [ObservationProcessor.iter_observations](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1095-L1254) |
| `Predictor.predict_series(observations, *, snapshot_time=None, snapshot_wavelength=193)` | Predict a sequence and return a `PredictionSeries`; optionally retain an AIA snapshot. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#predictorpredict_series) | [Predictor.predict_series](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1678-L1735) |
| `plot_flux_map(prediction, *, ax=None, cmap='inferno', vmin=None, vmax=None, show_harp_boxes=False, harp_regions=None, timestamp=None, jsoc_email=None, min_harp_flux=1e-07, harp_label_fontsize=6, show_harp_uncertainty=True, harp_label_offsets=None, harp_label_max_shift=80, aia_image=None, harp_colors=None)` | Plot patch flux contributions, with optional HARP boxes. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#plot_flux_map) | [plot_flux_map](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1940-L2203) |
| `plot_uncertainty_map(prediction, *, ax=None, cmap='magma', vmin=None, vmax=None)` | Plot Gaussian patch variance attribution when available. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#plot_uncertainty_map) | [plot_uncertainty_map](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L2205-L2252) |
| `PredictionSeries.plot(*, harpnums=None, ax=None, include_full_disk=False, include_sxr=False, show_snapshot=False, snapshot_ax=None, aia_ax=None, show_uncertainty=True, min_peak_flux=0.0, legend='auto')` | Plot HARP light curves, with optional full-disk prediction, GOES comparison, and snapshot panels. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#predictionseriesplot) | [PredictionSeries.plot](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1458-L1546) |
| `PredictionSeries.to_dataframe()` | Return full-disk predictions and measured XRS-B flux indexed by UTC time. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#predictionseriesto_dataframe) | [PredictionSeries.to_dataframe](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1379-L1387) |
| `PredictionSeries.harp_light_curves()` | Return a dictionary of HARP numbers and regional light-curve DataFrames. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#predictionseriesharp_light_curves) | [PredictionSeries.harp_light_curves](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1389-L1411) |
| `Observation.from_array(data, *, timestamp, wavelengths=(94, 131, 171, 193, 211, 304, 335), padding_factor=0.1, harp_regions=None, sxr=None)` | Wrap your processed AIA array and its metadata as an observation. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#observationfrom_array) | [Observation.from_array](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L224-L269) |
| `Observation.from_npy(path, *, timestamp, **kwargs)` | Create an observation from a processed `.npy` stack. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#observationfrom_npy) | [Observation.from_npy](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L272-L292) |
| `Observation.save(path, *, overwrite=False)` / `Observation.load(path)` | Save or restore one complete observation as a `.npz` archive. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#observationsave) · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#observationload) | [Observation.save](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L294-L353), [Observation.load](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L356-L382) |
| `load_observations(directory, *, progress=False)` | Lazily load saved `.npz` observations for reuse. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#load_observations) | [load_observations](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L385-L416) |
| `parse_timestamp(value)` | Convert supported timestamp formats to a UTC datetime. · [Docs](https://github.com/griffin-goodwin/FOXES/blob/main/docs/api_reference.md#parse_timestamp) | [parse_timestamp](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L28-L99) |

For individual data sources, use [AIAProcessor.fetch_date(date)](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L697-L754),
[HARPProcessor.fetch_date(date)](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L456-L506), or [SXRProcessor.fetch_date(date)](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L857-L951) on an
instance of the corresponding `helio_foxes` class.
[AIAProcessor.process_files(paths)](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L756-L796) processes local AIA FITS files.

The public data classes are [ObservationMetadata](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L133-L179), [Observation](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L183-L382), [Prediction](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1258-L1309),
and [PredictionSeries](https://github.com/griffin-goodwin/FOXES/blob/main/helio_foxes/api.py#L1335-L1626). A prediction exposes `flux`, `flux_map`, `harp_regions`,
`sxr`, `intervals`, and `patch_uncertainty_map`. Gaussian uncertainty fields are
`None` for deterministic checkpoints.

### Processor and predictor constructors

```python
ObservationProcessor(*, email=None, resolution=512, tolerance=120, n_workers=4, aia_norm_path=None, include_aia=True, include_harps=True, harp_bbox_expansion=0.0, include_sxr=True, sxr_tolerance=120, sxr_satellite=None)
```

```python
AIAProcessor(*, email=None, resolution=512, tolerance=120, n_workers=4, aia_norm_path=None)
```

```python
HARPProcessor(*, email=None, resolution=512, padding_factor=0.1, bbox_expansion=0.0)
```

```python
SXRProcessor(*, tolerance=120, satellite=None)
```

```python
Predictor(checkpoint_path, *, device='auto', aia_norm_path=None)
```

`iter_observations(..., **kwargs)` forwards `include_aia`, `include_harps`,
and `include_sxr` to `fetch_observation`. `Observation.from_npy(..., **kwargs)`
accepts the `wavelengths`, `padding_factor`, `harp_regions`, and `sxr` options
from `Observation.from_array`. See the complete reference for their meanings.

### Discover functions and arguments interactively

```python
import inspect
import helio_foxes

print(helio_foxes.__all__)  # Names exported by the public API.
print(inspect.signature(helio_foxes.load_model))  # Arguments and defaults.
help(helio_foxes.ObservationProcessor.fetch_observation)  # Detailed documentation.
help(helio_foxes.Prediction)  # Returned fields and their meanings.
```

In Jupyter, type `helio_foxes.` and press **Tab** to browse names, or run
`helio_foxes.load_model?` to see its documentation. Use
`help(helio_foxes.PredictionSeries.plot)` to inspect all light-curve plot options.



## Install

From a local checkout:

```bash
python -m pip install -e .
```

The package is named `helio-foxes`; import it as `helio_foxes`.
For a fresh environment, editable tests, or notebook kernel setup, follow the
[development setup](docs/workflows.md#2-create-an-environment).

## Predict one observation

The examples below use the same timestamps, checkpoints, and plotting settings
as the saved outputs in the
[heliofoxes notebook](https://github.com/griffin-goodwin/FOXES/blob/main/tutorials/heliofoxes_demo/heliofoxes.ipynb).
Use a registered JSOC email and download the checkpoints below.
Load the model once and reuse it for each observation.

### Download checkpoints

The single-observation and time-range examples use **Complex Gaussian**.
Download the other two checkpoints to run the comparison example.
Save each file at the path shown, relative to the directory where you run the code.
These downloads are separate from the PyPI package.

| Model | Download | Save as |
| --- | --- | --- |
| Complex Gaussian | [ComplexModel.ckpt](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/forecasting/trained_weights_and_normalization/updated_architecture_checkpoints/ComplexModel.ckpt) | `forecasting/trained_weights_and_normalization/updated_architecture_checkpoints/ComplexModel.ckpt` |
| Lighter Gaussian | [SlightlyLighterModel.ckpt](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/forecasting/trained_weights_and_normalization/updated_architecture_checkpoints/SlightlyLighterModel.ckpt) | `forecasting/trained_weights_and_normalization/updated_architecture_checkpoints/SlightlyLighterModel.ckpt` |
| Deterministic | [inverted-attention-mask.ckpt](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/forecasting/trained_weights_and_normalization/inverted-attention-mask.ckpt) | `forecasting/trained_weights_and_normalization/inverted-attention-mask.ckpt` |

### Load the model and predict

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import helio_foxes

weights = Path("forecasting/trained_weights_and_normalization")
model = helio_foxes.load_model(
    weights / "updated_architecture_checkpoints" / "ComplexModel.ckpt",
    device="auto",
)
processor = helio_foxes.ObservationProcessor(
    email="you@example.com", harp_bbox_expansion=0.2,
)
observation = processor.fetch_observation("2023-07-11T01:30:00Z")
result = model.predict(observation)

print(f"Predicted XRS-B: {result.flux:.3e} W/m²")
if observation.sxr is not None:
    print(f"Measured XRS-B: {observation.sxr['xrsb_flux']:.3e} W/m²")

fig, ax = plt.subplots(figsize=(10, 7))
helio_foxes.plot_flux_map(
    result, ax=ax, show_harp_boxes=True, min_harp_flux=1e-7,
)
plt.show()
```

Saved notebook output for **2023-07-11 at 01:30 UTC**:

```text
Predicted XRS-B: 2.695e-06 W/m²
Measured XRS-B: 2.941e-06 W/m²
```

![Complex Gaussian patch flux map with HARP active-region boxes for 2023-07-11 at 01:30 UTC](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/docs/images/notebook-flux-map.png)


### Inspect the AIA inputs

Plot the seven processed channels from the same `observation`:

```python
fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
for ax, wavelength, image in zip(
    axes.flat, observation.metadata.wavelengths, observation.data,
):
    low, high = np.percentile(image, [1, 99])
    ax.imshow(image, origin="lower", cmap="gray", vmin=low, vmax=high)
    ax.set_title(f"AIA {wavelength} Å")
    ax.axis("off")
for ax in axes.flat[len(observation.metadata.wavelengths):]:
    ax.axis("off")
plt.show()
```

![Seven processed AIA input channels for 2023-07-11 at 01:30 UTC](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/docs/images/notebook-aia-inputs.png)

### Compare checkpoints

Reuse the same observation to compare the three notebook checkpoints:

```python
checkpoints = {
    "Complex Gaussian": weights / "updated_architecture_checkpoints" / "ComplexModel.ckpt",
    "Lighter Gaussian": weights / "updated_architecture_checkpoints" / "SlightlyLighterModel.ckpt",
    "Deterministic": weights / "inverted-attention-mask.ckpt",
}
for name, checkpoint in checkpoints.items():
    predictor = helio_foxes.load_model(checkpoint, device="auto")
    prediction = predictor.predict(observation)
    print(f"{name}: {prediction.flux:.3e} W/m²")
    del predictor
```

| Checkpoint | Saved prediction (W/m²) |
| --- | ---: |
| Complex Gaussian | 2.695 × 10⁻⁶ |
| Lighter Gaussian | 2.947 × 10⁻⁶ |
| Deterministic | 2.899 × 10⁻⁶ |

The measured GOES XRS-B flux at the same time is **2.941 × 10⁻⁶ W/m²**,
using GOES-16 and GOES-18. These are individual notebook examples.

Fetching downloads and preprocesses AIA data and obtains supplementary HARP
regions and GOES measurements. Raw downloads are temporary. Missing HARP or
GOES data do not discard a usable AIA observation.

**Already have data?** See [loading your own processed arrays or local FITS files](docs/python_api.md#predicting-your-own-observations).
Inputs must match the checkpoint's training preprocessing and wavelength order:
`[94, 131, 171, 193, 211, 304, 335]` Å.

## Predict a time range

```python
observations = processor.iter_observations(
    "2024-03-23 00:45", "2024-03-23 02:45",
    cadence="1min", progress=True, save_dir="saved_observations",
    on_error="skip",
)
series = model.predict_series(
    observations, snapshot_time="2024-03-23 01:45", snapshot_wavelength=193,
)
ax = series.plot(
    show_snapshot=True, show_uncertainty=True, min_peak_flux=1e-7, legend="auto",
)
plt.show()

totals = series.to_dataframe()
harp_curves = series.harp_light_curves()  # {HARP number: DataFrame}
```

Using the Complex Gaussian `model` loaded above, the saved notebook run produced
**121 predictions**. The flux-map and AIA panels show the retained snapshot at
**2024-03-23 01:45 UTC**.

![HARP light curves from 2024-03-23 00:45 to 02:45 UTC alongside the flux map and AIA 193 angstrom snapshot at 01:45 UTC](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/docs/images/notebook-light-curves.png)

*Shading and regional ± values show the model's variance attribution;
they are not calibrated yet.

The iterator downloads and preprocesses observations one at a time. Existing
saved files are reused, so rerunning a range skips completed downloads and
preprocessing. AIA record selection uses a bulk JSOC query.

`snapshot_time` keeps the nearest observation's AIA channel for the flux-map
and image panels. Omit it to retain no image; use `series.plot()` for curves only.
Gaussian models show uncertainty when available; deterministic models do not.

Reload saved observations without downloading or preprocessing again:

```python
saved = helio_foxes.load_observations("saved_observations")
series = model.predict_series(saved)
series.plot()
```

See the [API guide](docs/python_api.md) for timestamp formats, cadence,
source toggles, snapshot selection, uncertainty, plot controls, and saved data.

## Models and documentation

The repository includes the original deterministic models and newer local +
global Gaussian checkpoints. The newer checkpoints have not yet been validated
for prediction quality, spatial maps, or uncertainty calibration. Patch and
HARP uncertainties attribute global model variance; they are not calibrated
local error bars.

- [Python API](docs/python_api.md): observations, prediction, light curves, and plotting.
- [Development and CLI workflows](docs/workflows.md): setup, checkpoint paths, data preparation, inference, evaluation, training, and repository structure.
- [Model background](docs/models.md): original abstract and architecture discussion.

After editing installed source, restart an existing notebook kernel and rerun
the setup cell so it loads the current API.

## Citation
If you use this code or data in your work, please cite:

```bibtex
@software{FOXES,
    title           = {{FOXES: A Framework For Operational X-ray Emission Synthesis}},
    institution     = {Frontier Development Lab (FDL)},
    repository-code = {https://github.com/griffin-goodwin/FOXES},
    version         = {v1.0},
    year            = {2026}
}
```

## Acknowledgement
This work is a research product of Heliolab (heliolab.ai), an initiative of the Frontier Development Lab (FDL.ai). FDL is a public–private partnership between NASA, Trillium Technologies (trillium.tech), and commercial AI partners including Google Cloud and NVIDIA.
Heliolab was designed, delivered, and managed by Trillium Technologies Inc., a research and development company focused on intelligent systems and collaborative communities for Heliophysics, planetary stewardship and space exploration.
We gratefully acknowledge Google Cloud for extensive computational resources and NVIDIA Corporation.
This material is based upon work supported by NASA under award number No. 80GSFC23CA040. Any opinions, findings, conclusions or recommendations expressed are those of the authors and do not necessarily reflect the views of the National Aeronautics and Space Administration.

Large language models were used as brainstorming tools to discuss possible training strategies and methodological considerations. The authors retained full responsibility for all research decisions, interpretations, and conclusions presented in this work.
