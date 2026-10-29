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

# FOXES: A Framework for Operational X-ray Emission Synthesis

<p align="center">
  <img src="https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/foxes_logo.png" alt="FOXES Logo" width="300"/>
</p>

FOXES predicts full-disk solar soft X-ray (SXR) flux and spatial patch contributions
from seven-channel SDO/AIA extreme-ultraviolet images. This repository contains
the Python API, data tools, training code, and evaluation workflows developed
through Frontier Development Lab's Heliolab 2025.

[Model demo](https://huggingface.co/spaces/griffingoodwin04/FOXES-model) ·
[Dataset](https://huggingface.co/datasets/griffingoodwin04/FOXES-Data) ·
[Notebook walkthrough](tutorials/heliofoxes_demo/heliofoxes.ipynb)

## Install

From a local checkout:

```bash
python -m pip install -e .
```

The package is named `helio-foxes`; import it as `helio_foxes`.
For a fresh environment, editable tests, or notebook kernel setup, follow the
[development setup](docs/workflows.md#2-create-an-environment).

## Predict one observation

Use a registered JSOC email and a compatible local checkpoint. Load the model
once and reuse it for each observation.

```python
import helio_foxes

model = helio_foxes.load_model(
    "forecasting/trained_weights_and_normalization/inverted-attention-mask.ckpt",
    device="auto",
)
processor = helio_foxes.ObservationProcessor(email="you@example.com")
observation = processor.fetch_observation("2023-08-01T00:00:00Z")
result = model.predict(observation)

print(result.flux)      # Full-disk SXR flux, W/m².
print(result.flux_map)  # Spatial patch contributions, W/m².
ax = helio_foxes.plot_flux_map(result, show_harp_boxes=True)
```

Fetching downloads and preprocesses AIA data and obtains supplementary HARP
regions and GOES measurements. Raw downloads are temporary. Missing HARP or
GOES data do not discard a usable AIA observation.

**Already have data?** See [loading your own processed arrays or local FITS files](docs/python_api.md#predicting-your-own-observations).
Inputs must match the checkpoint's training preprocessing and wavelength order:
`[94, 131, 171, 193, 211, 304, 335]` Å.

## Predict a time range

```python
observations = processor.iter_observations(
    "2024-03-23 01:00", "2024-03-23 02:30",
    cadence="2min", progress=True, save_dir="saved_observations",
)
series = model.predict_series(observations, snapshot_time="2024-03-23 01:45")
ax = series.plot(show_snapshot=True, min_peak_flux=1e-7)

totals = series.to_dataframe()
harp_curves = series.harp_light_curves()  # {HARP number: DataFrame}
```

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
