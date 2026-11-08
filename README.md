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

FOXES predicts full-disk solar soft X-ray flux and spatial patch contributions
from seven-channel SDO/AIA images. This repository includes the Python package,
data tools, training code, and evaluation workflows.

**[Read the Sphinx documentation](https://griffin-goodwin.github.io/FOXES/)**
for getting started, checkpoint downloads, examples, results, and the API reference.

[Current model results](https://griffin-goodwin.github.io/FOXES/results.html) ·
[Uncertainty calculation](https://griffin-goodwin.github.io/FOXES/uncertainties.html)

## Install

```bash
python -m pip install helio-foxes
```

Import the package as `helio_foxes`. Model checkpoints are downloaded separately;
follow the [getting started guide](https://griffin-goodwin.github.io/FOXES/python_api.html)
for your first prediction.

[Notebook walkthrough](https://github.com/griffin-goodwin/FOXES/blob/main/tutorials/heliofoxes_demo/heliofoxes.ipynb) ·
[Model demo](https://huggingface.co/spaces/griffingoodwin04/FOXES-model) ·
[Dataset](https://huggingface.co/datasets/griffingoodwin04/FOXES-Data)

## Citation

If you use this code or data in your work, please cite:

```bibtex
@software{FOXES,
    title           = {{FOXES: A Framework For Operational X-ray Emission Synthesis}},
    institution     = {Frontier Development Lab (FDL)},
    repository-code = {https://github.com/griffin-goodwin/FOXES},
    version         = {v2.0.2},
    year            = {2026}
}
```

## Acknowledgement

This work is a research product of Heliolab (heliolab.ai), an initiative of the Frontier Development Lab (FDL.ai). FDL is a public–private partnership between NASA, Trillium Technologies (trillium.tech), and commercial AI partners including Google Cloud and NVIDIA.
Heliolab was designed, delivered, and managed by Trillium Technologies Inc., a research and development company focused on intelligent systems and collaborative communities for Heliophysics, planetary stewardship and space exploration.
We gratefully acknowledge Google Cloud for extensive computational resources and NVIDIA Corporation.
This material is based upon work supported by NASA under award number No. 80GSFC23CA040. Any opinions, findings, conclusions or recommendations expressed are those of the authors and do not necessarily reflect the views of the National Aeronautics and Space Administration.

Large language models were used as brainstorming tools to discuss possible training strategies and methodological considerations. The authors retained full responsibility for all research decisions, interpretations, and conclusions presented in this work.
