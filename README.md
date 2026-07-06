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

# FOXES: A Framework For Operational X-ray Emission Synthesis

<p align="center">
  <img src="foxes_logo.png" alt="FOXES Logo" width="300"/>
</p>

This repository contains the code and resources for **FOXES**, a project developed as part of the _**Frontier Development Lab**_'s Heliolab 2025!

Model / Data:

https://huggingface.co/spaces/griffingoodwin04/FOXES-model

https://huggingface.co/datasets/griffingoodwin04/FOXES-Data

### Abstract
The solar soft X-ray (SXR) irradiance is a long-standing proxy of solar activity, used for the classification of flare strength. As a result, the flare class, along with the SXR light curve, are routinely used as the primary input to many forecasting methods, from coronal mass ejection speed to energetic particle output. However, the SXR irradiance lacks spatial information leading to dubious classification during periods of high activity, and is  applicable only for observations from Earth orbit, hindering forecasting for other places in the heliosphere. This work introduces the Framework for Operational X-ray Emission Synthesis (FOXES), a Vision Transformer-based approach for translating Extreme Ultraviolet (EUV) spatially-resolved observations into SXR irradiance predictions. The model produces two outputs: (1) a global SXR flux prediction and (2) per-patch flux contributions, which offer a spatially resolved interpretation of where the model attributes SXR emission. This paves the way for EUV-based flare detection to be extended beyond Earth's line of sight, allowing for a more comprehensive and reliable flare catalog to support robust, scalable, and real-time forecasting, extending our monitoring into a true multiviewpoint system.

**Team**: Griffin Goodwin, Alison March, Jayant Biradar, Christopher Schirninger, Robert Jarolim, Angelos Vourlidas, Viacheslav Sadykov, Lorien Pratt

---

## Repository Structure

This repository is intentionally scoped to **running and evaluating** a trained
FOXES model — it does not include the data download, preprocessing, or training
code used to produce the released checkpoint.

```text
FOXES
├── forecasting
│   ├── dataset.py            # AIA_GOESDataset: loads paired AIA + SXR .npy files
│   ├── model.py               # ViTLocal: Vision Transformer with patch flux heads
│   ├── model_test.py          # Unit tests for the model's attention masking
│   ├── inference.py           # Run a checkpoint over a folder of data; writes predictions.csv
│   ├── inference_config.yaml  # Config for inference.py
│   ├── evaluation.py          # Compute metrics and generate evaluation plots
│   └── evaluation_config.yaml # Config for evaluation.py
└── requirements.txt            # Python dependencies
```

---

## Setup

### 1) Clone
```bash
git clone https://github.com/griffin-goodwin/FOXES.git
cd FOXES
```

### 2) Create an environment

**Option A — pip:**
```bash
conda create -n foxes python=3.14 -y
conda activate foxes
pip install -r requirements.txt
```

**Option B — conda (full environment):**
```bash
conda env create -f foxes.yml
conda activate foxes
```

---

## Running the Model

FOXES is run in two steps: **inference** (run a checkpoint over your data) and
**evaluation** (score the predictions and generate plots). Both are driven by a
YAML config — edit the config, then run the script.

### 1) Data format

Point `inference.py` at a folder of paired `.npy` files, one file per timestamp:

```text
/your/aia_dir/
├── 2023-08-01T00:00:00.npy   # (7, 512, 512) float32 — one channel per AIA wavelength
├── 2023-08-01T00:01:00.npy
└── ...

/your/sxr_dir/                # only needed if you have ground truth to compare against
├── 2023-08-01T00:00:00.npy   # scalar xrsb_flux value
└── ...
```

Timestamps are matched by filename between `aia_dir` and `sxr_dir` — there's no
required subfolder name (no `train/`, `val/`, or `test/`). Just point the config
at whichever folder holds the data you want to run.

If you don't have ground-truth SXR data (e.g. scoring new/live data), set
`prediction_only: "true"` in the config and `data.sxr_dir` is ignored entirely.

### 2) Run inference

Edit `forecasting/inference_config.yaml`:

```yaml
data:
  aia_dir:         "/path/to/your/aia_data"
  sxr_dir:         "/path/to/your/sxr_data"   # omit/ignore if prediction_only
  sxr_norm_path:   "/path/to/normalized_sxr.npy"
  checkpoint_path: "/path/to/checkpoint.ckpt"

output_path: "/path/to/predictions.csv"
```

Then run:

```bash
python forecasting/inference.py -config forecasting/inference_config.yaml
```

This writes `output_path` (a CSV of timestamp/prediction/groundtruth). Per-patch
**flux contribution maps** and per-image **attention weights** are saved
automatically alongside it whenever `flux_path` / `weight_path` are set in the
config — set `model_params.no_flux: true` or `model_params.no_weights: true` to
skip either.

### 3) Evaluate

Edit `forecasting/evaluation_config.yaml` to point at the predictions
CSV and data directories, then run:

```bash
python forecasting/evaluation.py -config forecasting/evaluation_config.yaml
```

This computes metrics (MSE, MAE, R²) and generates plots under `evaluation.output_dir`.

---

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
