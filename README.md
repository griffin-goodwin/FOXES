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

This repository covers the full loop: getting data, training, running
inference, and evaluating a FOXES model.

```text
FOXES
├── download
│   ├── hugging_face_data_download.py # Recommended: HF Hub -> .npy (streamed, or from local parquet)
│   ├── hf_download_config.yaml       # Config for hugging_face_data_download.py
│   ├── download_sdo.py               # Advanced: raw AIA download from JSOC (needs data/build_dataset.py after)
│   ├── sdo_download_config.yaml      # Config for download_sdo.py
│   ├── download_sxr.py               # Advanced: raw GOES SXR download via SunPy Fido (needs data/build_dataset.py after)
│   └── sxr_download_config.yaml      # Config for download_sxr.py
├── data
│   ├── build_dataset.py         # Runs the full raw -> processed pipeline below in one command
│   ├── build_dataset_config.yaml # Config for build_dataset.py
│   ├── clean_aia.py             # Drop AIA FITS files with a bad DATE-OBS timestamp
│   ├── convert_aia.py           # Raw AIA FITS -> paired 512x512 .npy stacks (itipy)
│   ├── combine_sxr.py           # Combine raw multi-satellite GOES files into per-satellite CSVs
│   ├── align_aia_sxr.py         # Match AIA timestamps to GOES CSVs -> per-timestamp SXR .npy
│   ├── split_train_val_test.py  # Split processed AIA/SXR into train/val/test (training only)
│   └── sxr_normalization.py     # Compute log-space mean/std over SXR .npy files for training
├── forecasting
│   ├── dataset.py            # AIAGOESDataset / AIAGOESDataModule: loads paired AIA + SXR .npy files
│   ├── model.py               # ViTLocal: Vision Transformer with patch flux heads
│   ├── inference.py           # Run a checkpoint over a folder of data; writes predictions.csv
│   ├── inference_config.yaml  # Config for inference.py
│   ├── evaluation.py          # Compute metrics and generate evaluation plots
│   └── evaluation_config.yaml # Config for evaluation.py
├── training
│   ├── train.py                # Train ViTLocal with PyTorch Lightning + Weights & Biases logging
│   ├── train_config.yaml       # Config for train.py
│   └── callbacks.py            # W&B callbacks: SXR pred-vs-true plots, attention map visualization
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

FOXES is run in four steps: **get data**, **inference** (run a checkpoint over
your data), **evaluation** (score the predictions and generate plots), and
optionally **training** your own model. All are driven by a YAML config — edit
the config, then run the script.

### 0) Get data

**Recommended:** stream the released dataset straight from Hugging Face Hub
into the paired `.npy` layout inference expects — no separate processing step:

```bash
python download/hugging_face_data_download.py --config download/hf_download_config.yaml
```

Edit `download/hf_download_config.yaml` first to set `aia_dir`/`sxr_dir`, which
splits to pull, and whether to subsample.

If you've already downloaded the HF parquet files locally instead of
streaming, set `local_parquet_dir` in that same config to the root folder
containing your per-split subdirs (`train/`, `validation/` or `val/`, `test/`)
— streaming and the HF Hub login are skipped entirely in that case.

**Advanced:** to acquire and process raw data yourself instead, edit and run
the two download configs, then the one dataset-build config, in order:

```bash
# 1) Raw AIA FITS from JSOC (requires a registered email)
python download/download_sdo.py --config download/sdo_download_config.yaml

# 2) Raw GOES XRS data via SunPy Fido
python download/download_sxr.py --config download/sxr_download_config.yaml

# 3) Clean + convert AIA, combine + align SXR -> paired .npy (see data/build_dataset_config.yaml)
python data/build_dataset.py --config data/build_dataset_config.yaml
```

`data/build_dataset.py` runs the full raw-to-processed pipeline in one command
(clean AIA → convert AIA → combine GOES → align AIA/SXR); each step can be
skipped via the `steps:` block in its config if you've already run it.
Inference/evaluation just need the flat output of that — training needs two
more things, both off by default and only relevant if you're training:
- `steps.split: true` — splits `aia.processed_dir`/`output.sxr_dir` into
  `train/`/`val/`/`test/` subfolders (date ranges or a month-based default;
  see the `split:` block in the config).
- `sxr_normalization.compute: true` — computes SXR normalization stats from
  the train split (requires `steps.split` to have run first).

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
python forecasting/inference.py --config forecasting/inference_config.yaml
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
python forecasting/evaluation.py --config forecasting/evaluation_config.yaml
```

This computes metrics (MSE, MAE, R²) and generates plots under `evaluation.output_dir`.

### 4) Train your own model (optional)

Unlike inference, training expects `aia_dir`/`sxr_dir` each to have `train/`,
`val/`, and `test/` subfolders of paired `.npy` files — exactly what
`data/build_dataset.py` and the Hugging Face download path produce.

Edit `training/train_config.yaml`:

```yaml
base_data_dir: "/path/to/processed_data"        # holds AIA_processed/ and SXR_processed/
base_checkpoint_dir: "/path/to/checkpoints"

gpu_ids: -1        # -1 = CPU, 0 = GPU 0, [0,1] = specific GPUs, "all" = every GPU
batch_size: 6
epochs: 150

vit_architecture:
  mask_mode: inverted   # inverted (released model) | local | none (full/global attention)
  local_window: 9

wandb:
  entity: ""   # your W&B username or team name
```

Then run:

```bash
python training/train.py --config training/train_config.yaml
```

Training logs to Weights & Biases (predicted-vs-true SXR plots and attention
map visualizations each validation epoch — see `training/callbacks.py`) and
saves the top 10 checkpoints by validation loss to `data.checkpoints_dir`,
ready to point `forecasting/inference_config.yaml` at.

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
