# FOXES: A Framework For Operational X-ray Emission Synthesis

This repository contains the code and resources for **FOXES**, a project developed as part of the _**Frontier Development Lab**_'s Heliolab 2025.

Model / Data:

https://huggingface.co/spaces/griffingoodwin04/FOXES-model

https://huggingface.co/datasets/griffingoodwin04/FOXES-Data

### Abstract
The solar soft X-ray (SXR) irradiance is a long-standing proxy of solar activity, used for the classification of flare strength. As a result, the flare class, along with the SXR light curve, are routinely used as the primary input to many forecasting methods, from coronal mass ejection speed to energetic particle output. However, the SXR irradiance lacks spatial information leading to dubious classification during periods of high activity, and is  applicable only for observations from Earth orbit, hindering forecasting for other places in the heliosphere. This work introduces the Framework for Operational X-ray Emission Synthesis (FOXES), a Vision Transformer-based approach for translating Extreme Ultraviolet (EUV) spatially-resolved observations into SXR irradiance predictions. The model produces two outputs: (1) a global SXR flux prediction and (2) per-patch flux contributions, which offer a spatially resolved interpretation of where the model attributes SXR emission. This paves the way for EUV-based flare detection to be extended beyond Earth's line of sight, allowing for a more comprehensive and reliable flare catalog to support robust, scalable, and real-time forecasting, extending our monitoring into a true multiviewpoint system.

**Team**: Griffin Goodwin, Alison March, Jayant Biradar, Christopher Schirninger, Robert Jarolim, Angelos Vourlidas, Viacheslav Sadykov, Lorien Pratt

---

## Repository Structure

```text
FOXES
├── data                         # Data cleaning and preprocessing
│   ├── align_data.py            # Align AIA and SXR timestamps; save matched pairs
│   ├── euv_data_cleaning.py     # EUV image quality filtering and cleaning
│   ├── iti_data_processing.py   # ITI (image-to-image translation) preprocessing
│   ├── process_data_pipeline.py # End-to-end preprocessing orchestrator
│   ├── split_data.py            # Split processed data into train/val/test by date
│   ├── sxr_data_processing.py   # Combine raw GOES .nc files into per-satellite CSVs
│   ├── sxr_normalization.py     # Compute log-normalization stats (mean/std) on SXR
│   ├── pipeline_config.py       # Dataclass config for process_data_pipeline.py
│   └── pipeline_config.yaml     # YAML config for process_data_pipeline.py
├── download                     # Dataset download utilities
│   ├── download_sdo.py          # Download SDO/AIA EUV images from JSOC
│   ├── sxr_downloader.py        # Download GOES SXR flux data
│   ├── hugging_face_data_download.py  # Download pre-processed data from HuggingFace Hub
│   └── hf_download_config.yaml  # Config for HuggingFace downloader
├── forecasting                  # Model training and inference
│   ├── data_loaders
│   │   ├── SDOAIA_dataloader.py # PyTorch Lightning DataModule for AIA+SXR
│   │   └── patch_flux_dataloader.py
│   ├── inference
│   │   ├── inference.py         # Batch inference; writes predictions.csv
│   │   ├── evaluation.py        # Compute metrics and generate evaluation plots
│   │   ├── flare_analysis.py    # Detect, track, and match flares; generate plots
│   │   ├── local_config.yaml    # Config for inference.py and flare_analysis.py
│   │   └── evaluation_config.yaml  # Config for evaluation.py
│   ├── models
│   │   └── vit_patch_model_local.py   # ViTLocal: Vision Transformer with patch flux heads
│   └── training
│       ├── train.py             # Train the ViTLocal model
│       └── train_config.yaml    # Training hyperparameters and data paths
├── pipeline_config.yaml         # Top-level pipeline orchestration config
├── run_pipeline.py              # End-to-end pipeline orchestrator
└── requirements.txt             # Python dependencies
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

## Running the Pipeline

FOXES uses a single orchestrator script (`run_pipeline.py`) and a top-level config (`pipeline_config.yaml`) to run any combination of pipeline steps in order.

### Pipeline Steps

| # | Step | Description                                                                    |
|---|------|--------------------------------------------------------------------------------|
| 0 | `hf_download` | Download pre-processed, pre-split data from HuggingFace *(replaces steps 1–6)* |
| 1 | `download_aia` | Download SDO/AIA EUV images from JSOC                                          |
| 2 | `download_sxr` | Download GOES SXR flux data                                                    |
| 3 | `combine_sxr` | Combine raw GOES `.nc` files into per-satellite CSVs                           |
| 4 | `preprocess` | EUV cleaning, ITI processing, and AIA/SXR alignment                            |
| 5 | `split` | Split AIA and SXR data into train/val/test by date range                       |
| 6 | `normalize` | Compute SXR log-normalization statistics (mean/std)                            |
| 7 | `train` | Train the ViTLocal solar flare forecasting model                               |
| 8 | `inference` | Run batch inference and save a predictions CSV                                 |
| 9 | `evaluate` | Compute metrics and generate evaluation plots                                  |
| 10 | `flare_analysis` | Detect, track, and match flares; generate plots/movies                         |

### Usage

```bash
# List all available steps
python run_pipeline.py --list

# Run the full pipeline (from raw data)
python run_pipeline.py --config pipeline_config.yaml --steps all

# Quick-start: download pre-processed data from HuggingFace, then train
python run_pipeline.py --config pipeline_config.yaml --steps hf_download,train,inference,evaluate

# Run specific steps
python run_pipeline.py --config pipeline_config.yaml --steps train,inference,evaluate

# Force re-run of preprocessing even if outputs already exist
python run_pipeline.py --config pipeline_config.yaml --steps preprocess --force
```

### Downloading Data from HuggingFace

The `hf_download` step pulls pre-processed, pre-split AIA and SXR data directly from the [FOXES HuggingFace dataset](https://huggingface.co/datasets/griffingoodwin04/FOXES-Data), skipping the raw download, preprocessing, and split steps entirely. Configure it via `download/hf_download_config.yaml`:

```yaml
repo_id: "griffingoodwin04/FOXES"
aia_dir: "/Volumes/T9/AIA_hg_processed"   # where AIA .npy files land
sxr_dir: "/Volumes/T9/SXR_hg_processed"   # where SXR .npy files land
splits:
  - train
  - validation   # maps to local "val/" directory
  - test

# Optional: download a random subset instead of the full dataset
subsample: false
subsample_n: 1000          # exact count per split (or use subsample_frac)
subsample_frac: 0.1        # fraction per split, used when subsample_n is null
shuffle_buffer_size: 500   # rows buffered before sampling; larger = more random

num_workers: 8             # parallel disk-write threads
```

Run the downloader standalone:
```bash
python download/hugging_face_data_download.py --config download/hf_download_config.yaml
```

### Configuration

Edit `pipeline_config.yaml` to set data paths, date ranges, and hyperparameters. Each step has its own section, and an `overrides` block lets you override values from the step's base config without editing it directly.

```yaml
# Example: override training hyperparameters from the top-level config
train:
  config: "forecasting/training/train_config.yaml"
  overrides:
    epochs: 150
    batch_size: 6

# Example: override inference data paths
inference:
  config: "forecasting/inference/local_config.yaml"
  overrides:
    data:
      checkpoint_path: "/path/to/your/checkpoint.ckpt"
    output_path: "/path/to/predictions.csv"
```

Steps can also be run individually by calling their scripts directly:

```bash
python forecasting/training/train.py -config forecasting/training/train_config.yaml
python forecasting/inference/inference.py -config forecasting/inference/local_config.yaml
python forecasting/inference/evaluation.py -config forecasting/inference/evaluation_config.yaml
```

---

## Data Directory Layout

After preprocessing and splitting, data should be organized as follows:

```text
/your/data/dir/FOXES/
├── AIA_raw/                    # Raw downloaded AIA FITS files
├── AIA_processed/              # ITI-processed AIA .npy arrays
│   ├── train/
│   ├── val/
│   └── test/
├── SXR_raw/                    # Raw GOES .nc files
│   └── combined/               # Per-satellite combined CSVs (from combine_sxr step)
├── SXR_processed/              # Aligned SXR .npy scalars (xrsb_flux, one per timestamp)
│   ├── train/
│   ├── val/
│   ├── test/
│   └── normalized_sxr.npy      # Log-normalization stats [mean, std]
└── inference/
    ├── predictions.csv          # Model output from inference step
    ├── weights/                 # Per-image attention maps (optional)
    ├── flux/                    # Map of flux contributions from each patch (optional)
    └── evaluation/              # Metrics and plots from evaluation step
```

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
