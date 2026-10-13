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

### Original FOXES abstract
The solar soft X-ray (SXR) irradiance is a long-standing proxy of solar activity, used for the classification of flare strength. As a result, the flare class, along with the SXR light curve, are routinely used as the primary input to many forecasting methods, from coronal mass ejection speed to energetic particle output. However, the SXR irradiance lacks spatial information leading to dubious classification during periods of high activity, and is  applicable only for observations from Earth orbit, hindering forecasting for other places in the heliosphere. This work introduces the Framework for Operational X-ray Emission Synthesis (FOXES), a Vision Transformer-based approach for translating Extreme Ultraviolet (EUV) spatially-resolved observations into SXR irradiance predictions. The model produces two outputs: (1) a global SXR flux prediction and (2) per-patch flux contributions, which offer a spatially resolved interpretation of where the model attributes SXR emission. This paves the way for EUV-based flare detection to be extended beyond Earth's line of sight, allowing for a more comprehensive and reliable flare catalog to support robust, scalable, and real-time forecasting, extending our monitoring into a true multiviewpoint system.

**Team**: Griffin Goodwin, Alison March, Jayant Biradar, Christopher Schirninger, Robert Jarolim, Angelos Vourlidas, Viacheslav Sadykov, Lorien Pratt

---

## Current local + global FOXES model

The maintained model in this repository extends the released FOXES patch
predictor. Local attention builds a feature for each AIA image patch. A global
query then summarizes the full AIA image and uses that context, together with
each patch's local features, to adjust its flux contribution. The corrected
patch fluxes sum to the full-disk SXR prediction.

This local + global design produces **visually better spatial flux maps** in
the examples examined so far. It also predicts a global Gaussian uncertainty
and a map of each patch's contribution to that variance. The uncertainty map
is an attribution of global uncertainty; there are no patch-level SXR labels
to establish calibrated local error bars.

The tradeoff is **somewhat worse global SXR prediction performance** than the
original released FOXES model in the evaluations run so far. The visual map
improvement is qualitative, and this repository does not yet include a
matched, held-out benchmark quantifying the accuracy difference.

The three bundled checkpoints below are the **original released models**.
The local + global Gaussian model is `CoupledGaussianViTLocal`; provide its
trained checkpoint separately through `forecasting/inference_config.yaml`.

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
│   ├── split_train_val_test.py  # Split processed AIA/SXR into train/val/test (use for training a new model only)
│   └── sxr_normalization.py     # Compute log-space mean/std over SXR .npy files for training
├── forecasting
│   ├── dataset.py            # AIAGOESDataset / AIAGOESDataModule: loads paired AIA + SXR .npy files
│   ├── model.py               # Released deterministic and current local + global Gaussian models
│   ├── inference.py           # Run a checkpoint over a folder of data; writes predictions.csv
│   ├── inference_config.yaml  # Config for inference.py
│   ├── evaluation.py          # Compute metrics and generate evaluation plots
│   ├── evaluation_config.yaml # Config for evaluation.py
│   └── trained_weights_and_normalization/  # Original released checkpoints and SXR normalization
├── training
│   ├── train.py                # Train released deterministic or current local + global model
│   ├── train_config.yaml       # Points to the maintained coupled Gaussian config
│   ├── train_coupled_gaussian.yaml
│   ├── train_deterministic.yaml
│   └── callbacks.py            # W&B prediction, attention, and spatial-map callbacks
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

To regenerate only the SXR normalization artifact for an existing split (the
training convention is `log10(SXR + 1e-8)`), run:

```bash
python data/sxr_normalization.py \
  --sxr-dir /data/SXR_processed/train \
  --output /data/SXR_processed/normalized_sxr.npy
```

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

The three original released checkpoints are included under
`forecasting/trained_weights_and_normalization/`, differing only in the
self-attention mask they were trained with. Each checkpoint carries its own
mask, so switch between those original models with `checkpoint_path`:

| Checkpoint | `mask_mode` | Description |
| --- | --- | --- |
| `inverted-attention-mask.ckpt` | `inverted` | **Original released FOXES model.** Each patch attends to *distant* patches (the flipped local-attention mask it was actually trained with). |
| `localized-attention-mask.ckpt` | `local` | True local attention — the opposite of the original model: each patch attends only to its own neighborhood. |
| `no-attention-mask.ckpt` | `none` | Global ViT — standard full/global attention, no masking at all. |

Edit `forecasting/inference_config.yaml`:

```yaml
data:
  aia_dir:         "/path/to/your/aia_data"
  sxr_dir:         "/path/to/your/sxr_data"   # omit/ignore if prediction_only
  sxr_norm_path:   "forecasting/trained_weights_and_normalization/normalized_sxr.npy"
  checkpoint_path: "forecasting/trained_weights_and_normalization/inverted-attention-mask.ckpt"

output_path: "/path/to/predictions.csv"

device: auto      # auto selects CUDA, Apple MPS, or CPU; set mps to require MPS

# Optional: run a reproducible random subset of eligible images.
sampling:
  count: 1000    # or use fraction: 0.1; leave both null for all images
  seed: 42
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

For the current local + global model, set `data.checkpoint_path` to its trained
checkpoint and use `model_type: coupled_uncertainty` (or `auto` to detect the
maintained model type from the checkpoint). The implementation is
`CoupledGaussianViTLocal` in `forecasting/model.py`; the checkpoint supplies
its layer count, attention heads, and local window size. The prediction CSV
additionally contains normalized mean/variance,
normalized sigma, sigma in dex, and asymmetric 68%/95% intervals in physical
W/m². Set `patch_uncertainty_path` to save a raw-flux spatial uncertainty map
for each image, or set `model_params.no_patch_uncertainty: true` to skip the
maps while retaining the global uncertainty columns. These maps contain the
square root of each patch's contribution to global raw-flux variance in W/m².
The shared uncertainty head sees only each detached mean-head patch logit, so
uncertainty gradients cannot alter the mean prediction. It predicts a
dimensionless relative standard deviation for that patch,
which scales the patch's physical flux before independent variance
contributions are summed. Because training has only a global SXR target, the
maps are variance attributions, not independently calibrated local error bars.

### 3) Evaluate

Edit `forecasting/evaluation_config.yaml` to point at the predictions
CSV and data directories, then run:

```bash
python forecasting/evaluation.py --config forecasting/evaluation_config.yaml
```

This computes metrics (MSE, MAE, R²) and generates plots under
`evaluation.output_dir`. When the CSV contains Gaussian uncertainty columns,
evaluation also writes `metrics/uncertainty_metrics.csv` with NLL, 68%/95%
coverage, sharpness, and standardized-residual diagnostics for the full dataset
and each Quiet/C/M/X class. The corresponding reliability curve and residual
histogram are saved as `plots/uncertainty_calibration.png`.

Set `plotting.inference_patch_dir` to inference's patch output folder to save
one figure per matching CSV timestamp under `plots/patch_maps/`. Each figure
shows available flux and square-root variance contribution maps. Set
`plotting.aia_dir` and `plotting.aia_wavelength` to place the matching AIA
channel beside them. All panels use the lower-left image origin;
`plotting.max_patch_maps` can limit large runs. Flux and uncertainty panels
share one logarithmic scale from the smallest positive to largest value
across all matching patch maps. Zero values are masked on the log scale.

### 4) Train your own model (optional)

Unlike inference, training expects `aia_dir`/`sxr_dir` each to have `train/`,
`val/`, and `test/` subfolders of paired `.npy` files — exactly what
`data/build_dataset.py` and the Hugging Face download path produce.

The maintained local + global Gaussian model uses one global query and a
weighted final-prediction Huber loss. Edit
`training/train_coupled_gaussian.yaml` for its architecture, mean loss,
uncertainty settings, data paths, and W&B settings. For the released
deterministic model, use `training/train_deterministic.yaml`.

To encourage predictions above and below the one-to-one line to balance, set
`mean_loss.bias_penalty_weight` in either training config. The default is `0`;
`0.1` is a starting point for a sweep. The added loss is the weight times the
mean of the **squared signed bias within each GOES class present in the batch**,
using normalized log10 flux. Each represented class contributes equally; the
existing Huber loss retains its configured class weights. Compare
`val_class/<class>/bias_dex` with validation MSE and RMSE before choosing a
weight. A class contributes to the penalty only in batches where it appears.

Optional `vit_architecture.global_attention_heads`,
`global_head_hidden_dim`, and `global_attention_dropout` change only the
coupled model's global head. Omit them to use the architecture defaults saved
in the maintained checkpoint.

```bash
python training/train.py --config training/train_coupled_gaussian.yaml
```

Training logs to Weights & Biases and saves checkpoints to
`data.checkpoints_dir`.

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
