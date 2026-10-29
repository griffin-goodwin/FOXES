# Development and command-line workflows

[Back to the README](../README.md)

Run the commands below from the repository root.

- [Clone](#1-clone)
- [Development environment](#2-create-an-environment)
- [Data, inference, evaluation, and training](#running-the-model)
- [Repository structure](#repository-structure)

## 1) Clone
```bash
git clone https://github.com/griffin-goodwin/FOXES.git
cd FOXES
```

## 2) Create an environment

For editable package development and tests, run these commands from the repository root:

```bash
conda create -n helio-foxes-dev python=3.12 pip -y
conda activate helio-foxes-dev
python -m pip install --upgrade pip
python -m pip install -e ".[dev,evaluation,training]"
python -m ipykernel install --sys-prefix --name helio-foxes-dev --display-name "Python (helio-foxes-dev)"
python -m pip check
python -m pytest -q
```

The editable install uses this checkout directly: edits to Python source take
effect on the next import without reinstalling. Reinstall after changing package
dependencies, and restart an existing notebook kernel after changing imported
code. `dev` adds pytest and notebook tools and pins Matplotlib 3.10.7, the
reference renderer for the HARP label-placement tests. The evaluation/training
extras include dependencies used by the full test suite. Checkpoints and live
solar data are separate from the environment.

An existing notebook kernel keeps imported code in memory. If it still reports
`snapshot_time requires retain_snapshot=True`, restart the kernel and rerun the
setup cell before sections 7 or 8 of the demo notebook. The current API accepts
`snapshot_time` directly and has no `retain_snapshot` argument. Confirm that
`helio_foxes.__file__` points to this checkout if the error persists.

In VS Code, select the **helio-foxes-dev** Python interpreter and choose
**Python (helio-foxes-dev)** in the notebook kernel picker. Confirm the imported
package comes from your checkout:

```bash
python -c "import sys, helio_foxes; print(sys.executable); print(helio_foxes.__file__)"
```

For just the public API and new observation/time-series tests:

```bash
python -m pytest tests/test_public_api.py tests/test_observation_series.py tests/test_harp_api.py tests/test_sxr_api.py tests/test_harp_uncertainty.py tests/test_series_snapshot.py tests/test_series_plot_filtering.py -q
```

The broader download/training scripts may need the additional dependencies in
`requirements.txt`: install them with `python -m pip install -r requirements.txt`
when using those workflows.

## Running the Model

FOXES is run in four steps: **get data**, **inference** (run a checkpoint over
your data), **evaluation** (score the predictions and generate plots), and
optionally **training** your own model. The command-line workflows below use
YAML configs: edit the config, then run the script. The Python API examples above
do not require a YAML config.

The checked-in configs use paths relative to the repository root:
`./foxes_data/` for downloaded or processed data and `./foxes_outputs/` for
predictions and plots. Run the commands below from the repository root, or
replace those paths with locations on your machine. For inference, set
`data.checkpoint_path` to a full checkpoint file available on your machine.


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

This working copy also has two newer local + global Gaussian checkpoint files
under `forecasting/trained_weights_and_normalization/updated_architecture_checkpoints/`. Their
architecture and normalization details below come from the local checkpoint
metadata; **neither has yet been validated for prediction quality, spatial
maps, or uncertainty calibration**.

| Local checkpoint | Architecture recorded in checkpoint | Status |
| --- | --- | --- |
| [ComplexModel.ckpt](../forecasting/trained_weights_and_normalization/updated_architecture_checkpoints/ComplexModel.ckpt) | 12 layers, 8 attention heads, local mask, global patch adjustment | Local file; unvalidated; remote availability unconfirmed |
| [SlightlyLighterModel.ckpt](../forecasting/trained_weights_and_normalization/updated_architecture_checkpoints/SlightlyLighterModel.ckpt) | 4 layers, 4 attention heads, local mask, global patch adjustment | Local file; unvalidated; remote availability unconfirmed |


Both newer checkpoints contain SXR normalization values matching the locally
staged `forecasting/trained_weights_and_normalization/updated_architecture_checkpoints/normalized_sxr_updated_FOXES.npy`.
The original three checkpoints contain values matching the locally staged
`forecasting/trained_weights_and_normalization/normalized_sxr_FOXES_paper.npy` instead.

Edit `forecasting/inference_config.yaml`:

```yaml
data:
  aia_dir:         "/path/to/your/aia_data"
  sxr_dir:         "/path/to/your/sxr_data"   # omit/ignore if prediction_only
  sxr_norm_path:   null  # optional: set to the matching local normalization file
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
**flux contribution maps** are saved when `flux_path` is set and
`model_params.no_flux` is false. To save per-image **attention weights**, set
`weight_path` and `model_params.no_weights: false`; the checked-in config
disables attention saving. Gaussian patch uncertainty maps are saved when
`patch_uncertainty_path` is set and `model_params.no_patch_uncertainty` is false.
These maps contain standard deviations in W/m², not variances.

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
Set `plotting.patch_map_class_source` to `groundtruth` or `predictions` to save
figures under `plots/patch_maps/Below-B/`, `B/`, `C/`, `M/`, or `X/` using the
same flux boundaries as evaluation metrics. Leave it `null` for a flat folder.
Evaluation shows progress while scanning the maps for the shared color scale
and while saving the figures. Existing PNGs are skipped when evaluation is
rerun, so an interrupted plot run resumes with the remaining figures.

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

## Repository Structure

This repository covers the full loop: getting data, training, running
inference, and evaluating a FOXES model.

```text
FOXES
├── helio_foxes
│   └── api.py                  # Observation processing, prediction, and plotting API
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
│   └── trained_weights_and_normalization/  # Original weights and updated_architecture_checkpoints/
├── training
│   ├── train.py                # Train released deterministic or current local + global model
│   ├── train_config.yaml       # Points to the maintained coupled Gaussian config
│   ├── train_coupled_gaussian.yaml
│   ├── train_deterministic.yaml
│   └── callbacks.py            # W&B prediction, attention, and spatial-map callbacks
├── tutorials/heliofoxes_demo/   # Notebook walkthrough and local demo checkpoints
├── pyproject.toml              # Package dependencies and optional development extras
└── requirements.txt            # Additional dependencies for repository workflows
```
