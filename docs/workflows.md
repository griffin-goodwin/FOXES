# Development and command-line workflows

Use the [Python guide](python_api.md) for individual observations. The scripts
below prepare datasets, run batch predictions, evaluate results, or train models.
Run commands from the repository root and edit each YAML config before use.

## Set up a checkout

```bash
git clone https://github.com/griffin-goodwin/FOXES.git
cd FOXES
conda create -n helio-foxes-dev python=3.12 pip -y
conda activate helio-foxes-dev
python -m pip install -e ".[dev,evaluation,training]"
git lfs pull
```

For raw-data processing, also install `requirements.txt`. In notebooks, select
this environment's kernel and restart it after changing imported source.
Run `python -m pytest -q` when checking code changes.

Paths inside a YAML config are relative to that file, rather than your terminal
or notebook's working directory. The bundled configs set `path_root: ..` so
`./foxes_data/` and `./foxes_outputs/` refer to folders in the repository root.
You can set `path_root` to your data drive instead, or use absolute paths;
`~` and environment variables such as `${FOXES_DATA}` also work. When launching
from elsewhere, pass the script and `--config` locations explicitly.
The example data folders must exist or be replaced with your actual dataset paths.

## Get data

Download the released dataset into paired AIA/SXR `.npy` files:

```bash
python download/hugging_face_data_download.py --config download/hf_download_config.yaml
```

Set the output directories and desired splits in `download/hf_download_config.yaml`.
If you already have local Parquet files, set `local_parquet_dir` in that config.

For raw AIA and GOES data instead, edit the corresponding configs, then run:

```bash
python download/download_sdo.py --config download/sdo_download_config.yaml
python download/download_sxr.py --config download/sxr_download_config.yaml
python data/build_dataset.py --config data/build_dataset_config.yaml
```

AIA downloads require a registered JSOC email. The build script cleans and
converts AIA, then combines and aligns GOES measurements with the images.

## Run inference

Edit `forecasting/inference_config.yaml` to set `data.aia_dir`, `data.sxr_dir`,
`data.checkpoint_path`, and `output_path`. AIA files are processed
`(7, 512, 512)` arrays; matching SXR files contain scalar flux values and use
the same timestamp filenames. Set `prediction_only: "true"` when SXR targets
are unavailable.

The default checkpoint is `model_assets/UncertaintyModel.ckpt` with
`legacy_model: false`. Leave `data.sxr_norm_path: null` to use its embedded
normalization. See [Models and checkpoints](models.md) for other weights.

```bash
python forecasting/inference.py --config forecasting/inference_config.yaml
```

The script writes a prediction CSV. Flux maps and patch uncertainty outputs
are controlled by the output paths and `model_params` settings in the config.
Inference uses one device selected by `device`; CUDA uses the first visible GPU.

## Evaluate predictions

Set the prediction CSV and output directory in
`forecasting/evaluation_config.yaml`, then run:

```bash
python forecasting/evaluation.py --config forecasting/evaluation_config.yaml
```

Evaluation writes regression metrics and figures, plus interval coverage and
calibration plots when Gaussian uncertainty is present. See the bundled
[Results](results.md) for an example.

## Train a model

Training expects paired AIA/SXR files in `train/`, `val/`, and `test/` folders.
For raw-data builds, enable `steps.split` and `sxr_normalization.compute` in
`data/build_dataset_config.yaml` to prepare these inputs.

Edit the data paths, model, and training settings, then run:

```bash
python training/train.py --config training/train_coupled_gaussian.yaml
```

Use `training/train_deterministic.yaml` for a deterministic model. The Gaussian
config matches the current checkpoint's architecture and losses.
