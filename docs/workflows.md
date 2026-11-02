# Command-Line Workflows

For single observations or short time ranges, the [Python Guide](python_api.md)
is the easier route. The scripts on this page are for heavier jobs: building
datasets, running batch predictions, evaluating results, and training. Each
script reads a YAML config, so open and edit the config before running
anything. Run all commands from the repository root.

## Set Up a Checkout

```bash
git clone https://github.com/griffin-goodwin/FOXES.git
cd FOXES
conda create -n helio-foxes-dev python=3.12 pip -y
conda activate helio-foxes-dev
python -m pip install -e ".[dev,evaluation,training]"
git lfs pull
```

If you plan to process raw data, install `requirements.txt` as well. When
working in a notebook, pick this environment's kernel, and restart the kernel
after editing any source it imports. Run `python -m pytest -q` to check your
changes.

Paths in a YAML config are resolved relative to the config file itself, not
to wherever your terminal or notebook happens to be. The bundled configs set
`path_root: ..`, which makes `./foxes_data/` and `./foxes_outputs/` point to
folders in the repository root. If your data lives somewhere else, point
`path_root` at it or use absolute paths; `~` and environment variables like
`${FOXES_DATA}` work too. If you launch a script from a different directory,
give the full paths to both the script and its `--config`. The data folders in
the example configs need to exist, so create them or swap in your own paths.

## Get Data

To download the released dataset as paired AIA/SXR `.npy` files:

```bash
python download/hugging_face_data_download.py --config download/hf_download_config.yaml
```

Set the output directories and the splits you want in
`download/hf_download_config.yaml`. If you already have the Parquet files
locally, set `local_parquet_dir` in that config to skip the download.

To build the dataset from raw AIA and GOES data instead, edit the matching
configs and run:

```bash
python download/download_sdo.py --config download/sdo_download_config.yaml
python download/download_sxr.py --config download/sxr_download_config.yaml
python data/build_dataset.py --config data/build_dataset_config.yaml
```

AIA downloads need a [registered JSOC email](http://jsoc.stanford.edu/ajax/register_email.html).
The build script cleans and processes the AIA images, then lines the GOES
measurements up with them.

## Run Inference

In `forecasting/inference_config.yaml`, set `data.aia_dir`, `data.sxr_dir`,
`data.checkpoint_path`, and `output_path`. The AIA files should be processed
`(7, 512, 512) -> (C, H, W)` arrays, and each matching SXR file holds a single flux value
under the same timestamp filename. If you don't have SXR targets, set
`prediction_only: "true"`.

By default the config uses `model_assets/UncertaintyModel.ckpt` with
`legacy_model: false`. Leave `data.sxr_norm_path: null` so it uses the
normalization stored in the checkpoint. Other weights are listed under
[Models and Checkpoints](models.md).

```bash
python forecasting/inference.py --config forecasting/inference_config.yaml
```

The script writes a CSV of predictions. Whether it also saves flux maps and
patch uncertainty maps depends on the output paths and `model_params` in the
config. Inference runs on the single device set by `device`; with CUDA, that's
the first visible GPU.

## Evaluate Predictions

Point `forecasting/evaluation_config.yaml` at your prediction CSV and an
output directory, then run:

```bash
python forecasting/evaluation.py --config forecasting/evaluation_config.yaml
```

You'll get regression metrics and figures, plus interval coverage and
calibration plots if the predictions include Gaussian uncertainty. The
[Results](results.md) page is an example of the output.

## Train a Model

Training expects paired AIA/SXR files split into `train/`, `val/`, and `test/`
folders. If you're building from raw data, turn on `steps.split` and
`sxr_normalization.compute` in `data/build_dataset_config.yaml` to set those
up.

Edit the data paths, model, and training settings, then run:

```bash
python training/train.py --config training/train_coupled_gaussian.yaml
```

The Gaussian config matches the architecture and losses of the current
checkpoint. For a deterministic model, use `training/train_deterministic.yaml`
instead.
