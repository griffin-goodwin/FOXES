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

## The Original FOXES Model

The original model predicts a raw SXR flux contribution for each image patch.
Its three trained checkpoints use **global**, **local**, or **inverted**
attention. The model converts patch logits to flux and clamps very small
contributions to a positive floor (`1e-15` W/m²), which can make the maps
relatively sparse. In the evaluations run so far, global RMSE and MSE were
good, but the spatial maps for weak flares and quiet-Sun images were less
convincing. This motivated the updated model below.

## The Updated (Post Publication) FOXES Model: Local + Global Context

The maintained model in this repository extends the released FOXES patch
predictor. Local attention builds a feature for each AIA image patch. Patch
flux starts from a positive, image-wide baseline divided across the patches;
each patch's output scales its share. A global query then summarizes the full
AIA image and uses that context, together with each patch's local features,
to adjust its flux contribution. The corrected patch fluxes sum to the
full-disk SXR prediction.

This local + global design produces **less sparse, visually better spatial
flux maps** in the examples examined so far. It also predicts a global
Gaussian uncertainty and a map of each patch's contribution to that variance.
The uncertainty map is an attribution of global uncertainty; there are no patch-level SXR labels
to establish calibrated local error bars.

The tradeoff is **somewhat worse global SXR prediction performance** than the
original released FOXES model in the evaluations run so far. The visual map
improvement is qualitative.
The newly added local + global checkpoints listed below have **not yet been
validated**; these earlier observations should not be read as evaluation
results for either new checkpoint file.

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
│   └── trained_weights_and_normalization/  # Tracked original weights; local newer weights may also be staged here
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

### Use FOXES from a Python UI

Install this checkout in the UI environment:

```bash
pip install -e .
```

Load a checkpoint once when the UI starts, then reuse it for each image:

```python
import foxes

model = foxes.load_model("/path/to/model.ckpt", device="auto")
processor = foxes.ObservationProcessor(email="you@example.com")
observation = processor.fetch_observation("2023-08-01T00:00:00Z")
print(observation.data.shape)  # (7, 512, 512)
result = foxes.predict(observation, model=model)

print(result.flux)       # full-disk SXR flux, W/m²
print(result.flux_map)   # patch contributions, W/m²
print(result.harp_regions)  # HARP numbers, locations, and flux sums, W/m²
print(result.sxr)        # measured GOES XRS-A/XRS-B fluxes and sample time
ax = foxes.plot_flux_map(result)  # optional: vmin=1e-10, vmax=1e-6
ax.figure.savefig("flux_map.png")
```
When plotting a prediction, the title automatically shows the predicted FOXES
flux and the observed SXR-B mean when available, in W/m². For Gaussian models,
it also shows the model's full-disk uncertainty beside the predicted flux as
`± value`, computed as the square root of the summed patch variances. This is
a model estimate, not a calibrated error bar.

`ObservationProcessor` calls `AIAProcessor`, `HARPProcessor`, and `SXRProcessor`
for the same requested timestamp. All three sources are fetched by default.
AIA and HARPs share the crop geometry and JSOC email; GOES SXR needs no email.
Prediction adds a `flux_sum` in W/m² and a canonical lowercase `harpnum` to
each region in `result.harp_regions`. Each result includes `top_pct`,
`left_pct`, `width_pct`, and `height_pct` for its location in the image.
Original JSOC fields, including `LON_MIN`, `LON_MAX`, `LAT_MIN`, and `LAT_MAX`,
are preserved. The observation's queried rows are not modified.

```python
for harp in result.harp_regions or []:
    print(harp["harpnum"], harp["flux_sum"], "W/m²")
```

Plotting reuses these regions and the same flux summation rule:

```python
ax = foxes.plot_flux_map(result, show_harp_boxes=True)
```

For Gaussian models, each HARP result also contains `flux_sum_uncertainty`
in W/m²: square the patch uncertainties, sum within the same box as the flux,
then take the square root. Empty boxes have zero uncertainty; deterministic
models return `None`. These values follow the model's additive variance
attribution; they have not been calibrated as regional error bars.

Uncertainty appears on a third label line as `± value` when available.
Hide it with `show_harp_uncertainty=False`:

```python
ax = foxes.plot_flux_map(result, show_harp_boxes=True, show_harp_uncertainty=False)
for harp in result.harp_regions or []:
    print(harp["harpnum"], harp["flux_sum"], harp["flux_sum_uncertainty"])
```

Each HARP box is labeled with its number and flux sum in W/m². Labels use a
fixed font size; set `harp_label_fontsize` to change it. A standard
Matplotlib annotation places each label diagonally outside a box corner with
a three-point gap in both x and y. The entire label extends outward from
the corner. The side
toward the image center is preferred. A simple overlap pass tries the other
corners before four-point nudges, keeping automatic positions outside their
HARP box and inside the map. Shifted
labels get connector lines. `harp_label_max_shift=80` limits movement to 80
points from a corner; lower it to keep labels closer, or set it
to zero for labels fixed at their preferred corner. If no free position is found within the limit, the
label stays at its starting position. Crowded cases may need manual offsets
or a higher `min_harp_flux`. Labels are not automatically hidden. This runs
once when plotting; replot after substantial size changes.

For crowded maps, set individual labels' starting positions. Offsets are in points
from the chosen corner: positive x moves right, positive y moves up. Manually
offset labels get matching connector lines:

```python
ax = foxes.plot_flux_map(
    result, show_harp_boxes=True,
    harp_label_offsets={1234: (15, 30), 5678: (-20, -30)},
)
```

Box outlines and labels share a fixed
color by flux rank (strongest: sky blue, second: orange, third: green).

Expand HARP boxes when fetching observations so prediction sums and plot
outlines use the same expanded region:

```python
processor = foxes.ObservationProcessor(email=email, harp_bbox_expansion=0.1)
observation = processor.fetch_observation(timestamp)
result = foxes.predict(observation, model=model)
```

`harp_bbox_expansion` defaults to zero. `0.1` adds 10% of each box's width and
height on each side (20% more total width and height), clipped to the image.
Original JSOC heliographic bounds are preserved. For direct HARP queries, use
`foxes.HARPProcessor(email=email, bbox_expansion=0.1)`. The expansion is carried
in observation metadata for HARP queries deferred until plotting.

Set `min_harp_flux` to hide regions below an absolute flux sum in W/m²:

```python
ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=1e-8)
```

The default threshold is `1e-7` W/m²; use `min_harp_flux=0` to show all regions.
Filtering only affects the plot;
`result.harp_regions` retains all predicted regional sums.

Set `include_aia=False`, `include_harps=False`, or `include_sxr=False` on
`ObservationProcessor` to disable a source. Override any setting for one
observation with `processor.fetch_observation(timestamp, include_sxr=False)`
(and similarly for AIA/HARPs). With AIA disabled, `observation.data` is `None`,
and prediction requires fetching AIA first. Each component processor handles
only its own source.
If the observation did not fetch HARPs, plotting with `show_harp_boxes=True`
queries them using the prediction metadata and its `jsoc_email` argument.
Pass `harp_regions=regions` to the plot to use an explicit set of regions.

`harp_regions=None` means no query was requested; `[]` means no usable HARP
regions were returned, including query failures. Plotting reuses an empty
result without repeating the query. HARP data is supplementary, so an empty
HARP result does not discard the AIA observation.

`observation.metadata` is preserved as `result.metadata`. It records the
requested UTC timestamp, channel order, original processed image shape, and
crop padding. The timestamp is the requested time, not the individual AIA
exposure times. The JSOC email stays on the processor or plotting call.

For array-based workflows, `foxes.AIAProcessor(email=email).fetch_date(timestamp)`
returns a NumPy stack. Supply its time once to `foxes.predict(image, model=model,
timestamp=timestamp)` to enable automatic HARP queries from the result. This
assumes the standard AIA channel order and crop padding of 0.1. For a different
crop, wrap the array in `foxes.Observation` with explicit
`foxes.ObservationMetadata`. Plotting a plain flux-map array also continues to
accept an explicit `timestamp`.

Each sum includes positive patch contributions whose centers fall inside the
HARP box. Boxes can overlap, so their sums need not add to the full-disk flux.

### Measured GOES SXR

The observation's `sxr` dict is also carried into `result.sxr`; it is a measured
comparison value and is not used as model input. XRS-B is the 1–8 Å band to
compare with the predicted `result.flux`; XRS-A is 0.5–4 Å. Fluxes are in W/m².

```python
processor = foxes.ObservationProcessor(
    email="you@example.com", include_sxr=True, sxr_tolerance=120,
    sxr_satellite=None,  # optionally restrict to e.g. 18
)
observation = processor.fetch_observation("2023-08-01T00:00:00Z")
result = foxes.predict(observation, model=model)
if result.sxr is not None:
    print("Predicted:", result.flux, "Observed:", result.sxr["xrsb_flux"])
    print(result.sxr["satellites"], result.sxr["timestamp"])
    print(result.sxr["measurements"])  # individual contributing satellite values
    print(result.sxr["time_offset_seconds"])  # sample minus requested UTC time

# Fetch just SXR, using the same datetime/ISO string format:
sxr = foxes.SXRProcessor(tolerance=120, satellite=18).fetch_date(timestamp)
# Or through the observation fetcher:
sxr_only = processor.fetch_observation(timestamp, include_aia=False, include_harps=False)
```

`SXRProcessor` downloads the GOES science `avg1m` product and finds the nearest
usable timestamp within `tolerance` seconds (120 by default; time ties favor
the earlier sample). At that timestamp, `xrsb_flux` is the arithmetic mean of
available satellites' finite positive XRS-B values with zero XRS-B quality
flags. This follows `data/align_aia_sxr.py`: average in physical W/m² before
the training transform applies log normalization. `result.flux` also uses
physical W/m², so no normalization is needed for comparison.

`satellites` names the contributors and `measurements` retains their individual
values. Each satellite counts once; measurements at different timestamps are
never averaged together. Restricting `satellite` yields a single-satellite
result. XRS-A is averaged over valid short-band measurements from the XRS-B
contributors, or `None` if unavailable. Bad XRS-A values do not exclude good
XRS-B measurements from the model comparison. The returned `timestamp` is the
actual sample time, while `observation.metadata.timestamp` remains the requested
time. Daily files are filtered to the requested window and deleted after use.
No interpolation or additional legacy scaling is applied to the reprocessed
science fluxes. `sxr=None` means fetching was disabled or no usable measurement
was available, including network or processing failures; these failures log a
warning and do not discard the AIA observation. The science archive may lag
recent observations. Check the calibration used for older training datasets
before interpreting measured-versus-predicted differences. Unlike the bulk
dataset converter, this fetcher does not fill missing fluxes using log-space
interpolation; only valid measured samples contribute.

GOES netCDF reading requires `h5netcdf` and `h5py`, both declared package
dependencies. If an existing environment reports either missing, run
`python -m pip install h5netcdf h5py` using that environment's Python. In a
notebook, run `%pip install h5netcdf h5py` and restart the kernel.

To install:
```bash
python -m pip install -e .
```
`AIAProcessor.fetch_date` writes the seven raw AIA FITS frames to a temporary
directory, processes them with ITI, and returns the stack in memory.
`ObservationProcessor.fetch_observation` uses this same processing path. The raw files are
removed when the call finishes, including when a download is incomplete.
The requested timestamp is interpreted as UTC. An ISO string with an offset
is converted to UTC; a timestamp without an offset is assumed to be UTC.
If you provide `aia_norm_path` to `AIAProcessor`, do not also pass that
artifact to `load_model`, since that would normalize the stack twice.

The input uses channel order `[94, 131, 171, 193, 211, 304, 335]` and must
already have the same preprocessing as the checkpoint's training data. If that
training used an AIA normalization artifact, pass `aia_norm_path` to
`foxes.load_model`. `result.intervals` and `result.patch_uncertainty_map` are
available for the coupled Gaussian model. Checkpoints are passed by path and
are not bundled into the Python package. Pass a path to a **full checkpoint
file** available on the UI machine.

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

---

## Running the Model

FOXES is run in four steps: **get data**, **inference** (run a checkpoint over
your data), **evaluation** (score the predictions and generate plots), and
optionally **training** your own model. All are driven by a YAML config — edit
the config, then run the script.

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
under `updated_architecture_checkpoints/`. Their
architecture and normalization details below come from the local checkpoint
metadata; **neither has yet been validated for prediction quality, spatial
maps, or uncertainty calibration**.

| Local checkpoint | Architecture recorded in checkpoint | Status |
| --- | --- | --- |
| `updated_architecture_checkpoints/ComplexModel.ckpt` | 12 layers, 8 attention heads, local mask, global patch adjustment | Local file; unvalidated; remote availability unconfirmed |
| `updated_architecture_checkpoints/SlightlyLighterModel.ckpt` | 4 layers, 4 attention heads, local mask, global patch adjustment | Local file; unvalidated; remote availability unconfirmed |


Both newer checkpoints contain SXR normalization values matching the locally
staged `updated_architecture_checkpoints/normalized_sxr_updated_FOXES.npy`.
The original three checkpoints contain values matching the locally staged
`normalized_sxr_FOXES_paper.npy` instead.

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
