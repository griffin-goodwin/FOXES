# Patch-mean parameterization comparison

This experiment compares three meanings for the model's scalar output at every
image patch:

- `original`: a direct normalized log10-SXR patch prediction, converted to
  nonnegative raw flux;
- `multiplier`: a log10 multiplier applied to the patch's equal share of the
  typical global training flux, with a zero-initialized head;
- `positive`: the same equal-share baseline multiplied by
  `10 ** (patch_logit * sxr_std)`, retaining the original PyTorch head
  initialization. It is strictly positive and has no hard zero boundary.

In every arm, raw patch fluxes are summed to produce the global prediction.
All arms use `/data/FOXES_screening/subset-og-quick`, the same FOXES_OG quick
subset used by the parameter-screening experiments. Its reduced training split
is paired with the complete validation and test splits, and its AIA arrays are
already scaled to `[-1, 1]`.

The seed, architecture, optimizer, weighted-Huber objective, batching, and
validation metrics are inherited from the same `training/train_config.yaml`.
All arms run for 15 epochs with the cosine scheduler's `T_max` also set to 15.
Uncertainty is disabled in every arm, so the only model difference is the
patch-mean parameterization and its required head initialization (PyTorch
default for `original` and `positive`, zeros for `multiplier`). Every arm
therefore receives the same per-class metrics, prediction plots, and
attention-map callbacks; the uncertainty-only spatial variance callback is
omitted from all of them.

Validate the matched configs and print the commands without starting training:

```bash
python experiments/patch_mean_comparison/run_comparison.py
```

Run all arms concurrently on three GPUs:

```bash
python experiments/patch_mean_comparison/run_comparison.py \
  --gpu-ids 0,1,2 \
  --parallel \
  --run
```

For a one-GPU sequential comparison, use `--gpu-ids 0 --run`. The existing
original result can be reused; run only the new follow-up with:

```bash
python experiments/patch_mean_comparison/run_comparison.py \
  --arm positive --gpu-ids 0 --run
```

Compare `val_class/macro_mse_dex2` (the checkpoint metric), `val/mse`,
`val/mae`, the per-class error metrics, and the patch-flux maps in W&B. Do not
compare uncertainty metrics: none of the arms contains an uncertainty head.

## Positive-mean contrast follow-up

`positive_contrast.yaml` keeps the positive mean arm and adds a single weak
spatial prior. Within each image and separately for 94 Å and 131 Å, patch
brightness above the patch median defines a normalized contrast map. A cosine
loss with weight `0.001` encourages normalized patch-flux contributions to
overlap that hot-channel contrast. Huber remains the integrated-flux objective
and is logged independently. Samples without measurable hot-channel contrast
receive zero auxiliary loss. Uncertainty remains disabled. Five validation
figures are logged under `Attention plots` every epoch; each includes the
learned patch-flux contribution map.

Run it with:

```bash
python training/train.py \
  --config experiments/patch_mean_comparison/positive_contrast.yaml
```

The stronger follow-up changes only the contrast weight from `0.001` to
`0.01`, and writes to a separate checkpoint directory and W&B run:

```bash
python training/train.py \
  --config experiments/patch_mean_comparison/positive_contrast_10x.yaml
```
