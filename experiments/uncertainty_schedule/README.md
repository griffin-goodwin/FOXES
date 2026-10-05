# Uncertainty start-epoch comparison

This stage keeps a selected patch-mean parameterization fixed and changes only
when the detached uncertainty head begins training. The head
predicts an absolute `log10(std [W/m²])` for every patch. These independent raw
patch variances are summed and converted to normalized log-SXR variance for
the Gaussian likelihood; they are never scaled by predicted patch flux.

- `start_epoch_0.yaml` and `start_epoch_8.yaml` retain the original patch mean.
- `positive_start_epoch_0.yaml` and `positive_start_epoch_8.yaml` use the
  positive patch mean. For each mean, uncertainty NLL contributes either from
  epoch 0 or from zero-based epoch 8; a delayed head remains initialized but
  receives no gradients in epochs 0–7.

Both configs inherit the same 15-epoch FOXES_OG screening setup used for the
patch-mean comparison. They select checkpoints by
`val_class/macro_nll`; mean MSE and MAE remain required diagnostics because the
detached uncertainty branch should leave them unchanged.

Each validation epoch also logs five class-stratified combined figures under
`Attention plots` in W&B. A figure contains the AIA image, mean and center-patch
attention, the learned patch-flux contributions, and the independent patch
standard deviations. Samples are forwarded one at a time to keep the 4096 by
4096 attention tensors within GPU memory. The separate uncertainty-only map
callback is disabled for these runs because those two spatial maps are already
included in the combined figure.

Run the arms separately with:

```bash
python training/train.py --config experiments/uncertainty_schedule/start_epoch_0.yaml
python training/train.py --config experiments/uncertainty_schedule/start_epoch_8.yaml
python training/train.py --config experiments/uncertainty_schedule/positive_start_epoch_0.yaml
python training/train.py --config experiments/uncertainty_schedule/positive_start_epoch_8.yaml
```

Compare validation macro NLL, 68% and 95% coverage, interval width, and mean
MSE/MAE. Do not change the variance architecture or likelihood in this stage.
