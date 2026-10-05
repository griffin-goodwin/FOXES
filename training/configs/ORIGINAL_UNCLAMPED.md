# Original head with unclamped positive patch flux

Run from the repository root in the `foxes` environment:

```bash
python training/train.py --config training/configs/original_unclamped_local_uncertainty.yaml
```

`mean_parameterization: original_unclamped` retains the original normalized-log
head scale and PyTorch initialization. For patch logit `z`, its physical flux is
`10 ** (z * sxr_std + sxr_mean)`. There is no mean-logit clamp, raw-flux clamp,
per-patch subtraction of `1e-8`, or division by the number of patches. This is
an exponential flux prediction, not an unconstrained linear raw-flux head.
The original `original` mode still uses its existing offset subtraction and
clamps; existing checkpoints keep their behavior.

The experiment uses local 3x3 attention at each transformer layer and no global
correction branches. Stacked layers expand the effective receptive field beyond
one 3x3 neighborhood. Raw patch contributions sum to the predicted total, which
is normalized with the existing whole-image `log10(total_flux + 1e-8)` transform
for the loss. That offset is applied once to the total, not to each patch.

The detached patch uncertainty head remains enabled and trains from epoch zero.
It predicts relative standard deviations, scales them by physical patch flux,
and converts summed raw variance to the existing normalized-log Gaussian NLL.
Its numerical bounds and interval handling remain in place; the unclamped
experiment concerns the mean output. Uncertainty gradients remain separate
from mean-head/backbone gradients.

This starts from fresh weights with square-root class-weighted Huber (delta
0.15), AdamW LR `5e-5`, and a 25-epoch cosine schedule. Early stopping watches
validation MSE with patience five. Every epoch and `last.ckpt` go into a new
checkpoint directory. Evaluate total-flux MSE/MAE, class errors, uncertainty
coverage, and spatial maps at the same validation timestamps.

The exponential is evaluated in float32 even for reduced-precision inputs.
Without bounds, extreme logits can overflow or underflow; no hidden mean clamp
is added. Ordinary finite outputs and their gradients are checked in the tests.
