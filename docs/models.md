# Models and checkpoints

FOXES predicts a flux contribution for each image patch. These contributions
sum to the full-disk soft X-ray flux. HARP regions summarize contributions
inside active-region boxes; overlapping boxes can count the same patches.

## Download a checkpoint

Save weights under `model_assets/`, using the paths below. They are separate
from the Python package; after cloning this repository, `git lfs pull` retrieves
the checkpoint files.

| Checkpoint path | Model | Download |
| --- | --- | --- |
| `model_assets/UncertaintyModel.ckpt` | Current local + global model, with Gaussian uncertainty | [Weights](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/model_assets/UncertaintyModel.ckpt) |
| `model_assets/LegacyModels/inverted-attention-mask.ckpt` | Legacy deterministic, distant-patch attention | [Weights](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/model_assets/LegacyModels/inverted-attention-mask.ckpt) |
| `model_assets/LegacyModels/localized-attention-mask.ckpt` | Legacy deterministic, local attention | [Weights](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/model_assets/LegacyModels/localized-attention-mask.ckpt) |
| `model_assets/LegacyModels/no-attention-mask.ckpt` | Legacy deterministic, full attention | [Weights](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/model_assets/LegacyModels/no-attention-mask.ckpt) |

## Current uncertainty model

`UncertaintyModel.ckpt` combines local patch features with shared global
context. A seven-channel `(7, 512, 512)` AIA input produces a 64 × 64 flux map,
full-disk flux, and Gaussian prediction intervals.

Load it with the default `legacy_model=False`. Its SXR normalization is embedded
in the checkpoint; no separate SXR normalization file is needed.

Patch variances add to the full-disk variance. The patch uncertainty map
attributes that variance across the image; there are no local SXR labels to
validate regional error bars. See [Results](results.md) for full-disk metrics,
figures, and measured interval coverage.

## Legacy models

The three deterministic checkpoints use the older SXR targets and do not
produce uncertainty estimates. Enable their approximate output correction
when loading them:

```python
import helio_foxes

model = helio_foxes.load_model(
    "model_assets/LegacyModels/inverted-attention-mask.ckpt",
    legacy_model=True,
)
```

Predictions strictly before **2016-11-19 00:00 UTC** receive a **0.7** flux
factor; later predictions use 1.0. The same factor applies to full-disk and
patch flux, preserving their sum. Timestamps are required; an `Observation`
supplies its time automatically.

This date rule approximates the older mixed-satellite targets and does not
establish predictive calibration. Keep each checkpoint's original weights
and embedded normalization. The original normalization artifact is
`model_assets/LegacyModels/normalized_sxr_FOXES_paper.npy`.

## Team

Griffin Goodwin, Alison March, Jayant Biradar, Christopher Schirninger,
Robert Jarolim, Angelos Vourlidas, Viacheslav Sadykov, and Lorien Pratt.
