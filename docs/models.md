# Models and Checkpoints

## Download a Checkpoint

Checkpoints are distributed separately from the Python package. Pass a name
from the first column to `FOXESModel` and it's downloaded the first time and
cached in `~/.cache/helio_foxes` (or `HELIO_FOXES_CACHE`). You can also
download a file yourself, or run `git lfs pull` from a clone of the repository,
and pass its path instead.

| Name | Repository path | Model | Download |
| --- | --- | --- | --- |
| `"uncertainty"` (default) | `model_assets/UncertaintyModel.ckpt` | Current model: local + global attention, with Gaussian uncertainty | [Weights](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/model_assets/UncertaintyModel.ckpt) |
| `"inverted-attention-mask"` | `model_assets/LegacyModels/inverted-attention-mask.ckpt` | Legacy deterministic, distant-patch attention *(paper model)* | [Weights](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/model_assets/LegacyModels/inverted-attention-mask.ckpt) |
| `"localized-attention-mask"` | `model_assets/LegacyModels/localized-attention-mask.ckpt` | Legacy deterministic, local attention | [Weights](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/model_assets/LegacyModels/localized-attention-mask.ckpt) |
| `"no-attention-mask"` | `model_assets/LegacyModels/no-attention-mask.ckpt` | Legacy deterministic, full attention | [Weights](https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/model_assets/LegacyModels/no-attention-mask.ckpt) |

## Current Uncertainty Model

`UncertaintyModel.ckpt` consists of a local transformer backbone (3 x 3 attention) + a global attention transformer with an uncertainty MLP head.

No legacy correction is applied to it. The SXR normalization is stored
inside the checkpoint, so you don't need a separate normalization file.

## Legacy Models

The three deterministic checkpoints were trained on an older version of the
SXR targets and don't produce uncertainty estimates. Loading one by name turns
on the legacy correction automatically:

```python
import helio_foxes

model = helio_foxes.FOXESModel("inverted-attention-mask")
```

If you load a legacy checkpoint from a file path instead, pass
`legacy_correction_enabled=True` yourself, since FOXES can't tell from the path.
Pass `legacy_correction_enabled=False` to see the uncorrected output.

With the correction on, any prediction before **2016-11-19 00:00 UTC** is
multiplied by **0.7** (to remove a SWPC scaling bias), and anything after is left alone. The factor applies to
both the full-disk flux and the patch map, so they still add up. Each
prediction needs a timestamp for this to work; an `Observation` already
carries one.
