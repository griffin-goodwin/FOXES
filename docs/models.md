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

`UncertaintyModel.ckpt` uses 12 local transformer blocks with 3 × 3 attention
windows and a shared global attention query that adjusts the patch fluxes.
Its uncertainty MLP uses `uncertainty_input: context`: the corrected patch
logit, encoded patch features, raw AIA token projection, and global summary.
These inputs are detached from the flux model when training uncertainty.

The predicted raw patch variances add to the full-disk variance. The model
uses an exact shifted log-normal conversion to obtain the normalized
Gaussian variance and asymmetric prediction intervals; see
[Uncertainties](uncertainties.md) for the calculation and
[Results](results.md) for measured test-set coverage.

No legacy correction is applied to it. The SXR normalization is stored
inside the checkpoint, so you don't need a separate normalization file.

Each new `FOXESModel("uncertainty")` checks whether the published checkpoint
has changed. Unchanged weights are reused; changed weights are downloaded
and replace the cached copy after the download succeeds. If the server
cannot be reached, FOXES logs a warning and uses the cached weights. Existing
caches without download-version metadata are downloaded once to establish it.
Legacy checkpoint names reuse their cached files without an update check.

The check happens when creating the model, not when importing or reloading
the package. Updated weights must be published to the repository before
they can be downloaded by name. Already-loaded model instances keep their
weights. To select a specific checkpoint, pass its local file path:

```python
import helio_foxes

model = helio_foxes.FOXESModel("/path/to/model_assets/UncertaintyModel.ckpt")
```

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
