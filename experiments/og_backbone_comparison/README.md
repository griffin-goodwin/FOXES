# Published-backbone subset comparison

All six standalone YAML files use `/data/FOXES_screening/subset-og-quick`:
14,250 training pairs, 16,317 validation pairs, and 100,092 test pairs.
The subset contains reduced training data and the full validation/test splits.
Training uses only train and validation; leave test evaluation until the
experiment decisions are fixed. AIA arrays are already normalized. Every arm
uses the same supplied SXR normalization file.

## Arms and comparisons

| Config | Implementation | Attention | Patch mean | Mean objective |
|---|---|---|---|---|
| `original_legacy.yaml` | `forecasting/model.py:ViTLocal` | Inverted 9x9 | Original subtraction + clamp | Legacy adaptive Huber |
| `original_matched.yaml` | Uncertainty wrapper, uncertainty off | Inverted 9x9 | Original subtraction + clamp | Fixed sqrt-frequency Huber |
| `local_original.yaml` | Same mean-only wrapper | Local 3x3 | Original subtraction + clamp | Same fixed Huber |
| `local_multiplier.yaml` | Same mean-only wrapper | Local 3x3 | Log10 multiplier of flat baseline | Same fixed Huber |
| `local_global_multiplier.yaml` | Same mean-only wrapper + global correction | Local 3x3 + global AIA summary | Same multiplier + bounded correction | Same fixed Huber |
| `local_unclamped.yaml` | Same mean-only wrapper | Local 3x3 | Direct positive exponential | Same fixed Huber |

The first arm actually trains the original `model.py` class. It retains the
current fix that prevents validation/test from updating adaptive-loss history.
Its base weights are copied from the published run export. It is **not an exact
historical reproduction**: batch size and training budget are matched by request,
data are the existing subset, and the exact historical commit is unavailable
locally. Published weight decay was not included in the export. The current
legacy loss implementation/default adaptive rules are used with Huber delta 0.3.

Compare the first two arms to assess objective/wrapper differences. Compare
`original_matched` with `local_original` to change inverted 9x9 to local 3x3
attention (both masking direction and window size change). Compare
`local_original` with `local_multiplier` for the output parameterization, including
its intentional initialization and bound differences. Compare `local_multiplier`
with `local_global_multiplier` to isolate the added global correction. Compare
`local_original` with `local_unclamped` to remove the per-patch subtraction and
clamp while retaining the original exponential scale and default initialization.

## Matched settings

- Published backbone: 256 embedding, 1,024 hidden, eight transformer layers,
  eight heads, 8-pixel patches, 4,096 patches, seven channels, dropout 0.1.
- Batch 48, one GPU per arm, no gradient accumulation, seed 42, fresh weights.
  Edit `batch_size` and `accumulate_grad_batches` directly in all six YAML files
  to change them; the runner accepts positive integers and checks that arms match.
  Validation also uses batch 48.
- AdamW LR `1e-4`, weight decay `1e-4`; 20 epochs and matching cosine decay
  to `1e-6`. These are shared experiment choices, not recovered paper settings.
- No early stopping. Save the best validation-MAE checkpoint and `last.ckpt`.
- Uncertainty disabled in all arms; no contrast prior. All new-head arms use
  training-subset-derived square-root inverse-frequency weights and Huber 0.3.
- Prediction/per-class metrics and five stratified spatial examples are logged.
  Spatial maps are included by the attention callback every five epochs.
- Independent checkpoint directories and W&B run names for every arm.
  This batch-48, 20-epoch comparison uses the `b48-e20/` checkpoint subdirectory
  and `-B48-E20` run-name suffix to separate it from previous runs.

Twenty epochs provide a screening comparison, not a guarantee that every arm
has converged. Examine validation-MAE trajectories as well as best checkpoints
before ruling out a slower-learning arm.

The multiplier is `b * 10^clamp(z, -8, 8)`, where
`b = (10^sxr_mean - 1e-8) / 4096` (with the implementation's positive numerical
guard). Its zero-initialized head starts with a flat map. The global correction
multiplies this map by `10^delta`, where delta is bounded to +/-0.3 dex and starts
at zero. It reads local features and globally summarized original AIA tokens;
it is not global attention over scalar flux predictions. Both stages train jointly.
Eight local 3x3 blocks have a maximum 17x17-patch receptive field, not a single
3x3 receptive field. Global correction adds parameters, so backbone capacity is
matched but total parameter counts are intentionally different.

## Run

Activate the `foxes` environment first (or use its absolute Python path).

```bash
# Validate all configs, subset paths, and paired filenames without training.
python experiments/og_backbone_comparison/run_comparison.py

# CPU constructor/forward/backward/optimizer checks on reduced spatial grids.
python experiments/og_backbone_comparison/run_comparison.py --smoke

# Run all six sequentially on one GPU.
python experiments/og_backbone_comparison/run_comparison.py --gpu 0 --run

# Recommended for two GPUs: one active run per GPU, three runs per queue.
python experiments/og_backbone_comparison/run_comparison.py \
  --parallel --gpu-ids 0,1 --run

# Run all six simultaneously, one arm per GPU (GPUs 6 and 7 remain unused).
python experiments/og_backbone_comparison/run_comparison.py \
  --parallel --gpu-ids 0,1,2,3,4,5 --run

# Run selected arms only.
python experiments/og_backbone_comparison/run_comparison.py \
  --arm original_legacy local_multiplier local_global_multiplier --gpu 0 --run
```

The runner requires matching configs and rejects full-data training paths.
Parallel runs set a separate CUDA_VISIBLE_DEVICES value for each child, so
each YAML still addresses its sole visible GPU as device 0. Arms are assigned
round-robin to the supplied GPUs, with one active run per GPU. Each queue
advances independently when its current run finishes; it does not wait for
the other GPU. Extra supplied GPUs are left unused.

With `--gpu-ids 0,1`, the queues are:

- GPU 0: `original_legacy` → `local_original` → `local_global_multiplier`.
- GPU 1: `original_matched` → `local_multiplier` → `local_unclamped`.

Each arm gets a console log under a timestamped `logs/` directory; use
`--log-dir` to select another directory. Use `tail -f` on an arm's log to watch it.
Six concurrent runs create up to 48 data-loader workers during a phase with the
current eight workers per run, so shared CPU, RAM, and storage can limit speed.

Sequential execution stops on training failure. Parallel execution lets the
other arms (including queued runs) finish and reports all failed arms with their log paths. Interrupting
the runner terminates its active training children. It refuses to restart into a directory containing
checkpoints. To repeat with another seed, update **every** config's seed and
use new checkpoint directories and run names. Other user runs are not modified.
You can also launch any YAML directly with `training/train.py --config ...`.

## Compare results

Use `val_mae_loss` for the legacy arm and `val/mae` for the other arms; they both
measure normalized-log target MAE. All configs use the same normalization, so
these values are comparable within this experiment. Multiply by SXR standard
deviation to express MAE in dex. Do not compare legacy adaptive total loss with
the fixed-weight total loss as though they were the same objective.

Also inspect `val_class/*/mae_dex`, `val_class/macro_mae_dex`, the common
prediction plots, and patch-flux maps at matching timestamps and color scales.
Spatial appearance is a plausibility check, not independent spatial ground truth.
Select checkpoints by validation MAE, not the last epoch. Rerun the most
promising controlled pair across seeds before treating a small difference as
reliable. Do not tune architectures against the held-out test split.
