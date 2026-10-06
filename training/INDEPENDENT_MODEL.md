# Independent local, global, and uncertainty model

`forecasting/uncertainty_model.py` now implements one architecture. Use
`training/train_config.yaml` for the full-data configuration, or the standalone
configs in `experiments/patch_global_independent_sweep/` for the current subset
Huber sweep. Training starts from scratch unless explicitly configured otherwise.

## Model and gradient paths

1. **Local:** local-attention AIA features predict a log10 multiplier per patch.
   Each patch's physical flux is the multiplier times the equal share of the
   characteristic training flux: `(10^sxr_mean - 1e-8) / number_of_patches`.
   Multipliers are positive, with a configurable logit bound for numerical range.
   There is no original per-patch offset subtraction or hard zero-flux cutoff.
   The local head retains PyTorch's default random initialization.
2. **Global:** each detached local patch feature queries a pooled spatial memory
   containing detached local features and projected original AIA image patches.
   A configurable MLP reads local features plus the resulting global context
   and predicts a bounded log10 correction. Corrected flux is
   `detach(local_flux) * 10^correction`. The local total and corrected total
   each receive their own Huber objective, with weight one each.
3. **Uncertainty:** a small MLP reads detached local patch logits and predicts
   relative standard deviation. This scales detached final patch flux; patch
   variances sum and are converted to normalized-log variance for Gaussian NLL.
   The NLL mean and variance conversion scale are detached from both mean paths.

Consequently local Huber updates only the local model, global Huber updates only
its correction head, and NLL updates only uncertainty. Global and uncertainty
heads are enabled by default, and each can be disabled independently.

## Config

```yaml
local_head:
  max_abs_log10_multiplier: 8.0

global_head:
  enabled: true
  grid_size: 32          # Side length of pooled global spatial memory.
  attention_dim: 128     # Must divide evenly by num_heads.
  num_heads: 4
  num_layers: 1          # Cross-attention depth; extra layers use residual updates.
  hidden_dim: 128        # Output MLP width.
  mlp_depth: 1           # Number of hidden Linear/GELU stages.
  dropout: 0.0          # Attention dropout.
  max_abs_adjustment: 1.5 # Bounds the log10 correction, not raw flux.

mean_loss:
  huber_delta: 0.15      # Applies to both independent Huber objectives.
  class_weighting: sqrt_inverse_frequency

uncertainty:
  enabled: true
  initial_relative_std: 1.0
  min_log10_relative_std: -6.0
  max_log10_relative_std: 6.0
  uncertainty_hidden_dim: 16
```

Set `global_head.enabled: false` for local-only means. The corrected prediction
then equals the local prediction and global loss is zero; local loss is counted
only once. Set `uncertainty.enabled: false` to omit the uncertainty module and
its NLL entirely. The logged NLL is then zero and total equals mean loss.

The local and global heads use PyTorch's default random Linear initialization.
There are no alternate global initialization modes. The uncertainty output is
initialized to `initial_relative_std` and trains from the first update.

Architecture sizes live in `vit_architecture` (local) and `global_head` (global).
The config loader translates these readable sections into checkpointed model
constructor settings. Do not specify both `global_head` and the older nested
`vit_architecture.patch_global_context` field in the same config.

## Metrics

The following names have `train/`, `val/`, or `test/` prefixes:

| Metric | Meaning |
|---|---|
| `local_mse`, `local_rmse`, `local_mae` | Error of the local flux sum |
| `local_huber` | Local objective, including configured class weights |
| `global_huber` | Corrected prediction objective, or zero when global is disabled |
| `mean_loss` | Local Huber + global Huber |
| `nll` | Uncertainty Gaussian NLL, or zero when uncertainty is disabled |
| `total_loss` | Mean loss + NLL |
| `mse`, `rmse`, `mae` | Error of the final corrected prediction (local when global is off) |
| `patch_global_context_abs_dex` | Mean absolute log10 global correction |
| `coverage_68`, `coverage_95`, `mean_sigma_dex` | Uncertainty diagnostics when enabled |

RMSE is the square root of the **whole epoch's** MSE, not the average of batch
RMSEs. Prediction errors are in normalized-log target units; multiply MAE/RMSE
by the SXR normalization standard deviation to get dex. Huber is class-weighted
when configured; MSE/MAE/RMSE remain unweighted. Enabled uncertainty trains from the first update.

`val/mae` is the final prediction MAE used by the subset sweep. There are no
duplicate loss aliases; use the metric names above. Validation callbacks still
receive local/final predictions and uncertainty where enabled.

## Removed alternatives and existing artifacts

The uncertainty module no longer contains original/clamped or original-unclamped
mean heads, the old single-summary correction head, joint local/global gradient
training, or contrast priors. `forecasting/model.py` remains the legacy model.
Old configs selecting removed alternatives fail rather than silently run another
architecture. Use an earlier code checkout for those historical experiments.
Their configs, checkpoint files, and results have not been deleted.

The model has no mean-mode selector, compatibility aliases, old global-head
flags, contrast-prior arguments, uncertainty warm-up schedule, or head-only
freezing helpers. Independent gradient routing is always enforced.

Checkpoints saved by this simplified version include an architecture version
and can be reloaded or resumed. Older checkpoints require their original code;
the loader rejects them rather than silently reinterpreting their parameters.
Changing architecture dimensions/depth generally requires fresh training.
Existing running processes do not adopt these changes; start a new run.
