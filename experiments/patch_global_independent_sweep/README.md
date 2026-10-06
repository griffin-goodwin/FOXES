# Independent local/global subset sweep

This is the focused follow-up to the architecture comparison. It preserves the
setup of the saved `OG-PatchGlobal-Independent` run and changes only Huber
delta between arms. Every YAML is standalone.
The old architecture-comparison configs and runs are left intact.

## Three Huber settings, eight epochs each

Test Huber deltas **0.15, 0.30, and 1.0**. Global scaling stays fixed at the
reference **+/-1.5 dex** (multipliers from approximately 0.0316 to 31.62).
`range1p5_huber0p15.yaml` is the reference; `range1p5_huber1p00.yaml` tests 1.0.
The larger delta keeps more residuals in the quadratic region and allows larger
loss gradients for large errors. It is measured in normalized-log target units.
Old range-sweep configs are archived in `archived_range_sweep/` and are not run.

The active range setting is
`global_head.max_abs_adjustment`. This is the only global correction branch.
Correction is `local_flux * 10^(range * tanh(global_output))`.
`mean_loss.huber_delta` controls **both** local and corrected Huber objectives
in normalized-log target units. Each mean objective has weight one. The
uncertainty NLL is detached from both mean objectives.

## Preserved model and shared settings

- Local 3x3 attention, 12 layers, embedding 512, hidden width 2048, eight heads,
  seven AIA channels, 8-pixel patches, 4096 patches, dropout 0.1.
- The local head predicts log10 multipliers of a positive equal-share baseline.
  It has no per-patch subtraction or zero cutoff.
- Independence is built into the architecture: the global objective cannot update local
  features, positions, shared raw input features, or base patch flux.
- Patch-specific global attention: pooled 32x32 memory, attention width 128,
  four heads, zero dropout. The older single-summary global branch is disabled.
- Global final layer uses PyTorch's default random weight and bias initialization,
  allowing attention gradients from the first backward pass. The correction
  limit remains fixed at +/-1.5 dex.
- Detached uncertainty enabled from epoch zero, all modules trainable.
- Fresh model weights, optimizer, and scheduler for every arm. Both
  `checkpoint.initialize_from` and `checkpoint.resume_from` are null.
- LR `5e-5`, AdamW decay `1e-5`, cosine `T_max: 8`, minimum LR `1e-7`.
- Batch 48 per GPU, no gradient accumulation; seed 42. Edit batch and
  accumulation settings in all three configs together if needed.
- Eight epochs, no early stopping, best corrected `val/mae` plus last checkpoint.
- Square-root class weights are recomputed from the subset's training targets.
- Low-memory spatial flux/uncertainty callbacks, not full attention-map dumps.

All arms use `/data/FOXES_screening/subset-og-quick`: 14,250 training pairs and
16,317 validation pairs. Its test split is not evaluated by this runner.
Every arm is trained from scratch using only the subset training split.
## Run on two GPUs

From the repository root in the `foxes` environment:

```bash
# Validate without starting training.
python experiments/patch_global_independent_sweep/run_sweep.py --parallel --gpu-ids 0,1

# One active run on each GPU; each GPU advances its own queue.
python experiments/patch_global_independent_sweep/run_sweep.py --parallel --gpu-ids 0,1 --run
```

GPU 0 runs delta 0.15 followed by 1.0; GPU 1 runs delta 0.30.
To run only the reference arm:

```bash
python experiments/patch_global_independent_sweep/run_sweep.py \
  --arm range1p5_huber0p15 --gpu 0 --run
```

Outputs are under `/data/checkpoints/FOXES-PATCH-GLOBAL-INDEPENDENT-SUBSET/independent-clean-e8/`.
Each arm has its own W&B name and console log in timestamped `logs/` folders.
The launcher shares the tested GPU-queue utility with the earlier experiment,
but has independent arm definitions and config validation. It never invokes
the earlier experiment's configuration validator. Existing checkpoint outputs
are not overwritten.

## Compare

Use best corrected `val/mae` as the primary metric. Compare it with
`val/local_mae` to quantify the added value of global correction. Compare each
Huber setting at the same fixed scaling range. Also inspect
`val/local_huber`, `val/global_huber`, per-class MAE in dex, uncertainty
coverage, and `val/patch_global_context_abs_dex` alongside spatial maps. A mean
absolute correction near the allowed maximum suggests heavy use of the range,
but is not a per-patch saturation measurement.

Eight epochs is a quick training screen, not proof of convergence. If a
promising arm is still improving, follow up with a longer matched comparison.
Do not compare total losses across different Huber deltas as though their
scales were identical. Do not select sweep settings on the test split.

To run only Huber delta 1.0:

```bash
python experiments/patch_global_independent_sweep/run_sweep.py \
  --arm range1p5_huber1p00 --gpu 0 --run
```

Initialization applies to fresh models only. Existing checkpoints retain their saved weights.

The model/config cleanup is documented in [INDEPENDENT_MODEL.md](../../training/INDEPENDENT_MODEL.md).
