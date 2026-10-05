# Independent local/global subset sweep

This is the focused follow-up to the architecture comparison. It preserves the
setup of the saved `OG-PatchGlobal-Independent` run and changes only the global
correction range and Huber delta between arms. Every YAML is standalone.
The old architecture-comparison configs and runs are left intact.

## Six arms, eight epochs each

| Global limit (dex) | Allowed patch multiplier | Huber deltas |
|---|---|---|
| +/-0.3 | 0.501 to 1.995 | 0.15, 0.30 |
| +/-1.5 | 0.0316 to 31.62 | 0.15, 0.30 |
| +/-3.0 | 0.001 to 1000 | 0.15, 0.30 |

`range1p5_huber0p15.yaml` is the reference hyperparameter pair from the successful
run. The large range is deliberately permissive; it can amplify quiet patches
substantially, so compare the corrected maps as well as prediction error.

The active range setting is
`vit_architecture.patch_global_context.max_abs_adjustment`. The older
`max_abs_log10_global_adjustment` applies to a disabled branch and is not swept.
Correction is `local_flux * 10^(range * tanh(global_output))`.
`uncertainty.huber_delta` controls **both** local and corrected Huber objectives
in normalized-log target units. Each mean objective has weight one. The
uncertainty NLL is detached from both mean objectives.

## Preserved model and shared settings

- Local 3x3 attention, 12 layers, embedding 512, hidden width 2048, eight heads,
  seven AIA channels, 8-pixel patches, 4096 patches, dropout 0.1.
- `mean_parameterization: positive`, as in the successful run. This is a positive
  equal-share-baseline head, not `original_unclamped` or the legacy zero cutoff.
- `independent_local_global: true`: the global objective cannot update local
  features, positions, shared raw input features, or base patch flux.
- Patch-specific global attention: pooled 32x32 memory, attention width 128,
  four heads, zero dropout. The older single-summary global branch is disabled.
- Detached uncertainty enabled from epoch zero, all modules trainable.
- Same initializer as the successful run:
  `/data/checkpoints/FOXES-OG-SQRT-E10-PATCH-GLOBAL-REPLACEMENT/last.ckpt`.
  This loads model weights with a fresh optimizer/scheduler, not a full resume.
- LR `5e-5`, AdamW decay `1e-5`, cosine `T_max: 8`, minimum LR `1e-7`.
- Microbatch 8, accumulation 6, nominal effective batch 48; seed 42.
  These smaller batches address the observed GPU memory failures. Full GPU
  memory use has not been measured for this sweep. Edit batch/accumulation in
  every config together if needed.
- Eight epochs, no early stopping, best corrected `val/mae` plus last checkpoint.
- Square-root class weights are recomputed from the subset's training targets.
- Low-memory spatial flux/uncertainty callbacks, not full attention-map dumps.

All arms use `/data/FOXES_screening/subset-og-quick`: 14,250 training pairs and
16,317 validation pairs. Its test split is not evaluated by this runner.
The initializer was trained before this subset sweep; these are fine-tuning
experiments, not models trained exclusively on the subset from scratch.
Changing the correction range also changes initial corrected predictions when
loading nonzero global-head weights. All arms share weights, not necessarily
identical initial predictions. Range changes also change correction gradients
through the tanh multiplier; this sweep measures their combined effect.

## Run on two GPUs

From the repository root in the `foxes` environment:

```bash
# Validate without starting training.
python experiments/patch_global_independent_sweep/run_sweep.py --parallel --gpu-ids 0,1

# One active run on each GPU; each GPU advances its own queue.
python experiments/patch_global_independent_sweep/run_sweep.py --parallel --gpu-ids 0,1 --run
```

The GPUs run paired Huber settings at each range, progressing from 0.3 to 1.5
and then 3.0. To run only the reference arm:

```bash
python experiments/patch_global_independent_sweep/run_sweep.py \
  --arm range1p5_huber0p15 --gpu 0 --run
```

Outputs are under `/data/checkpoints/FOXES-PATCH-GLOBAL-INDEPENDENT-SUBSET/e8/`.
Each arm has its own W&B name and console log in timestamped `logs/` folders.
The launcher shares the tested GPU-queue utility with the earlier experiment,
but has independent arm definitions and config validation. It never invokes
the earlier experiment's configuration validator. Existing checkpoint outputs
are not overwritten.

## Compare

Use best corrected `val/mae` as the primary metric. Compare it with
`val/local_mae` to quantify the added value of global correction. Compare each
Huber setting at fixed range and each range at fixed Huber setting. Also inspect
`val/local_mean_loss`, `val/global_mean_loss`, per-class MAE in dex, uncertainty
coverage, and `val/patch_global_context_abs_dex` alongside spatial maps. A mean
absolute correction near the allowed maximum suggests heavy use of the range,
but is not a per-patch saturation measurement.

Eight epochs is a quick fine-tuning screen, not proof of convergence. If a
promising arm is still improving, follow up with a longer matched comparison.
Do not compare total losses across different Huber deltas as though their
scales were identical. Do not select sweep settings on the test split.
