Patch-specific global attention experiment
==========================================

Run `finetune_sqrt_e10_patch_global_queries.yaml` from the repository root in
the `foxes` environment. The run loads the square-root-weighted epoch-10
checkpoint (validation normalized-log MSE 0.0151221), starts a fresh optimizer,
and trains five new epochs. Checkpoints are kept in a separate directory.
Microbatch 16 with three accumulated batches preserves the source run's
effective batch of 48 and leaves GPU memory for the extra attention branch.

The source checkpoint's old global branch has one learned query over all
original AIA patches; its context is shared by all patch corrections. This
experiment skips that branch entirely. Each locally encoded patch instead
queries a 16x16 memory grid made from raw AIA patch features and local
transformer features. The attention cost grows with 4096 queries times 256
memory tokens, rather than 4096 squared. The new correction is bounded to
+/-1.5 dex and starts at zero, so initial predictions equal the source
checkpoint's *local patch mean before its old global correction*, not its
final predictions. The old global weights in the checkpoint are ignored.

Only the new global branch trains at an initial learning rate of 5e-5. The
pretrained backbone, local mean head, and uncertainty head stay fixed, with
dropout disabled in those frozen modules. The mean objective uses square-root
class weighting and Huber delta 0.15. Validation MSE and MAE decide whether
the experiment helps; `patch_global_context_abs_dex` shows whether the new
branch moves away from zero. The saved uncertainty intervals may need
recalibration after mean predictions change.

For an interrupted run, clear `checkpoint.initialize_from` and resume the
experiment's own `last.ckpt` with `--ckpt-path`. Keep `epochs: 5` to complete
five total fine-tuning epochs, including epochs already finished.

Independent local and global training
-------------------------------------

`finetune_patch_global_independent.yaml` inherits the current full-unfreeze
configuration, including its initialization checkpoint, optimizer, class
weighting, and attention grid. It writes to a separate checkpoint directory:

```bash
python training/train.py --config training/configs/finetune_patch_global_independent.yaml
```

Set `checkpoint.initialize_from` to the checkpoint you want to fine-tune.
Use weights-only initialization with a fresh optimizer when switching from
the joint objective; use full resume for an interrupted independent run.
To train from scratch instead, set `checkpoint.initialize_from: null`.

The option `vit_architecture.independent_local_global: true` works with both
the original single-query correction and the patch-specific global context.
It requires at least one global correction branch. Existing configurations
default to joint training when the option is absent or false.

The mean objective is exactly `local_mean_loss + global_mean_loss`, with
coefficient **1** on each term. Both use the same GOES target, normalized
log-SXR space, Huber delta, and training-derived class weights. The local loss
trains the uncorrected patch flux sum. The global loss trains the corrected
sum, with the base flux, local features, and any shared projection/position
features detached, so it cannot update local parameters. Both objectives use
one local backbone pass and one optimizer step. If a contrast prior is
enabled, it applies to the uncorrected local map in this mode.

Active uncertainty NLL is added separately as before and only updates the
uncertainty head. Thus the full objective is `local_mean_loss +
global_mean_loss + uncertainty_nll` (plus any configured contrast prior).
The global branch still depends on local evidence in its forward pass;
independence here means no global-loss gradient reaches the local model.

Compare `train/local_mean_loss`, `train/global_mean_loss`, and their validation
counterparts. `val/local_mse` and `val/local_mae` measure the uncorrected
prediction; existing `val/mse` and `val/mae` measure the final corrected
prediction. `mean_loss` logs the sum of both Huber terms. The new mean/total
loss scale differs from a joint run, so compare final prediction metrics and
select checkpoints by `val/mse`, as this configuration does.
