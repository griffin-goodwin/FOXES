Fine-tune the 2/3 model for overall prediction error
=================================================

From the repository root, in the `foxes` environment:

```bash
python training/train.py --config training/configs/finetune_power23_epoch14_unweighted.yaml
```

This initializes all model weights from the 2/3 run's epoch-14 checkpoint,
then trains five new epochs (0–4) with unweighted Huber, delta 0.15. Architecture,
data, normalization, batch size and weight decay match the source configuration.
The detached uncertainty branch stays enabled. Checkpoint selection uses mean
prediction error, not uncertainty NLL.

AdamW starts fresh at 1e-5, with cosine T_max=5 and eta_min=1e-6. The five epoch
learning rates are approximately 1e-5, 9.14e-6, 6.89e-6, 4.11e-6 and 1.86e-6.
The schedule reaches 1e-6 after the final epoch.

`checkpoint.initialize_from` restores only model weights. Do not supply
`--ckpt-path`: that option resumes optimizer, scheduler and epoch state and is
intentionally incompatible with initialization. To recover an interrupted
fine-tuning run, copy this config, clear `initialize_from`, and set
`resume_from` to the new run's `last.ckpt`. Keep `epochs: 5` for five total
fine-tuning epochs, including those already completed.

Every epoch and `last.ckpt` are saved in the separate directory
`/data/checkpoints/FOXES-OG-POWER23-E14-FINETUNE-UNWEIGHTED-LR1E5/`.
Keep the source checkpoint as the baseline. Compare `val/mse` and `val/mae`
along with the per-class metrics before selecting an epoch for test inference.
Source epoch 14 has validation normalized-log MSE 0.0162225 and MAE 0.0766417.
Training validation losses are in
normalized log10(flux + offset) space; the inference evaluator's dex metrics
use a different scale and should not be compared directly to `val/mse`.

This is the first unweighted-Huber experiment only. A larger Huber delta or MSE
objective should be a separate comparison from the same source checkpoint.
