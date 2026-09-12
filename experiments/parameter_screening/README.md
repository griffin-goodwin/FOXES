# FOXES parameter screening

This workflow screens loss and optimizer settings on a reproducible,
time-stratified subset before promoting the winner to full-data training.
Images are symlinked, not copied, and the original dataset is never modified.

## 1. Check the source data

Run the full integrity check first. Add `--move-invalid-to` if invalid pairs
should be quarantined after the reports are written.

```bash
python data/check_dataset_integrity.py \
  --aia-dir /data/AIA_processed \
  --sxr-dir /data/SXR_processed \
  --report /data/dataset_integrity_report.json
```

## 2. Create the fixed screening subset

```bash
python experiments/parameter_screening/create_subset.py \
  --aia-dir /data/AIA_processed \
  --sxr-dir /data/SXR_processed \
  --output-root /data/FOXES_screening/subset-quick \
  --preset quick \
  --seed 42
```

The quick preset is intended for objective comparisons:

| Split | Quiet | C | M | X |
|---|---:|---:|---:|---:|
| train | 10,000 | 3,200 | 1,000 | 50 |
| val | all | all | all | all |
| test | all | all | all | all |

The training counts approximately retain the full dataset's class proportions;
otherwise keeping every X flare while removing most Quiet/C samples would
silently turn the quick screen into an oversampling experiment. Complete
validation is retained so aggregate NLL and coverage remain meaningful.

The default `standard` preset uses:

| Split | Quiet | C | M | X |
|---|---:|---:|---:|---:|
| train | 12,000 | 8,000 | 4,000 | all |
| val | 2,000 | 2,000 | 1,000 | all |
| test | 1,000 | 2,000 | 1,000 | all |

Selections are proportional across calendar months within each class. Exact
timestamps, targets, and source paths are recorded in per-split manifests.
Rerunning with the same seed is idempotent; existing files are never replaced.

## 3. Screen one parameter family at a time

To run one focused spatial-quantile experiment on the quick subset:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage quantile \
  --subset-root /data/FOXES_screening/subset-quick \
  --runs-root /data/FOXES_screening/runs-quantile \
  --sparsity-weight 0.1 \
  --top-fraction 0.02 \
  --scheduler-t-max 10 \
  --max-steps 3000 \
  --gpu-ids 0 \
  --run
```

This launches exactly one run. Its name records sparsity, learning rate, and
step budget—for the command above,
`screen-spatial-quantile-s0.1-top0.02-lr0.0001-steps3000-tmax10`. This prevents
a new setting from overwriting or colliding with an earlier quantile configuration. The model
predicts ordered q02.5/q16/q50/q84/q97.5 values at every patch, uses q50 as the
point prediction, and trains the five global quantiles with class-balanced
pinball loss. Checkpoints are selected by `val_class/macro_pinball`.

The experiment disables full attention-map generation, but logs one
low-memory spatial quantile figure for each `<B`, `B`, `C`, `M`, and `X` class
per epoch when those classes are available. Each figure contains q50 patch
flux and the spatial 68%/95% interval-width maps. This lets you inspect whether
uncertainty is concentrated around the flare region without constructing the
large patch-to-patch attention matrices.

For a first smoke test, `--max-steps 1000` is acceptable. Use 3000 steps for a
more meaningful comparison with the Gaussian screens; judge the run from the
full validation split rather than its short training loss.

The Gaussian model uses summed patch flux for the global mean and summed patch
variance for its global uncertainty. The recommended focused regularization
screen is:

To test whether the MSE-like mean gradient is causing single-patch collapse,
compare it directly with Huber while keeping the Gaussian architecture,
beta=1 variance-head training, data, seed, and optimizer settings fixed:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage mean-loss \
  --dataset-root /data/FOXES_OG \
  --runs-root /data/FOXES_screening/runs-mean-loss \
  --learning-rate 0.00005 \
  --max-steps 3000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

Both runs disable spatial sparsity regardless of `--sparsity-weight`. The
control uses detached beta-NLL with beta=1 for both heads. The Huber arm uses
Huber loss (delta 1.0 in normalized log-flux units) for the mean/backbone and
the same detached beta=1 NLL gradient as the control for the variance head.
Thus the intended experimental difference is only the global-mean gradient.
Both runs log the same class-stratified Gaussian flux maps for comparison.

To compare the fully detached standard NLL control (beta=0) against the
MSE-like detached beta-NLL candidate (beta=1), run:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage beta-detached \
  --subset-root /data/FOXES_screening/subset \
  --runs-root /data/FOXES_screening/runs-beta-detached \
  --sparsity-weight 0.1 \
  --top-fraction 0.05 \
  --learning-rate 0.00005 \
  --max-steps 2500 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This launches the two otherwise identical runs simultaneously, one per GPU.

To compare the size of the local-attention neighbourhood independently of
the detached beta-NLL choice, run the 3-by-2 FOXES_OG subset screen:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage local-window-beta \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-local-window-beta \
  --local-window-sizes 3,5,9 \
  --learning-rate 0.00005 \
  --scheduler-t-max 10 \
  --max-steps 2000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This launches six runs with local windows of 3, 5, and 9 patches and beta 0
or 1. Spatial sparsity is disabled in every arm so it cannot confound the
localization comparison. GPU 0 runs the beta-0 queue and GPU 1 runs the
beta-1 queue. All runs retain class-stratified patch-flux callback figures.

To test finer 4x4 image patches with beta 0 after the local-window screen:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage patch-size-beta0 \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-patch4-beta0 \
  --patch-batch-size 4 \
  --patch-accumulate-grad-batches 16 \
  --learning-rate 0.00005 \
  --scheduler-t-max 6 \
  --max-steps 1200 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This creates two beta-0, no-sparsity runs. Patch-4/window-5 matches the
approximate physical receptive field of patch-8/window-3, while
patch-4/window-3 deliberately tests a narrower receptive field. The patch
count is derived as 16,384 for 512x512 inputs. A microbatch of 4 with 16-way
gradient accumulation preserves an effective batch size of 64.

When `--run` is supplied, the sweep snapshots its base YAML and materializes
every run config before training begins. Later edits to the normal training
config therefore cannot silently change queued experiments.

To screen patch-8 transformer capacity with beta 0 and local window 3:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage architecture-capacity-beta0 \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-architecture-p8-beta0 \
  --learning-rate 0.00005 \
  --scheduler-t-max 10 \
  --max-steps 2000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

The existing 256-embedding/8-block/1024-MLP run is the control. The eleven new
runs change exactly one axis: embedding dimensions 64, 128, or 512;
transformer depths 2, 4, 12, or 16; and MLP hidden widths 256, 512, 2048, or
4096. Patch size 8, local window 3, beta 0, sparsity 0, effective batch size
64, data, seed, and optimizer are held fixed. Two runs execute concurrently,
one per GPU, making this suitable for an overnight screen.

After embedding 512 and depth 16 have been selected independently, run the
joint higher-capacity follow-up with:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage architecture-combos-beta0 \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-architecture-combos-p8-beta0 \
  --learning-rate 0.00005 \
  --scheduler-t-max 10 \
  --max-steps 2000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This tests embed/depth pairs 512/16, 768/16, 1024/16, 512/24, and 768/24.
The MLP stays fixed at 1024, and patch size 8, local window 3, beta 0,
sparsity 0, data, seed, optimizer, and effective batch 64 remain unchanged.
Larger runs use smaller microbatches plus gradient accumulation. A failed run
is reported, but no longer prevents later experiments in its GPU queue from
running.

To compare global attention against local windows 1, 3, and 5 using the
original FOXES patch forecast, run:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage og-patch-attention-beta \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-ogpatch-attention-beta \
  --learning-rate 0.00005 \
  --scheduler-t-max 10 \
  --max-steps 2000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This queues eight `gaussian_nll_og` runs: global attention with beta 0 and 1,
then local windows 1, 3, and 5 with beta 0 and 1. Every run uses embedding 512,
8 transformer layers, patch size 8, no spatial sparsity, the same data and
seed, and the same optimizer-step budget. GPU 0 receives the beta-0 queue and
GPU 1 receives the beta-1 queue, so at most one model runs on each GPU.

To test whether uncertainty training or global gradient clipping is reducing
point-prediction accuracy in the newer multiplier-patch model, run:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage mean-gradient-controls \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-mean-gradient-controls \
  --learning-rate 0.00005 \
  --scheduler-t-max 25 \
  --max-steps 5000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This creates four embedding-512, patch-size-8, beta-0, no-sparsity runs:

- a local-window-3 mean-only MSE control with the base config's clipping;
- local-window-1 Gaussian training with gradient clipping disabled;
- local-window-3 Gaussian training with gradient clipping disabled;
- global-attention Gaussian training with gradient clipping disabled.

The mean-only control keeps the Gaussian forward interface so the usual point
metrics and plots still work, but the uncertainty loss is excluded from its
training graph. Its uncertainty/calibration metrics are therefore diagnostic
noise; compare its `val/mse` and per-class point metrics only. The two Gaussian
runs retain normal detached uncertainty training. This stage defaults to
`/data/FOXES_screening/subset-og-quick`, so the explicit `--subset-root` line
may be omitted. The subset reduces only the training split; validation and
test retain the complete FOXES_OG evaluation splits.

To compare genuinely shallow local receptive fields and a contextual
background-plus-excess mean while retaining uncertainty training, run:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage contextual-local-depth \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-contextual-local-depth \
  --learning-rate 0.00005 \
  --scheduler-t-max 25 \
  --max-steps 5000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This creates five embedding-512, patch-size-8, local-window-3 runs: ordinary
additive Gaussian models with 1, 2, 4, and 8 transformer layers, plus a
two-layer `gaussian_nll_background_excess` model. Every run uses the small
FOXES_OG training subset, direct MSE for the mean, detached beta-0 NLL for
uncertainty, no spatial sparsity, and no gradient clipping. Direct MSE means
the uncertainty estimate cannot numerically reweight the mean gradient.

The contextual model predicts a capped shared background and distributes it
over a fixed solar-disk mask. A one-way global query estimates that scalar but
never feeds context back into patch tokens. The local branch predicts positive
excess, and total patch means still sum exactly to global SXR. Its raw
variance is `background variance + sum(local excess variances)`; the returned
patch variance map is an additive variance attribution whose entries sum to
the global raw variance. Its plotted square root is labeled as a variance
contribution rather than a patch standard deviation. Monitor
`background_fraction` and
`background_cap_fraction` for branch collapse or cap saturation.

To isolate the original inverted-mask geometry in the winning shallow
embedding-512 model, run:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage inverted-mask-window \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-inverted-mask-window \
  --learning-rate 0.00005 \
  --scheduler-t-max 25 \
  --max-steps 5000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This creates four otherwise matched standard Gaussian runs with inverted
attention masks: 3x3, 5x5, 9x9, and 17x17 blocked neighborhoods. The final
arm represents the requested size 16 with the nearest centered symmetric
window; an exactly centered discrete square cannot have even width. Its run
name explicitly records `requestedw16-effectivew17`. Every arm uses the small
FOXES_OG training subset, embedding 512, eight transformer layers, patch size 8,
direct MSE for the mean, detached beta-0 NLL uncertainty, no sparsity, and no
gradient clipping. Checkpoints are selected using `val_class/X/rmse_dex`;
compare their X-class 68%/95% coverage and NLL as secondary diagnostics.

To replace the unsuccessful inverted mask with a class-balanced local Huber
objective, run the single matched experiment:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage local-weighted-huber \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-local-weighted-huber \
  --learning-rate 0.00005 \
  --scheduler-t-max 25 \
  --max-steps 5000 \
  --gpu-ids 0 \
  --run
```

This uses the standard additive Gaussian model with embedding 512, eight
transformer layers, patch size 8, and true local 3x3 attention. The mean path
uses Huber loss with delta 0.3. At startup, the training SXR targets are
counted in quiet/C/M/X bins and assigned weights `N / (4 * N_class)`, giving
each class equal expected Huber influence while keeping the average weight at
one. The detached beta-0 Gaussian NLL trains the uncertainty path only.
Sparsity and gradient clipping are disabled, and checkpoints are selected by
`val_class/X/rmse_dex`.

Once that delta-0.3, eight-layer inverse-frequency reference is running, use
the same immutable base snapshot for the focused follow-up:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage weighted-huber-followup \
  --base-config /data/FOXES_screening/runs-local-weighted-huber/sweep_base_config_0112b324e938.yaml \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-weighted-huber-followup \
  --learning-rate 0.00005 \
  --scheduler-t-max 25 \
  --max-steps 5000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This adds nine focused comparisons around the existing reference. At eight
layers, deltas 0.15 and 1.0 (the standard Huber default) each run with full
inverse-frequency and square-root inverse-frequency weights; delta 0.3 gets
the square-root counterpart to the existing full-inverse reference. At delta
0.3, two- and four-layer models also run under both weighting schemes.
Two additional eight-layer, delta-0.3, full-inverse runs compare local
attention widths 5 and 9 against the existing width-3 reference.
Square-root weights are renormalized to an expected sample weight of one. This
changes rare-class pressure without also changing the Huber-to-NLL loss scale.
Run ordering balances the two sequential GPU queues by total transformer depth.

To screen the requested attention/depth/feed-forward capacity combinations on
the small FOXES_OG subset, run:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage attention-capacity-weighted-huber \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-attention-capacity-weighted-huber \
  --learning-rate 0.00005 \
  --scheduler-t-max 25 \
  --max-steps 5000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

The four arms are `(layers, heads, MLP hidden) = (8, 8, 2048)`,
`(12, 8, 2048)`, `(12, 16, 2048)`, and `(12, 8, 4096)`. Every arm keeps
embedding 512, **patch size 8**, true local 3x3 attention, delta-0.15
inverse-frequency weighted Huber, detached Gaussian uncertainty, no sparsity,
and no gradient clipping. All use microbatch 8 with eight accumulated batches,
preserving an effective batch size of 64 while accommodating the largest arm.
Checkpoints are selected by `val_class/macro_mse_dex2` rather than X-only RMSE.
At embedding 512, eight heads give 64 dimensions per head, while 16 heads give
32 dimensions per head; the 4096 setting changes the transformer feed-forward
network rather than attention-head width.

The ordinary depths 1/2/4/8 and the original background-plus-excess depth-2
run are already complete in `runs-contextual-local-depth`. Complete the matched
background-plus-excess depth matrix with the missing depths 4, 8, and 1:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage background-excess-depth-completion \
  --base-config /data/FOXES_screening/runs-contextual-local-depth/sweep_base_config_0112b324e938.yaml \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-contextual-local-depth \
  --learning-rate 0.00005 \
  --scheduler-t-max 25 \
  --max-steps 5000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

The execution order is depth 4, 8, 1 so two GPU queues are approximately
balanced; together with the completed depth-2 run, results should be reported
in scientific order 1/2/4/8. All four use the same immutable base-config
snapshot and otherwise identical controls. For a fresh run directory, use
`--stage background-excess-depth` instead; that self-contained stage emits all
four depths 1/2/4/8.

To compare detached beta-NLL and localization strength on the complete
Hugging Face FOXES_OG dataset, run the dedicated beta-by-sparsity matrix:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage og-beta-sparsity \
  --dataset-root /data/FOXES_OG \
  --runs-root /data/FOXES_screening/runs-og-beta-sparsity \
  --sparsity-weights 0.1,0.3,1.0 \
  --top-fraction 0.05 \
  --learning-rate 0.00005 \
  --max-steps 3000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This creates six runs: beta 0 and beta 1 at each of the three sparsity
weights. Only two models run concurrently; GPU 0 processes the beta-0 queue
and GPU 1 processes the beta-1 queue. The stage uses
`/data/FOXES_OG/SXR_processed/normalized_sxr.npy`, disables extra AIA
normalization because FOXES_OG images are already scaled to `[-1, 1]`, and
selects checkpoints using `val/mse`. Early stopping is disabled so all six
runs receive the same optimizer-step budget. To launch only four runs, pass
`--sparsity-weights 0.1,0.3`.

For the faster FOXES_OG screen, first create a 14,250-sample training subset
that retains the original training class proportions. Validation and test are
kept complete, and the source SXR normalization artifact is linked into the
subset:

```bash
python experiments/parameter_screening/create_subset.py \
  --aia-dir /data/FOXES_OG/AIA_processed \
  --sxr-dir /data/FOXES_OG/SXR_processed \
  --output-root /data/FOXES_screening/subset-og-quick \
  --preset og-quick \
  --seed 42
```

Then run the same six comparisons on that subset:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage og-beta-sparsity \
  --subset-root /data/FOXES_screening/subset-og-quick \
  --runs-root /data/FOXES_screening/runs-og-beta-sparsity-subset \
  --sparsity-weights 0.1,0.3,1.0 \
  --top-fraction 0.05 \
  --learning-rate 0.00005 \
  --max-steps 3000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

The `og-beta-sparsity` stage now defaults to this subset when neither
`--subset-root` nor `--dataset-root` is provided. Pass
`--dataset-root /data/FOXES_OG` explicitly to use all 77,809 training samples.
Subset run names include `screen-og-subset`; full-data run names include
`screen-og-full`, so their configs and W&B runs cannot be confused.
Each Gaussian screening epoch also logs one low-memory patch-flux, fractional
flux, and spatial-uncertainty figure for every available `<B`, `B`, `C`, `M`,
and `X` class. The background-plus-excess run labels its map as square-root
variance contribution; other Gaussian runs show patch standard deviation.
These figures do not construct attention matrices.

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage patch-gaussian \
  --subset-root /data/FOXES_screening/subset-quick \
  --runs-root /data/FOXES_screening/runs-patch-gaussian \
  --sparsity-weight 0.1 \
  --top-fraction 0.05 \
  --learning-rate 0.00005 \
  --max-steps 3000 \
  --parallel 2 \
  --gpu-ids 0,1 \
  --run
```

This compares exactly three configurations: the current baseline (`dropout`
0.1, weight decay `1e-4`), stronger weight decay (`1e-3`), and higher dropout
(`0.2`). All three use detached beta-NLL with beta=1, the same isolated
patch-variance parameterization, and the same gated patch-flux sparsity
(`0.1`, top 5%), data, seed, learning rate, and step budget. `--parallel 2` keeps
at most one run on each GPU; the third starts when one finishes.

Finally carry both winners into the learning-rate screen:

```bash
python experiments/parameter_screening/run_sweep.py \
  --stage learning-rate \
  --subset-root /data/FOXES_screening/subset-quick \
  --max-steps 1000 \
  --run
```

The primary tested values are:

- Patch-model regularization: baseline, weight decay `1e-3`, dropout `0.2`
- Learning rate: `3e-5`, `1e-4`, `3e-4`

Every run uses the same subset, seed, and optimizer-step budget. Attention-map
generation is disabled during screening to reduce memory and runtime.

## 4. Select using full per-class validation metrics

Each validation epoch logs these W&B metrics for Below-B, B, C, M, and X:

- `mae_dex`
- `rmse_dex`
- `bias_dex`
- `mean_sigma_dex`
- `coverage_95`
- `class_accuracy`
- `nll`
- `val_class/macro_mae_dex`
- `val_class/macro_nll`
- `val/nll` (the directly optimized calibration objective)

For the quantile experiment, use the analogous metrics:

- `val_class/<class>/pinball`
- `val_class/<class>/mae_dex`
- `val_class/<class>/coverage_68`
- `val_class/<class>/coverage_95`
- `val_class/<class>/mean_width_68_dex`
- `val_class/<class>/mean_width_95_dex`
- `val_class/macro_pinball`
- `val_class/macro_mse_dex2`
- `val_class/macro_calibration_error`

Prefer low M/X MAE and bias, coverage close to 0.95, and narrower uncertainty
at comparable coverage. Do not choose solely from aggregate validation loss.
The test subset is not used during screening.

## 5. Promote the winner to full-data training

```bash
python experiments/parameter_screening/promote_to_full.py \
  --name full-screen-winner \
  --learning-rate 0.0001 \
  --epochs 50
```

This generates an isolated full-data config. Inspect it, then rerun with
`--run` or execute the printed training command. Full-data paths and
normalization artifacts come from `training/train_config.yaml`.
