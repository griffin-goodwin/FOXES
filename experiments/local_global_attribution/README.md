# Local patch map with global AIA context

`per_patch_global_aia.yaml` uses the same small-subset data, positive patch
mean, uncertainty objective, batch size 32, learning rate 0.00001, and 15-epoch
schedule as the matched attention-mask runs. All 12 existing transformer blocks
use local 3×3 attention.

The local transformer first predicts positive flux for each patch. A separate
global summary query reads **all original seven-channel AIA patches** before
the local transformer. Each patch's local feature, original AIA token, and the
shared global summary then produce a bounded correction to that patch's flux.
The correction is initialized to zero and limited to ±0.3 dex (roughly a
factor of two). This lets global context redistribute some flux while the
local patch map remains the main source of spatial evidence. The corrected
patch fluxes still sum exactly to the global prediction and also scale the
patch uncertainty contributions.

Run from scratch:

```bash
python training/train.py --config experiments/local_global_attribution/per_patch_global_aia.yaml
```

Compare with `experiments/inverted_attention_windows/local_3_control.yaml`.
Open **Global patch adjustment plots** in W&B to see the local and corrected
flux maps on the same color scale beside the AIA image and the correction map.
The run also logs `val/global_patch_adjustment_dex`; a value near zero means
the global branch has barely changed the local map. Because GOES supplies only
an image-level flux target, map plausibility still needs inspection against
the AIA image or an independent spatial reference.

For a controlled initialization comparison, run
`per_patch_global_aia_random_init.yaml`. It keeps the same subset, model,
learning rate, and ±0.3 dex correction limit, but initializes the correction
head's final weights from a normal distribution with standard deviation 0.01
(bias zero). This sends gradients into the global branch from the first step
and initially gives small spatially varying corrections. It has a separate
checkpoint directory and W&B run name:

```bash
python training/train.py --config experiments/local_global_attribution/per_patch_global_aia_random_init.yaml
```

Compare `val/global_patch_adjustment_dex`, validation loss, and the actual
patch maps with the zero-initialized run. The scalar GOES target alone does
not require the correction branch to redistribute flux toward active regions.
