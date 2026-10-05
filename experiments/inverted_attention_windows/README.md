# Attention mask comparison on the small FOXES-OG subset

These runs use `/data/FOXES_screening/subset-og-quick` with the current OG
uncertainty model: positive patch-mean parameterization, detached Gaussian
uncertainty, weighted Huber plus NLL objective, 12 transformer layers, seed 42,
batch size 32, and learning rate 0.00001. The shared `subset_base.yaml` sets 15
epochs and a matching 15-epoch cosine schedule. The three runs train from
scratch and differ only in their attention masks and run identifiers.

| Config | Mask in every transformer layer | W&B run |
| --- | --- | --- |
| `local_3_control.yaml` | Attends only within nearby 3×3 patches | `OG-Subset-Local-3-LR1e5` |
| `small_inverted_3.yaml` | Blocks nearby 3×3 patches; attends elsewhere | `OG-Subset-Inverted-3-LR1e5` |
| `large_inverted_9.yaml` | Blocks nearby 9×9 patches; attends elsewhere | `OG-Subset-Inverted-9-LR1e5` |

The window dimensions are in patches; each patch is 8×8 image pixels. A larger
*inverted* window blocks more nearby patches. The 3×3 and 9×9 inverted masks
both retain access to most of the 64×64 patch grid. The local control makes the
broad-context comparison valid on the same subset. Earlier local subset runs
used different training settings, so use this control for a matched comparison.
The released OG model also used a different patch-mean rule, which would mix
two architectural changes in the comparison.

The first inverted runs used learning rate 0.001. In epoch 0, every inspected
patch logit fell below the positive head's -8 lower clamp, leaving predictions
at about 3.34×10⁻¹⁴ W/m² with zero mean-head gradient. A 20-step probe at
0.0001 also hit the clamp. A 20-step probe at 0.00001 kept predictions on the
physical flux scale, but does not guarantee stability for all 15 epochs. The
new names and checkpoint directories keep those collapsed runs separate from
these reruns. Start from scratch; do not resume the collapsed checkpoints.

Run one config at a time:

```bash
python training/train.py --config experiments/inverted_attention_windows/local_3_control.yaml
python training/train.py --config experiments/inverted_attention_windows/small_inverted_3.yaml
python training/train.py --config experiments/inverted_attention_windows/large_inverted_9.yaml
```

For visual inspection, open **Spatial uncertainty plots** in each W&B run.
Each validation epoch logs five class-stratified examples with the AIA image,
patch flux contribution map, share of total flux, and patch uncertainty map.
The map panels use independent logarithmic color scales, so use their colorbars
when comparing values across runs. Compare `val/total_loss`, `val/mse`,
`val_class/X/rmse_dex`, and the predicted-versus-true plot alongside the maps.
Checkpoint filenames include the epoch and monitored `val/total_loss`.
