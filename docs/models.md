# Model background

[Back to the README](../README.md) · [Checkpoint files and inference](workflows.md#2-run-inference)

## Original FOXES abstract
The solar soft X-ray (SXR) irradiance is a long-standing proxy of solar activity, used for the classification of flare strength. As a result, the flare class, along with the SXR light curve, are routinely used as the primary input to many forecasting methods, from coronal mass ejection speed to energetic particle output. However, the SXR irradiance lacks spatial information leading to dubious classification during periods of high activity, and is  applicable only for observations from Earth orbit, hindering forecasting for other places in the heliosphere. This work introduces the Framework for Operational X-ray Emission Synthesis (FOXES), a Vision Transformer-based approach for translating Extreme Ultraviolet (EUV) spatially-resolved observations into SXR irradiance predictions. The model produces two outputs: (1) a global SXR flux prediction and (2) per-patch flux contributions, which offer a spatially resolved interpretation of where the model attributes SXR emission. This paves the way for EUV-based flare detection to be extended beyond Earth's line of sight, allowing for a more comprehensive and reliable flare catalog to support robust, scalable, and real-time forecasting, extending our monitoring into a true multiviewpoint system.

**Team**: Griffin Goodwin, Alison March, Jayant Biradar, Christopher Schirninger, Robert Jarolim, Angelos Vourlidas, Viacheslav Sadykov, Lorien Pratt

---

## The Original FOXES Model

The original model predicts a raw SXR flux contribution for each image patch.
Its three trained checkpoints use **global**, **local**, or **inverted**
attention. The model converts patch logits to flux and clamps very small
contributions to a positive floor (`1e-15` W/m²). Maps with many contributions
near that floor can appear sparse. In the evaluations run so far, global RMSE and MSE were
good, but the spatial maps for weak flares and quiet-Sun images were less
convincing. This motivated the updated model below.

## The Updated (Post Publication) FOXES Model: Local + Global Context

The maintained model in this repository extends the released FOXES patch
predictor. Local attention builds a feature for each AIA image patch. Patch
flux starts from a positive, image-wide baseline divided across the patches;
each patch's output scales its share. A global query summarizes projected
patches from the original AIA image. Each patch's correction combines that
shared context, its projected image patch, and its local Transformer features
to adjust its flux contribution. The corrected patch fluxes sum to the
full-disk SXR prediction.

This local + global design produces **less sparse, visually better spatial
flux maps** in the examples examined so far. It also predicts a global
Gaussian uncertainty and a map of each patch's contribution to that variance.
The uncertainty map is an attribution of global uncertainty; there are no patch-level SXR labels
to establish calibrated local error bars.

The tradeoff is **somewhat worse global SXR prediction performance** than the
original released FOXES model in the evaluations run so far. The visual map
improvement is qualitative.
The newer local + global checkpoints listed in the [workflow guide](workflows.md#2-run-inference) have **not yet been
validated**; these earlier observations should not be read as evaluation
results for either new checkpoint file.
