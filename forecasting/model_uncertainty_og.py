"""Gaussian uncertainty model with the original FOXES patch forecast.

The uncertainty branch is inherited from :mod:`forecasting.model_uncertainty`:
patch uncertainty features and the raw-flux scale used by that branch remain
detached from the mean model.  Only the patch-mean parameterization changes.

Unlike the newer log10-multiplier model, each patch output here is interpreted
as a normalized log-SXR prediction and passed through the original FOXES
inverse transform before patch fluxes are summed.  Subtracting the SXR offset
and clamping at zero restores the original model's exact zero-flux region.
"""

import torch

from forecasting.model import SXR_LOG_OFFSET
from forecasting.model_uncertainty import (
    GaussianNLLViTLocal,
    GaussianVisionTransformerLocal,
)


class OGGaussianVisionTransformerLocal(GaussianVisionTransformerLocal):
    """Patch Gaussian model using the released FOXES mean conversion."""

    MEAN_PARAMETERIZATION_VERSION = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The multiplier model deliberately starts from a uniform zero-logit
        # map. The original FOXES head used PyTorch's normal Linear-layer
        # initialization, so restore that behavior for a faithful ablation.
        self.mlp_head[-1].reset_parameters()

    def _patch_flux_from_logits(self, patch_logits, sxr_norm):
        """Invert normalized per-patch log-SXR exactly as original FOXES did."""
        sxr_mean, sxr_std = sxr_norm
        patch_log10_sxr = (
            patch_logits.float() * sxr_std.float() + sxr_mean.float()
        )
        patch_flux_raw = torch.pow(
            torch.tensor(
                10.0,
                device=patch_logits.device,
                dtype=torch.float32,
            ),
            patch_log10_sxr,
        ) - SXR_LOG_OFFSET
        return torch.clamp(patch_flux_raw, min=0.0, max=1.0)


class OGGaussianNLLViTLocal(GaussianNLLViTLocal):
    """Detached patch uncertainty around the original FOXES mean forecast."""

    NETWORK_CLASS = OGGaussianVisionTransformerLocal
