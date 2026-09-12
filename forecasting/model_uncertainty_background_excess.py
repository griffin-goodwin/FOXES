"""Gaussian FOXES model with local excess plus shared background flux.

The mean remains an exact additive patch forecast. A shallow local transformer
predicts nonnegative excess flux for each patch, while a one-way global query
predicts one bounded background scalar. The background is distributed with a
fixed solar-disk area mask and is never fed back into the local patch tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from forecasting.model import SXR_LOG_OFFSET
from forecasting.model_uncertainty import (
    GaussianNLLViTLocal,
    GaussianVisionTransformerLocal,
)


class BackgroundExcessGaussianVisionTransformerLocal(
    GaussianVisionTransformerLocal
):
    """Additive ``background + local excess`` Gaussian patch model."""

    MEAN_PARAMETERIZATION_VERSION = 4

    def __init__(
        self,
        *args,
        background_flux_max=5e-6,
        background_initial_cap_fraction=0.5,
        excess_initial_fraction=0.2,
        solar_disk_radius_fraction=0.48,
        background_scale_floor_fraction=0.05,
        background_uncertainty_initial_raw=-3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        embed_dim = self.input_layer.out_features
        num_heads = self.transformer_blocks[0].attn.num_heads

        self.background_flux_max = float(background_flux_max)
        self.background_initial_cap_fraction = float(
            background_initial_cap_fraction
        )
        self.excess_initial_fraction = float(excess_initial_fraction)
        self.solar_disk_radius_fraction = float(
            solar_disk_radius_fraction
        )
        self.background_scale_floor_fraction = float(
            background_scale_floor_fraction
        )
        self.background_uncertainty_initial_raw = float(
            background_uncertainty_initial_raw
        )
        if self.background_flux_max <= 0:
            raise ValueError("background_flux_max must be positive")
        if not 0 < self.background_initial_cap_fraction < 1:
            raise ValueError(
                "background_initial_cap_fraction must be in (0, 1)"
            )
        if not 0 < self.excess_initial_fraction <= 1:
            raise ValueError("excess_initial_fraction must be in (0, 1]")
        if not 0 < self.solar_disk_radius_fraction <= 0.5:
            raise ValueError(
                "solar_disk_radius_fraction must be in (0, 0.5]"
            )
        if self.background_scale_floor_fraction <= 0:
            raise ValueError(
                "background_scale_floor_fraction must be positive"
            )

        # Begin with a small local-excess share instead of adding a full second
        # typical-flux prediction on top of the new background component.
        with torch.no_grad():
            self.mlp_head[-1].bias.fill_(
                math.log10(self.excess_initial_fraction)
            )

        self.background_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.background_query, std=0.02)
        self.background_source_norm = nn.LayerNorm(embed_dim)
        self.background_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.background_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )
        nn.init.zeros_(self.background_head[-1].weight)
        nn.init.constant_(
            self.background_head[-1].bias,
            math.log(
                self.background_initial_cap_fraction
                / (1.0 - self.background_initial_cap_fraction)
            ),
        )

        self.background_uncertainty_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )
        nn.init.zeros_(self.background_uncertainty_head[-1].weight)
        nn.init.constant_(
            self.background_uncertainty_head[-1].bias,
            self.background_uncertainty_initial_raw,
        )

        rows = (torch.arange(self.grid_h, dtype=torch.float32) + 0.5) / self.grid_h
        cols = (torch.arange(self.grid_w, dtype=torch.float32) + 0.5) / self.grid_w
        yy, xx = torch.meshgrid(rows - 0.5, cols - 0.5, indexing='ij')
        disk_mask = (
            torch.sqrt(xx.square() + yy.square())
            <= self.solar_disk_radius_fraction
        ).reshape(1, -1).float()
        if not torch.any(disk_mask):
            raise ValueError("solar disk mask contains no patch centers")
        self.register_buffer('solar_disk_patch_mask', disk_mask)
        self.register_buffer(
            'background_patch_weights', disk_mask / disk_mask.sum()
        )

        self.last_background_flux_raw = None
        self.last_background_patch_flux_raw = None
        self.last_excess_patch_flux_raw = None
        self.last_background_cap_fraction = None
        self.last_background_context = None
        self.last_background_variance_raw = None
        self.last_excess_variance_contribution_raw = None
        self.last_background_variance_attribution_raw = None
        self._latest_excess_patch_flux_for_loss = None

    def _background_from_embeddings(self, patch_embeddings):
        query = self.background_query.expand(patch_embeddings.shape[0], -1, -1)
        source = self.background_source_norm(patch_embeddings)
        context, _ = self.background_attention(
            query, source, source, need_weights=False
        )
        context = context.squeeze(1)
        cap_fraction = torch.sigmoid(
            self.background_head(context).squeeze(-1).float()
        )
        background_flux = cap_fraction * self.background_flux_max
        return background_flux, cap_fraction, context

    def _patch_mean_from_embeddings(self, patch_embeddings, sxr_norm):
        excess_patch_flux = super()._patch_mean_from_embeddings(
            patch_embeddings, sxr_norm
        )
        background_flux, cap_fraction, context = (
            self._background_from_embeddings(patch_embeddings)
        )
        background_patch_flux = (
            background_flux.unsqueeze(-1) * self.background_patch_weights
        )

        # Store graph-free diagnostics and scales for the detached uncertainty
        # path. They are overwritten on every forward pass.
        self.last_background_flux_raw = background_flux.detach()
        self.last_background_patch_flux_raw = background_patch_flux.detach()
        self.last_excess_patch_flux_raw = excess_patch_flux.detach()
        self.last_background_cap_fraction = cap_fraction.detach()
        self.last_background_context = context.detach()
        # Kept only until the wrapper computes its same-forward sparsity term.
        # Unlike the fixed background, local excess may legitimately occur
        # beyond the optical limb, so it is not multiplied by the disk mask.
        self._latest_excess_patch_flux_for_loss = excess_patch_flux
        return background_patch_flux + excess_patch_flux

    def _patch_variance_from_embeddings(
        self, patch_embeddings, patch_flux_raw, sxr_norm,
    ):
        if self.last_excess_patch_flux_raw is None:
            raise RuntimeError("patch mean must be evaluated before variance")

        raw_excess_uncertainty = self.patch_uncertainty_head(
            patch_embeddings.detach()
        ).squeeze(-1)
        relative_excess_std = torch.clamp(
            F.softplus(raw_excess_uncertainty.float())
            + self.relative_std_floor,
            max=self.relative_std_max,
        )
        typical_global_flux = torch.clamp(
            10 ** sxr_norm[0].float() - SXR_LOG_OFFSET,
            min=torch.finfo(torch.float32).tiny,
        )
        excess_floor = (
            typical_global_flux / patch_flux_raw.shape[1]
            * self.patch_scale_floor_fraction
        )
        excess_scale = self.last_excess_patch_flux_raw + excess_floor
        excess_variance = (relative_excess_std * excess_scale).square()

        raw_background_uncertainty = self.background_uncertainty_head(
            self.last_background_context
        ).squeeze(-1)
        relative_background_std = torch.clamp(
            F.softplus(raw_background_uncertainty.float())
            + self.relative_std_floor,
            max=self.relative_std_max,
        )
        background_scale = (
            self.last_background_flux_raw
            + typical_global_flux * self.background_scale_floor_fraction
        )
        background_variance = (
            relative_background_std * background_scale
        ).square()

        # This is an additive variance attribution. The shared background term
        # is added once globally and allocated with weights summing to one; it
        # is not a claim that patch background errors are independent.
        background_variance_contribution = (
            background_variance.unsqueeze(-1) * self.background_patch_weights
        )
        self.last_background_variance_raw = background_variance.detach()
        self.last_excess_variance_contribution_raw = excess_variance.detach()
        self.last_background_variance_attribution_raw = (
            background_variance_contribution.detach()
        )
        return excess_variance + background_variance_contribution

    def forward(
        self, x, sxr_norm, return_attention=False, return_components=False,
    ):
        """Optionally append explicit background/excess diagnostics."""
        outputs = super().forward(
            x, sxr_norm, return_attention=return_attention
        )
        if not return_components:
            return outputs
        return (*outputs, self.latest_components())

    def consume_excess_patch_flux_for_loss(self):
        """Return the differentiable excess map from the latest forward."""
        excess_patch_flux = self._latest_excess_patch_flux_for_loss
        if excess_patch_flux is None:
            raise RuntimeError(
                "patch mean must be evaluated before the sparsity loss"
            )
        self._latest_excess_patch_flux_for_loss = None
        return excess_patch_flux

    def latest_mean_components(self):
        """Return graph-free component diagnostics from the latest forward."""
        if self.last_background_flux_raw is None:
            raise RuntimeError("no forward pass has been evaluated")
        return {
            'background_flux_raw': self.last_background_flux_raw,
            'background_patch_flux_raw': self.last_background_patch_flux_raw,
            'excess_patch_flux_raw': self.last_excess_patch_flux_raw,
            'background_cap_fraction': self.last_background_cap_fraction,
        }

    def latest_components(self):
        """Return batch-first, graph-free mean and variance components.

        The background allocation is fixed bookkeeping, not a learned spatial
        background prediction. Likewise, total variance is an attribution map
        whose sum equals global raw variance, not marginal patch variance.
        """
        mean = self.latest_mean_components()
        if self.last_background_variance_raw is None:
            raise RuntimeError("no uncertainty forward pass has been evaluated")
        excess_variance = self.last_excess_variance_contribution_raw
        background_attribution = (
            self.last_background_variance_attribution_raw
        )
        total_accounting = (
            mean['background_patch_flux_raw']
            + mean['excess_patch_flux_raw']
        )
        total_variance_attribution = (
            background_attribution + excess_variance
        )
        return {
            'background_flux_raw': mean['background_flux_raw'],
            'background_cap_fraction': mean['background_cap_fraction'],
            'excess_patch_flux_raw': mean['excess_patch_flux_raw'],
            'background_allocation_raw': mean[
                'background_patch_flux_raw'
            ],
            'total_accounting_patch_flux_raw': total_accounting,
            'background_variance_raw': self.last_background_variance_raw,
            'excess_variance_contribution_raw': excess_variance,
            'total_variance_attribution_raw': total_variance_attribution,
            'global_variance_raw': (
                self.last_background_variance_raw
                + excess_variance.sum(dim=-1)
            ),
        }


class BackgroundExcessGaussianNLLViTLocal(GaussianNLLViTLocal):
    """Lightning wrapper for the contextual background-plus-excess mean."""

    NETWORK_CLASS = BackgroundExcessGaussianVisionTransformerLocal
    predicts_background_excess_components = True
    patch_uncertainty_semantics = 'variance_attribution'
    component_uncertainty_semantics = (
        'global_background_plus_local_excess_attribution'
    )
    NETWORK_UNCERTAINTY_KEYS = (
        *GaussianNLLViTLocal.NETWORK_UNCERTAINTY_KEYS,
        'background_flux_max',
        'background_initial_cap_fraction',
        'excess_initial_fraction',
        'solar_disk_radius_fraction',
        'background_scale_floor_fraction',
        'background_uncertainty_initial_raw',
    )

    def forward(
        self, x, return_attention=False, return_components=False,
    ):
        """Preserve the public tuple unless components are requested."""
        outputs = self.model(
            x,
            self.sxr_norm,
            return_attention=return_attention,
            return_components=return_components,
        )
        components = outputs[-1] if return_components else None
        if return_components:
            outputs = outputs[:-1]
        if return_attention:
            raw, _, variance, attention, patches, patch_variance = outputs
            result = raw, variance, attention, patches, patch_variance
        else:
            raw, _, variance, patches, patch_variance = outputs
            result = raw, variance, patches, patch_variance
        return (*result, components) if return_components else result

    def predict_components(self, x):
        """Predict explicitly named whole-image and local components."""
        return self.forward(x, return_components=True)[-1]

    def _calculate_loss(self, batch, mode):
        result = super()._calculate_loss(batch, mode)
        components = self.model.latest_components()
        total_flux = (
            components['background_flux_raw']
            + components['excess_patch_flux_raw'].sum(dim=-1)
        ).clamp_min(torch.finfo(torch.float32).tiny)
        background_fraction = components['background_flux_raw'] / total_flux
        variance_total = components['global_variance_raw'].clamp_min(
            torch.finfo(torch.float32).tiny
        )
        background_variance_fraction = (
            components['background_variance_raw'] / variance_total
        )
        on_step = mode == 'train'
        self.log(
            f'{mode}/background_flux_raw',
            components['background_flux_raw'].mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{mode}/background_fraction', background_fraction.mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{mode}/background_cap_fraction',
            components['background_cap_fraction'].mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{mode}/excess_flux_raw',
            components['excess_patch_flux_raw'].sum(dim=-1).mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{mode}/background_std_raw',
            torch.sqrt(components['background_variance_raw']).mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{mode}/summed_excess_variance_raw',
            components['excess_variance_contribution_raw'].sum(
                dim=-1
            ).mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{mode}/background_variance_fraction',
            background_variance_fraction.mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        return result
