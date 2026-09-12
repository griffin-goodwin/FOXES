"""Spatial quantile-regression variant of the FOXES Vision Transformer.

The model predicts ordered q02.5/q16/q50/q84/q97.5 flux contributions for
every image patch. Summing a quantile map gives the corresponding global SXR
quantile, which is trained directly with pinball loss in normalized log-flux
space. This supplies spatial interval-width maps without assuming Gaussian
residuals or predicting a variance.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from forecasting.model import (
    SXR_LOG_OFFSET,
    VisionTransformerLocal,
    img_to_patch,
    normalize_sxr,
    unnormalize_sxr,
)


QUANTILE_LEVELS = (0.025, 0.16, 0.5, 0.84, 0.975)
MEDIAN_QUANTILE_INDEX = 2


class QuantileVisionTransformerLocal(VisionTransformerLocal):
    """FOXES backbone with five ordered flux quantiles per spatial patch."""

    MEAN_PARAMETERIZATION_VERSION = 2

    def __init__(self, *args, patch_flux_scale_multiplier=1.0,
                 max_abs_log10_patch_multiplier=8.0,
                 initial_log10_quantile_step=0.1,
                 max_log10_quantile_step=4.0, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = self.input_layer.out_features

        # q50 uses the same positive additive log10-multiplier
        # parameterization as the Gaussian model.
        nn.init.zeros_(self.mlp_head[-1].weight)
        nn.init.zeros_(self.mlp_head[-1].bias)
        self.register_buffer(
            'mean_parameterization_version',
            torch.tensor(self.MEAN_PARAMETERIZATION_VERSION, dtype=torch.int64),
        )

        self.patch_quantile_width_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4),
        )
        self.patch_flux_scale_multiplier = float(patch_flux_scale_multiplier)
        self.max_abs_log10_patch_multiplier = float(
            max_abs_log10_patch_multiplier
        )
        self.initial_log10_quantile_step = float(
            initial_log10_quantile_step
        )
        self.max_log10_quantile_step = float(max_log10_quantile_step)
        if self.patch_flux_scale_multiplier <= 0:
            raise ValueError("patch_flux_scale_multiplier must be positive")
        if self.max_abs_log10_patch_multiplier <= 0:
            raise ValueError(
                "max_abs_log10_patch_multiplier must be positive"
            )
        if self.initial_log10_quantile_step <= 0:
            raise ValueError("initial_log10_quantile_step must be positive")
        if self.max_log10_quantile_step <= 0:
            raise ValueError("max_log10_quantile_step must be positive")
        if (
            self.initial_log10_quantile_step
            > self.max_log10_quantile_step
        ):
            raise ValueError(
                "initial_log10_quantile_step cannot exceed "
                "max_log10_quantile_step"
            )

        # Each output is a positive log10 step away from the adjacent central
        # quantile. Zero weights give every patch a small, ordered interval at
        # initialization while gradients can immediately make it spatially
        # dependent.
        width_output = self.patch_quantile_width_head[-1]
        nn.init.zeros_(width_output.weight)
        nn.init.constant_(
            width_output.bias,
            math.log(math.expm1(self.initial_log10_quantile_step)),
        )

    def _patch_flux_from_logits(self, patch_logits, sxr_norm):
        """Convert q50 patch log10 multipliers into additive raw flux."""
        typical_global_flux = torch.clamp(
            10 ** sxr_norm[0] - SXR_LOG_OFFSET,
            min=torch.finfo(torch.float32).tiny,
        )
        log10_multiplier = torch.clamp(
            patch_logits.float(),
            min=-self.max_abs_log10_patch_multiplier,
            max=self.max_abs_log10_patch_multiplier,
        )
        patch_multiplier = torch.exp(
            log10_multiplier * torch.log(torch.tensor(
                10.0, device=patch_logits.device, dtype=torch.float32
            ))
        )
        patch_flux_scale = (
            typical_global_flux.float() / patch_logits.shape[1]
            * self.patch_flux_scale_multiplier
        )
        return patch_multiplier * patch_flux_scale

    def _ordered_patch_quantiles(self, patch_median_raw, patch_embeddings):
        """Build positive, non-crossing patch quantiles around q50.

        The width head predicts four positive log10 steps in this order:
        q50->q16, q16->q02.5, q50->q84, and q84->q97.5.
        """
        steps = torch.clamp(
            F.softplus(
                self.patch_quantile_width_head(patch_embeddings).float()
            ),
            max=self.max_log10_quantile_step,
        )
        lower_inner, lower_outer, upper_inner, upper_outer = steps.unbind(-1)
        zero = torch.zeros_like(lower_inner)
        log10_offsets = torch.stack((
            -(lower_inner + lower_outer),
            -lower_inner,
            zero,
            upper_inner,
            upper_inner + upper_outer,
        ), dim=-1)
        multipliers = torch.exp(
            log10_offsets * torch.log(torch.tensor(
                10.0,
                device=patch_median_raw.device,
                dtype=torch.float32,
            ))
        )
        return patch_median_raw.float().unsqueeze(-1) * multipliers

    def forward(self, x, sxr_norm, return_attention=False):
        x = img_to_patch(x, self.patch_size)
        x = self.input_layer(x)
        x = self._add_2d_positional_encoding(x)
        x = self.dropout(x).transpose(0, 1)

        attention_weights = []
        for block in self.transformer_blocks:
            if return_attention:
                x, weights = block(x, return_attention=True)
                attention_weights.append(weights)
            else:
                x = block(x)

        patch_embeddings = x.transpose(0, 1)
        patch_logits = self.mlp_head(patch_embeddings).squeeze(-1)
        patch_median_raw = self._patch_flux_from_logits(
            patch_logits, sxr_norm
        )
        patch_quantiles_raw = self._ordered_patch_quantiles(
            patch_median_raw, patch_embeddings
        )
        global_quantiles_raw = patch_quantiles_raw.sum(dim=1)
        quantiles_normalized = normalize_sxr(
            global_quantiles_raw, sxr_norm
        )

        if return_attention:
            return (
                global_quantiles_raw,
                quantiles_normalized,
                attention_weights,
                patch_quantiles_raw,
            )
        return (
            global_quantiles_raw,
            quantiles_normalized,
            patch_quantiles_raw,
        )


class QuantileViTLocal(pl.LightningModule):
    """Train ordered spatial SXR quantiles with class-balanced pinball loss."""

    DEFAULT_SCHEDULER_KWARGS = {'T_max': 250, 'eta_min': 1e-7}
    predicts_uncertainty = True
    uncertainty_kind = 'quantile'
    quantile_levels = QUANTILE_LEVELS

    def __init__(self, model_kwargs, sxr_norm, base_weights=None,
                 weight_decay=1e-5, scheduler_kwargs=None,
                 quantile_kwargs=None):
        super().__init__()
        self.lr = model_kwargs.get(
            'learning_rate', model_kwargs.get('lr', 1e-4)
        )
        self.save_hyperparameters()

        filtered_kwargs = dict(model_kwargs)
        filtered_kwargs.pop('learning_rate', None)
        filtered_kwargs.pop('lr', None)
        filtered_kwargs.pop('num_classes', None)
        quantile_kwargs = dict(quantile_kwargs or {})
        network_keys = {
            'patch_flux_scale_multiplier',
            'max_abs_log10_patch_multiplier',
            'initial_log10_quantile_step',
            'max_log10_quantile_step',
        }
        allowed_keys = network_keys | {
            'loss_weighting', 'five_class_weights', 'spatial_sparsity',
        }
        unknown_keys = set(quantile_kwargs) - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown quantile settings: {sorted(unknown_keys)}"
            )
        self.model = QuantileVisionTransformerLocal(
            **filtered_kwargs,
            **{
                key: quantile_kwargs[key]
                for key in network_keys
                if key in quantile_kwargs
            },
        )
        self.register_buffer(
            'sxr_norm', torch.as_tensor(sxr_norm, dtype=torch.float32)
        )
        self.register_buffer(
            '_quantile_levels_tensor',
            torch.tensor(QUANTILE_LEVELS, dtype=torch.float32),
        )
        self.weight_decay = weight_decay
        self.scheduler_kwargs = {
            **self.DEFAULT_SCHEDULER_KWARGS, **(scheduler_kwargs or {})
        }
        self.loss_weighting = quantile_kwargs.get(
            'loss_weighting', 'unweighted'
        )
        if self.loss_weighting not in {'unweighted', 'five_class'}:
            raise ValueError(
                "quantile.loss_weighting must be 'unweighted' or "
                "'five_class'"
            )
        self.five_class_weights = dict(
            quantile_kwargs.get('five_class_weights') or {}
        )
        if self.loss_weighting == 'five_class':
            required = {
                'below_b', 'b_class', 'c_class', 'm_class', 'x_class',
            }
            missing = required.difference(self.five_class_weights)
            if missing:
                raise ValueError(
                    "Missing five-class pinball weights: "
                    f"{sorted(missing)}"
                )
            if any(
                float(self.five_class_weights[name]) <= 0
                for name in required
            ):
                raise ValueError(
                    "Five-class pinball weights must all be positive"
                )

        spatial_sparsity = quantile_kwargs.get('spatial_sparsity', {})
        if not isinstance(spatial_sparsity, dict):
            raise TypeError("quantile.spatial_sparsity must be a mapping")
        unknown_sparsity_keys = set(spatial_sparsity) - {
            'weight', 'top_fraction', 'gate_center_flux', 'gate_width_dex',
        }
        if unknown_sparsity_keys:
            raise ValueError(
                "Unknown spatial sparsity settings: "
                f"{sorted(unknown_sparsity_keys)}"
            )
        self.spatial_sparsity_weight = float(
            spatial_sparsity.get('weight', 0.0)
        )
        self.spatial_sparsity_top_fraction = float(
            spatial_sparsity.get('top_fraction', 0.05)
        )
        self.spatial_sparsity_gate_center_flux = float(
            spatial_sparsity.get('gate_center_flux', 5e-6)
        )
        self.spatial_sparsity_gate_width_dex = float(
            spatial_sparsity.get('gate_width_dex', 0.25)
        )
        if self.spatial_sparsity_weight < 0:
            raise ValueError("spatial sparsity weight must be nonnegative")
        if not 0 < self.spatial_sparsity_top_fraction <= 1:
            raise ValueError(
                "spatial_sparsity.top_fraction must be in (0, 1]"
            )
        if self.spatial_sparsity_gate_center_flux <= 0:
            raise ValueError(
                "spatial_sparsity.gate_center_flux must be positive"
            )
        if self.spatial_sparsity_gate_width_dex <= 0:
            raise ValueError(
                "spatial_sparsity.gate_width_dex must be positive"
            )

    def on_load_checkpoint(self, checkpoint):
        version_key = 'model.mean_parameterization_version'
        if version_key not in checkpoint.get('state_dict', {}):
            raise RuntimeError(
                'This checkpoint does not use the spatial quantile '
                'log10-multiplier parameterization.'
            )

    def forward(self, x, return_attention=False):
        outputs = self.model(
            x, self.sxr_norm, return_attention=return_attention
        )
        if return_attention:
            global_raw, _, attention, patch_raw = outputs
            return (
                global_raw[:, MEDIAN_QUANTILE_INDEX].unsqueeze(-1),
                global_raw,
                attention,
                patch_raw[:, :, MEDIAN_QUANTILE_INDEX],
                patch_raw,
            )
        global_raw, _, patch_raw = outputs
        return (
            global_raw[:, MEDIAN_QUANTILE_INDEX].unsqueeze(-1),
            global_raw,
            patch_raw[:, :, MEDIAN_QUANTILE_INDEX],
            patch_raw,
        )

    def predict_quantiles(self, x, return_attention=False):
        """Return global and spatial quantiles in raw and normalized units."""
        return self.model(
            x, self.sxr_norm, return_attention=return_attention
        )

    def predict_interval(self, x, z=1.96):
        """Return q50 and either the central 68% or 95% raw interval."""
        global_raw, _, _ = self.predict_quantiles(
            x, return_attention=False
        )
        lower_index, upper_index = (
            (1, 3) if z <= 1.0 else (0, 4)
        )
        return (
            global_raw[:, MEDIAN_QUANTILE_INDEX],
            global_raw[:, lower_index],
            global_raw[:, upper_index],
        )

    def _five_class_weights(self, target_raw):
        weights = torch.full_like(
            target_raw, float(self.five_class_weights['below_b'])
        )
        weights = torch.where(
            target_raw >= 1e-7,
            float(self.five_class_weights['b_class']), weights,
        )
        weights = torch.where(
            target_raw >= 1e-6,
            float(self.five_class_weights['c_class']), weights,
        )
        weights = torch.where(
            target_raw >= 1e-5,
            float(self.five_class_weights['m_class']), weights,
        )
        return torch.where(
            target_raw >= 1e-4,
            float(self.five_class_weights['x_class']), weights,
        )

    def _loss_weights(self, target_raw):
        if self.loss_weighting == 'unweighted':
            return torch.ones_like(target_raw)
        return self._five_class_weights(target_raw)

    def _pinball_per_sample(self, quantiles_norm, target_norm):
        target_norm = target_norm.reshape(-1, 1)
        error = target_norm - quantiles_norm
        levels = self._quantile_levels_tensor.to(
            device=quantiles_norm.device,
            dtype=quantiles_norm.dtype,
        )
        return torch.maximum(
            levels * error,
            (levels - 1.0) * error,
        ).mean(dim=-1)

    def _spatial_sparsity_gate(self, target_raw):
        target_log_flux = torch.log10(
            target_raw.reshape(-1) + SXR_LOG_OFFSET
        )
        center_log_flux = torch.log10(torch.as_tensor(
            self.spatial_sparsity_gate_center_flux + SXR_LOG_OFFSET,
            device=target_raw.device,
            dtype=target_raw.dtype,
        ))
        return torch.sigmoid(
            (target_log_flux - center_log_flux)
            / self.spatial_sparsity_gate_width_dex
        )

    def _spatial_sparsity_loss(self, patch_median_raw, target_raw):
        total_flux = patch_median_raw.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(patch_median_raw.dtype).tiny)
        patch_fractions = patch_median_raw / total_flux
        num_top_patches = max(
            1,
            int(round(
                patch_median_raw.shape[-1]
                * self.spatial_sparsity_top_fraction
            )),
        )
        top_concentration = torch.topk(
            patch_fractions, num_top_patches, dim=-1
        ).values.sum(dim=-1)
        gate = self._spatial_sparsity_gate(target_raw).to(
            dtype=patch_median_raw.dtype
        )
        sparsity_loss = ((1.0 - top_concentration) * gate).mean()
        concentration = (
            (top_concentration * gate).sum()
            / gate.sum().clamp_min(1e-12)
        )
        return sparsity_loss, concentration, gate.mean()

    def _calculate_loss(self, batch, mode):
        images, target_norm = batch
        global_raw, quantiles_norm, patch_quantiles_raw = self.model(
            images, self.sxr_norm, return_attention=False
        )
        target_norm = target_norm.reshape(-1)
        target_raw = unnormalize_sxr(target_norm, self.sxr_norm)
        pinball_per_sample = self._pinball_per_sample(
            quantiles_norm, target_norm
        )
        pinball_unweighted = pinball_per_sample.mean()
        pinball_loss = (
            pinball_per_sample * self._loss_weights(target_raw)
        ).mean()

        median_norm = quantiles_norm[:, MEDIAN_QUANTILE_INDEX]
        median_raw = global_raw[:, MEDIAN_QUANTILE_INDEX]
        patch_median_raw = patch_quantiles_raw[
            :, :, MEDIAN_QUANTILE_INDEX
        ]
        sparsity_loss, top_concentration, sparsity_gate_mean = (
            self._spatial_sparsity_loss(patch_median_raw, target_raw)
        )
        loss = (
            pinball_loss
            + self.spatial_sparsity_weight * sparsity_loss
        )
        mse = F.mse_loss(median_norm, target_norm)
        mae = F.l1_loss(median_norm, target_norm)
        coverage_68 = (
            (target_norm >= quantiles_norm[:, 1])
            & (target_norm <= quantiles_norm[:, 3])
        ).float().mean()
        coverage_95 = (
            (target_norm >= quantiles_norm[:, 0])
            & (target_norm <= quantiles_norm[:, 4])
        ).float().mean()
        sxr_std = self.sxr_norm[1].float()
        width_68_dex = (
            quantiles_norm[:, 3] - quantiles_norm[:, 1]
        ) * sxr_std
        width_95_dex = (
            quantiles_norm[:, 4] - quantiles_norm[:, 0]
        ) * sxr_std
        patch_width_68_raw = (
            patch_quantiles_raw[:, :, 3]
            - patch_quantiles_raw[:, :, 1]
        )
        patch_width_95_raw = (
            patch_quantiles_raw[:, :, 4]
            - patch_quantiles_raw[:, :, 0]
        )

        prefix = mode
        on_step = mode == 'train'
        self.log(
            f'{prefix}/pinball', pinball_loss,
            on_step=on_step, on_epoch=True, prog_bar=True,
            logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/pinball_unweighted', pinball_unweighted,
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/spatial_sparsity', sparsity_loss,
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/top_patch_concentration', top_concentration,
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/sparsity_gate_mean', sparsity_gate_mean,
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/total_loss', loss,
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/mse', mse,
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/mae', mae,
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/coverage_68', coverage_68,
            on_step=False, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/coverage_95', coverage_95,
            on_step=False, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/mean_width_68_dex', width_68_dex.mean(),
            on_step=False, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/mean_width_95_dex', width_95_dex.mean(),
            on_step=False, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/mean_patch_width_68_raw',
            patch_width_68_raw.mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/mean_patch_width_95_raw',
            patch_width_95_raw.mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        if mode == 'val':
            self.log(
                'val_total_loss', loss,
                on_step=False, on_epoch=True, prog_bar=True,
                logger=True, sync_dist=True,
            )
        if mode == 'train':
            learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log(
                'train/learning_rate', learning_rate,
                on_step=True, on_epoch=False, prog_bar=True,
                logger=True, sync_dist=True,
            )
        if mode == 'val':
            return {
                'loss': loss.detach(),
                'mean_norm': median_norm.detach(),
                'target_norm': target_norm.detach(),
                'target_raw': target_raw.detach(),
                'prediction_raw': median_raw.detach(),
                'quantiles_norm': quantiles_norm.detach(),
                'pinball_per_sample': pinball_per_sample.detach(),
            }
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, **self.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'learning_rate',
            },
        }
