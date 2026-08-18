"""Heteroscedastic Gaussian uncertainty variant of the FOXES ViT.

The predictive distribution is Gaussian in normalized log10-SXR space. This
keeps the variance numerically well scaled and produces positive, asymmetric
intervals after conversion back to physical flux.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from forecasting.model import (
    SXRRegressionDynamicLoss,
    VisionTransformerLocal,
    img_to_patch,
    normalize_sxr,
    unnormalize_sxr,
)


class GaussianVisionTransformerLocal(VisionTransformerLocal):
    """FOXES backbone with a variance prediction for every flux patch.

    Patch means add in raw flux units, so patch variances are also constructed
    in raw flux units and summed under an explicit independence assumption.
    The summed raw variance is converted to normalized log10-SXR variance for
    the Gaussian NLL using a first-order (delta-method) approximation.
    """

    def __init__(self, *args, relative_std_floor=0.0025,
                 relative_std_max=20.0,
                 patch_scale_floor_fraction=1.0,
                 patch_flux_scale_multiplier=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = self.input_layer.out_features

        # One unconstrained uncertainty score per spatial patch. Softplus below
        # converts it into a positive relative standard deviation.
        self.patch_uncertainty_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        self.relative_std_floor = float(relative_std_floor)
        self.relative_std_max = float(relative_std_max)
        self.patch_scale_floor_fraction = float(patch_scale_floor_fraction)
        self.patch_flux_scale_multiplier = float(patch_flux_scale_multiplier)
        if self.relative_std_floor <= 0:
            raise ValueError("relative_std_floor must be positive")
        if self.relative_std_max <= self.relative_std_floor:
            raise ValueError("relative_std_max must exceed relative_std_floor")
        if self.patch_scale_floor_fraction <= 0:
            raise ValueError("patch_scale_floor_fraction must be positive")
        if self.patch_flux_scale_multiplier <= 0:
            raise ValueError("patch_flux_scale_multiplier must be positive")

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

        # Treat patch logits as unconstrained contribution scores, rather than
        # normalized global-SXR values. The normalization mean supplies a
        # typical total flux scale, divided across all patches; softplus makes
        # each raw contribution positive without a hard clamp/dead gradient.
        patch_logits = self.mlp_head(patch_embeddings).squeeze(-1)
        sxr_mean, sxr_std = sxr_norm
        typical_global_flux = torch.clamp(
            10 ** sxr_mean - 1e-8,
            min=torch.finfo(patch_logits.dtype).tiny,
        )
        patch_flux_scale = (
            typical_global_flux / patch_logits.shape[1]
            * self.patch_flux_scale_multiplier
        )
        patch_flux_raw = F.softplus(patch_logits) * patch_flux_scale
        global_flux_raw = patch_flux_raw.sum(dim=1, keepdim=True)
        mean_normalized = normalize_sxr(global_flux_raw.squeeze(-1), sxr_norm)

        # Each patch predicts uncertainty relative to its flux scale. The small
        # baseline scale lets a currently dark patch still express uncertainty.
        raw_uncertainty = self.patch_uncertainty_head(
            patch_embeddings
        ).squeeze(-1)
        relative_std = torch.clamp(
            F.softplus(raw_uncertainty) + self.relative_std_floor,
            max=self.relative_std_max,
        )

        baseline_patch_flux = (
            typical_global_flux / patch_flux_raw.shape[1]
            * self.patch_scale_floor_fraction
        )
        # Detaching the mean-derived scale keeps the variance branch from
        # changing patch means merely to manipulate its predicted uncertainty.
        patch_scale = patch_flux_raw.detach() + baseline_patch_flux
        patch_std_raw = relative_std * patch_scale
        patch_variance_raw = patch_std_raw.square()

        # Independent patch errors add in variance, not standard deviation.
        # This is the exact aggregation rule under conditional independence.
        global_variance_raw = patch_variance_raw.sum(dim=1)

        # The likelihood is evaluated in the same normalized log10 space as
        # the SXR targets. For g(F)=log10(F+eps)/sxr_std,
        # Var[g(F)] ~= Var[F] * g'(E[F])^2 (the delta method).
        log_derivative = 1.0 / (
            torch.log(torch.tensor(10.0, device=global_flux_raw.device))
            * sxr_std
            * (global_flux_raw.squeeze(-1) + 1e-8)
        )
        # No model-level floor is imposed here: positivity comes from the
        # per-patch softplus scale. Gaussian NLL applies its own numerical eps.
        variance_normalized = global_variance_raw * log_derivative.square()

        if return_attention:
            return (
                global_flux_raw, mean_normalized, variance_normalized,
                attention_weights, patch_flux_raw, patch_variance_raw,
            )
        return (
            global_flux_raw, mean_normalized, variance_normalized,
            patch_flux_raw, patch_variance_raw,
        )


class GaussianNLLViTLocal(pl.LightningModule):
    """Train FOXES with Gaussian NLL and return calibrated uncertainty.

    The predicted variance is in normalized log10-SXR units. Use
    ``predict_interval`` to obtain intervals in physical W/m^2 units.
    """

    DEFAULT_SCHEDULER_KWARGS = {'T_0': 250, 'T_mult': 2, 'eta_min': 1e-7}
    predicts_uncertainty = True

    def __init__(self, model_kwargs, sxr_norm, base_weights=None,
                 weight_decay=1e-5, scheduler_kwargs=None,
                 uncertainty_kwargs=None):
        super().__init__()
        self.lr = model_kwargs.get('learning_rate', model_kwargs.get('lr', 1e-4))
        self.save_hyperparameters()

        filtered_kwargs = dict(model_kwargs)
        filtered_kwargs.pop('learning_rate', None)
        filtered_kwargs.pop('lr', None)
        filtered_kwargs.pop('num_classes', None)
        uncertainty_kwargs = dict(uncertainty_kwargs or {})
        network_uncertainty_keys = (
            'relative_std_floor', 'relative_std_max',
            'patch_scale_floor_fraction',
            'patch_flux_scale_multiplier',
        )
        allowed_uncertainty_keys = set(network_uncertainty_keys) | {
            'use_class_weights',
        }
        unknown_keys = set(uncertainty_kwargs) - allowed_uncertainty_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown uncertainty settings: {sorted(unknown_keys)}"
            )
        network_uncertainty_kwargs = {
            key: uncertainty_kwargs[key]
            for key in network_uncertainty_keys
            if key in uncertainty_kwargs
        }
        self.model = GaussianVisionTransformerLocal(
            **filtered_kwargs, **network_uncertainty_kwargs
        )
        self.register_buffer(
            'sxr_norm', torch.as_tensor(sxr_norm, dtype=torch.float32)
        )
        self.base_weights = (
            dict(base_weights) if base_weights is not None
            else dict(SXRRegressionDynamicLoss.DEFAULT_BASE_WEIGHTS)
        )
        self.weight_decay = weight_decay
        self.scheduler_kwargs = {
            **self.DEFAULT_SCHEDULER_KWARGS, **(scheduler_kwargs or {})
        }
        self.use_class_weights = bool(uncertainty_kwargs.get('use_class_weights', True))

    def forward(self, x, return_attention=True):
        """Return point prediction, global variance, maps, and patch variance."""
        outputs = self.model(x, self.sxr_norm, return_attention=return_attention)
        if return_attention:
            raw, _, variance, attention, patches, patch_variance = outputs
            return raw, variance, attention, patches, patch_variance
        raw, _, variance, patches, patch_variance = outputs
        return raw, variance, patches, patch_variance

    def predict_distribution(self, x, return_attention=False):
        """Return global Gaussian parameters and raw patch uncertainty map."""
        return self.model(x, self.sxr_norm, return_attention=return_attention)

    def predict_interval(self, x, z=1.96):
        """Return point, lower, and upper physical-flux estimates."""
        raw, mean_norm, variance_norm, _, _ = self.predict_distribution(
            x, return_attention=False
        )
        sigma = torch.sqrt(variance_norm)
        lower = unnormalize_sxr(mean_norm - z * sigma, self.sxr_norm)
        upper = unnormalize_sxr(mean_norm + z * sigma, self.sxr_norm)
        return raw.squeeze(-1), torch.clamp(lower, min=0), torch.clamp(upper, min=0)

    def _class_weights(self, target_raw):
        if not self.use_class_weights:
            return torch.ones_like(target_raw)
        thresholds = SXRRegressionDynamicLoss.CLASS_THRESHOLDS
        weights = torch.full_like(target_raw, float(self.base_weights['quiet']))
        weights = torch.where(
            target_raw >= thresholds['c'],
            float(self.base_weights['c_class']), weights,
        )
        weights = torch.where(
            target_raw >= thresholds['m'],
            float(self.base_weights['m_class']), weights,
        )
        weights = torch.where(
            target_raw >= thresholds['x'],
            float(self.base_weights['x_class']), weights,
        )
        return weights / weights.mean().clamp_min(1e-12)

    def _calculate_loss(self, batch, mode):
        images, target_norm = batch
        _, mean_norm, variance_norm, _, patch_variance_raw = self.model(
            images, self.sxr_norm, return_attention=False
        )
        target_norm = target_norm.reshape_as(mean_norm)
        target_raw = unnormalize_sxr(target_norm, self.sxr_norm)

        nll_per_sample = F.gaussian_nll_loss(
            mean_norm, target_norm, variance_norm,
            full=True, reduction='none',
        )
        loss = (nll_per_sample * self._class_weights(target_raw)).mean()
        mse = F.mse_loss(mean_norm, target_norm)
        mae = F.l1_loss(mean_norm, target_norm)
        sigma = torch.sqrt(variance_norm)
        standardized = torch.abs(target_norm - mean_norm) / sigma
        coverage_68 = (standardized <= 1.0).float().mean()
        coverage_95 = (standardized <= 1.96).float().mean()

        prefix = mode
        on_step = mode == 'train'
        self.log(f'{prefix}/nll', loss, on_step=on_step, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{prefix}/mse', mse, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log(f'{prefix}/mae', mae, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log(f'{prefix}/mean_sigma', sigma.mean(), on_step=on_step,
                 on_epoch=True, logger=True, sync_dist=True)
        self.log(
            f'{prefix}/mean_patch_std_raw',
            torch.sqrt(patch_variance_raw).mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        self.log(f'{prefix}/coverage_68', coverage_68, on_step=False,
                 on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{prefix}/coverage_95', coverage_95, on_step=False,
                 on_epoch=True, logger=True, sync_dist=True)
        if mode == 'val':
            self.log('val_total_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
        if mode == 'train':
            learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/learning_rate', learning_rate, on_step=True,
                     on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, **self.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1,
                'name': 'learning_rate',
            },
        }
