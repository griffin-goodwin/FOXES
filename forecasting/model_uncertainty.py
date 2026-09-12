"""Heteroscedastic Gaussian uncertainty variant of the FOXES ViT.

The predictive distribution is Gaussian in normalized ``log10(SXR + 1e-8)``
space. This keeps the variance numerically well scaled and produces positive,
asymmetric intervals after conversion back to physical flux.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from forecasting.model import (
    SXR_LOG_OFFSET,
    SXRRegressionDynamicLoss,
    VisionTransformerLocal,
    img_to_patch,
    normalize_sxr,
    unnormalize_sxr,
)


class GaussianVisionTransformerLocal(VisionTransformerLocal):
    """FOXES backbone with additive patch means and patch-based variance.

    Positive patch fluxes sum to the global mean. Every patch also predicts a
    positive raw-flux standard deviation; patch variances sum under the same
    conditional-independence assumption used by the earlier FOXES uncertainty
    model. The summed variance is converted to normalized log-SXR space for
    Gaussian NLL.
    """

    MEAN_PARAMETERIZATION_VERSION = 2

    def __init__(self, *args, patch_flux_scale_multiplier=1.0,
                 max_abs_log10_patch_multiplier=8.0,
                 relative_std_floor=0.0025, relative_std_max=20.0,
                 patch_scale_floor_fraction=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = self.input_layer.out_features

        # Version 2 interprets this head as a log10 multiplier. At zero, every
        # patch contributes an equal share of the typical global training flux.
        nn.init.zeros_(self.mlp_head[-1].weight)
        nn.init.zeros_(self.mlp_head[-1].bias)
        self.register_buffer(
            'mean_parameterization_version',
            torch.tensor(self.MEAN_PARAMETERIZATION_VERSION, dtype=torch.int64),
        )

        self.patch_uncertainty_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        self.patch_flux_scale_multiplier = float(patch_flux_scale_multiplier)
        self.max_abs_log10_patch_multiplier = float(
            max_abs_log10_patch_multiplier
        )
        self.relative_std_floor = float(relative_std_floor)
        self.relative_std_max = float(relative_std_max)
        self.patch_scale_floor_fraction = float(
            patch_scale_floor_fraction
        )
        if self.patch_flux_scale_multiplier <= 0:
            raise ValueError("patch_flux_scale_multiplier must be positive")
        if self.max_abs_log10_patch_multiplier <= 0:
            raise ValueError("max_abs_log10_patch_multiplier must be positive")
        if self.relative_std_floor <= 0:
            raise ValueError("relative_std_floor must be positive")
        if self.relative_std_max <= self.relative_std_floor:
            raise ValueError("relative_std_max must exceed relative_std_floor")
        if self.patch_scale_floor_fraction <= 0:
            raise ValueError("patch_scale_floor_fraction must be positive")

    def _patch_flux_from_logits(self, patch_logits, sxr_norm):
        """Convert per-patch log10 multipliers to additive raw flux."""
        typical_global_flux = torch.clamp(
            10 ** sxr_norm[0] - SXR_LOG_OFFSET,
            min=torch.finfo(torch.float32).tiny,
        )
        # Evaluate the exponent in float32 so this remains safe under mixed
        # precision. The bound is a numerical guard far beyond the multiplier
        # needed for a single patch to carry an X-class prediction.
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

    def _encode_patch_embeddings(self, x, return_attention):
        """Encode image patches while preserving the public attention output."""
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
        return x.transpose(0, 1), attention_weights

    def _patch_mean_from_embeddings(self, patch_embeddings, sxr_norm):
        """Return nonnegative additive means for the encoded image patches."""
        patch_logits = self.mlp_head(patch_embeddings).squeeze(-1)
        return self._patch_flux_from_logits(patch_logits, sxr_norm)

    def _patch_variance_from_embeddings(
        self, patch_embeddings, patch_flux_raw, sxr_norm,
    ):
        """Return additive raw-flux variance contributions per patch."""
        uncertainty_embeddings = patch_embeddings.detach()
        raw_patch_uncertainty = self.patch_uncertainty_head(
            uncertainty_embeddings
        ).squeeze(-1)
        relative_std = torch.clamp(
            F.softplus(raw_patch_uncertainty.float())
            + self.relative_std_floor,
            max=self.relative_std_max,
        )
        typical_global_flux = torch.clamp(
            10 ** sxr_norm[0].float() - SXR_LOG_OFFSET,
            min=torch.finfo(torch.float32).tiny,
        )
        baseline_patch_flux = (
            typical_global_flux / patch_flux_raw.shape[1]
            * self.patch_scale_floor_fraction
        )
        patch_scale = patch_flux_raw.detach() + baseline_patch_flux
        return (relative_std * patch_scale).square()

    def forward(self, x, sxr_norm, return_attention=False):
        patch_embeddings, attention_weights = self._encode_patch_embeddings(
            x, return_attention
        )
        _, sxr_std = sxr_norm
        patch_flux_raw = self._patch_mean_from_embeddings(
            patch_embeddings, sxr_norm
        )
        global_flux_raw = patch_flux_raw.sum(dim=1, keepdim=True)
        mean_normalized = normalize_sxr(global_flux_raw.squeeze(-1), sxr_norm)

        patch_variance_raw = self._patch_variance_from_embeddings(
            patch_embeddings, patch_flux_raw, sxr_norm
        )
        global_variance_raw = patch_variance_raw.sum(dim=1)

        # Delta-method conversion for
        # y=(log10(raw_flux + offset)-mean)/std.
        # The delta-method scale depends numerically on the predicted global
        # flux, but its variance gradient must not flow into the mean branch.
        # Without this detach, NLL can move an otherwise-correct mean merely to
        # shrink or enlarge its normalized variance.
        global_flux_for_variance = global_flux_raw.squeeze(-1).float().detach()
        log_derivative = 1.0 / (
            torch.log(torch.tensor(
                10.0, device=global_flux_raw.device, dtype=torch.float32
            ))
            * sxr_std.float()
            * (global_flux_for_variance + SXR_LOG_OFFSET)
        )
        variance_normalized = (
            global_variance_raw * log_derivative.square()
        )

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
    """Train the canonical FOXES patch model with calibrated uncertainty.

    The mean/backbone is optimized only by an optionally class-weighted Huber
    loss.  A separate Gaussian NLL trains the patch uncertainty head with the
    predicted mean detached.  The uncertainty branch also consumes detached
    patch embeddings and detached patch flux scales, so it cannot alter the
    spatial mean prediction.

    The predicted variance is in normalized log10-SXR units. Use
    :meth:`predict_interval` to obtain intervals in physical W/m^2 units.
    """

    DEFAULT_SCHEDULER_KWARGS = {'T_max': 250, 'eta_min': 1e-7}
    predicts_uncertainty = True
    uncertainty_kind = 'gaussian'
    predicts_patch_uncertainty = True
    patch_uncertainty_semantics = 'marginal_variance'
    NETWORK_CLASS = GaussianVisionTransformerLocal
    NETWORK_UNCERTAINTY_KEYS = (
        'patch_flux_scale_multiplier',
        'max_abs_log10_patch_multiplier',
        'relative_std_floor',
        'relative_std_max',
        'patch_scale_floor_fraction',
    )

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
        allowed_uncertainty_keys = set(self.NETWORK_UNCERTAINTY_KEYS) | {
            'huber_delta',
            'class_weighting',
            'class_weights',
        }
        # Old checkpoints include objective settings in their saved
        # hyperparameters. They do not affect inference, so accept them only
        # for loading compatibility; new configs never emit them.
        legacy_loss_only_keys = {
            'beta_nll_beta',
            'mean_loss',
            'train_uncertainty',
            'spatial_sparsity',
            'mean_huber_delta',
            'mean_loss_weighting',
            'mean_class_weights',
            'use_class_weights',
            'mean_huber_weight',
            'mean_huber_use_class_weights',
            'nll_weighting',
            'five_class_weights',
            'detach_uncertainty_features',
            'global_sigma_init_dex',
            'global_sigma_floor_dex',
        }
        unknown_keys = (
            set(uncertainty_kwargs)
            - allowed_uncertainty_keys
            - legacy_loss_only_keys
        )
        if unknown_keys:
            raise ValueError(
                f"Unknown uncertainty settings: {sorted(unknown_keys)}"
            )
        network_uncertainty_kwargs = {
            key: uncertainty_kwargs[key]
            for key in self.NETWORK_UNCERTAINTY_KEYS
            if key in uncertainty_kwargs
        }
        self.model = self.NETWORK_CLASS(
            **filtered_kwargs, **network_uncertainty_kwargs
        )
        self.register_buffer(
            'sxr_norm', torch.as_tensor(sxr_norm, dtype=torch.float32)
        )
        self.weight_decay = weight_decay
        self.scheduler_kwargs = {
            **self.DEFAULT_SCHEDULER_KWARGS, **(scheduler_kwargs or {})
        }
        self.huber_delta = float(
            uncertainty_kwargs.get(
                'huber_delta',
                uncertainty_kwargs.get('mean_huber_delta', 1.0),
            )
        )
        if self.huber_delta <= 0:
            raise ValueError("uncertainty.huber_delta must be positive")
        legacy_weighting_names = {
            'none': 'none',
            'four_class_macro': 'inverse_frequency',
            'sqrt_four_class_macro': 'sqrt_inverse_frequency',
        }
        legacy_weighting = uncertainty_kwargs.get(
            'mean_loss_weighting', 'none'
        )
        self.class_weighting = str(
            uncertainty_kwargs.get(
                'class_weighting',
                legacy_weighting_names.get(
                    legacy_weighting, legacy_weighting,
                ),
            )
        )
        if self.class_weighting not in {
            'none', 'inverse_frequency', 'sqrt_inverse_frequency',
        }:
            raise ValueError(
                "uncertainty.class_weighting must be 'none', "
                "'inverse_frequency', or 'sqrt_inverse_frequency'"
            )
        class_names = ('quiet', 'c_class', 'm_class', 'x_class')
        configured_weights = uncertainty_kwargs.get(
            'class_weights', uncertainty_kwargs.get('mean_class_weights')
        )
        if self.class_weighting != 'none':
            if not isinstance(configured_weights, dict):
                raise ValueError(
                    "class-weighted mean loss requires training-derived "
                    "uncertainty.class_weights"
                )
            if set(configured_weights) != set(class_names):
                raise ValueError(
                    "uncertainty.class_weights must contain exactly "
                    f"{list(class_names)}"
                )
            weights = torch.tensor(
                [configured_weights[name] for name in class_names],
                dtype=torch.float32,
            )
            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError(
                    "uncertainty.class_weights must be finite and positive"
                )
        else:
            weights = torch.ones(len(class_names), dtype=torch.float32)
        # These weights are derived deterministically from the configured
        # training split. Keeping them out of the state dict preserves loading
        # compatibility while save_hyperparameters records their provenance.
        self.register_buffer(
            'class_weight_values', weights, persistent=False,
        )

    def on_load_checkpoint(self, checkpoint):
        version_key = 'model.mean_parameterization_version'
        state_dict = checkpoint.get('state_dict', {})
        if version_key not in state_dict:
            raise RuntimeError(
                'This checkpoint predates guarded patch-mean parameterizations, '
                'so its mean conversion cannot be identified safely. Use the '
                'model code that created it or start a fresh training run.'
            )
        checkpoint_version = int(state_dict[version_key].item())
        expected_version = int(
            self.model.mean_parameterization_version.item()
        )
        if checkpoint_version != expected_version:
            raise RuntimeError(
                'Checkpoint mean parameterization version '
                f'{checkpoint_version} does not match this model version '
                f'{expected_version}. Select the model_type used to train the '
                'checkpoint.'
            )

    def forward(self, x, return_attention=False):
        """Return global parameters plus patch flux and variance maps."""
        outputs = self.model(x, self.sxr_norm, return_attention=return_attention)
        if return_attention:
            raw, _, variance, attention, patches, patch_variance = outputs
            return raw, variance, attention, patches, patch_variance
        raw, _, variance, patches, patch_variance = outputs
        return raw, variance, patches, patch_variance

    def predict_distribution(self, x, return_attention=False):
        """Return global Gaussian parameters and contributing patch maps."""
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

    def _weighted_huber_loss(self, mean, target, target_raw):
        """Return Huber loss, optionally balanced by the four GOES classes."""
        per_sample = F.huber_loss(
            mean, target, delta=self.huber_delta, reduction='none',
        )

        if self.class_weighting != 'none':
            thresholds = SXRRegressionDynamicLoss.CLASS_THRESHOLDS
            class_indices = torch.zeros_like(target_raw, dtype=torch.long)
            class_indices = torch.where(
                target_raw >= thresholds['c'], 1, class_indices,
            )
            class_indices = torch.where(
                target_raw >= thresholds['m'], 2, class_indices,
            )
            class_indices = torch.where(
                target_raw >= thresholds['x'], 3, class_indices,
            )
            sample_weights = self.class_weight_values[class_indices]
            per_sample = per_sample * sample_weights.reshape_as(per_sample)
        return per_sample.mean()

    @staticmethod
    def _uncertainty_nll_loss(mean, target, variance):
        """Train variance without allowing NLL gradients into the mean."""
        return F.gaussian_nll_loss(
            mean.detach(), target, variance, full=True, reduction='mean',
        )

    def _calculate_loss(self, batch, mode):
        images, target_norm = batch
        (
            raw_prediction, mean_norm, variance_norm,
            patch_flux_raw, patch_variance_raw,
        ) = self.model(
            images, self.sxr_norm, return_attention=False,
        )
        target_norm = target_norm.reshape_as(mean_norm)
        target_raw = unnormalize_sxr(target_norm, self.sxr_norm)

        mean_loss = self._weighted_huber_loss(
            mean_norm, target_norm, target_raw,
        )
        uncertainty_loss = self._uncertainty_nll_loss(
            mean_norm, target_norm, variance_norm,
        )
        loss = mean_loss + uncertainty_loss
        mse = F.mse_loss(mean_norm, target_norm)
        mae = F.l1_loss(mean_norm, target_norm)
        sigma = torch.sqrt(variance_norm)
        sigma_dex = sigma * self.sxr_norm[1]
        standardized = torch.abs(target_norm - mean_norm) / sigma
        coverage_68 = (standardized <= 1.0).float().mean()
        coverage_95 = (standardized <= 1.96).float().mean()

        prefix = mode
        on_step = mode == 'train'
        self.log(f'{prefix}/nll', uncertainty_loss, on_step=on_step,
                 on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log(
            f'{prefix}/uncertainty_nll', uncertainty_loss, on_step=on_step,
            on_epoch=True,
            logger=True, sync_dist=True,
        )
        self.log(
            f'{prefix}/mean_loss', mean_loss, on_step=on_step, on_epoch=True,
            logger=True, sync_dist=True,
        )
        self.log(f'{prefix}/total_loss', loss, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log(f'{prefix}/mse', mse, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log(f'{prefix}/mae', mae, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log(f'{prefix}/mean_sigma', sigma.mean(), on_step=on_step,
                 on_epoch=True, logger=True, sync_dist=True)
        self.log(
            f'{prefix}/mean_sigma_dex', sigma_dex.mean(),
            on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
        )
        patch_uncertainty_metric = (
            'mean_sqrt_variance_contribution_raw'
            if self.patch_uncertainty_semantics == 'variance_attribution'
            else 'mean_patch_std_raw'
        )
        self.log(
            f'{prefix}/{patch_uncertainty_metric}',
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
        if mode == 'val':
            return {
                'loss': loss.detach(),
                'nll': uncertainty_loss.detach(),
                'mean_norm': mean_norm.detach(),
                'target_norm': target_norm.detach(),
                'target_raw': target_raw.detach(),
                'prediction_raw': raw_prediction.squeeze(-1).detach(),
                'variance_norm': variance_norm.detach(),
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
                'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1,
                'name': 'learning_rate',
            },
        }
