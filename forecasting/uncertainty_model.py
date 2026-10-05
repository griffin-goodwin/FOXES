"""The single uncertainty-aware FOXES model.

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


class PatchGlobalContext(nn.Module):
    """Each patch queries a compact grid of encoded and raw image features."""

    def __init__(self, embed_dim, grid_size=16, attention_dim=128,
                 num_heads=4, dropout=0.0, max_abs_adjustment=0.3):
        super().__init__()
        if grid_size <= 0 or attention_dim <= 0 or num_heads <= 0:
            raise ValueError('patch_global_context dimensions must be positive')
        if attention_dim % num_heads:
            raise ValueError('patch_global_context attention_dim must divide num_heads')
        if max_abs_adjustment <= 0:
            raise ValueError('patch_global_context max_abs_adjustment must be positive')
        self.grid_size = grid_size
        self.max_abs_adjustment = max_abs_adjustment
        self.query_projection = nn.Linear(embed_dim, attention_dim)
        self.memory_projection = nn.Linear(2 * embed_dim, attention_dim)
        self.attention = nn.MultiheadAttention(
            attention_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(embed_dim + attention_dim),
            nn.Linear(embed_dim + attention_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, 1),
        )
        # The new global correction starts at zero, exposing the local mean.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, local_features, raw_features):
        batch, patches, channels = local_features.shape
        side = int(patches ** 0.5)
        if side * side != patches or self.grid_size > side:
            raise ValueError('patch_global_context needs a square patch grid')

        def pool(features):
            grid = features.transpose(1, 2).reshape(batch, channels, side, side)
            return F.adaptive_avg_pool2d(
                grid, (self.grid_size, self.grid_size)
            ).flatten(2).transpose(1, 2)

        memory = self.memory_projection(torch.cat((
            pool(local_features), pool(raw_features),
        ), dim=-1))
        # need_weights=False permits PyTorch's memory-efficient attention path.
        context, _ = self.attention(
            self.query_projection(local_features), memory, memory,
            need_weights=False,
        )
        return self.max_abs_adjustment * torch.tanh(
            self.output(torch.cat((local_features, context), dim=-1))
            .squeeze(-1)
        )


class GaussianVisionTransformerLocal(VisionTransformerLocal):
    """FOXES backbone with additive means and logit-conditioned variance.

    Positive patch fluxes sum to the global mean. A shared head independently
    maps each detached mean-head patch logit to a dimensionless relative
    standard deviation. The raw-flux
    variance contributions sum under a conditional-independence assumption,
    then are converted to normalized log-SXR space for Gaussian NLL.
    """

    MEAN_PARAMETERIZATIONS = (
        'original', 'original_unclamped', 'multiplier', 'positive',
    )
    MEAN_HEAD_INITIALIZATIONS = {
        'original': 'pytorch_linear_default',
        'original_unclamped': 'pytorch_linear_default',
        'multiplier': 'zeros',
        'positive': 'pytorch_linear_default',
    }

    def __init__(self, *args, mean_parameterization='multiplier',
                 uncertainty_enabled=True, patch_flux_scale_multiplier=1.0,
                 max_abs_log10_patch_multiplier=8.0,
                 global_patch_adjustment=False,
                 max_abs_log10_global_adjustment=0.3,
                 global_patch_head_init_std=0.0,
                 independent_local_global=False,
                 patch_global_context=None,
                 initial_relative_std=1.0,
                 min_log10_relative_std=-4.0,
                 max_log10_relative_std=4.0,
                 uncertainty_hidden_dim=16, **kwargs):
        super().__init__(*args, **kwargs)

        self.mean_parameterization = str(mean_parameterization).lower()
        if self.mean_parameterization not in self.MEAN_PARAMETERIZATIONS:
            raise ValueError(
                'mean_parameterization must be one of '
                f'{self.MEAN_PARAMETERIZATIONS}'
            )
        self.mean_head_initialization = self.MEAN_HEAD_INITIALIZATIONS[
            self.mean_parameterization
        ]
        if self.mean_parameterization == 'multiplier':
            # At zero, every patch contributes an equal share of the typical
            # global training flux. The original mode deliberately retains the
            # Linear layer's standard PyTorch initialization from super().__init__.
            nn.init.zeros_(self.mlp_head[-1].weight)
            nn.init.zeros_(self.mlp_head[-1].bias)
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.patch_flux_scale_multiplier = float(patch_flux_scale_multiplier)
        self.max_abs_log10_patch_multiplier = float(
            max_abs_log10_patch_multiplier
        )
        self.global_patch_adjustment = bool(global_patch_adjustment)
        self.max_abs_log10_global_adjustment = float(
            max_abs_log10_global_adjustment
        )
        self.global_patch_head_init_std = float(global_patch_head_init_std)
        patch_global_context = dict(patch_global_context or {})
        self.patch_global_context_enabled = bool(
            patch_global_context.pop('enabled', False)
        )
        if not isinstance(independent_local_global, bool):
            raise ValueError('independent_local_global must be a boolean')
        self.independent_local_global = independent_local_global
        if self.independent_local_global and not (
            self.global_patch_adjustment or self.patch_global_context_enabled
        ):
            raise ValueError(
                'independent_local_global requires a global patch correction'
            )
        if not self.patch_global_context_enabled and patch_global_context:
            raise ValueError('patch_global_context settings require enabled=true')
        self.initial_relative_std = float(initial_relative_std)
        self.min_log10_relative_std = float(min_log10_relative_std)
        self.max_log10_relative_std = float(max_log10_relative_std)
        self.uncertainty_hidden_dim = int(uncertainty_hidden_dim)
        if self.patch_flux_scale_multiplier <= 0:
            raise ValueError("patch_flux_scale_multiplier must be positive")
        if self.max_abs_log10_patch_multiplier <= 0:
            raise ValueError("max_abs_log10_patch_multiplier must be positive")
        if self.max_abs_log10_global_adjustment <= 0:
            raise ValueError("max_abs_log10_global_adjustment must be positive")
        if self.global_patch_head_init_std < 0:
            raise ValueError("global_patch_head_init_std must be nonnegative")
        if self.global_patch_adjustment and self.mask_mode != 'local':
            raise ValueError(
                "global_patch_adjustment requires local patch attention so the "
                "spatial head cannot import distant activity"
            )
        if self.initial_relative_std <= 0:
            raise ValueError("initial_relative_std must be positive")
        if self.max_log10_relative_std <= self.min_log10_relative_std:
            raise ValueError(
                "max_log10_relative_std must exceed "
                "min_log10_relative_std"
            )
        if self.uncertainty_hidden_dim <= 0:
            raise ValueError("uncertainty_hidden_dim must be positive")

        self.last_log10_global_patch_adjustment = None
        embed_dim = self.mlp_head[-1].in_features
        if self.global_patch_adjustment:
            # A summary query sees every original AIA patch. The per-patch
            # correction also receives that patch's raw AIA token and its
            # locally encoded feature, so remote activity cannot directly
            # replace the local evidence at a quiet patch.
            self.global_raw_patch_projection = nn.Linear(
                self.input_layer.in_features, embed_dim
            )
            self.global_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.global_attention = nn.MultiheadAttention(
                embed_dim,
                self.transformer_blocks[0].num_heads,
                dropout=self.dropout.p,
                batch_first=True,
            )
            self.global_patch_head = nn.Sequential(
                nn.LayerNorm(3 * embed_dim),
                nn.Linear(3 * embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
            )
            if self.global_patch_head_init_std:
                nn.init.normal_(
                    self.global_patch_head[-1].weight,
                    mean=0.0,
                    std=self.global_patch_head_init_std,
                )
            else:
                nn.init.zeros_(self.global_patch_head[-1].weight)
            nn.init.zeros_(self.global_patch_head[-1].bias)

        self.patch_global_context = None
        self.last_log10_patch_global_adjustment = None
        if self.patch_global_context_enabled:
            allowed = {
                'grid_size', 'attention_dim', 'num_heads',
                'dropout', 'max_abs_adjustment',
            }
            unknown = set(patch_global_context) - allowed
            if unknown:
                raise ValueError(
                    f'Unknown patch_global_context settings: {sorted(unknown)}'
                )
            self.patch_global_context = PatchGlobalContext(
                embed_dim, **patch_global_context,
            )

        self.patch_uncertainty_head = None
        self.initial_log10_relative_std = None
        if self.uncertainty_enabled:
            self.initial_log10_relative_std = float(
                torch.log10(torch.tensor(self.initial_relative_std)).item()
            )
            if not (
                self.min_log10_relative_std
                < self.initial_log10_relative_std
                < self.max_log10_relative_std
            ):
                raise ValueError(
                    "initial log10 relative standard deviation falls outside "
                    "the configured numerical bounds"
                )
            self.patch_uncertainty_head = nn.Sequential(
                nn.Linear(1, self.uncertainty_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.uncertainty_hidden_dim, 1),
            )
            nn.init.zeros_(self.patch_uncertainty_head[-1].weight)
            nn.init.constant_(
                self.patch_uncertainty_head[-1].bias,
                self.initial_log10_relative_std,
            )

    def _patch_flux_from_logits(self, patch_logits, sxr_norm):
        """Convert patch outputs to nonnegative additive raw flux."""
        if self.mean_parameterization in ('original', 'original_unclamped'):
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
            )
            if self.mean_parameterization == 'original_unclamped':
                # Direct positive physical patch flux, without hard bounds or
                # subtracting the whole-image normalization offset per patch.
                return patch_flux_raw
            patch_flux_raw = patch_flux_raw - SXR_LOG_OFFSET
            # Match the released FOXES patch conversion exactly, including
            # its per-patch upper guard.
            return torch.clamp(patch_flux_raw, min=0.0, max=1.0)

        typical_global_flux = torch.clamp(
            10 ** sxr_norm[0] - SXR_LOG_OFFSET,
            min=torch.finfo(torch.float32).tiny,
        )
        if self.mean_parameterization == 'positive':
            # Preserve the original normalized-log head scale and default
            # initialization, but assign the whole-image training scale across
            # patches before applying each patch's multiplicative adjustment.
            log10_adjustment = torch.clamp(
                patch_logits.float(),
                min=-self.max_abs_log10_patch_multiplier,
                max=self.max_abs_log10_patch_multiplier,
            )
            patch_multiplier = torch.exp(
                log10_adjustment * torch.log(torch.tensor(
                    10.0,
                    device=patch_logits.device,
                    dtype=torch.float32,
                ))
            )
            return (
                typical_global_flux.float()
                / patch_logits.shape[1]
                * patch_multiplier
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

    def _patch_mean_from_embeddings(self, patch_embeddings, sxr_norm,
                                    images=None, return_local=False):
        """Return logits and nonnegative additive patch means."""
        patch_logits = self.mlp_head(patch_embeddings).squeeze(-1)
        patch_flux_raw = self._patch_flux_from_logits(patch_logits, sxr_norm)
        local_patch_flux_raw = patch_flux_raw
        correction_features = patch_embeddings
        correction_positions = self.pos_embedding_2d
        if self.independent_local_global:
            # The global objective may read local evidence but must not update
            # its flux head, transformer, projection, or positional embeddings.
            patch_flux_raw = patch_flux_raw.detach()
            correction_features = patch_embeddings.detach()
            correction_positions = correction_positions.detach()
        if (self.global_patch_adjustment or self.patch_global_context is not None):
            if images is None:
                raise ValueError('Global patch correction requires original AIA images')
        if self.global_patch_adjustment:
            raw_tokens = self.global_raw_patch_projection(
                img_to_patch(images, self.patch_size)
            )
            raw_tokens = raw_tokens + correction_positions.reshape(
                1, -1, raw_tokens.shape[-1]
            )
            query = self.global_query.expand(patch_embeddings.shape[0], -1, -1)
            context, _ = self.global_attention(
                query, raw_tokens, raw_tokens, need_weights=False
            )
            context_per_patch = context.expand(-1, patch_embeddings.shape[1], -1)
            correction_input = torch.cat(
                (correction_features, raw_tokens, context_per_patch), dim=-1
            )
            log10_adjustment = (
                self.max_abs_log10_global_adjustment
                * torch.tanh(self.global_patch_head(correction_input).squeeze(-1))
            )
            self.last_log10_global_patch_adjustment = log10_adjustment.detach()
            patch_flux_raw = patch_flux_raw * torch.pow(
                torch.tensor(10.0, device=patch_flux_raw.device),
                log10_adjustment.float(),
            )
        if self.patch_global_context is not None:
            # Reuse the pretrained backbone's input projection. The old
            # single-query global path is absent when its flag is false.
            raw_tokens = self.input_layer(img_to_patch(images, self.patch_size))
            raw_tokens = raw_tokens + self.pos_embedding_2d.reshape(
                1, -1, raw_tokens.shape[-1]
            )
            if self.independent_local_global:
                # This path reuses the local input projection as well as its
                # positions. Detach the entire shared raw-feature path.
                raw_tokens = raw_tokens.detach()
            patch_specific_adjustment = self.patch_global_context(
                correction_features, raw_tokens,
            )
            self.last_log10_patch_global_adjustment = (
                patch_specific_adjustment.detach()
            )
            patch_flux_raw = patch_flux_raw * torch.pow(
                torch.tensor(10.0, device=patch_flux_raw.device),
                patch_specific_adjustment.float(),
            )
            combined = (
                patch_specific_adjustment + log10_adjustment
                if self.global_patch_adjustment else patch_specific_adjustment
            )
            self.last_log10_global_patch_adjustment = combined.detach()
        if return_local:
            return patch_logits, patch_flux_raw, local_patch_flux_raw
        return patch_logits, patch_flux_raw

    def _patch_variance_from_logits(self, patch_logits, patch_flux_raw):
        """Predict raw variance contributions from detached patch logits."""
        if not self.uncertainty_enabled:
            raise RuntimeError("uncertainty is disabled for this model")
        detached_patch_logits = patch_logits.detach().float()
        detached_patch_flux = patch_flux_raw.detach().float()
        patch_log10_relative_std = self.patch_uncertainty_head(
            detached_patch_logits.unsqueeze(-1)
        ).squeeze(-1).float()
        patch_log10_relative_std = torch.clamp(
            patch_log10_relative_std,
            min=self.min_log10_relative_std,
            max=self.max_log10_relative_std,
        )
        relative_std = torch.pow(
            torch.tensor(
                10.0,
                device=patch_log10_relative_std.device,
                dtype=torch.float32,
            ),
            patch_log10_relative_std,
        )
        patch_std_raw = relative_std * detached_patch_flux
        return patch_std_raw.square()

    def forward(self, x, sxr_norm, return_attention=False, return_local=False):
        """Return corrected predictions, optionally appending local patch flux.

        The optional local map retains its graph for the independent local
        objective. Default inference and attention output formats are unchanged.
        """
        patch_embeddings, attention_weights = self._encode_patch_embeddings(
            x, return_attention
        )
        _, sxr_std = sxr_norm
        patch_outputs = self._patch_mean_from_embeddings(
            patch_embeddings, sxr_norm, images=x, return_local=True
        )
        patch_logits, patch_flux_raw, local_patch_flux_raw = patch_outputs
        global_flux_raw = patch_flux_raw.sum(dim=1, keepdim=True)
        mean_normalized = normalize_sxr(global_flux_raw.squeeze(-1), sxr_norm)

        if not self.uncertainty_enabled:
            if return_attention:
                outputs = (
                    global_flux_raw, mean_normalized,
                    attention_weights, patch_flux_raw,
                )
            else:
                outputs = global_flux_raw, mean_normalized, patch_flux_raw
            return outputs + (local_patch_flux_raw,) if return_local else outputs
        patch_variance_raw = self._patch_variance_from_logits(
            patch_logits, patch_flux_raw
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
            outputs = (
                global_flux_raw, mean_normalized, variance_normalized,
                attention_weights, patch_flux_raw, patch_variance_raw,
            )
        else:
            outputs = (
                global_flux_raw, mean_normalized, variance_normalized,
                patch_flux_raw, patch_variance_raw,
            )
        return outputs + (local_patch_flux_raw,) if return_local else outputs


class GaussianNLLViTLocal(pl.LightningModule):
    """Train the canonical FOXES patch model with calibrated uncertainty.

    The mean/backbone is optimized only by an optionally class-weighted Huber
    loss. A separate Gaussian NLL trains relative patch standard deviations
    from detached patch logits, so it cannot alter the spatial mean
    prediction.

    With ``model_kwargs.independent_local_global``, equally weighted Huber
    losses train the uncorrected local total and corrected global total.
    Global-loss gradients cannot reach the local model.

    The predicted variance is in normalized log10-SXR units. Use
    :meth:`predict_interval` to obtain intervals in physical W/m^2 units.
    """

    DEFAULT_SCHEDULER_KWARGS = {'T_max': 250, 'eta_min': 1e-7}
    predicts_uncertainty = True
    uncertainty_kind = 'gaussian'
    predicts_patch_uncertainty = True
    patch_uncertainty_semantics = 'variance_attribution'
    uncertainty_parameterization = 'logit_conditioned_relative_log10_patch_std'
    NETWORK_CLASS = GaussianVisionTransformerLocal
    NETWORK_UNCERTAINTY_KEYS = (
        'patch_flux_scale_multiplier',
        'max_abs_log10_patch_multiplier',
        'initial_relative_std',
        'min_log10_relative_std',
        'max_log10_relative_std',
        'uncertainty_hidden_dim',
    )

    def __init__(self, model_kwargs, sxr_norm, base_weights=None,
                 weight_decay=1e-5, scheduler_kwargs=None,
                 uncertainty_kwargs=None,
                 mean_parameterization='multiplier',
                 contrast_prior_kwargs=None):
        super().__init__()
        self._trainable_modules = ()
        self.lr = model_kwargs.get('learning_rate', model_kwargs.get('lr', 1e-4))
        self.save_hyperparameters()

        filtered_kwargs = dict(model_kwargs)
        filtered_kwargs.pop('learning_rate', None)
        filtered_kwargs.pop('lr', None)
        filtered_kwargs.pop('num_classes', None)
        uncertainty_kwargs = dict(uncertainty_kwargs or {})
        incompatible_uncertainty_keys = {
            'relative_std_floor',
            'relative_std_max',
            'patch_scale_floor_fraction',
            'initial_global_sigma_normalized',
            'min_log10_patch_std',
            'max_log10_patch_std',
        }
        configured_incompatible_keys = (
            incompatible_uncertainty_keys & set(uncertainty_kwargs)
        )
        if configured_incompatible_keys:
            raise ValueError(
                "Incompatible legacy patch uncertainty settings: "
                + ', '.join(sorted(configured_incompatible_keys))
            )
        self.uncertainty_enabled = bool(
            uncertainty_kwargs.pop('enabled', True)
        )
        configured_start_epoch = uncertainty_kwargs.pop('start_epoch', None)
        if configured_start_epoch is None:
            self.uncertainty_start_epoch = 0
        elif (
            isinstance(configured_start_epoch, bool)
            or not isinstance(configured_start_epoch, int)
            or configured_start_epoch < 0
        ):
            raise ValueError(
                "uncertainty.start_epoch must be a nonnegative integer or null"
            )
        else:
            self.uncertainty_start_epoch = configured_start_epoch
        self.predicts_uncertainty = self.uncertainty_enabled
        self.predicts_patch_uncertainty = self.uncertainty_enabled
        allowed_uncertainty_keys = set(self.NETWORK_UNCERTAINTY_KEYS) | {
            'huber_delta',
            'class_weighting',
            'class_weight_exponent',
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
            **filtered_kwargs,
            **network_uncertainty_kwargs,
            mean_parameterization=mean_parameterization,
            uncertainty_enabled=self.uncertainty_enabled,
        )
        self.mean_parameterization = self.model.mean_parameterization
        self.mean_head_initialization = self.model.mean_head_initialization
        contrast_prior_kwargs = dict(contrast_prior_kwargs or {})
        self.contrast_prior_enabled = bool(
            contrast_prior_kwargs.pop('enabled', False)
        )
        self.contrast_prior_weight = float(
            contrast_prior_kwargs.pop('weight', 0.0)
        )
        self.contrast_channel_indices = tuple(
            int(index) for index in contrast_prior_kwargs.pop(
                'channel_indices', ()
            )
        )
        self.contrast_minimum_sum = float(
            contrast_prior_kwargs.pop('minimum_contrast_sum', 1e-6)
        )
        if contrast_prior_kwargs:
            raise ValueError(
                "Unknown contrast_prior settings: "
                f"{sorted(contrast_prior_kwargs)}"
            )
        if self.contrast_prior_enabled:
            if self.mean_parameterization != 'positive':
                raise ValueError(
                    "The contrast prior is currently restricted to the "
                    "positive mean parameterization."
                )
            if self.contrast_prior_weight <= 0:
                raise ValueError("contrast_prior.weight must be positive")
            if not self.contrast_channel_indices:
                raise ValueError(
                    "contrast_prior.channel_indices must not be empty"
                )
            num_channels = int(filtered_kwargs.get('num_channels', 0))
            if (
                min(self.contrast_channel_indices) < 0
                or max(self.contrast_channel_indices) >= num_channels
            ):
                raise ValueError(
                    "contrast_prior.channel_indices contains an index outside "
                    f"the model's {num_channels} input channels"
                )
            if self.contrast_minimum_sum < 0:
                raise ValueError(
                    "contrast_prior.minimum_contrast_sum must be nonnegative"
                )
        self.hparams['mean_head_initialization'] = (
            self.mean_head_initialization
        )
        self.hparams['uncertainty_parameterization'] = (
            self.uncertainty_parameterization
        )
        self.hparams['initial_log10_relative_std'] = (
            self.model.initial_log10_relative_std
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
            'power_inverse_frequency',
        }:
            raise ValueError(
                "uncertainty.class_weighting must be 'none', "
                "'inverse_frequency', 'sqrt_inverse_frequency', "
                "or 'power_inverse_frequency'"
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
        # These weights are derived deterministically from the training split.
        self.register_buffer(
            'class_weight_values', weights, persistent=False,
        )

    def freeze_except(self, module_names):
        """Train selected modules while leaving the pretrained feature path fixed."""
        if not module_names:
            raise ValueError('At least one trainable module is required')
        modules = tuple(self.get_submodule(name) for name in module_names)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in modules:
            module.requires_grad_(True)
        self._trainable_modules = tuple(module_names)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and self._trainable_modules:
            # Frozen dropout must stay inactive during head-only training.
            self.model.eval()
            for name in self._trainable_modules:
                self.get_submodule(name).train()
        return self

    def on_load_checkpoint(self, checkpoint):
        checkpoint_hparams = checkpoint.get('hyper_parameters', {})
        checkpoint_parameterization = checkpoint_hparams.get(
            'mean_parameterization'
        )
        if (
            checkpoint_parameterization is not None
            and checkpoint_parameterization != self.mean_parameterization
        ):
            raise RuntimeError(
                'Checkpoint mean_parameterization '
                f'{checkpoint_parameterization!r} does not match configured '
                f'{self.mean_parameterization!r}.'
            )
        checkpoint_uncertainty_parameterization = checkpoint_hparams.get(
            'uncertainty_parameterization'
        )
        state_dict = checkpoint.get('state_dict', {})
        has_uncertainty_head = any(
            'patch_uncertainty_head' in key for key in state_dict
        )
        if (
            has_uncertainty_head
            and checkpoint_uncertainty_parameterization
            != self.uncertainty_parameterization
        ):
            raise RuntimeError(
                "Checkpoint uncertainty parameterization is missing or "
                "incompatible with the logit-conditioned relative-standard-"
                "deviation model."
            )

    def forward(self, x, return_attention=False):
        """Return Gaussian outputs, or deterministic outputs when disabled."""
        outputs = self.model(x, self.sxr_norm, return_attention=return_attention)
        if not self.uncertainty_enabled:
            if return_attention:
                raw, _, attention, patches = outputs
                return raw, attention, patches
            raw, _, patches = outputs
            return raw, patches
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
        if not self.uncertainty_enabled:
            raise RuntimeError("predict_interval requires uncertainty.enabled=true")
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

    def _hot_channel_contrast(self, images):
        """Return nonnegative per-patch contrast for configured hot channels."""
        if images.ndim != 4:
            raise ValueError(
                f"Expected channel-last image batch, got shape {images.shape}"
            )
        patch_size = int(self.model.patch_size)
        batch_size, height, width, _ = images.shape
        if height % patch_size or width % patch_size:
            raise ValueError(
                "Image height and width must be divisible by patch size"
            )
        selected = images[..., self.contrast_channel_indices].float()
        grid_h, grid_w = height // patch_size, width // patch_size
        patch_brightness = selected.reshape(
            batch_size, grid_h, patch_size,
            grid_w, patch_size, len(self.contrast_channel_indices),
        ).mean(dim=(2, 4))
        patch_brightness = patch_brightness.reshape(
            batch_size, grid_h * grid_w,
            len(self.contrast_channel_indices),
        )
        channel_medians = patch_brightness.median(
            dim=1, keepdim=True
        ).values
        return torch.relu(patch_brightness - channel_medians).mean(dim=-1)

    def _contrast_alignment_loss(self, images, patch_flux_raw):
        """Align positive patch-flux fractions with 94/131 spatial contrast."""
        contrast = self._hot_channel_contrast(images)
        if contrast.shape != patch_flux_raw.shape:
            raise ValueError(
                "Contrast and patch-flux grids differ: "
                f"{contrast.shape} versus {patch_flux_raw.shape}"
            )
        contrast_sum = contrast.sum(dim=1, keepdim=True)
        flux_sum = patch_flux_raw.sum(dim=1, keepdim=True)
        contrast_fraction = contrast / contrast_sum.clamp_min(1e-12)
        flux_fraction = patch_flux_raw.float() / flux_sum.float().clamp_min(1e-20)
        per_sample = 1.0 - F.cosine_similarity(
            flux_fraction, contrast_fraction, dim=1, eps=1e-12
        )
        valid = contrast_sum.squeeze(1) > self.contrast_minimum_sum
        return (
            (per_sample * valid.to(per_sample.dtype)).sum()
            / valid.sum().clamp_min(1).to(per_sample.dtype)
        )

    @staticmethod
    def _uncertainty_nll_loss(mean, target, variance):
        """Train variance without allowing NLL gradients into the mean."""
        return F.gaussian_nll_loss(
            mean.detach(), target, variance, full=True, reduction='mean',
        )

    def _uncertainty_active_for_epoch(self, epoch):
        return (
            self.uncertainty_enabled
            and epoch >= self.uncertainty_start_epoch
        )

    def _uncertainty_training_is_active(self):
        trainer = getattr(self, '_trainer', None)
        epoch = trainer.current_epoch if trainer is not None else 0
        return self._uncertainty_active_for_epoch(epoch)

    def _calculate_loss(self, batch, mode):
        images, target_norm = batch
        independent = self.model.independent_local_global
        outputs = self.model(
            images, self.sxr_norm, return_attention=False,
            return_local=independent,
        )
        if independent:
            outputs, local_patch_flux_raw = outputs[:-1], outputs[-1]
            local_mean_norm = normalize_sxr(
                local_patch_flux_raw.sum(dim=1), self.sxr_norm,
            )
        if self.uncertainty_enabled:
            (
                raw_prediction, mean_norm, variance_norm,
                patch_flux_raw, patch_variance_raw,
            ) = outputs
        else:
            raw_prediction, mean_norm, patch_flux_raw = outputs
            variance_norm = patch_variance_raw = None
        target_norm = target_norm.reshape_as(mean_norm)
        target_raw = unnormalize_sxr(target_norm, self.sxr_norm)

        global_mean_loss = self._weighted_huber_loss(
            mean_norm, target_norm, target_raw,
        )
        if independent:
            local_mean_loss = self._weighted_huber_loss(
                local_mean_norm, target_norm, target_raw,
            )
            # Weight 1 for each objective; do not halve either gradient.
            mean_loss = local_mean_loss + global_mean_loss
        else:
            mean_loss = global_mean_loss
        contrast_loss = (
            self._contrast_alignment_loss(
                images, local_patch_flux_raw if independent else patch_flux_raw,
            )
            if self.contrast_prior_enabled
            else mean_loss.new_zeros(())
        )
        mean_objective = (
            mean_loss + self.contrast_prior_weight * contrast_loss
        )
        uncertainty_active = self._uncertainty_training_is_active()
        uncertainty_loss = (
            self._uncertainty_nll_loss(
                mean_norm, target_norm, variance_norm,
            )
            if self.uncertainty_enabled else None
        )
        loss = (
            mean_objective + uncertainty_loss
            if uncertainty_active else mean_objective
        )
        mse = F.mse_loss(mean_norm, target_norm)
        mae = F.l1_loss(mean_norm, target_norm)

        prefix = mode
        on_step = mode == 'train'
        if self.uncertainty_enabled:
            sigma = torch.sqrt(variance_norm)
            sigma_dex = sigma * self.sxr_norm[1]
            standardized = torch.abs(target_norm - mean_norm) / sigma
            coverage_68 = (standardized <= 1.0).float().mean()
            coverage_95 = (standardized <= 1.96).float().mean()
            self.log(f'{prefix}/nll', uncertainty_loss, on_step=on_step,
                     on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            self.log(
                f'{prefix}/uncertainty_nll', uncertainty_loss,
                on_step=on_step, on_epoch=True,
                logger=True, sync_dist=True,
            )
            self.log(
                f'{prefix}/uncertainty_active',
                float(uncertainty_active), on_step=on_step, on_epoch=True,
                logger=True, sync_dist=True,
            )
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
        self.log(
            f'{prefix}/mean_loss', mean_loss, on_step=on_step, on_epoch=True,
            logger=True, sync_dist=True,
        )
        if independent:
            for name, value in (
                ('local_mean_loss', local_mean_loss),
                ('global_mean_loss', global_mean_loss),
                ('local_mse', F.mse_loss(local_mean_norm, target_norm)),
                ('local_mae', F.l1_loss(local_mean_norm, target_norm)),
            ):
                self.log(
                    f'{prefix}/{name}', value, on_step=on_step, on_epoch=True,
                    logger=True, sync_dist=True,
                )
        if self.contrast_prior_enabled:
            self.log(
                f'{prefix}/contrast_loss', contrast_loss,
                on_step=on_step, on_epoch=True,
                logger=True, sync_dist=True,
            )
            self.log(
                f'{prefix}/weighted_contrast_loss',
                self.contrast_prior_weight * contrast_loss,
                on_step=on_step, on_epoch=True,
                logger=True, sync_dist=True,
            )
        self.log(f'{prefix}/total_loss', loss, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log(f'{prefix}/mse', mse, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log(f'{prefix}/mae', mae, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        if self.model.global_patch_adjustment or self.model.patch_global_context is not None:
            self.log(
                f'{prefix}/global_patch_adjustment_dex',
                self.model.last_log10_global_patch_adjustment.abs().mean(),
                on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
            )
        if self.model.patch_global_context is not None:
            self.log(
                f'{prefix}/patch_global_context_abs_dex',
                self.model.last_log10_patch_global_adjustment.abs().mean(),
                on_step=on_step, on_epoch=True, logger=True, sync_dist=True,
            )
        if mode == 'val':
            self.log('val_total_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
        if mode == 'train':
            learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/learning_rate', learning_rate, on_step=True,
                     on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        if mode == 'val':
            result = {
                'loss': loss.detach(),
                'mean_norm': mean_norm.detach(),
                'target_norm': target_norm.detach(),
                'target_raw': target_raw.detach(),
                'prediction_raw': raw_prediction.squeeze(-1).detach(),
            }
            if independent:
                result.update({
                    'local_mean_norm': local_mean_norm.detach(),
                    'local_prediction_raw': (
                        local_patch_flux_raw.sum(dim=1).detach()
                    ),
                })
            if self.uncertainty_enabled:
                result.update({
                    'nll': uncertainty_loss.detach(),
                    'variance_norm': variance_norm.detach(),
                })
            return result
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, 'test')

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError('No trainable parameters remain')
        optimizer = torch.optim.AdamW(
            trainable, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, **self.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1,
                'name': 'learning_rate',
            },
        }
