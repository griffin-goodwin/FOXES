"""Independent local flux, optional global correction, and detached uncertainty.

Local patch flux = a flat equal-share baseline times a learned log10 multiplier.
Local Huber trains only the local model. Global Huber trains only the correction
head. Gaussian NLL trains only patch uncertainty. All sums are in physical flux.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MeanSquaredError
from forecasting.model import (
    SXR_LOG_OFFSET, SXRRegressionDynamicLoss, VisionTransformerLocal,
    img_to_patch, normalize_sxr, unnormalize_sxr,
)


class PatchGlobalContext(nn.Module):
    """Local queries attend to pooled local features and original AIA tokens."""

    def __init__(self, embed_dim, grid_size=32, attention_dim=128,
                 num_heads=4, dropout=0.0, max_abs_adjustment=1.5,
                 num_layers=1, hidden_dim=None,
                 mlp_depth=1):
        super().__init__()
        hidden_dim = attention_dim if hidden_dim is None else hidden_dim
        for name, value in dict(grid_size=grid_size, attention_dim=attention_dim,
                                num_heads=num_heads, num_layers=num_layers,
                                hidden_dim=hidden_dim, mlp_depth=mlp_depth).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f'global head {name} must be a positive integer')
        if attention_dim % num_heads:
            raise ValueError('global attention_dim must be divisible by num_heads')
        if not math.isfinite(max_abs_adjustment) or max_abs_adjustment <= 0:
            raise ValueError('global max_abs_adjustment must be finite and positive')
        if not 0 <= dropout < 1:
            raise ValueError('global dropout must be in [0, 1)')
        self.grid_size = grid_size
        self.max_abs_adjustment = max_abs_adjustment
        self.query_projection = nn.Linear(embed_dim, attention_dim)
        self.memory_projection = nn.Linear(2 * embed_dim, attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.extra_attention = nn.ModuleList([
            nn.MultiheadAttention(attention_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers - 1)
        ])
        layers = [nn.LayerNorm(embed_dim + attention_dim),
                  nn.Linear(embed_dim + attention_dim, hidden_dim), nn.GELU()]
        for _ in range(mlp_depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.output = nn.Sequential(*layers)
        # Keep PyTorch default random initialization for every Linear layer.

    def forward(self, local_features, raw_features):
        batch, patches, channels = local_features.shape
        side = math.isqrt(patches)
        if side * side != patches or self.grid_size > side:
            raise ValueError('global grid_size must fit the square patch grid')
        def pool(features):
            grid = features.transpose(1, 2).reshape(batch, channels, side, side)
            return F.adaptive_avg_pool2d(grid, (self.grid_size, self.grid_size)).flatten(2).transpose(1, 2)
        memory = self.memory_projection(torch.cat((pool(local_features), pool(raw_features)), dim=-1))
        context, _ = self.attention(self.query_projection(local_features), memory, memory, need_weights=False)
        for attention in self.extra_attention:
            update, _ = attention(context, memory, memory, need_weights=False)
            context = context + update
        return self.max_abs_adjustment * torch.tanh(
            self.output(torch.cat((local_features, context), dim=-1)).squeeze(-1)
        )


class GaussianVisionTransformerLocal(VisionTransformerLocal):
    """One local multiplier architecture with optional independent heads."""

    def __init__(self, *args, uncertainty_enabled=True,
                 max_abs_log10_patch_multiplier=8.0,
                 patch_global_context=None, initial_relative_std=1.0,
                 min_log10_relative_std=-6.0, max_log10_relative_std=6.0,
                 uncertainty_hidden_dim=16, **kwargs):
        kwargs.setdefault('mask_mode', 'local')
        kwargs.setdefault('local_window', 3)
        if kwargs['mask_mode'] != 'local':
            raise ValueError('The multiplier backbone requires local attention')
        super().__init__(*args, **kwargs)
        self.uncertainty_enabled = bool(uncertainty_enabled)
        if not math.isfinite(max_abs_log10_patch_multiplier) or max_abs_log10_patch_multiplier <= 0:
            raise ValueError('max_abs_log10_patch_multiplier must be finite and positive')
        self.max_abs_log10_patch_multiplier = float(max_abs_log10_patch_multiplier)
        context = dict(patch_global_context or {})
        self.patch_global_context_enabled = bool(context.pop('enabled', True))
        self.patch_global_context = None
        if self.patch_global_context_enabled:
            context.setdefault('grid_size', min(32, self.grid_h))
            self.patch_global_context = PatchGlobalContext(self.mlp_head[-1].in_features, **context)
        self.last_log10_patch_global_adjustment = None
        self.initial_relative_std = float(initial_relative_std)
        self.min_log10_relative_std = float(min_log10_relative_std)
        self.max_log10_relative_std = float(max_log10_relative_std)
        self.uncertainty_hidden_dim = uncertainty_hidden_dim
        self.patch_uncertainty_head = None
        self.initial_log10_relative_std = None
        if self.uncertainty_enabled:
            if not math.isfinite(self.initial_relative_std) or self.initial_relative_std <= 0:
                raise ValueError('initial_relative_std must be finite and positive')
            self.initial_log10_relative_std = math.log10(self.initial_relative_std)
            if not (math.isfinite(self.min_log10_relative_std) and math.isfinite(self.max_log10_relative_std)
                    and self.min_log10_relative_std < self.initial_log10_relative_std < self.max_log10_relative_std):
                raise ValueError('Initial uncertainty must lie inside finite log10 bounds')
            if type(uncertainty_hidden_dim) is not int or uncertainty_hidden_dim <= 0:
                raise ValueError('uncertainty_hidden_dim must be a positive integer')
            self.patch_uncertainty_head = nn.Sequential(nn.Linear(1, uncertainty_hidden_dim),
                                                       nn.GELU(), nn.Linear(uncertainty_hidden_dim, 1))
            nn.init.zeros_(self.patch_uncertainty_head[-1].weight)
            nn.init.constant_(self.patch_uncertainty_head[-1].bias, self.initial_log10_relative_std)

    def _patch_flux_from_logits(self, patch_logits, sxr_norm):
        baseline = (10 ** sxr_norm[0].float() - SXR_LOG_OFFSET).clamp_min(torch.finfo(torch.float32).tiny)
        logits = patch_logits.float().clamp(-self.max_abs_log10_patch_multiplier, self.max_abs_log10_patch_multiplier)
        return baseline / patch_logits.shape[1] * torch.pow(10.0, logits)

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

    def _patch_mean_from_embeddings(self, patch_embeddings, sxr_norm, images=None, return_local=False):
        logits = self.mlp_head(patch_embeddings).squeeze(-1)
        local_flux = self._patch_flux_from_logits(logits, sxr_norm)
        corrected_flux = local_flux
        if self.patch_global_context is not None:
            if images is None:
                raise ValueError('The global head requires the original AIA image')
            # Frozen inputs, trainable correction: global loss cannot change local features.
            with torch.no_grad():
                raw = self.input_layer(img_to_patch(images, self.patch_size))
                raw = raw + self.pos_embedding_2d.reshape(1, -1, raw.shape[-1])
            adjustment = self.patch_global_context(patch_embeddings.detach(), raw.detach())
            self.last_log10_patch_global_adjustment = adjustment.detach()
            corrected_flux = local_flux.detach() * torch.pow(10.0, adjustment.float())
        result = (logits, corrected_flux)
        return result + (local_flux,) if return_local else result

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
    """Separate local/global Huber objectives and an independent Gaussian NLL."""
    DEFAULT_SCHEDULER_KWARGS = {'T_max': 50, 'eta_min': 1e-7}
    predicts_uncertainty = True
    uncertainty_kind = 'gaussian'
    predicts_patch_uncertainty = True
    patch_uncertainty_semantics = 'variance_attribution'
    uncertainty_parameterization = 'logit_conditioned_relative_log10_patch_std'
    NETWORK_CLASS = GaussianVisionTransformerLocal
    NETWORK_UNCERTAINTY_KEYS = ('max_abs_log10_patch_multiplier', 'initial_relative_std',
        'min_log10_relative_std', 'max_log10_relative_std', 'uncertainty_hidden_dim')

    def __init__(self, model_kwargs, sxr_norm, weight_decay=1e-5,
                 scheduler_kwargs=None, uncertainty_kwargs=None):
        super().__init__()
        self.lr = model_kwargs.get('learning_rate', model_kwargs.get('lr', 5e-5))
        self.save_hyperparameters()
        architecture = dict(model_kwargs)
        for key in ('learning_rate', 'lr', 'num_classes'):
            architecture.pop(key, None)
        u = dict(uncertainty_kwargs or {})
        allowed = set(self.NETWORK_UNCERTAINTY_KEYS) | {
            'enabled', 'huber_delta', 'class_weighting',
            'class_weight_exponent', 'class_weights'}
        if set(u) - allowed:
            raise ValueError(f'Unknown uncertainty settings: {sorted(set(u) - allowed)}')
        self.uncertainty_enabled = bool(u.get('enabled', True))
        self.predicts_uncertainty = self.predicts_patch_uncertainty = self.uncertainty_enabled
        self.model = self.NETWORK_CLASS(**architecture,
            uncertainty_enabled=self.uncertainty_enabled,
            **{k:u[k] for k in self.NETWORK_UNCERTAINTY_KEYS if k in u})
        self.hparams['architecture_version'] = 'independent_multiplier_v1'
        self.register_buffer('sxr_norm', torch.as_tensor(sxr_norm, dtype=torch.float32))
        if self.sxr_norm.shape != (2,) or not torch.isfinite(self.sxr_norm).all() or self.sxr_norm[1] <= 0:
            raise ValueError('sxr_norm must be finite [mean, positive std]')
        self.weight_decay = weight_decay
        self.scheduler_kwargs = {**self.DEFAULT_SCHEDULER_KWARGS, **(scheduler_kwargs or {})}
        self.huber_delta = float(u.get('huber_delta', 0.15))
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError('huber_delta must be finite and positive')
        self.class_weighting = u.get('class_weighting', 'none')
        if self.class_weighting not in {'none', 'inverse_frequency', 'sqrt_inverse_frequency', 'power_inverse_frequency'}:
            raise ValueError('Unknown class_weighting')
        names = ('quiet', 'c_class', 'm_class', 'x_class')
        weights = torch.ones(4)
        if self.class_weighting != 'none':
            configured = u.get('class_weights')
            if not isinstance(configured, dict) or set(configured) != set(names):
                raise ValueError('class-weighted Huber requires training-derived class_weights')
            weights = torch.tensor([configured[k] for k in names], dtype=torch.float32)
            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('class_weights must be finite and positive')
        self.register_buffer('class_weight_values', weights, persistent=False)
        self.rmse_metrics = nn.ModuleDict({
            f'{mode}_{branch}': MeanSquaredError(squared=False)
            for mode in ('train', 'val', 'test') for branch in ('local', 'final')
        })

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('hyper_parameters', {}).get('architecture_version') != 'independent_multiplier_v1':
            raise ValueError('Checkpoint predates the simplified architecture; use its original code to load it')

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

    @staticmethod
    def _uncertainty_nll_loss(mean, target, variance):
        """Train variance without allowing NLL gradients into the mean."""
        return F.gaussian_nll_loss(
            mean.detach(), target, variance, full=True, reduction='mean',
        )

    def _calculate_loss(self, batch, mode):
        images, target = batch
        outputs = self.model(images, self.sxr_norm, return_local=True)
        local_flux = outputs[-1]
        if self.uncertainty_enabled:
            raw, mean, variance, patches, patch_variance = outputs[:-1]
        else:
            raw, mean, patches = outputs[:-1]
            variance = None
        target = target.reshape_as(mean)
        target_raw = unnormalize_sxr(target, self.sxr_norm)
        local_mean = normalize_sxr(local_flux.sum(dim=1), self.sxr_norm)
        local_huber = self._weighted_huber_loss(local_mean, target, target_raw)
        global_huber = (self._weighted_huber_loss(mean, target, target_raw)
                        if self.model.patch_global_context_enabled else local_huber.new_zeros(()))
        mean_loss = local_huber + global_huber
        nll = (self._uncertainty_nll_loss(mean, target, variance)
               if self.uncertainty_enabled else mean_loss.new_zeros(()))
        loss = mean_loss + nll
        metrics = {
            'local_huber': local_huber, 'global_huber': global_huber,
            'mean_loss': mean_loss, 'nll': nll, 'total_loss': loss,
            'local_mse': F.mse_loss(local_mean, target),
            'local_mae': F.l1_loss(local_mean, target),
            'mse': F.mse_loss(mean, target), 'mae': F.l1_loss(mean, target),
        }
        if self.uncertainty_enabled:
            sigma = variance.clamp_min(1e-12).sqrt()
            standardized = (target - mean).abs() / sigma
            metrics.update(mean_sigma=sigma.mean(), mean_sigma_dex=sigma.mean()*self.sxr_norm[1],
                           coverage_68=(standardized <= 1).float().mean(),
                           coverage_95=(standardized <= 1.96).float().mean())
        if self.model.patch_global_context_enabled:
            metrics['patch_global_context_abs_dex'] = self.model.last_log10_patch_global_adjustment.abs().mean()
        for key, value in metrics.items():
            self.log(f'{mode}/{key}', value, on_step=(mode == 'train'), on_epoch=True,
                     logger=True, sync_dist=True, batch_size=target.numel(),
                     prog_bar=(key == 'total_loss'))
        # Accumulate squared error over the entire epoch BEFORE taking its root.
        for branch, prediction, key in [('local', local_mean, 'local_rmse'), ('final', mean, 'rmse')]:
            metric = self.rmse_metrics[f'{mode}_{branch}']
            metric.update(prediction.detach(), target.detach())
            self.log(f'{mode}/{key}', metric, on_step=False, on_epoch=True, logger=True)
        if mode == 'train' and self._trainer is not None:
            self.log('train/learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'],
                     on_step=True, on_epoch=False, sync_dist=True)
        if mode == 'val':
            result = dict(loss=loss.detach(), mean_norm=mean.detach(), target_norm=target.detach(),
                          target_raw=target_raw.detach(), prediction_raw=raw.squeeze(-1).detach(),
                          local_mean_norm=local_mean.detach(), local_prediction_raw=local_flux.sum(dim=1).detach())
            if self.uncertainty_enabled:
                result.update(nll=nll.detach(), variance_norm=variance.detach())
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
