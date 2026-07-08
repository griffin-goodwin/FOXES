from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def normalize_sxr(unnormalized_values, sxr_norm):
    """Convert from unnormalized to normalized space"""
    log_values = torch.log10(unnormalized_values + 1e-8)
    normalized = (log_values - float(sxr_norm[0].item())) / float(sxr_norm[1].item())
    return normalized


def unnormalize_sxr(normalized_values, sxr_norm):
    return 10 ** (normalized_values * float(sxr_norm[1].item()) + float(sxr_norm[0].item())) - 1e-8


class ViTLocal(pl.LightningModule):
    """
    Parameters
    ----------
    model_kwargs : dict
        Forwarded to VisionTransformerLocal (embed_dim, num_heads, mask_mode, ...).
    sxr_norm : np.ndarray
        (mean, std) used to log-normalize SXR targets.
    base_weights : dict, optional
        Per-class loss weights (see SXRRegressionDynamicLoss). If None, falls
        back to SXRRegressionDynamicLoss's built-in defaults, which were fit to
        the originally released dataset — pass real weights (e.g. from
        training/train.py's get_base_weights) when training on different data.
    weight_decay : float
        AdamW weight decay.
    scheduler_kwargs : dict, optional
        Passed to CosineAnnealingWarmRestarts (T_0, T_mult, eta_min). Defaults
        match the released model's training run.
    loss_kwargs : dict, optional
        Forwarded to SXRRegressionDynamicLoss (window_size, huber_delta,
        adaptive_multipliers) — see that class for defaults.
    diagnostic_every_n_steps : int
        How often (in training steps) to log the adaptive loss's per-class
        multipliers to the logger.
    """

    DEFAULT_SCHEDULER_KWARGS = {'T_0': 250, 'T_mult': 2, 'eta_min': 1e-7}

    def __init__(self, model_kwargs, sxr_norm, base_weights=None, weight_decay=1e-5,
                 scheduler_kwargs=None, loss_kwargs=None, diagnostic_every_n_steps=200):
        super().__init__()
        self.model_kwargs = model_kwargs
        self.lr = model_kwargs.get('learning_rate', model_kwargs.get('lr', 1e-4))
        self.save_hyperparameters()
        filtered_kwargs = dict(model_kwargs)
        filtered_kwargs.pop('learning_rate', None)
        filtered_kwargs.pop('lr', None)
        filtered_kwargs.pop('num_classes', None)
        self.model = VisionTransformerLocal(**filtered_kwargs)
        self.base_weights = base_weights
        self.weight_decay = weight_decay
        self.scheduler_kwargs = {**self.DEFAULT_SCHEDULER_KWARGS, **(scheduler_kwargs or {})}
        self.diagnostic_every_n_steps = diagnostic_every_n_steps
        self.adaptive_loss = SXRRegressionDynamicLoss(base_weights=self.base_weights, **(loss_kwargs or {}))
        self.huber_delta = self.adaptive_loss.huber_delta
        self.sxr_norm = sxr_norm

    def forward(self, x, return_attention=True):
        return self.model(x, self.sxr_norm, return_attention=return_attention)

    def forward_for_callback(self, x, return_attention=True):
        return self.model(x, self.sxr_norm, return_attention=return_attention)

    def configure_optimizers(self):
        # Use AdamW with weight decay for better regularization
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = CosineAnnealingWarmRestarts(optimizer, **self.scheduler_kwargs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'learning_rate'
            }
        }

    def _calculate_loss(self, batch, mode="train"):
        imgs, sxr = batch
        raw_preds, raw_patch_contributions = self.model(imgs, self.sxr_norm)
        raw_preds_squeezed = torch.squeeze(raw_preds)
        sxr_un = unnormalize_sxr(sxr, self.sxr_norm)

        norm_preds_squeezed = normalize_sxr(raw_preds_squeezed, self.sxr_norm)
        # Use adaptive rare event loss
        loss, weights = self.adaptive_loss.calculate_loss(
            norm_preds_squeezed, sxr, sxr_un
        )

        # Also calculate huber loss for logging
        huber_loss = F.huber_loss(norm_preds_squeezed, sxr, delta=self.huber_delta)
        mse_loss = F.mse_loss(norm_preds_squeezed, sxr)
        mae_loss = F.l1_loss(norm_preds_squeezed, sxr)
        rmse_loss = torch.sqrt(mse_loss)

        # Log adaptation info
        if mode == "train":
            # Always log learning rate (every step)
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/learning_rate', current_lr, on_step=True, on_epoch=False,
                     prog_bar=True, logger=True, sync_dist=True)
            self.log("train/total_loss", loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            self.log("train/huber_loss", huber_loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            self.log("train/mse_loss", mse_loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            self.log("train/mae_loss", mae_loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            self.log("train/rmse_loss", rmse_loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            # Detailed diagnostics only every N steps
            if self.global_step % self.diagnostic_every_n_steps == 0:
                multipliers = self.adaptive_loss.get_current_multipliers()
                for key, value in multipliers.items():
                    self.log(f"adaptive/{key}", value, on_step=True, on_epoch=False, sync_dist=True)

        if mode == "val":
            # Validation: typically only log epoch aggregates
            multipliers = self.adaptive_loss.get_current_multipliers()
            for key, value in multipliers.items():
                self.log(f"val/adaptive/{key}", value, on_step=False, on_epoch=True)
            self.log("val_total_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_huber_loss", huber_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)
            self.log("val_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_mae_loss", mae_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_rmse_loss", rmse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class VisionTransformerLocal(nn.Module):
    def __init__(
            self,
            embed_dim,
            hidden_dim,
            num_channels,
            num_heads,
            num_layers,
            patch_size,
            num_patches,
            dropout,
            mask_mode='inverted',
            local_window=9,

    ):
        """Vision Transformer that outputs flux contributions per patch.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            patch_size: Number of pixels that the patches have per dimension
            num_patches: Maximum number of patches an image can have
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding
            mask_mode: Self-attention masking. 'inverted' (default) reproduces the
                      original released model exactly; 'local' is true local
                      attention; 'none' is full global attention. See
                      InvertedAttentionBlock.
            local_window: Side length (in patches) of the local neighbourhood used
                      by the 'inverted' and 'local' masks.

        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)

        self.mask_mode = mask_mode
        self.local_window = local_window
        self.transformer_blocks = nn.ModuleList([
            InvertedAttentionBlock(embed_dim, hidden_dim, num_heads, num_patches,
                                   dropout=dropout, local_window=local_window, mask_mode=mask_mode)
            for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings - using 2D positional encoding for local attention
        # No CLS token needed for local attention architecture
        self.grid_h = int(math.sqrt(num_patches))
        self.grid_w = int(math.sqrt(num_patches))
        self.pos_embedding_2d = nn.Parameter(torch.randn(1, self.grid_h, self.grid_w, embed_dim))

    def forward(self, x, sxr_norm, return_attention=False):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add positional encoding (no CLS token for local attention)
        x = self._add_2d_positional_encoding(x)

        # Apply Transformer blocks
        x = self.dropout(x)
        x = x.transpose(0, 1)  # [T, B, embed_dim]

        attention_weights = []
        for block in self.transformer_blocks:
            if return_attention:
                x, attn_weights = block(x, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                x = block(x)

        patch_embeddings = x.transpose(0, 1)  # [B, num_patches, embed_dim]
        patch_logits = self.mlp_head(patch_embeddings).squeeze(-1)  # normalized log predictions [B, num_patches]

        # --- Convert to raw SXR ---
        mean, std = sxr_norm  # in log10 space
        patch_flux_raw = torch.clamp(10 ** (patch_logits * std + mean) - 1e-8, min=1e-15, max=1)

        # Sum over patches for raw global flux
        global_flux_raw = patch_flux_raw.sum(dim=1, keepdim=True)

        # Ensure global flux is never zero (add small epsilon if needed)
        global_flux_raw = torch.clamp(global_flux_raw, min=1e-15)

        if return_attention:
            return global_flux_raw, attention_weights, patch_flux_raw
        else:
            return global_flux_raw, patch_flux_raw

    def _add_2d_positional_encoding(self, x):
        """Add learned 2D positional encoding to patch embeddings"""
        B, T, embed_dim = x.shape
        num_patches = T  # All tokens are patches (no CLS token)

        # Reshape patches to 2D grid: [B, grid_h, grid_w, embed_dim]
        patch_embeddings = x.reshape(B, self.grid_h, self.grid_w, embed_dim)

        # Add learned 2D positional encoding
        # Broadcasting: [B, grid_h, grid_w, embed_dim] + [1, grid_h, grid_w, embed_dim]
        patch_embeddings = patch_embeddings + self.pos_embedding_2d

        # Reshape back to sequence format: [B, num_patches, embed_dim]
        x = patch_embeddings.reshape(B, num_patches, embed_dim)

        return x

    def forward_for_callback(self, x, return_attention=True):
        """Forward method compatible with AttentionMapCallback"""
        global_flux_raw, attention_weights, patch_flux_raw = self.forward(x, return_attention=return_attention)
        # Callback expects (outputs, attention_weights, _)
        return global_flux_raw, attention_weights

    def set_mask_mode(self, mask_mode, local_window=None):
        """Override the attention mask in every block (e.g. to ablate a loaded
        checkpoint). Normal checkpoint loading keeps each block's saved mask."""
        self.mask_mode = mask_mode
        if local_window is not None:
            self.local_window = local_window
        for block in self.transformer_blocks:
            block.set_mask_mode(mask_mode, local_window=local_window)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """Attention Block.

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network

        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False):
        inp_x = self.layer_norm_1(x)

        if return_attention:
            attn_output, attn_weights = self.attn(inp_x, inp_x, inp_x, average_attn_weights=False)
            x = x + attn_output
            x = x + self.linear(self.layer_norm_2(x))
            return x, attn_weights
        else:
            attn_output = self.attn(inp_x, inp_x, inp_x)[0]
            x = x + attn_output
            x = x + self.linear(self.layer_norm_2(x))
            return x


class InvertedAttentionBlock(nn.Module):
    """Transformer block whose self-attention can be masked in three ways.

    mask_mode (passed through from the model config):
      'inverted' - ORIGINAL FOXES behaviour and the default, so released
                   checkpoints reproduce exactly. The boolean mask marks LOCAL
                   pairs as True, and nn.MultiheadAttention treats True as
                   "blocked", so nearby patches are blocked and every patch
                   attends only to DISTANT patches. (This is the flipped
                   local-attention syntax the model was actually trained with.)
      'local'    - Correct local attention: block everything OUTSIDE the local
                   window, so each patch attends only to its neighbours.
      'none'     - No mask at all; full global attention.

    The mask is registered as a PERSISTENT buffer, so it travels inside the
    checkpoint. Loading restores the exact mask a model was trained with -- the
    original inverted mask, or any hand-edited mask from past experiments --
    regardless of the mask_mode passed at construction. mask_mode/local_window
    only decide the mask for a *fresh* model; to deliberately change the mask of
    an already-loaded checkpoint (e.g. for an ablation), call set_mask_mode().
    """

    VALID_MASK_MODES = ('inverted', 'local', 'none')

    def __init__(self, embed_dim, hidden_dim, num_heads, num_patches, dropout=0.0,
                 local_window=9, mask_mode='inverted'):
        super().__init__()
        if mask_mode not in self.VALID_MASK_MODES:
            raise ValueError('mask_mode must be one of %s, got %r'
                             % (self.VALID_MASK_MODES, mask_mode))
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.local_window = local_window
        self.num_patches = num_patches
        self.mask_mode = mask_mode
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Persistent: the mask is saved with the weights so every checkpoint
        # reproduces exactly what it trained with. None => no masking ('none').
        self.register_buffer('attention_mask', self._build_attention_mask())

    def _build_attention_mask(self):
        """Boolean attn_mask for nn.MultiheadAttention (True == position blocked).
        """
        if self.mask_mode == 'none':
            return None
        grid_size = int(math.sqrt(self.num_patches))
        idx = torch.arange(self.num_patches)
        rows = (idx // grid_size).to(torch.int16)
        cols = (idx % grid_size).to(torch.int16)
        r = self.local_window // 2
        # local[i, j] True when patches i and j are within the local window.
        local = (((rows[:, None] - rows[None, :]).abs() <= r) &
                 ((cols[:, None] - cols[None, :]).abs() <= r))
        # 'inverted' blocks the local neighbourhood (original); 'local' blocks its
        # complement so attention stays inside the neighbourhood.
        return local if self.mask_mode == 'inverted' else ~local

    def set_mask_mode(self, mask_mode, local_window=None):
        """Rebuild the attention mask in place, overriding whatever is currently
        set (including a mask restored from a checkpoint).

        Use this to deliberately change a trained model's mask -- e.g. to ablate a
        released checkpoint under 'none' or 'local'. Loading a checkpoint normally
        keeps its own saved mask; this is the explicit opt-out.
        """
        if mask_mode not in self.VALID_MASK_MODES:
            raise ValueError('mask_mode must be one of %s, got %r'
                             % (self.VALID_MASK_MODES, mask_mode))
        self.mask_mode = mask_mode
        if local_window is not None:
            self.local_window = local_window
        mask = self._build_attention_mask()
        if mask is not None:
            try:
                mask = mask.to(self.attention_mask.device)
            except AttributeError:  # current buffer is None ('none' mode)
                mask = mask.to(next(self.parameters()).device)
        self.attention_mask = mask

    def forward(self, x, return_attention=False):
        inp_x = self.layer_norm_1(x)

        if return_attention:
            # attention_mask is None for mask_mode='none' -> standard full attention.
            attn_output, attn_weights = self.attn(
                inp_x, inp_x, inp_x,
                attn_mask=self.attention_mask,
                average_attn_weights=False
            )
            x = x + attn_output
            x = x + self.linear(self.layer_norm_2(x))
            return x, attn_weights
        else:
            attn_output = self.attn(
                inp_x, inp_x, inp_x,
                attn_mask=self.attention_mask
            )[0]
            x = x + attn_output
            x = x + self.linear(self.layer_norm_2(x))
            return x


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Args:
        x: Tensor representing the image of shape [B, H, W, C]
        patch_size: Number of pixels per dimension of the patches (integer)
        flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    x = x.permute(0, 3, 1, 2)
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x


class SXRRegressionDynamicLoss:
    """
    Huber loss with per-class weights (quiet/C/M/X) that adapt based on each
    class's recent vs. overall running error — classes that are currently
    performing worse than their own history get up-weighted.

    Parameters
    ----------
    window_size : int
        Max length of each class's rolling error history.
    base_weights : dict, optional
        Static per-class weight, multiplied by the adaptive multiplier below.
        If None, falls back to DEFAULT_BASE_WEIGHTS (fit to the originally
        released dataset — pass real weights fit to your own data instead,
        e.g. via training/train.py's get_base_weights).
    adaptive_multipliers : dict, optional
        Per-class {max_multiplier, min_multiplier, sensitivity, min_samples,
        recent_window} controlling how much recent performance can move a
        class's weight. Merged over DEFAULT_ADAPTIVE_MULTIPLIERS.
    huber_delta : float
        Delta for the underlying Huber loss.
    """

    # GOES flare-class flux boundaries (W/m^2) — a fixed physical definition,
    # not a training knob, so this is intentionally not a constructor param.
    CLASS_THRESHOLDS = {'c': 1e-6, 'm': 1e-5, 'x': 1e-4}

    DEFAULT_BASE_WEIGHTS = {
        'quiet': 6.643528005464481,
        'c_class': 1.626986450317832,
        'm_class': 4.724573441010383,
        'x_class': 43.13137472283814,
    }

    DEFAULT_ADAPTIVE_MULTIPLIERS = {
        'quiet':   {'max_multiplier': 1.5, 'min_multiplier': 0.6, 'sensitivity': 0.05, 'min_samples': 2500, 'recent_window': 800},
        'c_class': {'max_multiplier': 2.0, 'min_multiplier': 0.7, 'sensitivity': 0.08, 'min_samples': 2500, 'recent_window': 800},
        'm_class': {'max_multiplier': 5.0, 'min_multiplier': 0.8, 'sensitivity': 0.1,  'min_samples': 1500, 'recent_window': 500},
        'x_class': {'max_multiplier': 8.0, 'min_multiplier': 0.8, 'sensitivity': 0.12, 'min_samples': 1000, 'recent_window': 300},
    }

    def __init__(self, window_size=15000, base_weights=None, adaptive_multipliers=None, huber_delta=0.3):
        self.c_threshold = self.CLASS_THRESHOLDS['c']
        self.m_threshold = self.CLASS_THRESHOLDS['m']
        self.x_threshold = self.CLASS_THRESHOLDS['x']
        self.huber_delta = huber_delta

        self.window_size = window_size
        self.quiet_errors = deque(maxlen=window_size)
        self.c_errors = deque(maxlen=window_size)
        self.m_errors = deque(maxlen=window_size)
        self.x_errors = deque(maxlen=window_size)

        self.base_weights = base_weights if base_weights is not None else dict(self.DEFAULT_BASE_WEIGHTS)
        self.multiplier_params = {**self.DEFAULT_ADAPTIVE_MULTIPLIERS, **(adaptive_multipliers or {})}

    def calculate_loss(self, preds_norm, sxr_norm, sxr_un):
        base_loss = F.huber_loss(preds_norm, sxr_norm, delta=self.huber_delta, reduction='none')
        weights = self._get_adaptive_weights(sxr_un)
        self._update_tracking(sxr_un, sxr_norm, preds_norm)
        weighted_loss = base_loss * weights
        loss = weighted_loss.mean()
        return loss, weights

    def _class_weight(self, sxrclass, error_history):
        mult = self._get_performance_multiplier(error_history, sxrclass)
        return self.base_weights[sxrclass] * mult

    def _get_adaptive_weights(self, sxr_un):
        device = sxr_un.device

        quiet_weight = self._class_weight('quiet', self.quiet_errors)
        c_weight = self._class_weight('c_class', self.c_errors)
        m_weight = self._class_weight('m_class', self.m_errors)
        x_weight = self._class_weight('x_class', self.x_errors)

        weights = torch.ones_like(sxr_un, device=device)
        weights = torch.where(sxr_un < self.c_threshold, quiet_weight, weights)
        weights = torch.where((sxr_un >= self.c_threshold) & (sxr_un < self.m_threshold), c_weight, weights)
        weights = torch.where((sxr_un >= self.m_threshold) & (sxr_un < self.x_threshold), m_weight, weights)
        weights = torch.where(sxr_un >= self.x_threshold, x_weight, weights)

        # Normalize so mean weight ~1.0 (optional, helps stability)
        mean_weight = torch.mean(weights)
        weights = weights / (mean_weight)

        return weights

    def _get_performance_multiplier(self, error_history, sxrclass):
        """Class-dependent performance multiplier: how much recent error deviates
        from this class's overall running error, mapped through an exponential
        and clipped to [min_multiplier, max_multiplier]."""
        params = self.multiplier_params[sxrclass]

        if len(error_history) < params['min_samples']:
            return 1.0

        recent = np.mean(list(error_history)[-params['recent_window']:])
        overall = np.mean(list(error_history))

        ratio = recent / overall
        multiplier = np.exp(params['sensitivity'] * (ratio - 1))
        return np.clip(multiplier, params['min_multiplier'], params['max_multiplier'])

    def _update_tracking(self, sxr_un, sxr_norm, preds_norm):
        sxr_un_np = sxr_un.detach().cpu().numpy()

        error = F.huber_loss(preds_norm, sxr_norm, delta=self.huber_delta, reduction='none')
        error = error.detach().cpu().numpy()

        quiet_mask = sxr_un_np < self.c_threshold
        if quiet_mask.sum() > 0:
            self.quiet_errors.append(float(np.mean(error[quiet_mask])))

        c_mask = (sxr_un_np >= self.c_threshold) & (sxr_un_np < self.m_threshold)
        if c_mask.sum() > 0:
            self.c_errors.append(float(np.mean(error[c_mask])))

        m_mask = (sxr_un_np >= self.m_threshold) & (sxr_un_np < self.x_threshold)
        if m_mask.sum() > 0:
            self.m_errors.append(float(np.mean(error[m_mask])))

        x_mask = sxr_un_np >= self.x_threshold
        if x_mask.sum() > 0:
            self.x_errors.append(float(np.mean(error[x_mask])))

    def get_current_multipliers(self):
        """Current per-class multipliers/weights for logging — uses the exact
        same _get_performance_multiplier params as the loss itself, so what's
        logged always matches what was actually applied."""
        quiet_mult = self._get_performance_multiplier(self.quiet_errors, 'quiet')
        c_mult = self._get_performance_multiplier(self.c_errors, 'c_class')
        m_mult = self._get_performance_multiplier(self.m_errors, 'm_class')
        x_mult = self._get_performance_multiplier(self.x_errors, 'x_class')
        return {
            'quiet_mult': quiet_mult,
            'c_mult': c_mult,
            'm_mult': m_mult,
            'x_mult': x_mult,
            'quiet_count': len(self.quiet_errors),
            'c_count': len(self.c_errors),
            'm_count': len(self.m_errors),
            'x_count': len(self.x_errors),
            'quiet_error': np.mean(self.quiet_errors) if self.quiet_errors else 0.0,
            'c_error': np.mean(self.c_errors) if self.c_errors else 0.0,
            'm_error': np.mean(self.m_errors) if self.m_errors else 0.0,
            'x_error': np.mean(self.x_errors) if self.x_errors else 0.0,
            'quiet_weight': self.base_weights['quiet'] * quiet_mult,
            'c_weight': self.base_weights['c_class'] * c_mult,
            'm_weight': self.base_weights['m_class'] * m_mult,
            'x_weight': self.base_weights['x_class'] * x_mult,
        }