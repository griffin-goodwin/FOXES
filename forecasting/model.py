"""Core FOXES models: released deterministic and coupled single-query Gaussian."""

from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

SXR_LOG_OFFSET = 1e-8


def normalize_sxr(unnormalized_values, sxr_norm):
    """Convert physical flux using the FOXES ``log10(F + 1e-8)`` convention."""
    log_values = torch.log10(unnormalized_values + SXR_LOG_OFFSET)
    normalized = (log_values - float(sxr_norm[0].item())) / float(sxr_norm[1].item())
    return normalized


def unnormalize_sxr(normalized_values, sxr_norm):
    """Invert :func:`normalize_sxr` back to physical flux."""
    return (
        10 ** (
            normalized_values * float(sxr_norm[1].item())
            + float(sxr_norm[0].item())
        )
        - SXR_LOG_OFFSET
    )


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
        training/train.py's get_four_class_macro_weights) when training on different data.
    weight_decay : float
        AdamW weight decay.
    scheduler_kwargs : dict, optional
        Passed to CosineAnnealingLR (T_max, eta_min). ``T_max`` is measured in
        epochs because the scheduler is stepped once per epoch.
    loss_kwargs : dict, optional
        Forwarded to SXRRegressionDynamicLoss (window_size, huber_delta,
        adaptive_multipliers) — see that class for defaults.
    diagnostic_every_n_steps : int
        How often (in training steps) to log the adaptive loss's per-class
        multipliers to the logger.
    """

    DEFAULT_SCHEDULER_KWARGS = {'T_max': 250, 'eta_min': 1e-7}

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

    def forward(self, x, return_attention=False):
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

        scheduler = CosineAnnealingLR(optimizer, **self.scheduler_kwargs)

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

        if not torch.isfinite(raw_preds_squeezed).all():
            raise RuntimeError(
                f"Non-finite model output in {mode} step at global_step={self.global_step}: "
                f"raw_preds range=[{raw_preds_squeezed.min().item()}, {raw_preds_squeezed.max().item()}], "
                f"imgs range=[{imgs.min().item()}, {imgs.max().item()}], "
                f"sxr (normalized target) range=[{sxr.min().item()}, {sxr.max().item()}]. "
                "This means the model itself produced NaN/Inf before reaching the loss -- "
                "an activation blowup, not a data or loss-weighting issue."
            )

        sxr_un = unnormalize_sxr(sxr, self.sxr_norm)

        norm_preds_squeezed = normalize_sxr(raw_preds_squeezed, self.sxr_norm)
        # Use adaptive rare event loss. Only train batches should feed the
        # per-class error history the adaptive multipliers are computed from --
        # otherwise val/test performance silently leaks into training dynamics.
        loss, weights = self.adaptive_loss.calculate_loss(
            norm_preds_squeezed, sxr, sxr_un, update_tracking=(mode == "train")
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
        patch_flux_raw = torch.clamp(
            10 ** (patch_logits * std + mean) - SXR_LOG_OFFSET,
            min=1e-15,
            max=1
        )

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
            # Do not materialize the O(T^2) attention-weight tensor when the
            # caller only needs the transformed embeddings.  Besides saving
            # memory, need_weights=False lets PyTorch use its optimized scaled
            # dot-product attention implementation when available.
            attn_output = self.attn(
                inp_x, inp_x, inp_x, need_weights=False
            )[0]
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
            # Avoid computing and returning the O(T^2) attention weights on
            # the normal training/inference path.  Attention-map callers use
            # the explicit return_attention=True branch above.
            attn_output = self.attn(
                inp_x, inp_x, inp_x,
                attn_mask=self.attention_mask,
                need_weights=False,
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
        e.g. via training/train.py's get_four_class_macro_weights).
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

    def calculate_loss(self, preds_norm, sxr_norm, sxr_un, update_tracking=True):
        base_loss = F.huber_loss(preds_norm, sxr_norm, delta=self.huber_delta, reduction='none')
        weights = self._get_adaptive_weights(sxr_un)
        if update_tracking:
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

# Coupled single-query uncertainty model (shared by OG and depth variants).
class CoupledVisionTransformerLocal(VisionTransformerLocal):
    """Similar to the original FOXES model, but now starts with a local attention Transformer that outputs per-patch flux contributions, and adds a single global query that sees all patches and outputs a global correction to the sum of the local patch contributions. The global correction is initialized to zero, so the model starts with the same behavior as the original FOXES model. The global query can also be used to predict uncertainty in the global flux prediction. The predictions (starting on the local scale) are initialized as the average flux spread across the patches, the logits then scale each patch's contibution up or down... the global query can then adjust each patch by a certain dex by taking into account the global context.
    """

    def __init__(self, *args, mean_parameterization='positive',
                 uncertainty_enabled=True, patch_flux_scale_multiplier=1.0,
                 max_abs_log10_patch_multiplier=8.0,
                 global_patch_adjustment=False,
                 max_abs_log10_global_adjustment=0.3,
                 global_patch_head_init_std=0.0,
                 global_attention_heads=None,
                 global_head_hidden_dim=None,
                 global_attention_dropout=None,
                 independent_local_global=False,
                 patch_global_context=None,
                 initial_relative_std=1.0,
                 min_log10_relative_std=-4.0,
                 max_log10_relative_std=4.0,
                 uncertainty_hidden_dim=16, **kwargs):
        super().__init__(*args, **kwargs)

        if mean_parameterization != 'positive':
            raise ValueError("Coupled single-query model requires mean_parameterization='positive'")
        self.mean_parameterization = 'positive'
        self.mean_head_initialization = 'pytorch_linear_default'
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
        # Accept inactive legacy flags so saved OG hyperparameters still load.
        if independent_local_global or patch_global_context:
            raise ValueError('Coupled single-query model does not use a separate global head')
        if not global_patch_adjustment or global_patch_head_init_std != 0:
            raise ValueError('Coupled single-query model requires a zero-initialized global correction head')
        self.independent_local_global = False
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
        local_heads = self.transformer_blocks[0].num_heads
        self.global_attention_heads = (
            local_heads if global_attention_heads is None
            else global_attention_heads
        )
        self.global_head_hidden_dim = (
            embed_dim // 2 if global_head_hidden_dim is None
            else global_head_hidden_dim
        )
        self.global_attention_dropout = (
            self.dropout.p if global_attention_dropout is None
            else global_attention_dropout
        )
        if (isinstance(self.global_attention_heads, bool)
                or not isinstance(self.global_attention_heads, int)
                or self.global_attention_heads <= 0
                or embed_dim % self.global_attention_heads != 0):
            raise ValueError(
                'global_attention_heads must be a positive divisor of embed_dim'
            )
        if (isinstance(self.global_head_hidden_dim, bool)
                or not isinstance(self.global_head_hidden_dim, int)
                or self.global_head_hidden_dim <= 0):
            raise ValueError('global_head_hidden_dim must be a positive integer')
        if (isinstance(self.global_attention_dropout, bool)
                or not isinstance(self.global_attention_dropout, (int, float))
                or not 0 <= self.global_attention_dropout < 1):
            raise ValueError('global_attention_dropout must be in [0, 1)')
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
            self.global_attention_heads,
            dropout=self.global_attention_dropout,
            batch_first=True,
        )
        self.global_patch_head = nn.Sequential(
            nn.LayerNorm(3 * embed_dim),
            nn.Linear(3 * embed_dim, self.global_head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.global_head_hidden_dim, 1),
        )
        nn.init.zeros_(self.global_patch_head[-1].weight)
        nn.init.zeros_(self.global_patch_head[-1].bias)

        self.patch_global_context = None  # Checkpoint inspection compatibility.

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
        typical_global_flux = torch.clamp(
            10 ** sxr_norm[0] - SXR_LOG_OFFSET,
            min=torch.finfo(torch.float32).tiny,
        )
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
        if images is None:
            raise ValueError('Global correction requires original AIA images')
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

        The optional local map is diagnostic; training uses corrected flux.
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


class CoupledGaussianViTLocal(pl.LightningModule):
    """Train the canonical FOXES patch model with calibrated uncertainty.

    The mean/backbone is optimized only by an optionally class-weighted Huber
    loss. A separate Gaussian NLL trains relative patch standard deviations
    from detached patch logits, so it cannot alter the spatial mean
    prediction.

    The final corrected-flux Huber loss updates both local and global layers.

    The predicted variance is in normalized log10-SXR units. Use
    :meth:`predict_interval` to obtain intervals in physical W/m^2 units.
    """

    DEFAULT_SCHEDULER_KWARGS = {'T_max': 250, 'eta_min': 1e-7}
    predicts_uncertainty = True
    uncertainty_kind = 'gaussian'
    predicts_patch_uncertainty = True
    patch_uncertainty_semantics = 'variance_attribution'
    uncertainty_parameterization = 'logit_conditioned_relative_log10_patch_std'
    NETWORK_CLASS = CoupledVisionTransformerLocal
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
                 mean_parameterization='positive'):
        super().__init__()
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

    def predict_interval(self, x, z=1.96):
        """Return point, lower, and upper physical-flux estimates."""
        if not self.uncertainty_enabled:
            raise RuntimeError("predict_interval requires uncertainty.enabled=true")
        raw, variance_norm, _, _ = self.forward(x, return_attention=False)
        mean_norm = normalize_sxr(raw.squeeze(-1), self.sxr_norm)
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
        outputs = self.model(
            images, self.sxr_norm, return_attention=False,
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
        mean_loss = global_mean_loss
        mean_objective = mean_loss
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
        self.log(f'{prefix}/total_loss', loss, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log(f'{prefix}/mse', mse, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log(f'{prefix}/mae', mae, on_step=on_step, on_epoch=True,
                 logger=True, sync_dist=True)
        if self.model.global_patch_adjustment:
            self.log(
                f'{prefix}/global_patch_adjustment_dex',
                self.model.last_log10_global_patch_adjustment.abs().mean(),
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


# Imports created while developing the coupled model remain valid for scripts
# and any serialized objects that refer to those historical class names.
CoupledSingleQueryVisionTransformerLocal = CoupledVisionTransformerLocal
CoupledSingleQueryGaussianNLLViTLocal = CoupledGaussianViTLocal
