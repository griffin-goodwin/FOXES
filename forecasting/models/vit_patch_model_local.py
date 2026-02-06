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
    def __init__(self, model_kwargs, sxr_norm, base_weights=None):
        super().__init__()
        self.model_kwargs = model_kwargs
        self.lr = model_kwargs['lr']
        self.save_hyperparameters()
        filtered_kwargs = dict(model_kwargs)
        filtered_kwargs.pop('lr', None)
        filtered_kwargs.pop('num_classes', None)
        self.model = VisionTransformerLocal(**filtered_kwargs)
        # Set the base weights based on the number of samples in each class within training data
        self.base_weights = base_weights
        self.adaptive_loss = SXRRegressionDynamicLoss(window_size=15000, base_weights=self.base_weights)
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
            weight_decay=0.00001,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=250,
            T_mult=2,
            eta_min=1e-7
        )

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
        huber_loss = F.huber_loss(norm_preds_squeezed, sxr, delta=.3)
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
            # Detailed diagnostics only every 200 steps
            if self.global_step % 200 == 0:
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

        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)

        self.transformer_blocks = nn.ModuleList([
            InvertedAttentionBlock(embed_dim, hidden_dim, num_heads, num_patches, dropout=dropout)
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
    def __init__(self, embed_dim, hidden_dim, num_heads, num_patches, dropout=0.0, local_window=9):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.local_window = local_window
        self.num_patches = num_patches
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

        # Pre-compute attention mask for local interactions
        self.register_buffer('attention_mask', self._create_inverted_attention_mask())

    def _create_inverted_attention_mask(self):
        """Create attention mask for local interactions only"""
        # This creates a mask where only distant patches can attend to each other

        num_patches = self.num_patches
        grid_size = int(math.sqrt(num_patches))

        # Create mask for patches only: [num_patches, num_patches]
        mask = torch.zeros(num_patches, num_patches)

        # Patches can only attend to nearby patches
        for i in range(num_patches):
            row_i, col_i = i // grid_size, i % grid_size
            for j in range(num_patches):
                row_j, col_j = j // grid_size, j % grid_size
                # Only allow attention if patches are within local_window distance
                if abs(row_i - row_j) <= self.local_window // 2 and abs(col_i - col_j) <= self.local_window // 2:
                    mask[i, j] = 1

        return mask.bool()

    def forward(self, x, return_attention=False):
        inp_x = self.layer_norm_1(x)

        if return_attention:
            # Apply local attention mask
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
    def __init__(self, window_size=15000, base_weights=None):
        self.c_threshold = 1e-6
        self.m_threshold = 1e-5
        self.x_threshold = 1e-4

        self.window_size = window_size
        self.quiet_errors = deque(maxlen=window_size)
        self.c_errors = deque(maxlen=window_size)
        self.m_errors = deque(maxlen=window_size)
        self.x_errors = deque(maxlen=window_size)

        # Calculate the base weights based on the number of samples in each class within training data
        if base_weights is None:
            self.base_weights = self._get_base_weights()
        else:
            self.base_weights = base_weights

    def _get_base_weights(self):
        # Base weights based on the number of samples in each class within training data
        return {
            'quiet': 6.643528005464481,    # Increase from current value
            'c_class': 1.626986450317832,  # Keep as baseline
            'm_class': 4.724573441010383,  # Maintain M-class focus
            'x_class': 43.13137472283814  # Maintain X-class focus
        }

    def calculate_loss(self, preds_norm, sxr_norm, sxr_un):
        base_loss = F.huber_loss(preds_norm, sxr_norm, delta=.3, reduction='none')
        weights = self._get_adaptive_weights(sxr_un)
        self._update_tracking(sxr_un, sxr_norm, preds_norm)
        weighted_loss = base_loss * weights
        loss = weighted_loss.mean()
        return loss, weights

    def _get_adaptive_weights(self, sxr_un):
        device = sxr_un.device

        # Get continuous multipliers per class with custom params
        quiet_mult = self._get_performance_multiplier(
            self.quiet_errors, max_multiplier=1.5, min_multiplier=0.6, sensitivity=0.05, sxrclass='quiet'
        )
        c_mult = self._get_performance_multiplier(
            self.c_errors, max_multiplier=2, min_multiplier=0.7, sensitivity=0.08, sxrclass='c_class'
        )
        m_mult = self._get_performance_multiplier(
            self.m_errors, max_multiplier=5.0, min_multiplier=0.8, sensitivity=0.1, sxrclass='m_class'
        )
        x_mult = self._get_performance_multiplier(
            self.x_errors, max_multiplier=8.0, min_multiplier=0.8, sensitivity=0.12, sxrclass='x_class'
        )

        quiet_weight = self.base_weights['quiet'] * quiet_mult
        c_weight = self.base_weights['c_class'] * c_mult
        m_weight = self.base_weights['m_class'] * m_mult
        x_weight = self.base_weights['x_class'] * x_mult

        weights = torch.ones_like(sxr_un, device=device)
        weights = torch.where(sxr_un < self.c_threshold, quiet_weight, weights)
        weights = torch.where((sxr_un >= self.c_threshold) & (sxr_un < self.m_threshold), c_weight, weights)
        weights = torch.where((sxr_un >= self.m_threshold) & (sxr_un < self.x_threshold), m_weight, weights)
        weights = torch.where(sxr_un >= self.x_threshold, x_weight, weights)

        # Normalize so mean weight ~1.0 (optional, helps stability)
        mean_weight = torch.mean(weights)
        weights = weights / (mean_weight)

        # Save for logging
        self.current_multipliers = {
            'quiet_mult': quiet_mult,
            'c_mult': c_mult,
            'm_mult': m_mult,
            'x_mult': x_mult,
            'quiet_weight': quiet_weight,
            'c_weight': c_weight,
            'm_weight': m_weight,
            'x_weight': x_weight
        }

        return weights

    def _get_performance_multiplier(self, error_history, max_multiplier=10.0, min_multiplier=0.5, sensitivity=3.0,
                                    sxrclass='quiet'):
        """Class-dependent performance multiplier"""

        class_params = {
            'quiet': {'min_samples': 2500, 'recent_window': 800},
            'c_class': {'min_samples': 2500, 'recent_window': 800},
            'm_class': {'min_samples': 1500, 'recent_window': 500},
            'x_class': {'min_samples': 1000, 'recent_window': 300}
        }

        if len(error_history) < class_params[sxrclass]['min_samples']:
            return 1.0

        recent_window = class_params[sxrclass]['recent_window']
        recent = np.mean(list(error_history)[-recent_window:])
        overall = np.mean(list(error_history))

        ratio = recent / overall
        multiplier = np.exp(sensitivity * (ratio - 1))
        return np.clip(multiplier, min_multiplier, max_multiplier)

    def _update_tracking(self, sxr_un, sxr_norm, preds_norm):
        sxr_un_np = sxr_un.detach().cpu().numpy()

        # Huber loss
        error = F.huber_loss(preds_norm, sxr_norm, delta=.3, reduction='none')
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
        """Get current performance multipliers for logging"""
        return {
            'quiet_mult': self._get_performance_multiplier(
                self.quiet_errors, max_multiplier=1.5, min_multiplier=0.6, sensitivity=0.2, sxrclass='quiet'
            ),
            'c_mult': self._get_performance_multiplier(
                self.c_errors, max_multiplier=2, min_multiplier=0.7, sensitivity=0.3, sxrclass='c_class'
            ),
            'm_mult': self._get_performance_multiplier(
                self.m_errors, max_multiplier=5.0, min_multiplier=0.8, sensitivity=0.8, sxrclass='m_class'
            ),
            'x_mult': self._get_performance_multiplier(
                self.x_errors, max_multiplier=8.0, min_multiplier=0.8, sensitivity=1.0, sxrclass='x_class'
            ),
            'quiet_count': len(self.quiet_errors),
            'c_count': len(self.c_errors),
            'm_count': len(self.m_errors),
            'x_count': len(self.x_errors),
            'quiet_error': np.mean(self.quiet_errors) if self.quiet_errors else 0.0,
            'c_error': np.mean(self.c_errors) if self.c_errors else 0.0,
            'm_error': np.mean(self.m_errors) if self.m_errors else 0.0,
            'x_error': np.mean(self.x_errors) if self.x_errors else 0.0,
            'quiet_weight': getattr(self, 'current_multipliers', {}).get('quiet_weight', 0.0),
            'c_weight': getattr(self, 'current_multipliers', {}).get('c_weight', 0.0),
            'm_weight': getattr(self, 'current_multipliers', {}).get('m_weight', 0.0),
            'x_weight': getattr(self, 'current_multipliers', {}).get('x_weight', 0.0)
        }