from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def normalize_sxr(unnormalized_values, sxr_norm):
    """Convert from unnormalized to normalized space"""
    log_values = torch.log10(unnormalized_values + 1e-8)
    normalized = (log_values - float(sxr_norm[0].item())) / float(sxr_norm[1].item())
    return normalized

def unnormalize_sxr(normalized_values, sxr_norm):
    return 10 ** (normalized_values * float(sxr_norm[1].item()) + float(sxr_norm[0].item())) - 1e-8

class CNNPatch(pl.LightningModule):
    def __init__(self, model_kwargs, sxr_norm, base_weights=None):
        super().__init__()
        self.model_kwargs = model_kwargs
        self.lr = model_kwargs['lr']
        self.save_hyperparameters()
        filtered_kwargs = dict(model_kwargs)
        self.model = SXRCNNModel(**filtered_kwargs)
        #Set the base weights based on the number of samples in each class within training data
        self.base_weights = base_weights
        self.adaptive_loss = SXRRegressionDynamicLoss(window_size=15000, base_weights=self.base_weights)
        self.sxr_norm = sxr_norm

    def forward(self, x, return_attention=True):
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
            T_0=50,  # Restart every 50 epochs
            T_mult=2,  # Double the cycle length after each restart
            eta_min=1e-7  # Minimum learning rate
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

        #Also calculate huber loss for logging
        huber_loss = F.huber_loss(norm_preds_squeezed, sxr, delta=.3)

        # Log adaptation info
        if mode == "train":
            # Always log learning rate (every step)
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=True, on_epoch=False,
                     prog_bar=True, logger=True, sync_dist=True)

            self.log("train_total_loss", loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            self.log("train_huber_loss", huber_loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

            # Detailed diagnostics only every 200 steps
            if self.global_step % 200 == 0:
                multipliers = self.adaptive_loss.get_current_multipliers()
                for key, value in multipliers.items():
                    self.log(f"adaptive/{key}", value, on_step=True, on_epoch=False)

                self.log("adaptive/avg_weight", weights.mean(), on_step=True, on_epoch=False)
                self.log("adaptive/max_weight", weights.max(), on_step=True, on_epoch=False)

        if mode == "val":
            # Validation: typically only log epoch aggregates
            multipliers = self.adaptive_loss.get_current_multipliers()
            for key, value in multipliers.items():
                self.log(f"val/adaptive/{key}", value, on_step=False, on_epoch=True)
            self.log("val_total_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_huber_loss", huber_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

class SXRCNNModel(nn.Module):
    def __init__(
        self,
        num_channels,
        patch_size,
        num_patches,
        dropout=0.1,
        hidden_dim=256,
        **kwargs  # Accept other kwargs but don't use them
    ):
        """CNN model that processes patches independently for SXR prediction.
        
        Args:
            num_channels: Number of input channels
            patch_size: Size of each patch (assumed square)
            num_patches: Total number of patches
            dropout: Dropout rate
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.grid_h = int(math.sqrt(num_patches))
        self.grid_w = int(math.sqrt(num_patches))
        
        # CNN layers for processing each patch independently
        self.patch_processor = nn.Sequential(
            # First conv block
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Global average pooling to get single value per patch
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Final prediction head
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Optional: Add positional encoding for patches
        self.use_positional_encoding = True
        if self.use_positional_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
            self.pos_projection = nn.Linear(hidden_dim, 1)

    def forward(self, x, sxr_norm, return_attention=False):
        """
        Args:
            x: Input tensor of shape [B, H, W, C]
            sxr_norm: Normalization parameters
            return_attention: For compatibility (always returns None for attention)
        """
        B, H, W, C = x.shape
        
        # Convert to [B, C, H, W] for conv operations
        x = x.permute(0, 3, 1, 2)
        
        # Extract patches
        patch_size = self.patch_size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        # Reshape to process all patches at once
        # x: [B, C, H, W] -> [B, C, num_patches_h, patch_size, num_patches_w, patch_size]
        x = x.unfold(2, patch_size, patch_size).unfold(4, patch_size, patch_size)
        # x: [B, C, num_patches_h, patch_size, num_patches_w, patch_size]
        
        # Reshape to [B, num_patches, C, patch_size, patch_size]
        x = x.contiguous().view(B, C, num_patches_h * num_patches_w, patch_size, patch_size)
        x = x.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, patch_size, patch_size]
        x = x.contiguous().view(B * num_patches_h * num_patches_w, C, patch_size, patch_size)

        # Process ALL patches at once (vectorized!)
        patch_outputs = self.patch_processor(x)  # [B*num_patches, 1]

        # Reshape back to [B, num_patches]
        patch_logits = patch_outputs.view(B, num_patches_h * num_patches_w)
        
        # Optional: Add positional encoding
        if self.use_positional_encoding:
            pos_encoded = self.pos_embedding + patch_logits.unsqueeze(-1)  # [B, num_patches, hidden_dim]
            patch_logits = self.pos_projection(pos_encoded).squeeze(-1)  # [B, num_patches]
        
        # Convert to raw SXR
        mean, std = sxr_norm  # in log10 space
        patch_flux_raw = torch.clamp(10 ** (patch_logits * std + mean) - 1e-8, min=1e-15, max=1)
        
        # Sum over patches for global flux
        global_flux_raw = patch_flux_raw.sum(dim=1, keepdim=True)
        global_flux_raw = torch.clamp(global_flux_raw, min=1e-15)
        
        if return_attention:
            # Return None for attention weights (CNNs don't have attention)
            return global_flux_raw, None, patch_flux_raw
        else:
            return global_flux_raw, patch_flux_raw
    
    def forward_for_callback(self, x, return_attention=True):
        """Forward method compatible with AttentionMapCallback"""
        global_flux_raw, _, patch_flux_raw = self.forward(x, return_attention=return_attention)
        # Callback expects (outputs, attention_weights, _)
        return global_flux_raw, None  # No attention weights for CNN


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

        #Calculate the base weights based on the number of samples in each class within training data
        if base_weights is None:
            self.base_weights = self._get_base_weights()
        else:
            self.base_weights = base_weights

    def _get_base_weights(self):
        #Calculate the base weights based on the number of samples in each class within training data
        return {
            'quiet': 1.5,    # Increase from current value
            'c_class': 1.0,  # Keep as baseline
            'm_class': 8.0,  # Maintain M-class focus
            'x_class': 20.0  # Maintain X-class focus
        }

    def calculate_loss(self, preds_norm, sxr_norm, sxr_un):
        base_loss = F.huber_loss(preds_norm, sxr_norm, delta=.3, reduction='none')
        #base_loss = F.mse_loss(preds_norm, sxr_norm, reduction='none')
        weights = self._get_adaptive_weights(sxr_un)
        self._update_tracking(sxr_un, sxr_norm, preds_norm)
        weighted_loss = base_loss * weights
        loss = weighted_loss.mean()
        return loss, weights

    def _get_adaptive_weights(self, sxr_un):
        device = sxr_un.device

        # Get continuous multipliers per class with custom params
        quiet_mult = self._get_performance_multiplier(
            self.quiet_errors, max_multiplier=1.5, min_multiplier=0.6, sensitivity=0.05, sxrclass='quiet'  # Was 0.2
        )
        c_mult = self._get_performance_multiplier(
            self.c_errors, max_multiplier=2, min_multiplier=0.7, sensitivity=0.08, sxrclass='c_class'    # Was 0.3
        )
        m_mult = self._get_performance_multiplier(
            self.m_errors, max_multiplier=5.0, min_multiplier=0.8, sensitivity=0.1, sxrclass='m_class'   # Was 0.4
        )
        x_mult = self._get_performance_multiplier(
            self.x_errors, max_multiplier=8.0, min_multiplier=0.8, sensitivity=0.12, sxrclass='x_class'  # Was 0.5
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

        # Clamp extreme weights
        #weights = torch.clamp(weights, min=0.01, max=40.0)

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

    def _get_performance_multiplier(self, error_history, max_multiplier=10.0, min_multiplier=0.5, sensitivity=3.0, sxrclass='quiet'):
        """Class-dependent performance multiplier"""

        class_params = {
            'quiet': {'min_samples': 2500, 'recent_window': 800},
            'c_class': {'min_samples': 2500, 'recent_window': 800},
            'm_class': {'min_samples': 1500, 'recent_window': 500},
            'x_class': {'min_samples': 1000, 'recent_window': 300}
        }

        # target_errors = {
        #     'quiet': 0.15,
        #     'c_class': 0.08,
        #     'm_class': 0.05,
        #     'x_class': 0.05
        # }
        
        #target = target_errors[sxrclass]

        if len(error_history) < class_params[sxrclass]['min_samples']:
            return 1.0

        recent_window = class_params[sxrclass]['recent_window']
        recent = np.mean(list(error_history)[-recent_window:])
        overall = np.mean(list(error_history))

        # if overall < 1e-10:
        #     return 1.0

        ratio = recent / overall
        multiplier = np.exp(sensitivity * (ratio - 1))
        return np.clip(multiplier, min_multiplier, max_multiplier)


        
        # if len(error_history) < class_params[sxrclass]['min_samples']:
        #     return 1.0
        
        # recent = np.mean(list(error_history)[-class_params[sxrclass]['recent_window']:])
        
        # if recent > target:  # Not meeting target - increase weight
        #     excess_error = (recent - target) / target
        #     multiplier = 1.0 + sensitivity * excess_error
        # else:  # Meeting/exceeding target
        #     if sxrclass == 'quiet':
        #         # Can reduce quiet weight significantly
        #         multiplier = max(0.5, 1.0 - 0.5 * (target - recent) / target)
        #     else:
        #         # Keep important classes weighted well even when performing good
        #         multiplier = max(0.8, 1.0 - 0.2 * (target - recent) / target)
        
        # return np.clip(multiplier, min_multiplier, max_multiplier)

    def _update_tracking(self, sxr_un, sxr_norm, preds_norm):
        sxr_un_np = sxr_un.detach().cpu().numpy()

        #Huber loss
        error = F.huber_loss(preds_norm, sxr_norm, delta=.3, reduction='none')
        #error = F.mse_loss(preds_norm, sxr_norm, reduction='none')
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