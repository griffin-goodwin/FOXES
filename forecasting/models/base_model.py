import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from collections import deque
import numpy as np

# Import adaptive loss and normalization functions
from .vit_patch_model import SXRRegressionDynamicLoss, normalize_sxr, unnormalize_sxr

class BaseModel(LightningModule):
    def __init__(self, model, loss_func, lr, sxr_norm=None, weight_decay=1e-5, 
                 cosine_restart_T0=50, cosine_restart_Tmult=2, cosine_eta_min=1e-7):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.lr = lr
        self.sxr_norm = sxr_norm
        self.weight_decay = weight_decay
        self.cosine_restart_T0 = cosine_restart_T0
        self.cosine_restart_Tmult = cosine_restart_Tmult
        self.cosine_eta_min = cosine_eta_min
        
        # Initialize adaptive loss if sxr_norm is provided
        if sxr_norm is not None:
            self.adaptive_loss = SXRRegressionDynamicLoss(window_size=1500)
        else:
            self.adaptive_loss = None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cosine_restart_T0,
            T_mult=self.cosine_restart_Tmult,
            eta_min=self.cosine_eta_min,
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

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        
        # Use adaptive loss if available and sxr_norm is provided
        if self.adaptive_loss is not None and self.sxr_norm is not None:
            raw_preds_squeezed = torch.squeeze(pred)
            target_un = unnormalize_sxr(target, self.sxr_norm)
            norm_preds_squeezed = normalize_sxr(raw_preds_squeezed, self.sxr_norm)
            loss, weights = self.adaptive_loss.calculate_loss(
                norm_preds_squeezed, target, target_un, raw_preds_squeezed
            )
        else:
            loss = self.loss_func(torch.squeeze(pred), target)
        
        self.log('train_loss', loss)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        
        # Use adaptive loss if available and sxr_norm is provided
        if self.adaptive_loss is not None and self.sxr_norm is not None:
            raw_preds_squeezed = torch.squeeze(pred)
            target_un = unnormalize_sxr(target, self.sxr_norm)
            norm_preds_squeezed = normalize_sxr(raw_preds_squeezed, self.sxr_norm)
            loss, weights = self.adaptive_loss.calculate_loss(
                norm_preds_squeezed, target, target_un, raw_preds_squeezed
            )
        else:
            loss = self.loss_func(torch.squeeze(pred), target)
        
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        
        # Use adaptive loss if available and sxr_norm is provided
        if self.adaptive_loss is not None and self.sxr_norm is not None:
            raw_preds_squeezed = torch.squeeze(pred)
            target_un = unnormalize_sxr(target, self.sxr_norm)
            norm_preds_squeezed = normalize_sxr(raw_preds_squeezed, self.sxr_norm)
            loss, weights = self.adaptive_loss.calculate_loss(
                norm_preds_squeezed, target, target_un, raw_preds_squeezed
            )
        else:
            loss = self.loss_func(torch.squeeze(pred), target)
        
        self.log('test_loss', loss)
        return loss