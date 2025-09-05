import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class BaseModel(LightningModule):
    def __init__(self, model, loss_func, lr, weight_decay=1e-5, 
                 cosine_restart_T0=50, cosine_restart_Tmult=2, cosine_eta_min=1e-7):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.cosine_restart_T0 = int(cosine_restart_T0)
        self.cosine_restart_Tmult = int(cosine_restart_Tmult)
        self.cosine_eta_min = float(cosine_eta_min)

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
        loss = self.loss_func(torch.squeeze(pred), target)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.loss_func(torch.squeeze(pred), target)
        
        self.log('val_total_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.loss_func(torch.squeeze(pred), target)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss