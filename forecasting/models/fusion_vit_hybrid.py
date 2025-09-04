import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .vit_patch_model import VisionTransformer, SXRRegressionDynamicLoss, normalize_sxr, unnormalize_sxr
from .linear_and_hybrid import LinearIrradianceModel, HybridIrradianceModel


class FusionViTHybrid(pl.LightningModule):
    """End-to-end fused model: ViT for spatial patches + Linear/Hybrid for scalar.

    - ViT branch outputs per-patch raw flux and a ViT global (sum of patches).
    - Scalar branch (Linear or Hybrid) outputs a global scalar.
    - A learnable gate blends the two globals; the spatial map uses ViT's
      distribution but is calibrated to the fused/global prediction.

    Forward returns a 4-tuple compatible with existing inference utils:
        (global_fused, attention_weights, fused_patch_flux, global_fused)
    """

    def __init__(
        self,
        vit_kwargs: dict,
        scalar_branch: str,
        scalar_kwargs: dict,
        sxr_norm,
        lr: float = 1e-4,
        lambda_vit_to_target: float = 0.3,
        lambda_scalar_to_target: float = 0.1,
        use_attention: bool = True,
        learnable_gate: bool = True,
        gate_init_bias: float = 5.0,
        weight_decay: float = 1e-5,
        cosine_restart_T0: int = 50,
        cosine_restart_Tmult: int = 2,
        cosine_eta_min: float = 1e-7,
    ):
        super().__init__()

        # Save hyperparameters needed for checkpointing
        self.save_hyperparameters(ignore=["sxr_norm"])  # sxr_norm is a tensor/array

        # Branches: filter unsupported keys for VisionTransformer
        filtered_vit_kwargs = dict(vit_kwargs)
        filtered_vit_kwargs.pop('lr', None)
        filtered_vit_kwargs.pop('num_classes', None)
        self.vit = VisionTransformer(**filtered_vit_kwargs)

        if scalar_branch.lower() in ["linear", "lineairradiancemodel"]:
            self.scalar = LinearIrradianceModel(
                d_input=scalar_kwargs.get("d_input"),
                d_output=scalar_kwargs.get("d_output"),
                loss_func=scalar_kwargs.get("loss_func", nn.HuberLoss()),
                lr=scalar_kwargs.get("lr", lr),
            )
        elif scalar_branch.lower() in ["hybrid", "hybridirradiancemodel"]:
            self.scalar = HybridIrradianceModel(
                d_input=scalar_kwargs.get("d_input"),
                d_output=scalar_kwargs.get("d_output"),
                cnn_model=scalar_kwargs.get("cnn_model", "updated"),
                ln_model=scalar_kwargs.get("ln_model", True),
                ln_params=scalar_kwargs.get("ln_params", None),
                lr=scalar_kwargs.get("lr", lr),
                cnn_dp=scalar_kwargs.get("cnn_dp", 0.75),
                loss_func=scalar_kwargs.get("loss_func", nn.HuberLoss()),
            )
        else:
            raise ValueError(f"Unknown scalar_branch: {scalar_branch}")

        # Loss and normalization
        self.sxr_norm = sxr_norm
        self.adaptive_loss = SXRRegressionDynamicLoss(window_size=1500)

        # Gate: learnable scalar in [0,1] blending scalar vs vit global
        self.learnable_gate = learnable_gate
        if learnable_gate:
            self.gate_logit = nn.Parameter(torch.tensor(gate_init_bias, dtype=torch.float32))
        else:
            self.register_buffer("gate_logit", torch.tensor(gate_init_bias, dtype=torch.float32))

        # Optim params
        self.lr = lr
        self.weight_decay = weight_decay
        self.cosine_restart_T0 = cosine_restart_T0
        self.cosine_restart_Tmult = cosine_restart_Tmult
        self.cosine_eta_min = cosine_eta_min

        # Aux loss weights
        self.lambda_vit_to_target = lambda_vit_to_target
        self.lambda_scalar_to_target = lambda_scalar_to_target

        # Whether to compute/return attention
        self.use_attention = use_attention

    def forward(self, x, return_attention: bool = True):
        # ViT branch: returns different numbers of values based on return_attention
        vit_out = self.vit(x, self.sxr_norm, return_attention=(self.use_attention and return_attention))

        if self.use_attention and return_attention and len(vit_out) == 3:
            global_vit_raw, attention_weights, patch_flux_raw = vit_out
        else:
            global_vit_raw, patch_flux_raw = vit_out
            attention_weights = None

        # Scalar branch expects (B,H,W,C)
        global_scalar_raw = self.scalar(x)
        # Ensure positivity for SXR-like targets
        global_scalar_raw = F.softplus(global_scalar_raw)

        # Shapes: ensure tensors are shaped [B, 1]
        if global_vit_raw.dim() == 1:
            global_vit_raw = global_vit_raw.unsqueeze(-1)
        if global_scalar_raw.dim() == 1:
            global_scalar_raw = global_scalar_raw.unsqueeze(-1)

        # Patch weights from ViT distribution
        weights = patch_flux_raw / (global_vit_raw.clamp(min=1e-15))

        # Blend globals via sigmoid(gate_logit)
        gate = torch.sigmoid(self.gate_logit)
        global_fused = gate * global_scalar_raw + (1.0 - gate) * global_vit_raw
        # Avoid zeros/negatives before log normalization downstream
        global_fused = global_fused.clamp(min=1e-15)

        # Calibrated patch flux using fused global
        fused_patch_flux = global_fused * weights

        # Match inference API: (pred, attn, patch_flux, total_from_patches)
        return global_fused, attention_weights, fused_patch_flux, global_fused
    def forward_for_callback(self, x, return_attention: bool = True):
        """Forward method compatible with AttentionMapCallback"""
        global_fused, attention_weights, fused_patch_flux, _ = self.forward(x, return_attention)
        # Callback expects (outputs, attention_weights, _)
        return attention_weights
    def _calc_losses(self, imgs, sxr):
        # Forward
        global_fused, attention_weights, fused_patch_flux, _ = self(imgs, return_attention=True)

        # Main adaptive loss on fused global
        raw_preds_squeezed = torch.squeeze(global_fused)
        sxr_un = unnormalize_sxr(sxr, self.sxr_norm)
        norm_preds_squeezed = normalize_sxr(raw_preds_squeezed, self.sxr_norm)
        main_loss, weights_adapt = self.adaptive_loss.calculate_loss(
            norm_preds_squeezed, sxr, sxr_un, raw_preds_squeezed
        )

        # Auxiliary consistency losses (vit and scalar heads individually)
        # Recompute heads without extra forward
        # Extract vit global by re-running vit without attention to save memory
        with torch.no_grad():
            vit_out = self.vit(imgs, self.sxr_norm, return_attention=False)
        global_vit_raw = vit_out[0]
        if global_vit_raw.dim() > 1:
            global_vit_raw = torch.squeeze(global_vit_raw)
        global_vit_raw = global_vit_raw.clamp(min=1e-15)
        vit_norm = normalize_sxr(global_vit_raw, self.sxr_norm)
        loss_vit = F.huber_loss(vit_norm, sxr)

        global_scalar_raw = self.scalar(imgs)
        global_scalar_raw = F.softplus(global_scalar_raw)
        if global_scalar_raw.dim() > 1:
            global_scalar_raw = torch.squeeze(global_scalar_raw)
        global_scalar_raw = global_scalar_raw.clamp(min=1e-15)
        scalar_norm = normalize_sxr(global_scalar_raw, self.sxr_norm)
        loss_scalar = F.huber_loss(scalar_norm, sxr)

        total_loss = main_loss \
            + self.lambda_vit_to_target * loss_vit \
            + self.lambda_scalar_to_target * loss_scalar

        return total_loss, {
            "main_loss": main_loss.detach(),
            "loss_vit": loss_vit.detach(),
            "loss_scalar": loss_scalar.detach(),
        }

    def training_step(self, batch, batch_idx):
        imgs, sxr = batch
        total_loss, logs = self._calc_losses(imgs, sxr)

        # Logs
        self.log("train_main_loss", logs["main_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_vit_loss", logs["loss_vit"], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_scalar_loss", logs["loss_scalar"], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        imgs, sxr = batch
        total_loss, logs = self._calc_losses(imgs, sxr)
        self.log("val_main_loss", logs["main_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        imgs, sxr = batch
        total_loss, _ = self._calc_losses(imgs, sxr)
        self.log("test_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

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


