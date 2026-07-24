"""
PyTorch Lightning callbacks for visualizing training progress: predicted vs.
true SXR flux, and Vision Transformer attention maps, logged to Weights & Biases.

Used by train.py — not meant to be run standalone.
"""
import random

import wandb
from pytorch_lightning import Callback
import matplotlib.pyplot as plt
import numpy as np
import torch

from forecasting.model import unnormalize_sxr


class ImagePredictionLogger_SXR(Callback):
    """
    PyTorch Lightning callback for logging AIA input images and corresponding
    true vs predicted Soft X-Ray (SXR) flux values to Weights & Biases (wandb).

    This helps monitor model performance across validation epochs by
    comparing predicted vs. ground-truth flare intensities.
    """

    def __init__(self, val_ds, num_samples, sxr_norm):
        """
        Initialize callback with the validation dataset and normalization parameters.

        Parameters
        ----------
        val_ds : Dataset
            Validation dataset to draw a fresh random sample from each epoch.
        num_samples : int
            Number of samples to draw per validation epoch.
        sxr_norm : np.ndarray
            Normalization statistics used to unnormalize predicted flux values.
        """
        super().__init__()
        self.val_ds = val_ds
        self.num_samples = num_samples
        self.sxr_norm = sxr_norm

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Log scatter plots comparing predicted and true SXR flux values,
        sampled randomly from the validation set, at the end of each epoch.
        """
        true_sxr = []
        pred_sxr = []

        n = min(self.num_samples, len(self.val_ds))
        indices = random.sample(range(len(self.val_ds)), n)
        data_samples = [self.val_ds[i] for i in indices]

        for aia, target in data_samples:
            aia = aia.to(pl_module.device).unsqueeze(0)
            # forward() always returns a tuple (global_flux_raw, ...); we only need the flux.
            pred, *_ = pl_module(aia, return_attention=False)
            pred_sxr.append(pred.item())
            true_sxr.append(target.item())

        true_unorm = unnormalize_sxr(np.array(true_sxr, dtype=np.float32), self.sxr_norm)
        pred_unnorm = unnormalize_sxr(np.array(pred_sxr, dtype=np.float32), self.sxr_norm)

        fig1 = self.plot_aia_sxr(true_unorm, pred_unnorm)
        trainer.logger.experiment.log({"Soft X-ray flux plots": wandb.Image(fig1)})
        plt.close(fig1)

    # Flare-class range: A-class starts at 1e-8 W/m^2, X10 is 10x the 1e-4 X-class threshold.
    AXIS_MIN = 1e-8
    AXIS_MAX = 1e-3

    def plot_aia_sxr(self, val_sxr, pred_sxr):
        """Log-log parity plot: predicted vs. true SXR flux, with a 1:1 reference line."""
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.plot([self.AXIS_MIN, self.AXIS_MAX], [self.AXIS_MIN, self.AXIS_MAX],
                color='gray', linestyle='--', linewidth=1, label='Perfect prediction')
        ax.scatter(val_sxr, pred_sxr, color='blue', alpha=0.7,s=10, label='Predictions')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(self.AXIS_MIN, self.AXIS_MAX)
        ax.set_ylim(self.AXIS_MIN, self.AXIS_MAX)
        ax.set_xlabel("True SXR flux [W/m$^2$]")
        ax.set_ylabel("Predicted SXR flux [W/m$^2$]")
        ax.legend()
        fig.tight_layout()
        return fig


class AttentionMapCallback(Callback):
    """
    PyTorch Lightning callback for visualizing transformer attention maps
    during validation epochs.

    Supports CLS-token-based and local patch attention visualization.
    """

    def __init__(self, log_every_n_epochs=1, num_samples=4, patch_size=8, use_local_attention=False):
        """
        Initialize callback.

        Parameters
        ----------
        log_every_n_epochs : int
            Frequency of logging attention maps.
        num_samples : int
            Number of samples to visualize per epoch.
        patch_size : int
            Patch size used in the Vision Transformer.
        use_local_attention : bool
            If True, visualize local attention patterns instead of CLS attention.
        """
        super().__init__()
        self.patch_size = patch_size
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.use_local_attention = use_local_attention

    def on_validation_epoch_end(self, trainer, pl_module):
        """Trigger visualization of attention maps at the end of validation epochs."""
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self._visualize_attention(trainer, pl_module)

    def _visualize_attention(self, trainer, pl_module):
        """Generate and log attention maps from the model's attention weights,
        for a fresh random sample of the validation set each epoch."""
        val_ds = trainer.datamodule.val_ds if trainer.datamodule else None
        if not val_ds:
            return

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            n = min(self.num_samples, len(val_ds))
            indices = random.sample(range(len(val_ds)), n)
            imgs = torch.stack([val_ds[i][0] for i in indices]).to(pl_module.device)

            # forward(return_attention=True) -> (global_flux_raw, attention_weights, patch_flux_raw)
            _, attention_weights, patch_flux_raw = pl_module(imgs, return_attention=True)

            for sample_idx in range(min(self.num_samples, imgs.size(0))):
                fig = self._plot_attention_map(
                    imgs[sample_idx],
                    attention_weights,
                    sample_idx,
                    trainer.current_epoch,
                    patch_size=self.patch_size,
                    patch_flux=patch_flux_raw[sample_idx] if patch_flux_raw is not None else None
                )
                trainer.logger.experiment.log({"Attention plots": wandb.Image(fig)})
                plt.close(fig)

        if was_training:
            pl_module.train()

    def _plot_attention_map(self, image, attention_weights, sample_idx, epoch, patch_size, patch_flux=None):
        """Plot and return a visualization of the attention heatmaps for a single image."""
        img_np = image.cpu().numpy()
        if len(img_np.shape) == 3 and img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))

        H, W = img_np.shape[:2]
        grid_h, grid_w = H // patch_size, W // patch_size

        last_layer_attention = attention_weights[-1]
        sample_attention = last_layer_attention[sample_idx]
        avg_attention = sample_attention.mean(dim=0)

        if self.use_local_attention:
            # Spatial center of the grid, not the middle of the flattened sequence
            # (those only coincide when grid_w == 1) — row-major flatten: idx = row*grid_w + col.
            center_patch_idx = (grid_h // 2) * grid_w + (grid_w // 2)
            center_attention = avg_attention[center_patch_idx, :].cpu()
            avg_attention_map = avg_attention.mean(dim=0).cpu()
            attention_map = avg_attention_map.reshape(grid_h, grid_w)
            center_map = center_attention.reshape(grid_h, grid_w)
        else:
            cls_attention = avg_attention[0, 1:].cpu()
            attention_map = cls_attention.reshape(grid_h, grid_w)
            center_map = None

        if len(img_np[0, 0, :]) >= 6:
            rgb_channels = [0, 2, 4]
            img_display = np.stack([(img_np[:, :, i] + 1) / 2 for i in rgb_channels], axis=2)
            img_display = np.clip(img_display, 0, 1)
        else:
            img_display = (img_np[:, :, 0] + 1) / 2
            img_display = np.stack([img_display] * 3, axis=2)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Attention Visualization - Epoch {epoch}, Sample {sample_idx}', fontsize=16)

        axes[0, 0].imshow(img_display)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        im1 = axes[0, 1].imshow(attention_map, cmap='hot', interpolation='nearest')
        axes[0, 1].set_title('Attention Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])

        axes[0, 2].imshow(img_display)
        axes[0, 2].imshow(attention_map, cmap='hot', alpha=0.6, interpolation='nearest')
        axes[0, 2].set_title('Attention Overlay')
        axes[0, 2].axis('off')

        if center_map is not None:
            im2 = axes[1, 0].imshow(center_map, cmap='hot', interpolation='nearest')
            axes[1, 0].set_title('Center Patch Attention')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0])
        else:
            axes[1, 0].text(0.5, 0.5, 'Center attention\nnot available',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Center Patch Attention')
            axes[1, 0].axis('off')

        if patch_flux is not None:
            patch_flux_np = patch_flux.cpu().numpy().reshape(grid_h, grid_w)
            im3 = axes[1, 1].imshow(patch_flux_np, cmap='viridis', interpolation='nearest')
            axes[1, 1].set_title('Patch Flux')
            axes[1, 1].axis('off')
            plt.colorbar(im3, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'Patch flux\nnot available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Patch Flux')
            axes[1, 1].axis('off')

        axes[1, 2].hist(attention_map.flatten(), bins=50, alpha=0.7)
        axes[1, 2].set_title('Attention Distribution')
        axes[1, 2].set_xlabel('Attention Weight')
        axes[1, 2].set_ylabel('Frequency')

        plt.tight_layout()
        return fig
