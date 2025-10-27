import wandb
from pytorch_lightning import Callback
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import sunpy.visualization.colormaps as cm
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback
from PIL import Image
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.ndimage import zoom

# Custom Callback
sdoaia94 = matplotlib.colormaps['sdoaia94']


def unnormalize_sxr(normalized_values, sxr_norm):
    """
    Convert normalized SXR (soft X-ray) values back to their physical scale.

    Parameters
    ----------
    normalized_values : torch.Tensor or np.ndarray
        Normalized SXR flux values.
    sxr_norm : np.ndarray or torch.Tensor
        Normalization parameters (mean and std used during preprocessing).

    Returns
    -------
    np.ndarray
        Unnormalized SXR flux values on the original logarithmic scale.
    """
    if isinstance(normalized_values, torch.Tensor):
        normalized_values = normalized_values.cpu().numpy()
    normalized_values = np.array(normalized_values, dtype=np.float32)
    return 10 ** (normalized_values * float(sxr_norm[1].item()) + float(sxr_norm[0].item())) - 1e-8


class ImagePredictionLogger_SXR(Callback):
    """
    PyTorch Lightning callback for logging AIA input images and corresponding
    true vs predicted Soft X-Ray (SXR) flux values to Weights & Biases (wandb).

    This helps monitor model performance across validation epochs by
    comparing predicted vs. ground-truth flare intensities.
    """

    def __init__(self, data_samples, sxr_norm):
        """
        Initialize callback with validation samples and normalization parameters.

        Parameters
        ----------
        data_samples : list
            List of validation samples (AIA image, SXR target pairs).
        sxr_norm : np.ndarray
            Normalization statistics used to unnormalize predicted flux values.
        """
        super().__init__()
        self.data_samples = data_samples
        self.val_aia = data_samples[0]
        self.val_sxr = data_samples[1]
        self.sxr_norm = sxr_norm

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Log scatter plots comparing predicted and true SXR flux values
        at the end of each validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model being trained/validated.
        """
        aia_images = []
        true_sxr = []
        pred_sxr = []

        for aia, target in self.data_samples:
            aia = aia.to(pl_module.device).unsqueeze(0)
            pred = pl_module(aia)
            pred_sxr.append(pred.item())
            aia_images.append(aia.squeeze(0).cpu().numpy())
            true_sxr.append(target.item())

        true_unorm = unnormalize_sxr(true_sxr, self.sxr_norm)
        pred_unnorm = unnormalize_sxr(pred_sxr, self.sxr_norm)

        fig1 = self.plot_aia_sxr(aia_images, true_unorm, pred_unnorm)
        trainer.logger.experiment.log({"Soft X-ray flux plots": wandb.Image(fig1)})
        plt.close(fig1)

        fig2 = self.plot_aia_sxr_difference(aia_images, true_unorm, pred_unnorm)
        trainer.logger.experiment.log({"Soft X-ray flux difference plots": wandb.Image(fig2)})
        plt.close(fig2)

    def plot_aia_sxr(self, val_aia, val_sxr, pred_sxr):
        """
        Plot scatter of predicted vs true SXR flux values.

        Returns
        -------
        matplotlib.figure.Figure
            Scatter plot comparing true and predicted flux values.
        """
        num_samples = len(val_aia)
        fig, axes = plt.subplots(1, 1, figsize=(5, 2))

        for i in range(num_samples):
            axes.scatter(i, val_sxr[i], label='Ground truth' if i == 0 else "", color='blue')
            axes.scatter(i, pred_sxr[i], label='Prediction' if i == 0 else "", color='orange')
        axes.set_xlabel("Index")
        axes.set_ylabel("Soft x-ray flux [W/m2]")
        axes.set_yscale('log')
        axes.legend()

        fig.tight_layout()
        return fig

    def plot_aia_sxr_difference(self, val_aia, val_sxr, pred_sxr):
        """
        Plot difference between true and predicted SXR flux values.

        Returns
        -------
        matplotlib.figure.Figure
            Scatter plot of flux differences (true - predicted).
        """
        num_samples = len(val_aia)
        fig, axes = plt.subplots(1, 1, figsize=(5, 2))
        for i in range(num_samples):
            axes.scatter(i, val_sxr[i] - pred_sxr[i], label='Soft X-ray Flux Difference', color='blue')
            axes.set_xlabel("Index")
            axes.set_ylabel("Soft X-ray Flux Difference (True - Pred.) [W/m2]")

        fig.tight_layout()
        return fig


class AttentionMapCallback(Callback):
    """
    PyTorch Lightning callback for visualizing transformer attention maps
    during validation epochs.

    Supports CLS-token-based and local patch attention visualization.
    """

    def __init__(self, log_every_n_epochs=1, num_samples=4, save_dir="attention_maps",
                 patch_size=8, use_local_attention=False):
        """
        Initialize callback.

        Parameters
        ----------
        log_every_n_epochs : int
            Frequency of logging attention maps.
        num_samples : int
            Number of samples to visualize per epoch.
        save_dir : str
            Directory to save attention visualizations.
        patch_size : int
            Patch size used in the Vision Transformer.
        use_local_attention : bool
            If True, visualize local attention patterns instead of CLS attention.
        """
        super().__init__()
        self.patch_size = patch_size
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.save_dir = save_dir
        self.use_local_attention = use_local_attention

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Trigger visualization of attention maps at the end of validation epochs.
        """
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self._visualize_attention(trainer, pl_module)

    def _visualize_attention(self, trainer, pl_module):
        """
        Generate and log attention maps from the model's attention weights.
        """
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        pl_module.eval()
        with torch.no_grad():
            batch = next(iter(val_dataloader))
            imgs, labels = batch
            imgs = imgs[:self.num_samples].to(pl_module.device)

            patch_flux_raw = None
            try:
                outputs, attention_weights = pl_module(imgs, return_attention=True)
            except:
                if hasattr(pl_module, 'model') and hasattr(pl_module.model, 'forward'):
                    try:
                        print("Using model's forward method")
                        outputs, attention_weights, patch_flux_raw = pl_module.model(
                            imgs, pl_module.sxr_norm, return_attention=True)
                    except:
                        print("Using model's forward method failed")
                        outputs, attention_weights = pl_module.forward_for_callback(imgs, return_attention=True)
                else:
                    outputs, attention_weights = pl_module.forward_for_callback(imgs, return_attention=True)

            for sample_idx in range(min(self.num_samples, imgs.size(0))):
                map = self._plot_attention_map(
                    imgs[sample_idx],
                    attention_weights,
                    sample_idx,
                    trainer.current_epoch,
                    patch_size=self.patch_size,
                    patch_flux=patch_flux_raw[sample_idx] if patch_flux_raw is not None else None
                )
                trainer.logger.experiment.log({"Attention plots": wandb.Image(map)})
                plt.close(map)

    def _plot_attention_map(self, image, attention_weights, sample_idx, epoch, patch_size, patch_flux=None):
        """
        Plot and return a visualization of the attention heatmaps for a single image.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor.
        attention_weights : list[torch.Tensor]
            List of attention weight tensors from transformer layers.
        sample_idx : int
            Index of the sample in the batch.
        epoch : int
            Current training epoch.
        patch_size : int
            Patch size used in ViT.
        patch_flux : torch.Tensor, optional
            Optional tensor containing patch flux contributions.
        """
        img_np = image.cpu().numpy()
        if len(img_np.shape) == 3 and img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))

        H, W = img_np.shape[:2]
        grid_h, grid_w = H // patch_size, W // patch_size

        last_layer_attention = attention_weights[-1]
        sample_attention = last_layer_attention[sample_idx]
        avg_attention = sample_attention.mean(dim=0)

        if self.use_local_attention:
            center_patch_idx = (grid_h * grid_w) // 2
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

        # Create the figure and subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Attention Visualization - Epoch {epoch}, Sample {sample_idx}', fontsize=16)
        
        # Plot 1: Original image
        axes[0, 0].imshow(img_display)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Plot 2: Attention map
        im1 = axes[0, 1].imshow(attention_map, cmap='hot', interpolation='nearest')
        axes[0, 1].set_title('Attention Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Plot 3: Overlay
        axes[0, 2].imshow(img_display)
        axes[0, 2].imshow(attention_map, cmap='hot', alpha=0.6, interpolation='nearest')
        axes[0, 2].set_title('Attention Overlay')
        axes[0, 2].axis('off')
        
        # Plot 4: Center attention (if available)
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
        
        # Plot 5: Patch flux (if available)
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
        
        # Plot 6: Attention statistics
        axes[1, 2].hist(attention_map.flatten(), bins=50, alpha=0.7)
        axes[1, 2].set_title('Attention Distribution')
        axes[1, 2].set_xlabel('Attention Weight')
        axes[1, 2].set_ylabel('Frequency')

        plt.tight_layout()
        return fig


class MultiHeadAttentionCallback(AttentionMapCallback):
    """
    Extended callback that visualizes not only averaged attention maps
    but also the attention distributions of individual transformer heads.
    """

    def _plot_attention_map(self, image, attention_weights, sample_idx, epoch, patch_size):
        """
        Override: Plot both average and per-head attention maps.
        """
        super()._plot_attention_map(image, attention_weights, sample_idx, epoch, patch_size)
        self._plot_individual_heads(image, attention_weights, sample_idx, epoch, patch_size)

    def _plot_individual_heads(self, image, attention_weights, sample_idx, epoch, patch_size):
        """
        Visualize attention for each individual head separately.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor.
        attention_weights : list[torch.Tensor]
            List of attention tensors from model layers.
        sample_idx : int
            Sample index within the batch.
        epoch : int
            Current training epoch number.
        patch_size : int
            Patch size used in ViT.
        """
        img_np = image.cpu().numpy()
        last_layer_attention = attention_weights[-1][sample_idx]
        num_heads = last_layer_attention.size(0)

        H, W = img_np.shape[:2]
        grid_h, grid_w = H // patch_size, W // patch_size

        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if num_heads == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for head_idx in range(num_heads):
            row = head_idx // cols
            col = head_idx % cols
            head_attention = last_layer_attention[head_idx, 0, 1:].cpu()
            attention_map = head_attention.reshape(grid_h, grid_w)

            ax = axes[row, col] if rows > 1 else axes[col]
            im = ax.imshow(attention_map.numpy(), cmap='hot', interpolation='nearest')
            ax.set_title(f'Head {head_idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        for idx in range(num_heads, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/heads_epoch_{epoch}_sample_{sample_idx}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
