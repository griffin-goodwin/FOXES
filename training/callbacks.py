"""
PyTorch Lightning callbacks for visualizing training progress: predicted vs.
true SXR flux, and Vision Transformer attention maps, logged to Weights & Biases.

Used by train.py — not meant to be run standalone.
"""
import random
import warnings

import wandb
from pytorch_lightning import Callback
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from forecasting.model import unnormalize_sxr


FLARE_CLASSES = (
    ('Below-B', -np.inf, 1e-7),
    ('B', 1e-7, 1e-6),
    ('C', 1e-6, 1e-5),
    ('M', 1e-5, 1e-4),
    ('X', 1e-4, np.inf),
)


def _flare_class(flux):
    for name, lower, upper in FLARE_CLASSES:
        if lower <= flux < upper:
            return name
    raise ValueError(f"Cannot classify non-finite SXR flux {flux!r}")


def _validation_class_indices(dataset):
    """Cache validation indices grouped by raw GOES flare class."""
    cached = getattr(dataset, '_callback_flare_class_indices', None)
    if cached is not None:
        return cached

    groups = {name: [] for name, _, _ in FLARE_CLASSES}
    sxr_dir = getattr(dataset, 'sxr_dir', None)
    samples = getattr(dataset, 'samples', None)
    if sxr_dir is None or samples is None:
        warnings.warn(
            'Validation dataset does not expose raw SXR paths; using random sampling.',
            stacklevel=2,
        )
        return None

    for index, timestamp in enumerate(samples):
        try:
            value = np.load(sxr_dir / f'{timestamp}.npy', allow_pickle=False)
            if value.size != 1:
                raise ValueError(f'expected one value, found {value.size}')
            flux = float(value.reshape(-1)[0])
            if not np.isfinite(flux):
                raise ValueError(f'non-finite value {flux!r}')
            groups[_flare_class(flux)].append(index)
        except (OSError, ValueError) as error:
            warnings.warn(
                f'Excluding validation target {timestamp} from callback sampling: {error}',
                stacklevel=2,
            )

    dataset._callback_flare_class_indices = groups
    return groups


def stratified_validation_indices(dataset, num_samples):
    """Select validation examples round-robin across all available flare classes.

    If every <B/B/C/M/X group is available, at least five examples are returned
    even when ``num_samples`` is configured below five.
    """
    groups = _validation_class_indices(dataset)
    if not groups or not any(groups.values()):
        return random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    available = [name for name, _, _ in FLARE_CLASSES if groups[name]]
    target_count = min(len(dataset), max(num_samples, len(available)))
    class_order = []
    class_counts = {name: 0 for name in available}
    while len(class_order) < target_count:
        added = False
        for name in available:
            if class_counts[name] < len(groups[name]) and len(class_order) < target_count:
                class_order.append(name)
                class_counts[name] += 1
                added = True
        if not added:
            break

    sampled = {
        name: iter(random.sample(groups[name], count))
        for name, count in class_counts.items()
    }
    return [next(sampled[name]) for name in class_order]


class PerClassValidationMetrics(Callback):
    """Aggregate mean metrics and optional uncertainty diagnostics."""

    METRIC_COUNT = 9

    def on_validation_epoch_start(self, trainer, pl_module):
        self.has_uncertainty = bool(
            getattr(pl_module, 'predicts_uncertainty', False)
        )
        self.totals = torch.zeros(
            len(FLARE_CLASSES), self.METRIC_COUNT,
            device=pl_module.device, dtype=torch.float64,
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        required = {'target_raw', 'target_norm', 'mean_norm', 'prediction_raw'}
        if not isinstance(outputs, dict) or not required.issubset(outputs):
            return
        if 'variance_norm' in outputs:
            self.has_uncertainty = True
        target_raw = outputs['target_raw'].reshape(-1)
        target_norm = outputs['target_norm'].reshape(-1)
        mean_norm = outputs['mean_norm'].reshape(-1)
        prediction_raw = outputs['prediction_raw'].reshape(-1)
        norm_std = float(pl_module.sxr_norm[1].item())
        error_dex = (mean_norm - target_norm) * norm_std
        if self.has_uncertainty:
            variance_norm = outputs['variance_norm'].reshape(-1)
            sigma_dex = torch.sqrt(variance_norm) * norm_std
            nll = F.gaussian_nll_loss(
                mean_norm, target_norm, variance_norm,
                full=True, reduction='none',
            )
        else:
            sigma_dex = torch.zeros_like(error_dex)
            nll = torch.zeros_like(error_dex)

        predicted_classes = torch.zeros_like(target_raw, dtype=torch.long)
        predicted_classes = torch.where(
            prediction_raw >= 1e-7, 1, predicted_classes
        )
        predicted_classes = torch.where(
            prediction_raw >= 1e-6, 2, predicted_classes
        )
        predicted_classes = torch.where(
            prediction_raw >= 1e-5, 3, predicted_classes
        )
        predicted_classes = torch.where(
            prediction_raw >= 1e-4, 4, predicted_classes
        )

        for class_index, (_, lower, upper) in enumerate(FLARE_CLASSES):
            mask = (target_raw >= lower) & (target_raw < upper)
            if not mask.any():
                continue
            errors = error_dex[mask]
            sigmas = sigma_dex[mask]
            self.totals[class_index] += torch.stack((
                mask.sum(),
                errors.abs().sum(),
                errors.square().sum(),
                errors.sum(),
                sigmas.sum(),
                (errors.abs() <= sigmas).sum(),
                (errors.abs() <= 1.96 * sigmas).sum(),
                (predicted_classes[mask] == class_index).sum(),
                nll[mask].sum(),
            )).to(dtype=torch.float64)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not hasattr(self, 'totals'):
            return
        totals = self.totals.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        metrics = {}
        class_mae = []
        class_mse = []
        class_nll = []
        class_coverage_68 = []
        class_coverage_95 = []
        for class_index, (name, _, _) in enumerate(FLARE_CLASSES):
            (
                count, abs_error, squared_error, bias, sigma,
                covered_68, covered_95, correct, nll,
            ) = totals[class_index]
            count_value = count.item()
            if count_value == 0:
                continue
            prefix = f'val_class/{name}'
            mae = (abs_error / count).item()
            class_mae.append(mae)
            mse = (squared_error / count).item()
            class_mse.append(mse)
            metrics.update({
                f'{prefix}/count': count_value,
                f'{prefix}/mae_dex': mae,
                f'{prefix}/mse_dex2': mse,
                f'{prefix}/rmse_dex': np.sqrt(mse),
                f'{prefix}/bias_dex': (bias / count).item(),
                f'{prefix}/class_accuracy': (correct / count).item(),
            })
            if self.has_uncertainty:
                mean_nll = (nll / count).item()
                class_nll.append(mean_nll)
                coverage_68 = (covered_68 / count).item()
                class_coverage_68.append(coverage_68)
                coverage_95 = (covered_95 / count).item()
                class_coverage_95.append(coverage_95)
                metrics.update({
                    f'{prefix}/mean_sigma_dex': (sigma / count).item(),
                    f'{prefix}/coverage_68': coverage_68,
                    f'{prefix}/coverage_95': coverage_95,
                    f'{prefix}/nll': mean_nll,
                })
        if class_mae:
            metrics['val_class/macro_mae_dex'] = float(np.mean(class_mae))
            metrics['val_class/macro_mse_dex2'] = float(np.mean(class_mse))
            if self.has_uncertainty:
                metrics['val_class/macro_nll'] = float(np.mean(class_nll))
                metrics['val_class/macro_coverage_68'] = float(
                    np.mean(class_coverage_68)
                )
                metrics['val_class/macro_coverage_95'] = float(
                    np.mean(class_coverage_95)
                )
        # Logging through the LightningModule makes macro NLL available to
        # ModelCheckpoint as well as W&B. Every DDP rank has the same totals
        # after all-reduce, so no additional distributed reduction is needed.
        pl_module.log_dict(
            metrics, on_step=False, on_epoch=True, logger=True,
            sync_dist=False,
        )


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
        lower_sxr = []
        upper_sxr = []

        n = min(self.num_samples, len(self.val_ds))
        indices = stratified_validation_indices(self.val_ds, n)
        data_samples = [self.val_ds[i] for i in indices]

        with torch.no_grad():
            for aia, target in data_samples:
                aia = aia.to(pl_module.device).unsqueeze(0)
                if getattr(pl_module, 'predicts_uncertainty', False):
                    pred, lower, upper = pl_module.predict_interval(aia, z=1.96)
                    lower_sxr.append(lower.item())
                    upper_sxr.append(upper.item())
                else:
                    pred, *_ = pl_module(aia, return_attention=False)
                pred_sxr.append(pred.item())
                true_sxr.append(target.item())

        true_unorm = unnormalize_sxr(np.array(true_sxr, dtype=np.float32), self.sxr_norm)
        # Both model variants already return predictions in raw W/m^2 units.
        fig1 = self.plot_aia_sxr(
            true_unorm,
            np.asarray(pred_sxr),
            np.asarray(lower_sxr) if lower_sxr else None,
            np.asarray(upper_sxr) if upper_sxr else None,
        )
        trainer.logger.experiment.log({"Soft X-ray flux plots": wandb.Image(fig1)})
        plt.close(fig1)

    # Flare-class range: A-class starts at 1e-8 W/m^2, X10 is 10x the 1e-4 X-class threshold.
    AXIS_MIN = 1e-8
    AXIS_MAX = 1e-3

    def plot_aia_sxr(self, val_sxr, pred_sxr, lower_sxr=None, upper_sxr=None):
        """Log-log parity plot: predicted vs. true SXR flux, with a 1:1 reference line."""
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.plot([self.AXIS_MIN, self.AXIS_MAX], [self.AXIS_MIN, self.AXIS_MAX],
                color='gray', linestyle='--', linewidth=1, label='Perfect prediction')
        if lower_sxr is not None and upper_sxr is not None:
            # Log axes cannot display zero; clipping is visualization-only.
            visible_predictions = np.maximum(pred_sxr, self.AXIS_MIN)
            visible_lower = np.clip(lower_sxr, self.AXIS_MIN, visible_predictions)
            visible_upper = np.maximum(upper_sxr, visible_predictions)
            errors = np.vstack((
                visible_predictions - visible_lower,
                visible_upper - visible_predictions,
            ))
            ax.errorbar(
                val_sxr, visible_predictions, yerr=errors, fmt='o', color='blue',
                ecolor='cornflowerblue', alpha=0.7, markersize=3, capsize=2,
                linewidth=0.8, label='Predictions (95% interval)',
            )
        else:
            ax.scatter(
                val_sxr, pred_sxr, color='blue', alpha=0.7, s=10,
                label='Predictions',
            )
        for true_value, predicted_value in zip(val_sxr, pred_sxr):
            if np.isfinite(true_value) and np.isfinite(predicted_value):
                ax.annotate(
                    _flare_class(float(true_value)),
                    (true_value, max(predicted_value, self.AXIS_MIN)),
                    xytext=(3, 3), textcoords='offset points', fontsize=7,
                )

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
            indices = stratified_validation_indices(val_ds, n)
            samples = [val_ds[index] for index in indices]
            # Attention is O(num_patches^2) per head and layer. Forwarding all
            # visualization samples together can consume tens of GiB for a
            # 64x64 patch grid, even though each plot only needs one sample.
            for sample_idx, sample in enumerate(samples):
                img = sample[0].unsqueeze(0).to(pl_module.device)
                outputs = pl_module(img, return_attention=True)
                if getattr(pl_module, 'predicts_uncertainty', False):
                    (
                        _, _, attention_weights, patch_flux_raw,
                        patch_variance_raw,
                    ) = outputs
                    patch_uncertainty = torch.sqrt(patch_variance_raw)
                    patch_uncertainty_title = (
                        'Sqrt Global Variance Contribution [W/m²]'
                        if getattr(
                            pl_module, 'patch_uncertainty_semantics', None
                        ) == 'variance_attribution'
                        else 'Patch Flux Std [W/m²]'
                    )
                else:
                    # (global_flux_raw, attention, patch_flux_raw)
                    _, attention_weights, patch_flux_raw = outputs
                    patch_uncertainty = None
                    patch_uncertainty_title = None

                target_norm = float(sample[1].item())
                sxr_transform = getattr(val_ds, 'sxr_transform', None)
                if sxr_transform is not None:
                    target_raw = float(
                        unnormalize_sxr(
                            target_norm,
                            np.asarray(
                                [sxr_transform.mean, sxr_transform.std],
                                dtype=np.float64,
                            ),
                        )
                    )
                    sample_description = (
                        f'{_flare_class(target_raw)} class, true SXR={target_raw:.2e} W/m²'
                    )
                else:
                    sample_description = f'normalized target={target_norm:.3g}'
                fig = self._plot_attention_map(
                    img[0],
                    attention_weights,
                    0,
                    trainer.current_epoch,
                    patch_size=self.patch_size,
                    aia_transform=getattr(val_ds, 'aia_transform', None),
                    sample_description=sample_description,
                    patch_flux=patch_flux_raw[0] if patch_flux_raw is not None else None,
                    patch_std=(patch_uncertainty[0]
                               if patch_uncertainty is not None else None),
                    patch_uncertainty_title=patch_uncertainty_title,
                )
                trainer.logger.experiment.log({"Attention plots": wandb.Image(fig)})
                plt.close(fig)
                del outputs, attention_weights, patch_flux_raw, img

        if was_training:
            pl_module.train()

    @staticmethod
    def _display_image(image, aia_transform=None):
        """Convert a normalized channel-last AIA tensor to a display RGB image.

        Model inputs use a per-wavelength asinh/z-score transform and therefore
        are not in ``[-1, 1]``.  Undo the z-score and scale the asinh signal by
        the transform's physical clipping level.  This keeps zero signal black
        and prevents normal positive values from being clipped into saturated
        callback images.
        """
        img_np = image.detach().cpu().float().numpy()
        if img_np.ndim != 3:
            raise ValueError(f"Expected an HWC image, got shape {img_np.shape}")

        rgb_channels = [0, 2, 4] if img_np.shape[-1] >= 6 else [0] * 3
        selected = img_np[:, :, rgb_channels]
        if aia_transform is not None:
            means = aia_transform.asinh_mean[:, 0, 0].cpu().numpy()[rgb_channels]
            stds = aia_transform.asinh_std[:, 0, 0].cpu().numpy()[rgb_channels]
            q90 = aia_transform.q90[:, 0, 0].cpu().numpy()[rgb_channels]
            clip = aia_transform.clip_q99999[:, 0, 0].cpu().numpy()[rgb_channels]
            asinh_signal = selected * stds + means
            display_max = np.arcsinh(clip / q90)
            selected = np.divide(
                asinh_signal,
                display_max,
                out=np.zeros_like(asinh_signal),
                where=display_max > 0,
            )
        else:
            # Compatibility for datasets without an AIA normalization artifact.
            # Scale each selected channel independently instead of assuming a
            # normalization range that the dataset does not guarantee.
            selected = np.nan_to_num(
                selected, nan=0.0, posinf=0.0, neginf=0.0
            )
            lo = np.nanpercentile(selected, 1, axis=(0, 1), keepdims=True)
            hi = np.nanpercentile(selected, 99.9, axis=(0, 1), keepdims=True)
            selected = np.divide(
                selected - lo,
                hi - lo,
                out=np.zeros_like(selected),
                where=(hi - lo) > 0,
            )
        return np.clip(np.nan_to_num(selected), 0, 1)

    def _plot_attention_map(self, image, attention_weights, sample_idx, epoch,
                            patch_size, aia_transform=None, patch_flux=None,
                            patch_std=None, sample_description=None,
                            patch_uncertainty_title=None):
        """Plot and return a visualization of the attention heatmaps for a single image."""
        img_np = image.detach().cpu().numpy()

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

        img_display = self._display_image(image, aia_transform)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        title = f'Attention Visualization - Epoch {epoch}, Sample {sample_idx}'
        if sample_description:
            title += f'\n{sample_description}'
        fig.suptitle(title, fontsize=16)

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
            patch_flux_np = patch_flux.detach().cpu().float().numpy().reshape(grid_h, grid_w)
            patch_flux_np = np.nan_to_num(patch_flux_np, nan=0.0, posinf=0.0, neginf=0.0)
            positive = patch_flux_np[patch_flux_np > 0]
            # Patch contributions commonly span orders of magnitude. A linear,
            # independently auto-scaled map lets one outlier saturate the rest
            # and exaggerates tiny differences in nearly uniform maps.
            flux_norm = None
            if positive.size and positive.max() > positive.min():
                flux_norm = LogNorm(vmin=positive.min(), vmax=positive.max())
            im3 = axes[1, 1].imshow(
                patch_flux_np, cmap='viridis', norm=flux_norm,
                interpolation='nearest',
            )
            axes[1, 1].set_title(
                'Learned Patch Contribution\n'
                f'total={patch_flux_np.sum():.2e} W/m²'
            )
            axes[1, 1].axis('off')
            plt.colorbar(im3, ax=axes[1, 1], label='Contribution [W/m²]')
        else:
            axes[1, 1].text(0.5, 0.5, 'Patch flux\nnot available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Patch Flux')
            axes[1, 1].axis('off')

        if patch_std is not None:
            patch_std_np = patch_std.cpu().numpy().reshape(grid_h, grid_w)
            im4 = axes[1, 2].imshow(
                patch_std_np, cmap='magma', interpolation='nearest'
            )
            axes[1, 2].set_title(
                patch_uncertainty_title or 'Patch Flux Std [W/m²]'
            )
            axes[1, 2].axis('off')
            plt.colorbar(im4, ax=axes[1, 2])
        else:
            axes[1, 2].hist(attention_map.flatten(), bins=50, alpha=0.7)
            axes[1, 2].set_title('Attention Distribution')
            axes[1, 2].set_xlabel('Attention Weight')
            axes[1, 2].set_ylabel('Frequency')

        plt.tight_layout()
        return fig

class SpatialUncertaintyMapCallback(Callback):
    """Log patch flux and uncertainty maps without attention."""

    def __init__(self, log_every_n_epochs=1, num_samples=5, patch_size=8):
        super().__init__()
        self.log_every_n_epochs = int(log_every_n_epochs)
        self.num_samples = int(num_samples)
        self.patch_size = int(patch_size)
        if self.log_every_n_epochs <= 0:
            raise ValueError("log_every_n_epochs must be positive")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")

    @staticmethod
    def _positive_log_norm(values):
        positive = values[np.isfinite(values) & (values > 0)]
        if not positive.size or positive.max() <= positive.min():
            return None
        return LogNorm(vmin=positive.min(), vmax=positive.max())

    @classmethod
    def _plot_spatial_uncertainty_maps(
        cls, image, patch_flux_raw, patch_variance_raw, patch_size,
        aia_transform=None, sample_description=None,
        patch_uncertainty_title='Patch Flux Std',
        patch_uncertainty_label='Std [W/m²]',
    ):
        height, width = image.shape[:2]
        grid_h, grid_w = height // patch_size, width // patch_size
        patch_flux = patch_flux_raw.detach().cpu().float().numpy().reshape(
            grid_h, grid_w
        )
        patch_std = torch.sqrt(
            torch.clamp(patch_variance_raw.detach().cpu().float(), min=0)
        ).numpy().reshape(grid_h, grid_w)
        patch_flux = np.nan_to_num(
            patch_flux, nan=0.0, posinf=0.0, neginf=0.0
        )
        patch_std = np.nan_to_num(
            patch_std, nan=0.0, posinf=0.0, neginf=0.0
        )
        total_flux = float(patch_flux.sum())

        patch_fraction = np.divide(
            patch_flux,
            total_flux,
            out=np.zeros_like(patch_flux),
            where=total_flux > 0,
        )

        fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
        title = 'Gaussian Patch Flux Maps'
        if sample_description:
            title += f'\n{sample_description}'
        fig.suptitle(title, fontsize=14)

        axes[0].imshow(AttentionMapCallback._display_image(
            image, aia_transform
        ))
        axes[0].set_title('AIA Image')
        axes[0].axis('off')

        panels = (
            (
                axes[1], patch_flux, 'Patch Flux Contribution',
                'Flux [W/m²]', 'viridis', cls._positive_log_norm(patch_flux),
            ),
            (
                axes[2], 100.0 * patch_fraction,
                'Share of Global Flux', 'Percent [%]', 'viridis',
                cls._positive_log_norm(100.0 * patch_fraction),
            ),
            (
                axes[3], patch_std, patch_uncertainty_title,
                patch_uncertainty_label, 'magma',
                cls._positive_log_norm(patch_std),
            ),
        )
        for axis, values, panel_title, label, cmap, norm in panels:
            plotted = axis.imshow(
                values, cmap=cmap, norm=norm, interpolation='nearest'
            )
            axis.set_title(panel_title)
            axis.axis('off')
            fig.colorbar(plotted, ax=axis, label=label)
        axes[1].set_title(
            f'Patch Flux Contribution\ntotal={total_flux:.2e} W/m²'
        )
        fig.tight_layout()
        return fig

    @classmethod
    def _plot_global_patch_adjustment(
        cls, image, corrected_patch_flux, log10_adjustment, patch_size,
        max_abs_adjustment, aia_transform=None, sample_description=None,
    ):
        height, width = image.shape[:2]
        grid_shape = (height // patch_size, width // patch_size)
        corrected = corrected_patch_flux.detach().cpu().float().numpy().reshape(
            grid_shape
        )
        adjustment = log10_adjustment.detach().cpu().float().numpy().reshape(
            grid_shape
        )
        local = corrected / np.power(10.0, adjustment)
        positive = np.concatenate((local[local > 0], corrected[corrected > 0]))
        shared_norm = (
            LogNorm(vmin=positive.min(), vmax=positive.max())
            if positive.size and positive.max() > positive.min() else None
        )

        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        title = 'Local and Globally Adjusted Patch Flux'
        if sample_description:
            title += f'\n{sample_description}'
        fig.suptitle(title, fontsize=14)
        axes[0].imshow(AttentionMapCallback._display_image(
            image, aia_transform
        ))
        axes[0].set_title('AIA Image')
        axes[0].axis('off')

        for axis, values, label in (
            (axes[1], local, 'Local Patch Flux'),
            (axes[2], corrected, 'Adjusted Patch Flux'),
        ):
            image_plot = axis.imshow(
                values, cmap='viridis', norm=shared_norm,
                interpolation='nearest',
            )
            axis.set_title(f'{label}\ntotal={values.sum():.2e} W/m²')
            axis.axis('off')
            fig.colorbar(image_plot, ax=axis, label='Flux [W/m²]')

        image_plot = axes[3].imshow(
            adjustment, cmap='coolwarm', interpolation='nearest',
            vmin=-max_abs_adjustment, vmax=max_abs_adjustment,
        )
        axes[3].set_title('Global Patch Adjustment')
        axes[3].axis('off')
        fig.colorbar(image_plot, ax=axes[3], label='log10 multiplier [dex]')
        fig.tight_layout()
        return fig

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            not trainer.is_global_zero
            or trainer.current_epoch % self.log_every_n_epochs != 0
            or getattr(pl_module, 'uncertainty_kind', None) != 'gaussian'
        ):
            return
        val_ds = trainer.datamodule.val_ds if trainer.datamodule else None
        if val_ds is None or len(val_ds) == 0:
            return

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                count = min(self.num_samples, len(val_ds))
                indices = stratified_validation_indices(val_ds, count)
                samples = [val_ds[index] for index in indices]
                images = torch.stack([
                    sample[0] for sample in samples
                ]).to(pl_module.device)
                prediction_raw, _, patch_flux_raw, patch_variance_raw = (
                    pl_module(images, return_attention=False)
                )
                network = getattr(pl_module, 'model', None)
                global_adjustment = getattr(
                    network, 'last_log10_global_patch_adjustment', None
                )
                is_variance_attribution = getattr(
                    pl_module, 'patch_uncertainty_semantics', None
                ) == 'variance_attribution'

                sxr_transform = getattr(val_ds, 'sxr_transform', None)
                aia_transform = getattr(val_ds, 'aia_transform', None)
                for sample_index, sample in enumerate(samples):
                    target_norm = float(sample[1].item())
                    if sxr_transform is not None:
                        target_raw = float(unnormalize_sxr(
                            target_norm,
                            np.asarray([
                                sxr_transform.mean, sxr_transform.std,
                            ], dtype=np.float64),
                        ))
                        description = (
                            f'{_flare_class(target_raw)} class, '
                            f'true={target_raw:.2e}, '
                            f'predicted={float(prediction_raw[sample_index].item()):.2e} W/m²'
                        )
                    else:
                        description = f'normalized target={target_norm:.3g}'
                    figure = self._plot_spatial_uncertainty_maps(
                        images[sample_index],
                        patch_flux_raw[sample_index],
                        patch_variance_raw[sample_index],
                        patch_size=self.patch_size,
                        aia_transform=aia_transform,
                        sample_description=description,
                        patch_uncertainty_title=(
                            'Sqrt Global Variance Contribution'
                            if is_variance_attribution
                            else 'Patch Flux Std'
                        ),
                        patch_uncertainty_label=(
                            'Sqrt variance contribution [W/m²]'
                            if is_variance_attribution
                            else 'Std [W/m²]'
                        ),
                    )
                    trainer.logger.experiment.log({
                        'Spatial uncertainty plots': wandb.Image(figure)
                    })
                    plt.close(figure)
                    if global_adjustment is not None:
                        adjustment_figure = self._plot_global_patch_adjustment(
                            images[sample_index],
                            patch_flux_raw[sample_index],
                            global_adjustment[sample_index],
                            patch_size=self.patch_size,
                            max_abs_adjustment=(
                                network.max_abs_log10_global_adjustment
                            ),
                            aia_transform=aia_transform,
                            sample_description=description,
                        )
                        trainer.logger.experiment.log({
                            'Global patch adjustment plots': wandb.Image(
                                adjustment_figure
                            )
                        })
                        plt.close(adjustment_figure)
        finally:
            if was_training:
                pl_module.train()
