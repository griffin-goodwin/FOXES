"""
Inference Script for Solar Flare Prediction Models
=================================================

Runs a trained ViTLocal checkpoint over a folder of AIA data. Computes soft
X-ray (SXR) predictions, and saves attention weights, flux contributions, and
final outputs.

The workflow includes:
- Loading configuration parameters from a YAML file.
- Loading the model checkpoint and preparing it for inference.
- Performing batched evaluation over AIA/GOES datasets.
- Saving predicted fluxes, ground truth, and visualization-ready artifacts.

"""

import argparse
import sys
import gc
from time import perf_counter
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import yaml
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.dataset import AIAGOESDataset, AIANormTransform
from forecasting.model import (
    CoupledGaussianViTLocal,
    CoupledPinballViTLocal,
    ViTLocal,
    normalize_sxr,
    unnormalize_sxr,
)
from forecasting.quantiles import quantile_columns, quantile_label


def _as_bool(value):
    """Accept YAML booleans as well as the legacy ``"true"`` strings."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def resolve_inference_device(config_data):
    """Choose an available inference device from the YAML setting."""
    requested = str(config_data.get('device', 'auto')).strip().lower()
    if requested not in {'auto', 'cuda', 'mps', 'cpu'}:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if requested == 'auto':
        requested = 'cuda' if cuda_available else 'mps' if mps_available else 'cpu'
    elif requested == 'cuda' and not cuda_available:
        raise RuntimeError("device=cuda was requested, but CUDA is unavailable")
    elif requested == 'mps' and not mps_available:
        raise RuntimeError("device=mps was requested, but MPS is unavailable")
    return torch.device('cuda:0' if requested == 'cuda' else requested)


def select_inference_samples(dataset, sampling=None):
    """Select a reproducible random subset without loading AIA image data."""
    if sampling is None:
        sampling = {}
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be a mapping")
    unknown = set(sampling) - {'count', 'fraction', 'seed'}
    if unknown:
        raise ValueError(f"Unknown sampling settings: {sorted(unknown)}")
    count = sampling.get('count')
    fraction = sampling.get('fraction')
    if count is not None and fraction is not None:
        raise ValueError("Set only one of sampling.count or sampling.fraction")
    if count is None and fraction is None:
        return dataset, dataset.samples
    total = len(dataset)
    if total == 0:
        raise ValueError("No inference samples are available")
    if count is not None:
        if (
            isinstance(count, bool) or not isinstance(count, int)
            or not 1 <= count <= total
        ):
            raise ValueError(f"sampling.count must be an integer from 1 to {total}")
        sample_count = count
    else:
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not np.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError("sampling.fraction must be in (0, 1]")
        sample_count = max(1, int(np.ceil(total * fraction)))
    seed = sampling.get('seed', 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("sampling.seed must be a nonnegative integer")
    indices = np.sort(np.random.default_rng(seed).choice(
        total, size=sample_count, replace=False,
    ))
    times = [dataset.samples[index] for index in indices]
    print(f"Selected {sample_count} of {total} inference samples (seed={seed})")
    return Subset(dataset, indices.tolist()), times


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _auto_checkpoint_model_class(checkpoint):
    """Select the maintained deterministic, Gaussian, or pinball model."""
    state_dict = checkpoint.get('state_dict', {})
    state_keys = state_dict.keys()
    has_uncertainty_head = any(
        'patch_uncertainty_head' in key for key in state_keys
    )
    hyperparameters = checkpoint.get('hyper_parameters', {})
    if hyperparameters.get('objective') == 'pinball':
        return CoupledPinballViTLocal
    if ('model.global_query' in state_dict
            and hyperparameters.get('mean_parameterization') == 'positive'
            and not hyperparameters.get('model_kwargs', {}).get('independent_local_global', False)
            and not any(k.startswith('model.patch_global_context.') for k in state_keys)):
        return CoupledGaussianViTLocal
    is_uncertainty_checkpoint = (
        'uncertainty_kwargs' in hyperparameters
        or 'mean_parameterization' in hyperparameters
    )
    if has_uncertainty_head or is_uncertainty_checkpoint:
        raise ValueError(
            'Exploratory uncertainty checkpoint; use '
            'python -m forecasting.experimental.inference.'
        )
    return ViTLocal


def _resolve_amp_dtype(config_data, data_device):
    """Resolve the configured CUDA autocast dtype with an actionable error."""
    dtype_name = str(
        (config_data or {}).get('amp_dtype', 'float16')
    ).strip().lower()
    aliases = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'half': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    if dtype_name not in aliases:
        raise ValueError(
            "amp_dtype must be one of: float16, fp16, bfloat16, bf16"
        )
    dtype = aliases[dtype_name]
    if (
        data_device.type == 'cuda'
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "amp_dtype=bfloat16 was requested, but this GPU does not "
            "support BF16; use amp_dtype=float16 or disable AMP"
        )
    return dtype


class _AutocastForwardWrapper(torch.nn.Module):
    """Enter the requested autocast dtype inside DataParallel workers.

    DataParallel propagates whether autocast is enabled, but not its configured
    dtype. Without this wrapper, a BF16 context in the caller silently becomes
    FP16 in replica threads.
    """

    def __init__(self, module, device_type, dtype):
        super().__init__()
        self.wrapped_module = module
        self.device_type = str(device_type)
        self.amp_dtype = dtype

    def forward(self, *args, **kwargs):
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.amp_dtype,
            enabled=True,
        ):
            return self.wrapped_module(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped_module = super().__getattr__('wrapped_module')
            return getattr(wrapped_module, name)


def uncertainty_from_prediction(prediction_raw, variance_normalized, sxr_norm):
    """Convert normalized Gaussian variance into physical-flux intervals.

    The Gaussian lives in normalized ``log10(flux + 1e-8)`` space, so its
    physical intervals are asymmetric. Keeping both representations in the
    inference CSV lets evaluation compute the exact training-space NLL while
    still presenting intervals in W/m^2.
    """
    prediction_raw = prediction_raw.squeeze(-1)
    variance_normalized = variance_normalized.reshape_as(prediction_raw).float()
    variance_normalized = variance_normalized.clamp_min(
        torch.finfo(variance_normalized.dtype).tiny
    )
    prediction_normalized = normalize_sxr(
        prediction_raw.float(), sxr_norm
    )
    sigma_normalized = torch.sqrt(variance_normalized)

    intervals = {}
    for label, z_value in (('68', 1.0), ('95', 1.96)):
        lower = unnormalize_sxr(
            prediction_normalized - z_value * sigma_normalized, sxr_norm
        )
        upper = unnormalize_sxr(
            prediction_normalized + z_value * sigma_normalized, sxr_norm
        )
        intervals[f'lower_{label}'] = torch.clamp(lower, min=0)
        intervals[f'upper_{label}'] = torch.clamp(upper, min=0)

    return {
        'prediction_normalized': prediction_normalized,
        'variance_normalized': variance_normalized,
        'sigma_normalized': sigma_normalized,
        'sigma_dex': sigma_normalized * sxr_norm[1].float(),
        **intervals,
    }


def evaluate_model_on_dataset(model, dataset, batch_size=16, times=None, config_data=None,
                              save_weights=True, input_size=512, patch_size=16,
                              save_flux=False, save_patch_uncertainty=False):
    """
    Run batched inference on the dataset and yield predictions, attention maps, and flux data.
    
    Memory optimization: Moves enabled outputs to CPU once per batch before
    yielding individual records.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded solar flare prediction model.
    dataset : torch.utils.data.Dataset
        Dataset containing AIA images and corresponding SXR values.
    batch_size : int, default=16
        Number of samples per batch.
    times : list, optional
        List of timestamps corresponding to each sample.
    config_data : dict, optional
        YAML configuration dictionary.
    save_weights : bool, default=True
        Whether to save attention weights for visualization.
    input_size : int, default=512
        Input image resolution.
    patch_size : int, default=16
        Patch size for ViT-based models.
    save_flux : bool, default=False
        Whether to save flux contributions.

    Yields
    ------
    tuple
        Deterministic models retain the existing five-item tuple
        ``(prediction, ground_truth, attention_map, flux_map, global_index)``.
        Uncertainty models append a dictionary as item six. Quantile models
        use q0.5 for prediction/flux_map and include every raw and normalized
        head in that dictionary. Their per-head maps are saved under
        quantile_flux_path/q<probability>/<timestamp>.npy when save_flux=True.
    """
    model.eval()
    data_device = next(model.parameters()).device

    config_data = config_data or {}

    # Only pass multiprocessing-only options when workers exist; PyTorch
    # rejects them for num_workers=0.
    num_workers = int(config_data.get('num_workers', 4))
    pin_memory = _as_bool(config_data.get('pin_memory', True))
    
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    # Unwrapped model used as fallback for batches smaller than n_gpus
    base_model = _unwrap_model(model)
    predicts_uncertainty = bool(
        getattr(base_model, 'predicts_uncertainty', False)
    )
    is_quantile = getattr(base_model, 'uncertainty_kind', None) == 'quantile'
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'shuffle': False,
    }
    if num_workers > 0:
        prefetch_factor = int(config_data.get('prefetch_factor', 2))
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        loader_kwargs.update({
            'multiprocessing_context': config_data.get(
                'multiprocessing_context', 'spawn'
            ),
            'persistent_workers': _as_bool(
                config_data.get('persistent_workers', True)
            ),
            'prefetch_factor': prefetch_factor,
        })
    loader = DataLoader(dataset, **loader_kwargs)
    print(
        f"Data loader: {len(dataset):,} samples, {len(loader):,} batches, "
        f"{num_workers} workers; AIA arrays load one batch at a time.",
        flush=True,
    )
    
    # All models are ViTLocal with localized attention (no CLS token)
    grid_h, grid_w = input_size // patch_size, input_size // patch_size
    
    use_amp = (
        data_device.type == 'cuda'
        and _as_bool(config_data.get('use_amp', False))
    )
    amp_dtype = _resolve_amp_dtype(config_data, data_device)

    weight_dir = None
    if save_weights and config_data.get('weight_path'):
        weight_dir = Path(config_data['weight_path'])
        weight_dir.mkdir(parents=True, exist_ok=True)
    flux_dir = None
    if save_flux and config_data.get('flux_path'):
        flux_dir = Path(config_data['flux_path'])
        flux_dir.mkdir(parents=True, exist_ok=True)
    quantile_dirs = []
    if is_quantile and save_flux:
        quantile_root = config_data.get('quantile_flux_path')
        if quantile_root is None and flux_dir is not None:
            quantile_root = flux_dir.parent / 'quantile_flux'
        if quantile_root:
            for level in base_model.quantiles:
                directory = Path(quantile_root) / quantile_label(level)
                directory.mkdir(parents=True, exist_ok=True)
                quantile_dirs.append(directory)
    uncertainty_dir = None
    if save_patch_uncertainty and config_data.get('patch_uncertainty_path'):
        uncertainty_dir = Path(config_data['patch_uncertainty_path'])
        uncertainty_dir.mkdir(parents=True, exist_ok=True)
    try:
      with torch.inference_mode():
        print("Starting data loader workers...", flush=True)
        loader_iterator = iter(loader)
        print("Data loader ready.", flush=True)
        for batch_idx in range(len(loader)):
            print(
                f"Loading batch {batch_idx + 1}/{len(loader)}...",
                flush=True,
            )
            load_start = perf_counter()
            batch = next(loader_iterator)
            print(
                f"Loaded batch {batch_idx + 1}/{len(loader)} "
                f"({len(batch[0])} samples) in "
                f"{perf_counter() - load_start:.1f}s; running model...",
                flush=True,
            )
            model_start = perf_counter()
            aia_imgs = batch[0]
            sxr = batch[1]
            aia_imgs = aia_imgs.to(data_device, non_blocking=True)

            # Call model — optionally use AMP (set use_amp: true in config).
            # FP16 on V100 can spike peak memory due to FP32 fallbacks in attention.
            # Only compute the large attention-weight tensors when saving them.
            # Keep this explicit even though ViTLocal now defaults to False.
            # Fall back to single GPU for batches smaller than n_gpus to avoid
            # DataParallel crashing when some replicas receive empty inputs.
            active_model = (base_model
                            if isinstance(model, torch.nn.DataParallel) and aia_imgs.shape[0] < n_gpus
                            else model)
            with torch.autocast(
                device_type=data_device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                pred = active_model(
                    aia_imgs, return_attention=save_weights
                )

            # Gaussian, quantile and deterministic models have different
            # tuple layouts. Branch explicitly so variance is never mistaken
            # for an attention or flux map.
            variance_normalized = None
            patch_variance_raw = None
            quantile_patches = None
            if is_quantile:
                raw_quantiles, norm_quantiles, all_flux_contributions = pred[:3]
                weights = pred[3] if save_weights else None
                median_index = base_model.quantiles.index(0.5)
                predictions = raw_quantiles[:, median_index:median_index + 1]
                flux_contributions = (
                    all_flux_contributions[:, :, median_index] if save_flux else None
                )
                quantile_patches = all_flux_contributions if quantile_dirs else None
                uncertainty_batch = {
                    'prediction_normalized': norm_quantiles[:, median_index],
                    'quantile_crossing': (torch.diff(raw_quantiles, dim=1) < 0).any(dim=1),
                }
                for column, level in enumerate(base_model.quantiles):
                    raw_name, normalized_name = quantile_columns(level)
                    uncertainty_batch[raw_name] = raw_quantiles[:, column]
                    uncertainty_batch[normalized_name] = norm_quantiles[:, column]
                raw_quantiles = norm_quantiles = None
            elif predicts_uncertainty:
                if save_weights:
                    (
                        predictions, variance_normalized, weights,
                        all_flux_contributions, patch_variance_raw,
                    ) = pred
                else:
                    (
                        predictions, variance_normalized,
                        all_flux_contributions, patch_variance_raw,
                    ) = pred
                    weights = None
                flux_contributions = (
                    all_flux_contributions if save_flux else None
                )
                uncertainty_batch = uncertainty_from_prediction(
                    predictions, variance_normalized, base_model.sxr_norm
                )
            elif isinstance(pred, tuple) and len(pred) >= 3:
                predictions = pred[0]
                weights = pred[1] if save_weights else None
                flux_contributions = pred[2] if save_flux else None
                uncertainty_batch = None
            elif isinstance(pred, tuple) and len(pred) == 2:
                predictions = pred[0]
                weights = None
                flux_contributions = pred[1] if save_flux else None
                uncertainty_batch = None
            else:
                predictions = pred
                weights = None
                flux_contributions = None
                uncertainty_batch = None

            current_batch_size = predictions.shape[0]

            # Perform a small, fixed number of device-to-host transfers per
            # batch. Per-sample .cpu().item() calls serialize the GPU hundreds
            # of times and can dominate otherwise fast inference.
            predictions_cpu = predictions.detach().cpu().numpy()
            targets_cpu = sxr.detach().cpu().numpy()

            uncertainty_keys = None
            uncertainty_values_cpu = None
            if uncertainty_batch is not None:
                uncertainty_keys = tuple(uncertainty_batch)
                uncertainty_values_cpu = torch.stack(
                    [
                        uncertainty_batch[key].reshape(current_batch_size)
                        for key in uncertainty_keys
                    ],
                    dim=1,
                ).detach().float().cpu().numpy()

            weight_maps_cpu = None
            if save_weights and weights is not None:
                # [B, heads, queries, keys] -> [B, grid_h, grid_w].
                weight_maps_cpu = weights[-1].detach().mean(
                    dim=1
                ).mean(dim=1).reshape(
                    current_batch_size, grid_h, grid_w
                ).float().cpu().numpy()

            flux_maps_cpu = None
            if save_flux and flux_contributions is not None:
                flux_maps_cpu = flux_contributions.detach().reshape(
                    current_batch_size, grid_h, grid_w
                ).float().cpu().numpy()

            patch_std_maps_cpu = None
            quantile_maps_cpu = None
            if quantile_patches is not None:
                quantile_maps_cpu = quantile_patches.detach().reshape(
                    current_batch_size, grid_h, grid_w, len(base_model.quantiles)
                ).float().cpu().numpy()
            if save_patch_uncertainty and patch_variance_raw is not None:
                # Save the square root of each additive variance contribution.
                patch_std_maps_cpu = torch.sqrt(
                    torch.clamp(patch_variance_raw.detach(), min=0)
                ).reshape(
                    current_batch_size, grid_h, grid_w
                ).float().cpu().numpy()

            # Drop the CUDA output tree before yielding individual CPU records.
            # Keep allocator blocks cached so the next batch can reuse them.
            del pred, predictions, aia_imgs
            weights = None
            flux_contributions = None
            uncertainty_batch = None
            all_flux_contributions = None
            variance_normalized = None
            patch_variance_raw = None
            quantile_patches = None

            print(
                f"Model batch {batch_idx + 1}/{len(loader)} ready in "
                f"{perf_counter() - model_start:.1f}s; writing results...",
                flush=True,
            )

            for i in range(current_batch_size):
                global_idx = batch_idx * batch_size + i

                weight_data = None
                if weight_maps_cpu is not None:
                    candidate = weight_maps_cpu[i]
                    if not np.isnan(candidate).any():
                        weight_data = candidate
                        if weight_dir is not None and global_idx < len(times):
                            np.save(
                                weight_dir / f"{times[global_idx]}.npy",
                                weight_data,
                            )

                flux_data = None
                if flux_maps_cpu is not None:
                    flux_data = flux_maps_cpu[i]
                    if flux_dir is not None and global_idx < len(times):
                        np.save(
                            flux_dir / f"{times[global_idx]}.npy",
                            flux_data,
                        )

                if patch_std_maps_cpu is not None:
                    try:
                        if uncertainty_dir is not None and global_idx < len(times):
                            np.save(
                                uncertainty_dir / f"{times[global_idx]}.npy",
                                patch_std_maps_cpu[i],
                            )
                    except Exception as exc:
                        print(
                            f"Warning: could not save Gaussian patch "
                            f"uncertainty for sample {global_idx}: {exc}"
                        )
                if quantile_maps_cpu is not None:
                    if times is None:
                        raise ValueError('timestamps are required to save quantile maps')
                    for column, directory in enumerate(quantile_dirs):
                        np.save(directory / f'{times[global_idx]}.npy',
                                quantile_maps_cpu[i, :, :, column])
                uncertainty_data = None
                if uncertainty_values_cpu is not None:
                    uncertainty_data = {
                        key: uncertainty_values_cpu[i, column].item()
                        for column, key in enumerate(uncertainty_keys)
                    }
                result = (
                    predictions_cpu[i], targets_cpu[i], weight_data,
                    flux_data, global_idx,
                )
                yield (
                    (*result, uncertainty_data)
                    if uncertainty_data is not None else result
                )
    finally:
        # Explicitly shut down DataLoader workers so they don't linger into the
        # next condition and exhaust shared memory / semaphore limits.
        if 'loader_iterator' in locals():
            del loader_iterator
        del loader
        gc.collect()


def load_model_from_config(config_data):
    """
    Load the model from checkpoint based on configuration data.

    Parameters
    ----------
    config_data : dict
        Configuration dictionary from YAML file.

    Returns
    -------
    torch.nn.Module
        Loaded model ready for inference.
    """
    checkpoint_path = config_data['data']['checkpoint_path']

    load_device = resolve_inference_device(config_data)
    print(f"Using {load_device} for inference")

    if ".ckpt" in checkpoint_path:
        # Lightning checkpoint format
        requested_type = str(
            config_data.get('model_type', 'auto')
        ).strip().lower()
        model_types = {
            'deterministic': ViTLocal,
            'og_uncertainty': CoupledGaussianViTLocal,
            'uncertainty_coupled_single_query': CoupledGaussianViTLocal,
            'coupled_uncertainty': CoupledGaussianViTLocal,
            'coupled_pinball': CoupledPinballViTLocal,
        }
        if requested_type == 'auto':
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=False
            )
            model_class = _auto_checkpoint_model_class(checkpoint)
            del checkpoint
            gc.collect()
        elif requested_type in model_types:
            model_class = model_types[requested_type]
        else:
            raise ValueError(
                "model_type must be auto, deterministic, coupled_uncertainty, or coupled_pinball (legacy coupled aliases also work)"
            )
        print(f"Loading {model_class.__name__} model...")
        model = model_class.load_from_checkpoint(
            checkpoint_path, map_location=load_device, weights_only=False
        )
    else:
        state = torch.load(checkpoint_path, map_location=load_device, weights_only=False)
        model = state['model']
        model = model.to(load_device)
        print(f"Loaded serialized {type(model).__name__} model...")

    model.eval()

    inference_branch = str(config_data.get('inference_branch', 'final')).strip().lower()
    if inference_branch not in {'final', 'local'}:
        raise ValueError("inference_branch must be 'final' or 'local'")
    if inference_branch == 'local':
        raise ValueError(
            'Local-only inference belongs to the exploratory independent model; '
            'use python -m forecasting.experimental.inference.'
        )

    raw_multi_gpu = config_data.get('multi_gpu', False) if config_data else False
    use_multi_gpu = (_as_bool(raw_multi_gpu)
                     and load_device.type == 'cuda'
                     and torch.cuda.device_count() > 1)
    if use_multi_gpu:
        n_gpus = torch.cuda.device_count()
        if _as_bool(config_data.get('use_amp', False)):
            amp_dtype = _resolve_amp_dtype(config_data, load_device)
            model = _AutocastForwardWrapper(
                model, load_device.type, amp_dtype
            )
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel across {n_gpus} GPUs — ensure batch_size >= {n_gpus}")

    return model


def main():
    """
    Main function to execute solar flare model inference pipeline.

    Steps
    -----
    1. Parse YAML configuration.
    2. Load pretrained model and dataset.
    3. Run batched inference and optionally save attention/flux maps.
    4. Save predictions and ground truth results to CSV.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='inference_config.yaml', required=True,
                        help='Path to the inference configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    model_params = config_data.get('model_params', {})
    input_size = model_params.get('input_size', 512)
    patch_size = model_params.get('patch_size', 16)
    batch_size = model_params.get('batch_size', 10)
    no_weights = _as_bool(model_params.get('no_weights', False))
    no_flux = _as_bool(model_params.get('no_flux', False))
    no_patch_uncertainty = _as_bool(
        model_params.get('no_patch_uncertainty', False)
    )
    
    print(f"Using parameters from config:")
    print(f"  Input size: {input_size}\n  Patch size: {patch_size}\n  Batch size: {batch_size}\n  Skip weights: {no_weights}\n  Skip flux: {no_flux}")

    model = load_model_from_config(config_data)
    base_model = _unwrap_model(model)
    predicts_uncertainty = bool(
        getattr(base_model, 'predicts_uncertainty', False)
    )

    save_weights = not no_weights
    if no_weights:
        print("Skipping attention weight saving (no_weights=true).")
        print("  Note: This saves ~3GB per batch by not computing attention weights.")
    else:
        print("Will save attention weights during inference.")
        print("\n Memory note:")
        print("   - Attention weights from all layers use significant GPU memory")
        print("   - For ViT with 8 layers, 8 heads, 4096 patches: ~3GB+ per batch with attention!")
        print("   - If you get OOM errors, set no_weights=true to skip attention saving\n")

    save_flux = config_data and 'flux_path' in config_data and not no_flux
    if no_flux:
        print("Skipping flux contribution saving (no_flux=true).")
    elif config_data and 'flux_path' in config_data:
        print("Will save flux contributions during inference.")
    else:
        print("No flux path specified.")

    save_patch_uncertainty = (
        bool(getattr(base_model, 'predicts_patch_uncertainty', False))
        and bool(config_data.get('patch_uncertainty_path'))
        and not no_patch_uncertainty
    )
    if getattr(base_model, 'uncertainty_kind', None) == 'quantile':
        print(f'Quantile model detected: {base_model.quantiles}; CSV point prediction is q0.5.')
        if save_flux:
            print('Will save median flux maps and individual quantile contribution maps.')
    elif predicts_uncertainty:
        print("Uncertainty model detected; global intervals will be saved to CSV.")
        if save_patch_uncertainty:
            if getattr(
                base_model, 'patch_uncertainty_semantics', None
            ) == 'variance_attribution':
                print(
                    "Will save square-root global-variance contribution "
                    "maps in raw W/m² as NPY files."
                )
            else:
                print(
                    "Will save Gaussian patch standard-deviation maps "
                    "in raw W/m² as NPY files."
                )
        else:
            print("Patch uncertainty map saving is disabled.")
    elif config_data.get('patch_uncertainty_path'):
        print("Deterministic model detected; no uncertainty maps will be produced.")
    torch.backends.cudnn.benchmark = True
    matmul_precision = str(
        config_data.get('matmul_precision', 'high')
    ).strip().lower()
    if matmul_precision not in {'highest', 'high', 'medium'}:
        raise ValueError(
            "matmul_precision must be one of: highest, high, medium"
        )
    torch.set_float32_matmul_precision(matmul_precision)
    print(f"Float32 matmul precision: {matmul_precision}")

    print("Loading dataset...")
    # Check if running in prediction-only mode
    prediction_only = _as_bool(config_data.get('prediction_only', False))

    aia_norm_path = config_data['data'].get('aia_norm_path')
    aia_transform = (
        AIANormTransform.from_file(aia_norm_path, config_data['wavelengths'])
        if aia_norm_path else None
    )
    dataset = AIAGOESDataset(
        aia_dir=config_data['data']['aia_dir'],
        sxr_dir=config_data['data'].get('sxr_dir') if not prediction_only else None,
        wavelengths=config_data['wavelengths'],
        aia_transform=aia_transform,
        only_prediction=prediction_only,
        show_scan_progress=True,
    )

    dataset, times = select_inference_samples(
        dataset, config_data.get('sampling')
    )
    
    # The uncertainty checkpoint carries the exact training normalization. The
    # optional file is checked against it to catch accidentally mixed artifacts.
    checkpoint_sxr_norm = getattr(base_model, 'sxr_norm', None)
    if checkpoint_sxr_norm is not None:
        checkpoint_sxr_norm = torch.as_tensor(
            checkpoint_sxr_norm
        ).detach().cpu().float()

    if prediction_only:
        print("Running in prediction-only mode - no ground truth will be saved")
    elif config_data['data'].get('sxr_norm_path'):
        try:
            sxr_norm_file = np.load(config_data['data']['sxr_norm_path'])
            print("SXR normalization loaded successfully")
            if (checkpoint_sxr_norm is not None
                    and not np.allclose(
                        sxr_norm_file,
                        checkpoint_sxr_norm.numpy(),
                        rtol=1e-5,
                        atol=1e-7,
                    )):
                raise ValueError(
                    "SXR normalization file does not match the normalization "
                    f"stored in the checkpoint: file={sxr_norm_file.tolist()}, "
                    f"checkpoint={checkpoint_sxr_norm.tolist()}"
                )
        except FileNotFoundError:
            print(f"Warning: SXR normalization file not found: {config_data['data']['sxr_norm_path']}")

    timestamp, predictions, ground = [], [], []
    uncertainty_rows = []
    total_samples = len(times)
    print(f"Processing {total_samples} samples with batch size {batch_size}...")

    print("Running inference...")
    
    # Create progress bar
    pbar = tqdm(
        evaluate_model_on_dataset(
            model, dataset, batch_size, times, config_data,
            save_weights, input_size, patch_size, save_flux,
            save_patch_uncertainty,
        ),
        total=total_samples,
        desc="Inference",
        unit="sample",
        ncols=100
    )
    
    progress_update_interval = int(
        config_data.get('progress_update_interval', max(batch_size, 1))
    )
    if progress_update_interval <= 0:
        raise ValueError("progress_update_interval must be positive")

    for result in pbar:
        if len(result) == 6:
            prediction, sxr, weight, flux_data, idx, uncertainty = result
        else:
            prediction, sxr, weight, flux_data, idx = result
            uncertainty = None
        # ViTLocal models already return unnormalized predictions, so no need to unnormalize
        pred = prediction

        predictions.append(pred.item() if hasattr(pred, 'item') else float(pred))
        
        # Only collect ground truth if not in prediction-only mode
        if not prediction_only:
            ground.append(sxr.item() if hasattr(sxr, 'item') else float(sxr))
        else:
            ground.append(np.nan)

        if predicts_uncertainty:
            if uncertainty is None:
                raise RuntimeError(
                    "Uncertainty model did not return uncertainty outputs"
                )
            if prediction_only:
                uncertainty['groundtruth_normalized'] = np.nan
            else:
                target_raw = torch.tensor(
                    ground[-1], dtype=torch.float32
                )
                uncertainty['groundtruth_normalized'] = normalize_sxr(
                    target_raw, checkpoint_sxr_norm
                ).item()
            uncertainty_rows.append(uncertainty)
        
        timestamp.append(str(times[idx]))
        
        # set_postfix forces a terminal refresh, so do it periodically rather
        # than once for every sample in a large inference run.
        if (
            (idx + 1) % progress_update_interval == 0
            or idx + 1 == total_samples
        ):
            pbar.set_postfix({'sample': idx + 1, 'total': total_samples})

    output_df = pd.DataFrame({'timestamp': timestamp, 'predictions': predictions, 'groundtruth': ground})
    if predicts_uncertainty:
        uncertainty_df = pd.DataFrame(uncertainty_rows)
        output_df = pd.concat([output_df, uncertainty_df], axis=1)
    output_dir = Path(config_data['output_path']).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(config_data['output_path'], index=False)
    
    if prediction_only:
        print(f"Predictions saved to {config_data['output_path']} (prediction-only mode)")
    else:
        print(f"Predictions saved to {config_data['output_path']}")


if __name__ == '__main__':
    main()
