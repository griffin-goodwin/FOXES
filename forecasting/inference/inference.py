"""
Inference Script for Solar Flare Prediction Models
=================================================

This script performs automated inference on solar flare prediction datasets using trained models such as
Vision Transformers (ViT, ViT-Patch, ViT-Local), HybridIrradianceModel, or LinearIrradianceModel.
It computes soft X-ray (SXR) predictions, saves attention weights, flux contributions, and final outputs.

The workflow includes:
- Loading configuration parameters from a YAML file.
- Resolving dynamic variables in the config.
- Loading the model checkpoint and preparing it for inference.
- Performing batched evaluation over AIA/GOES datasets.
- Saving predicted fluxes, ground truth, and visualization-ready artifacts.

"""

import argparse
import re
import sys
import pandas as pd
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import yaml
import torch.nn.functional as F
from torch.nn import HuberLoss

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.data_loaders.SDOAIA_dataloader import AIA_GOESDataset
import forecasting.models as models
from forecasting.models.vit_patch_model_local import ViTLocal

from forecasting.training.callback import unnormalize_sxr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def has_attention_weights(model):
    """Check if model supports attention weights"""
    return hasattr(model, 'attention') or isinstance(model, ViTLocal)


def is_localized_attention_model(model):
    """
    Check if the model uses localized attention (no CLS token).

    Parameters
    ----------
    model : torch.nn.Module
        Model instance.

    Returns
    -------
    bool
        True if the model uses localized attention (ViTLocal).
    """
    return isinstance(model, ViTLocal)


def evaluate_model_on_dataset(model, dataset, batch_size=16, times=None, config_data=None,
                              save_weights=True, input_size=512, patch_size=16):
    """
    Run batched inference on the dataset and yield predictions, attention maps, and flux data.

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

    Yields
    ------
    tuple
        (predictions, ground_truth, attention_map, flux_map, global_index)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    supports_attention = has_attention_weights(model) and save_weights
    save_flux = config_data and 'flux_path' in config_data
    sxr_norm = np.load(config_data['data']['sxr_norm_path']) if config_data and 'data' in config_data and 'sxr_norm_path' in config_data['data'] else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            aia_imgs = batch[0]
            sxr = batch[1]
            aia_imgs = aia_imgs.to(device, non_blocking=True)

            if supports_attention:
                pred = model(aia_imgs, return_attention=True)
            else:
                pred = model(aia_imgs)

            if isinstance(pred, tuple) and len(pred) >= 3:
                predictions = pred[0]
                weights = pred[1] if supports_attention else None
                flux_contributions = pred[2] if save_flux else None
            elif isinstance(pred, tuple) and len(pred) > 1:
                predictions = pred[0]
                weights = pred[1] if supports_attention else None
                flux_contributions = None
            else:
                predictions = pred
                weights = None
                flux_contributions = None

            batch_weights = []
            if supports_attention and weights is not None:
                current_batch_size = predictions.shape[0]
                is_localized = is_localized_attention_model(model)

                for i in range(current_batch_size):
                    try:
                        last_layer_attention = weights[-1][i]
                        if last_layer_attention is None:
                            continue
                        avg_attention = last_layer_attention.mean(dim=0)
                        if torch.isnan(avg_attention).any():
                            continue

                        if is_localized:
                            patch_attention = avg_attention.mean(dim=0).cpu()
                        else:
                            cls_attention = avg_attention[0, 1:].cpu()
                            patch_attention = cls_attention

                        grid_h, grid_w = input_size // patch_size, input_size // patch_size
                        attention_map = patch_attention.reshape(grid_h, grid_w)
                        batch_weights.append(attention_map.numpy())

                    except Exception as e:
                        grid_h, grid_w = input_size // patch_size, input_size // patch_size
                        fallback_map = torch.zeros(grid_h * grid_w).reshape(grid_h, grid_w).numpy()
                        batch_weights.append(fallback_map)

                if config_data and 'weight_path' in config_data:
                    save_batch_weights(batch_weights, batch_idx, batch_size, times, config_data['weight_path'])

            batch_flux_contributions = []
            if save_flux and flux_contributions is not None:
                current_batch_size = predictions.shape[0]
                for i in range(current_batch_size):
                    flux_contrib = flux_contributions[i].cpu()
                    grid_h, grid_w = input_size // patch_size, input_size // patch_size
                    flux_contrib_map = flux_contrib.reshape(grid_h, grid_w)
                    batch_flux_contributions.append(flux_contrib_map.numpy())
                save_batch_flux_contributions(batch_flux_contributions, batch_idx, batch_size, times, config_data['flux_path'], sxr_norm)

            current_batch_size = predictions.shape[0]
            for i in range(current_batch_size):
                global_idx = batch_idx * batch_size + i
                weight_data = batch_weights[i] if (supports_attention and batch_weights) else None
                flux_data = batch_flux_contributions[i] if batch_flux_contributions else None
                yield (predictions[i].cpu().numpy(), sxr[i].cpu().numpy(), weight_data, flux_data, global_idx)


def save_batch_flux_contributions(batch_flux_contributions, batch_idx, batch_size, times, flux_path, sxr_norm=None):
    """
    Save all flux contributions in a batch efficiently using parallel threads.

    Parameters
    ----------
    batch_flux_contributions : list of np.ndarray
        List of flux contribution maps for each sample.
    batch_idx : int
        Batch index.
    batch_size : int
        Number of samples per batch.
    times : list of str
        Corresponding timestamps.
    flux_path : str
        Directory path to save flux files.
    sxr_norm : np.ndarray, optional
        Normalization constants for unnormalization.
    """
    flux_dir = Path(flux_path)
    flux_dir.mkdir(parents=True, exist_ok=True)

    def save_single_flux(args):
        flux_contrib, filepath = args
        np.savetxt(filepath, flux_contrib, delimiter=",")

    save_args = []
    for i, flux_contrib in enumerate(batch_flux_contributions):
        global_idx = batch_idx * batch_size + i
        if global_idx < len(times):
            filepath = flux_path + f"{times[global_idx]}"
            save_args.append((flux_contrib, filepath))

    with ThreadPoolExecutor(max_workers=min(11, len(save_args))) as executor:
        executor.map(save_single_flux, save_args)


def save_batch_weights(batch_weights, batch_idx, batch_size, times, weight_path):
    """
    Save all attention weights from a batch efficiently in parallel.

    Parameters
    ----------
    batch_weights : list of np.ndarray
        Attention maps for each sample.
    batch_idx : int
        Current batch index.
    batch_size : int
        Number of samples in batch.
    times : list of str
        List of timestamps.
    weight_path : str
        Output directory for weight files.
    """
    weight_dir = Path(weight_path)
    weight_dir.mkdir(parents=True, exist_ok=True)

    def save_single_weight(args):
        weight, filepath = args
        np.savetxt(filepath, weight, delimiter=",")

    save_args = []
    for i, weight in enumerate(batch_weights):
        global_idx = batch_idx * batch_size + i
        if global_idx < len(times):
            filepath = os.path.join(weight_path, f"{times[global_idx]}")
            save_args.append((weight, filepath))

    with ThreadPoolExecutor(max_workers=min(11, len(save_args))) as executor:
        executor.map(save_single_weight, save_args)


def save_weights_async(weight_data_queue, weight_path):
    """
    Asynchronously save attention weights to disk using threads.

    Parameters
    ----------
    weight_data_queue : list of tuple
        Each entry contains (weight_data, filepath).
    weight_path : str
        Output directory path for saving weights.
    """
    def save_single_weight(args):
        weight, filepath = args
        np.savetxt(filepath, weight, delimiter=",")

    with ThreadPoolExecutor(max_workers=11) as executor:
        executor.map(save_single_weight, weight_data_queue)


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
    model_type = config_data['model']
    wavelengths = config_data.get('wavelengths', [94, 131, 171, 193, 211, 304])

    print(f"Loading {model_type} model...")

    if ".ckpt" in checkpoint_path:
        # Lightning checkpoint format

        if model_type.lower() == 'vitlocal':
            model = ViTLocal.load_from_checkpoint(checkpoint_path)
        else:
            try:
                model_class = getattr(models, model_type)
                model = model_class.load_from_checkpoint(checkpoint_path)
            except AttributeError:
                raise ValueError(f"Unknown model type: {model_type}.")
    else:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = state['model']

    model.eval()
    model.to(device)
    return model


def main():
    """
    Main function to execute solar flare model inference pipeline.

    Steps
    -----
    1. Parse YAML configuration and resolve ${variable} placeholders.
    2. Load pretrained model and dataset.
    3. Run batched inference and optionally save attention/flux maps.
    4. Save predictions and ground truth results to CSV.
    """
    def resolve_config_variables(config_dict):
        """Recursively resolve ${variable} references within config."""
        variables = {}
        for key, value in config_dict.items():
            if isinstance(value, str) and not value.startswith('${'):
                variables[key] = value

        def substitute_value(value, variables):
            if isinstance(value, str):
                pattern = r'\$\{([^}]+)\}'
                for match in re.finditer(pattern, value):
                    var_name = match.group(1)
                    if var_name in variables:
                        value = value.replace(f'${{{var_name}}}', variables[var_name])
            return value

        def recursive_substitute(obj, variables):
            if isinstance(obj, dict):
                return {k: recursive_substitute(v, variables) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_substitute(item, variables) for item in obj]
            else:
                return substitute_value(obj, variables)

        return recursive_substitute(config_dict, variables)

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='inference_config.yaml', required=True,
                        help='Path to the inference configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    config_data = resolve_config_variables(config_data)
    sys.modules['models'] = models

    model_params = config_data.get('model_params', {})
    input_size = model_params.get('input_size', 512)
    patch_size = model_params.get('patch_size', 16)
    batch_size = model_params.get('batch_size', 10)
    no_weights = model_params.get('no_weights', False)

    print(f"Using parameters from config:")
    print(f"  Input size: {input_size}\n  Patch size: {patch_size}\n  Batch size: {batch_size}\n  Skip weights: {no_weights}")

    model = load_model_from_config(config_data)

    save_weights = not no_weights and has_attention_weights(model)
    if no_weights:
        print("Skipping attention weight saving (no_weights=true).")
    elif not has_attention_weights(model):
        print(f"Model {type(model).__name__} does not support attention weights.")
    else:
        print("Will save attention weights during inference.")

    save_flux = config_data and 'flux_path' in config_data
    if save_flux:
        print("Will save flux contributions during inference.")
    else:
        print("No flux path specified.")

    torch.backends.cudnn.benchmark = True

    print("Loading dataset...")
    if config_data['SolO'] == "true":
        dataset = AIA_GOESDataset(aia_dir=config_data['SolO_data']['solo_img_dir'] + '/test',
                                  sxr_dir=config_data['SolO_data']['sxr_dir'] + '/test',
                                  wavelengths=[94, 131], only_prediction=True)
    elif config_data['Stereo'] == "true":
        dataset = AIA_GOESDataset(aia_dir=config_data['Stereo_data']['stereo_img_dir'],
                                  sxr_dir=config_data['Stereo_data']['sxr_dir'] + '/test',
                                  wavelengths=[171, 193, 211, 304], only_prediction=True)
    else:
        dataset = AIA_GOESDataset(aia_dir=config_data['data']['aia_dir'] + '/test',
                                  sxr_dir=config_data['data']['sxr_dir'] + '/test',
                                  wavelengths=config_data['wavelengths'])

    times = dataset.samples
    sxr_norm = np.load(config_data['data']['sxr_norm_path'])

    timestamp, predictions, ground = [], [], []
    total_samples = len(times)
    print(f"Processing {total_samples} samples with batch size {batch_size}...")

    print("Running inference...")
    for prediction, sxr, weight, flux_data, idx in evaluate_model_on_dataset(
            model, dataset, batch_size, times, config_data, save_weights, input_size, patch_size
    ):
        # Unnormalize prediction only if not ViTPatch / ViTLocal
        if not isinstance(model, ViTLocal):
            pred = unnormalize_sxr(prediction, sxr_norm)
        else:
            pred = prediction

        predictions.append(pred.item() if hasattr(pred, 'item') else float(pred))
        ground.append(sxr.item() if hasattr(sxr, 'item') else float(sxr))
        timestamp.append(str(times[idx]))

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total_samples}")

    output_df = pd.DataFrame({'timestamp': timestamp, 'predictions': predictions, 'groundtruth': ground})
    output_dir = Path(config_data['output_path']).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(config_data['output_path'], index=False)
    print(f"Predictions saved to {config_data['output_path']}")


if __name__ == '__main__':
    main()
