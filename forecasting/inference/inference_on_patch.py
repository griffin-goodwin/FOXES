"""
----------------------------------

This script performs inference for solar flare prediction models such as ViT, Hybrid Irradiance, and FusionViT-Hybrid architectures. 
It includes functionality for efficient batch processing, attention weight extraction, and flux contribution visualization.
It supports AIA/GOES datasets for solar irradiance forecasting and outputs predictions, attention maps, and flux distributions.

The code maintains compatibility with multiple dataset configurations (AIA, SolO, Stereo) and supports both Lightning and standard PyTorch checkpoints.

"""

import argparse
import re
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from forecasting.data_loaders.SDOAIA_dataloader import AIA_GOESDataset
import forecasting.models as models
from forecasting.models.vit_patch_model import ViT
from forecasting.models import FusionViTHybrid
from forecasting.models.linear_and_hybrid import HybridIrradianceModel
from forecasting.training.callback import unnormalize_sxr
import yaml
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def has_attention_weights(model):
    """
    Check if a given model supports attention weight extraction.

    Parameters
    ----------
    model : torch.nn.Module
        The model to check (e.g., ViT, FusionViTHybrid).

    Returns
    -------
    bool
        True if the model supports attention extraction; False otherwise.
    """
    return hasattr(model, 'attention') or isinstance(model, ViT) or isinstance(model, FusionViTHybrid)


def save_batch_flux_contributions(batch_flux_contributions, batch_idx, batch_size, times, flux_path, sxr_norm=None):
    """
    Save flux contribution maps for each batch sample in parallel.

    Parameters
    ----------
    batch_flux_contributions : list of np.ndarray
        Flux contribution maps for the batch (each a 2D numpy array).
    batch_idx : int
        Index of the current batch in the overall dataset.
    batch_size : int
        Number of samples per batch.
    times : list of str
        List of timestamp identifiers for samples.
    flux_path : str
        Directory path to save flux contribution maps.
    sxr_norm : np.ndarray, optional
        Normalization factors for scaling saved flux values.
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


def evaluate_model_on_dataset(model, dataset, batch_size=16, times=None, config_data=None, save_weights=True,
                              input_size=512, patch_size=16):
    """
    Perform batched inference on the dataset and optionally save attention maps and flux contributions.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model for inference.
    dataset : torch.utils.data.Dataset
        Dataset object providing image and SXR pairs.
    batch_size : int, default=16
        Number of samples per inference batch.
    times : list of str, optional
        List of timestamps corresponding to dataset samples.
    config_data : dict, optional
        YAML configuration dictionary for inference paths and parameters.
    save_weights : bool, default=True
        Whether to save attention weights for visualization.
    input_size : int, default=512
        Image input resolution.
    patch_size : int, default=16
        Patch size used in ViT tokenization.

    Yields
    ------
    tuple
        Contains predictions, ground truth SXR values, attention maps, flux contributions, and global indices.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    supports_attention = has_attention_weights(model) and save_weights
    save_flux = config_data and 'flux_path' in config_data
    sxr_norm = np.load(config_data['data']['sxr_norm_path']) if config_data and 'data' in config_data and 'sxr_norm_path' in config_data['data'] else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            aia_imgs, sxr = batch[0], batch[1]
            aia_imgs = aia_imgs.to(device, non_blocking=True)
            pred = model(aia_imgs)

            if isinstance(pred, tuple) and len(pred) >= 4:
                predictions, weights, flux_contributions, total_flux_from_patches = pred[0], pred[1], pred[2], pred[3]
            elif isinstance(pred, tuple) and len(pred) > 1:
                predictions, weights, flux_contributions = pred[0], pred[1], pred[2]
                total_flux_from_patches = None
            else:
                predictions, weights, flux_contributions, total_flux_from_patches = pred, None, None, None

            batch_weights = []
            if supports_attention and weights is not None:
                current_batch_size = predictions.shape[0]
                for i in range(current_batch_size):
                    last_layer_attention = weights[-1][i]
                    avg_attention = last_layer_attention.mean(dim=0)
                    cls_attention = avg_attention[0, 1:].cpu()
                    grid_h, grid_w = input_size // patch_size, input_size // patch_size
                    attention_map = cls_attention.reshape(grid_h, grid_w)
                    batch_weights.append(attention_map.numpy())

                if config_data and 'weight_path' in config_data:
                    print(f"Saving {len(batch_weights)} attention weights to {config_data['weight_path']}")
                    save_batch_weights(batch_weights, batch_idx, batch_size, times, config_data['weight_path'])

            batch_flux_contributions = []
            if save_flux and flux_contributions is not None:
                current_batch_size = predictions.shape[0]
                for i in range(current_batch_size):
                    flux_contrib = flux_contributions[i].cpu()
                    grid_h, grid_w = input_size // patch_size, input_size // patch_size
                    flux_contrib_map = flux_contrib.reshape(grid_h, grid_w)
                    batch_flux_contributions.append(flux_contrib_map.numpy())
                save_batch_flux_contributions(batch_flux_contributions, batch_idx, batch_size, times,
                                              config_data['flux_path'], sxr_norm)

            current_batch_size = predictions.shape[0]
            for i in range(current_batch_size):
                global_idx = batch_idx * batch_size + i
                weight_data = batch_weights[i] if (supports_attention and batch_weights) else None
                flux_data = batch_flux_contributions[i] if batch_flux_contributions else None
                yield (predictions[i].cpu().numpy(), sxr[i].cpu().numpy(),
                       weight_data, flux_data, global_idx)


def save_batch_weights(batch_weights, batch_idx, batch_size, times, weight_path):
    """
    Save attention weight maps for a batch of samples.

    Parameters
    ----------
    batch_weights : list of np.ndarray
        List of attention maps per sample.
    batch_idx : int
        Index of the current batch.
    batch_size : int
        Number of samples per batch.
    times : list of str
        Timestamps corresponding to each sample.
    weight_path : str
        Directory path to save attention maps.
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
    Save attention weights asynchronously using multiple threads.

    Parameters
    ----------
    weight_data_queue : list of tuple
        Each element is a tuple of (weight_data, filepath).
    weight_path : str
        Directory path to save weights.
    """
    def save_single_weight(args):
        weight, filepath = args
        np.savetxt(filepath, weight, delimiter=",")

    with ThreadPoolExecutor(max_workers=11) as executor:
        executor.map(save_single_weight, weight_data_queue)


def load_model_from_config(config_data):
    """
    Load and initialize a model instance from a YAML configuration.

    Parameters
    ----------
    config_data : dict
        Configuration dictionary containing model name, checkpoint path, and type.

    Returns
    -------
    torch.nn.Module
        Loaded model ready for inference.
    """
    checkpoint_path = config_data['data']['checkpoint_path']
    model_type = config_data['model']

    print(f"Loading {model_type} model...")

    if ".ckpt" in checkpoint_path:
        if model_type.lower() == 'vit':
            model = ViT.load_from_checkpoint(checkpoint_path)
        elif model_type.lower() in ('hybrid', 'hybridirradiancemodel'):
            model = HybridIrradianceModel.load_from_checkpoint(checkpoint_path)
        else:
            try:
                model_class = getattr(models, model_type)
                model = model_class.load_from_checkpoint(checkpoint_path)
            except AttributeError:
                raise ValueError(f"Unknown model type: {model_type}")
    else:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = state['model']

    model.eval()
    model.to(device)
    return model


def main():
    """
    Main entry point for inference.

    Loads the configuration file, initializes the model and dataset, performs batched inference,
    saves predictions, and optionally exports attention weights and flux contributions.
    """
    def resolve_config_variables(config_dict):
        """Recursively resolve ${variable} references within YAML configs."""
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

    parser = argparse.ArgumentParser(description="Perform solar flare inference using ViT or Hybrid models")
    parser.add_argument('-config', type=str, default='config.yaml', required=True, help='Path to config YAML.')
    parser.add_argument('-input_size', type=int, default=512, help='Input image resolution')
    parser.add_argument('-patch_size', type=int, default=16, help='Patch size for tokenization')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--no_weights', action='store_true', help='Skip saving attention weights to speed up')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    config_data = resolve_config_variables(config_data)
    sys.modules['models'] = models
    model = load_model_from_config(config_data)

    save_weights = not args.no_weights and has_attention_weights(model)
    if args.no_weights:
        print("Skipping attention weight saving (--no_weights flag)")
    elif not has_attention_weights(model):
        print(f"Model {type(model).__name__} doesn't support attention weights")
    else:
        print("Will save attention weights during inference")

    save_flux = config_data and 'flux_path' in config_data
    if save_flux:
        print("Will save flux contributions during inference")
    else:
        print("No flux path specified - skipping flux contribution saving")

    torch.backends.cudnn.benchmark = True

    print("Loading dataset...")
    if config_data['SolO'] == "true":
        dataset = AIA_GOESDataset(
            aia_dir=config_data['SolO_data']['solo_img_dir'] + '/test',
            sxr_dir=config_data['SolO_data']['sxr_dir'] + '/test', wavelengths=[94, 131], only_prediction=True
        )
    elif config_data['Stereo'] == "true":
        dataset = AIA_GOESDataset(
            aia_dir=config_data['Stereo_data']['stereo_img_dir'],
            sxr_dir=config_data['Stereo_data']['sxr_dir'] + '/test', wavelengths=[94, 131, 171, 193],
            only_prediction=True
        )
    else:
        dataset = AIA_GOESDataset(
            aia_dir=config_data['data']['aia_dir'] + '/test',
            sxr_dir=config_data['data']['sxr_dir'] + '/test', wavelengths=config_data['wavelengths']
        )

    times = dataset.samples
    sxr_norm = np.load(config_data['data']['sxr_norm_path'])

    total_samples = len(times)
    timestamp, predictions, ground = [], [], []

    print(f"Processing {total_samples} samples with batch size {args.batch_size}...")

    if config_data['mc']['active'] == "false":
        print("Running inference without MC Dropout")
        for prediction, sxr, weight, flux_contrib, idx in evaluate_model_on_dataset(
                model, dataset, args.batch_size, times, config_data, save_weights, args.input_size, args.patch_size):
            predictions.append(prediction.item() if hasattr(prediction, 'item') else float(prediction))
            ground.append(sxr.item() if hasattr(sxr, 'item') else float(sxr))
            timestamp.append(str(times[idx]))

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{total_samples}")

        print("Creating output DataFrame...")
        output_df = pd.DataFrame({
            'timestamp': timestamp,
            'predictions': predictions,
            'groundtruth': ground
        })

        output_dir = Path(config_data['output_path']).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(config_data['output_path'], index=False)
        print(f"Predictions saved to {config_data['output_path']}")


if __name__ == '__main__':
    main()
