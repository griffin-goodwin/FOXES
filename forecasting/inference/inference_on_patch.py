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
from forecasting.models.linear_and_hybrid import HybridIrradianceModel  # Add your hybrid model import
from forecasting.training.callback import unnormalize_sxr
import yaml
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def has_attention_weights(model):
    """Check if model supports attention weights"""
    return hasattr(model, 'attention') or isinstance(model, ViT) or isinstance(model, FusionViTHybrid)


def save_batch_flux_contributions(batch_flux_contributions, batch_idx, batch_size, times, flux_path, sxr_norm=None):
    """Save all flux contributions in a batch efficiently - same format as attention weights"""
    flux_dir = Path(flux_path)
    flux_dir.mkdir(parents=True, exist_ok=True)

    # Use ThreadPoolExecutor to save files in parallel
    def save_single_flux(args):
        flux_contrib, filepath = args
        # Note: flux contributions are already reshaped to 2D spatial maps
        # Unnormalize flux contributions before saving if needed
        np.savetxt(filepath, flux_contrib, delimiter=",")

    # Prepare arguments for parallel saving
    save_args = []
    for i, flux_contrib in enumerate(batch_flux_contributions):
        global_idx = batch_idx * batch_size + i
        if global_idx < len(times):  # Make sure we don't go out of bounds
            # Save with same naming convention as attention weights (without "flux_" prefix)
            filepath = flux_path + f"{times[global_idx]}"
            save_args.append((flux_contrib, filepath))

    # Save all flux contributions in this batch in parallel
    with ThreadPoolExecutor(max_workers=min(11, len(save_args))) as executor:
        executor.map(save_single_flux, save_args)


def evaluate_model_on_dataset(model, dataset, batch_size=16, times=None, config_data=None, save_weights=True,
                              input_size=512, patch_size=16):
    """Optimized generator with batch processing and weight saving"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Check if this model supports attention weights
    supports_attention = has_attention_weights(model) and save_weights
    save_flux = config_data and 'flux_path' in config_data
    sxr_norm = np.load(
        config_data['data']['sxr_norm_path']) if config_data and 'data' in config_data and 'sxr_norm_path' in \
                                                 config_data['data'] else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Correct unpacking based on your data structure
            aia_imgs = batch[0]  # Get aia_img from inputs
            sxr = batch[1]
            # Move to device (it's already a tensor)
            aia_imgs = aia_imgs.to(device, non_blocking=True)

            # Get model predictions for entire batch
            pred = model(aia_imgs)

            # Handle different model output formats - now expecting 4 outputs
            if isinstance(pred, tuple) and len(pred) >= 4:
                predictions = pred[0]  # sxr predictions: [batch_size, ...]
                weights = pred[1] if supports_attention else None  # attention_weights: [batch_size, heads, L, S ...]
                flux_contributions = pred[2] if save_flux else None  # flux_contributions: [batch_size, num_patches]
                total_flux_from_patches = pred[3]  # total_flux_from_patches: [batch_size, ...]
            elif isinstance(pred, tuple) and len(pred) > 1:
                predictions = pred[0]  # Shape: [batch_size, ...]
                weights = pred[1] if supports_attention else None  # Shape: [batch_size, heads, L, S ...]
                flux_contributions = pred[2]
            else:
                predictions = pred
                weights = None
                flux_contributions = None
                total_flux_from_patches = None

            # Process entire batch at once for weights if needed
            batch_weights = []
            if supports_attention and weights is not None:
                current_batch_size = predictions.shape[0]
                for i in range(current_batch_size):
                    # Process attention weights for this item - matching callback approach
                    # select last layer and appropriate item from batch
                    last_layer_attention = weights[-1][i]  # Get i-th item from batch [num_heads, seq_len, seq_len]
                    # Average across attention heads
                    avg_attention = last_layer_attention.mean(dim=0)  # [seq_len, seq_len]

                    # Get attention from CLS token to patches (exclude CLS->CLS)
                    cls_attention = avg_attention[0, 1:].cpu()  # [num_patches] - 1D array

                    # Calculate grid size based on patch size (assuming 8x8 patches)
                    grid_h, grid_w = input_size // patch_size, input_size // patch_size  # Should be 64, 64

                    # Reshape CLS attention to spatial grid
                    attention_map = cls_attention.reshape(grid_h, grid_w)  # [64, 64]

                    batch_weights.append(attention_map.numpy())

                # Save all weights in this batch at once
                if config_data and 'weight_path' in config_data:
                    print(f"Saving {len(batch_weights)} attention weights to {config_data['weight_path']}")
                    save_batch_weights(batch_weights, batch_idx, batch_size, times, config_data['weight_path'])

            # Process and save flux contributions
            batch_flux_contributions = []
            if save_flux and flux_contributions is not None:
                current_batch_size = predictions.shape[0]
                for i in range(current_batch_size):
                    # Get flux contributions for this sample and reshape to spatial grid
                    flux_contrib = flux_contributions[i].cpu()  # [num_patches]                    # Reshape flux contributions to spatial grid (same as attention maps)
                    grid_h, grid_w = input_size // patch_size, input_size // patch_size  # Should be 32x32 for 512/16
                    flux_contrib_map = flux_contrib.reshape(grid_h, grid_w)  # [grid_h, grid_w]
                    batch_flux_contributions.append(flux_contrib_map.numpy())

                # Save flux contributions
                save_batch_flux_contributions(batch_flux_contributions, batch_idx, batch_size, times,
                                              config_data['flux_path'], sxr_norm)

            # Yield batch results
            current_batch_size = predictions.shape[0]
            for i in range(current_batch_size):
                global_idx = batch_idx * batch_size + i
                weight_data = batch_weights[i] if (supports_attention and batch_weights) else None
                flux_data = batch_flux_contributions[i] if batch_flux_contributions else None
                yield (predictions[i].cpu().numpy(), sxr[i].cpu().numpy(),
                       weight_data, flux_data, global_idx)


def save_batch_weights(batch_weights, batch_idx, batch_size, times, weight_path):
    """Save all weights in a batch efficiently"""
    weight_dir = Path(weight_path)
    weight_dir.mkdir(parents=True, exist_ok=True)

    # Use ThreadPoolExecutor to save files in parallel
    def save_single_weight(args):
        weight, filepath = args
        np.savetxt(filepath, weight, delimiter=",")

    # Prepare arguments for parallel saving
    save_args = []
    for i, weight in enumerate(batch_weights):
        global_idx = batch_idx * batch_size + i
        if global_idx < len(times):  # Make sure we don't go out of bounds
            filepath = os.path.join(weight_path, f"{times[global_idx]}")
            save_args.append((weight, filepath))

    # Save all weights in this batch in parallel
    with ThreadPoolExecutor(max_workers=min(11, len(save_args))) as executor:
        executor.map(save_single_weight, save_args)


def save_weights_async(weight_data_queue, weight_path):
    """Async function to save weights to disk"""

    def save_single_weight(args):
        weight, filepath = args
        np.savetxt(filepath, weight, delimiter=",")

    with ThreadPoolExecutor(max_workers=11) as executor:
        executor.map(save_single_weight, weight_data_queue)


def load_model_from_config(config_data):
    """Load model based on config specifications"""
    checkpoint_path = config_data['data']['checkpoint_path']
    model_type = config_data['model']  # Default to ViT for backward compatibility

    print(f"Loading {model_type} model...")

    if ".ckpt" in checkpoint_path:
        # Lightning checkpoint format
        if model_type.lower() == 'vit':
            model = ViT.load_from_checkpoint(checkpoint_path)
        elif model_type.lower() == 'hybrid' or model_type.lower() == 'hybridirradiancemodel':
            model = HybridIrradianceModel.load_from_checkpoint(checkpoint_path)
        else:
            # Try to dynamically load the model class
            try:
                model_class = getattr(models, model_type)
                model = model_class.load_from_checkpoint(checkpoint_path)
            except AttributeError:
                raise ValueError(f"Unknown model type: {model_type}. Available types include: ViT, HybridIrradianceModel, FusionViTHybrid")
    else:
        # Regular PyTorch checkpoint
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = state['model']

    model.eval()
    model.to(device)
    return model


def main():
    def resolve_config_variables(config_dict):
        """Recursively resolve ${variable} references within the config"""
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
    parser.add_argument('-config', type=str, default='config.yaml', required=True, help='Path to config YAML.')
    parser.add_argument('-input_size', type=int, default=512, help='Input size for the model')
    parser.add_argument('-patch_size', type=int, default=16, help='Patch size for the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--no_weights', action='store_true', help='Skip saving attention weights to speed up')
    args = parser.parse_args()

    # Load config with variable substitution
    with open(args.config, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    config_data = resolve_config_variables(config_data)
    sys.modules['models'] = models

    # Load model based on config
    model = load_model_from_config(config_data)

    # Check if model supports attention and user wants to save weights
    save_weights = not args.no_weights and has_attention_weights(model)

    if args.no_weights:
        print("Skipping attention weight saving (--no_weights flag)")
    elif not has_attention_weights(model):
        print(f"Model {type(model).__name__} doesn't support attention weights - skipping weight saving")
    else:
        print("Will save attention weights during inference")

    # Check if flux contributions should be saved
    save_flux = config_data and 'flux_path' in config_data
    if save_flux:
        print("Will save flux contributions during inference")
    else:
        print("No flux path specified - skipping flux contribution saving")

    # Enable optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

    # Dataset
    print("Loading dataset...")
    if config_data['SolO'] == "true":
        print("Using SolO dataset configuration")
        dataset = AIA_GOESDataset(
            aia_dir=config_data['SolO_data']['solo_img_dir'] + '/test',
            sxr_dir=config_data['SolO_data']['sxr_dir'] + '/test', wavelengths=[94, 131], only_prediction=True
        )
        print(dataset)
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

    # Pre-allocate lists for better memory performance
    total_samples = len(times)
    timestamp = []
    predictions = []
    ground = []

    print(f"Processing {total_samples} samples with batch size {args.batch_size}...")

    if config_data['mc']['active'] == "false":
        print("Running inference without MC Dropout")
        for prediction, sxr, weight, flux_contrib, idx in evaluate_model_on_dataset(
                model, dataset, args.batch_size, times, config_data, save_weights, args.input_size, args.patch_size
        ):
            # Unnormalize prediction


            # Store results
            predictions.append(prediction.item() if hasattr(prediction, 'item') else float(prediction))
            ground.append(sxr.item() if hasattr(sxr, 'item') else float(sxr))
            timestamp.append(str(times[idx]))

            # Progress update
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{total_samples}")

        if save_weights:
            print("All weights saved during batch processing!")
        else:
            print("Inference completed (no weights saved)!")

        if save_flux:
            print("All flux contributions saved during batch processing!")

        # Create and save results DataFrame
        print("Creating output DataFrame...")
        output_df = pd.DataFrame({
            'timestamp': timestamp,
            'predictions': predictions,
            'groundtruth': ground
        })

        print(output_df.head())
        # Make output directory if it doesn't exist
        output_dir = Path(config_data['output_path']).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(config_data['output_path'], index=False)
        print(f"Predictions saved to {config_data['output_path']}")



if __name__ == '__main__':
    main()