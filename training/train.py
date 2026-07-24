"""
Training script for the AIA-GOES multimodal solar flare forecasting model using PyTorch Lightning.

This script:
1. Loads configuration from a YAML file with variable substitution (e.g., ${base_dir} references).
2. Initializes the AIA-GOES DataModule.
3. Configures logging with Weights & Biases.
4. Builds and trains a Vision Transformer (ViTLocal) model.
5. Optionally computes dynamic base class weights for flare categories (Quiet, C, M, X).
6. Saves model checkpoints (.ckpt).

Usage:
    python training/train.py --config training/train_config.yaml
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from training.callbacks import AttentionMapCallback, ImagePredictionLogger_SXR
from forecasting.dataset import AIAGOESDataModule
from forecasting.model import ViTLocal, SXRRegressionDynamicLoss, unnormalize_sxr


def resolve_config_variables(config_dict):
    """
    Recursively resolve ${variable} references within the config.

    This function processes configuration dictionaries to substitute variable
    references of the form ${variable_name} with their actual values defined
    elsewhere in the configuration.
    """
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


def get_base_weights(data_module, sxr_norm):
    """
    Compute inverse-frequency weights for flare classes based on training data.

    The weights help balance loss contributions from imbalanced flare categories
    by making rare classes (M/X flares) count for more in the loss.

    Parameters
    ----------
    data_module : AIAGOESDataModule
        Initialized DataModule providing the train_dataloader.
    sxr_norm : np.ndarray
        Normalization parameters for SXR.

    Returns
    -------
    dict
        Class weights for quiet, C, M, and X classes.
    """
    print("Calculating base weights from training data...")
    # Fixed GOES flare-class boundaries — same source of truth the loss uses.
    thresholds = SXRRegressionDynamicLoss.CLASS_THRESHOLDS
    c_threshold, m_threshold, x_threshold = thresholds['c'], thresholds['m'], thresholds['x']

    quiet_count = c_count = m_count = x_count = total = 0
    train_loader = data_module.train_dataloader()
    print(f"Processing {len(train_loader)} batches...")

    for batch_idx, (aia_batch, sxr_batch) in enumerate(train_loader):
        if batch_idx % 50 == 0:
            print(f"Processed {batch_idx}/{len(train_loader)} batches...")

        sxr_un = unnormalize_sxr(sxr_batch, sxr_norm)
        sxr_un_flat = sxr_un.reshape(-1)

        total += len(sxr_un_flat)
        quiet_count += (sxr_un_flat < c_threshold).sum()
        c_count += ((sxr_un_flat >= c_threshold) & (sxr_un_flat < m_threshold)).sum()
        m_count += ((sxr_un_flat >= m_threshold) & (sxr_un_flat < x_threshold)).sum()
        x_count += (sxr_un_flat >= x_threshold).sum()

    quiet_count, c_count, m_count, x_count = (max(c, 1) for c in (quiet_count, c_count, m_count, x_count))

    weights = {
        'quiet': total / quiet_count,
        'c_class': total / c_count,
        'm_class': total / m_count,
        'x_class': total / x_count,
    }
    print(f"Total samples: {total}")
    print(f"Quiet: {quiet_count} (weight {weights['quiet']:.4f}), "
          f"C: {c_count} (weight {weights['c_class']:.4f}), "
          f"M: {m_count} (weight {weights['m_class']:.4f}), "
          f"X: {x_count} (weight {weights['x_class']:.4f})")
    return weights


def resolve_devices(gpu_config):
    """Resolve accelerator/devices/strategy from the gpu_ids config value."""
    if gpu_config == -1:
        print("Using CPU for training")
        return "cpu", 1, "auto"

    if not torch.cuda.is_available():
        print("No GPUs available, falling back to CPU")
        return "cpu", 1, "auto"

    if gpu_config == "all":
        print(f"Using all available GPUs ({torch.cuda.device_count()} GPUs)")
        return "gpu", -1, "auto"
    if isinstance(gpu_config, list):
        print(f"Using GPUs: {gpu_config}")
        return "gpu", gpu_config, "auto"
    print(f"Using GPU {gpu_config}")
    return "gpu", [gpu_config], "auto"


def main():
    parser = argparse.ArgumentParser(description='Train the FOXES ViTLocal model.')
    parser.add_argument('--config', type=str, default='training/train_config.yaml', required=True,
                        help='Path to train_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)
    config_data: dict = resolve_config_variables(config_data)

    print("Resolved paths:")
    print(f"AIA dir: {config_data['data']['aia_dir']}")
    print(f"SXR dir: {config_data['data']['sxr_dir']}")
    print(f"Checkpoints dir: {config_data['data']['checkpoints_dir']}")

    sxr_norm = np.load(config_data['data']['sxr_norm_path'])
    wavelengths = config_data['wavelengths']

    optimizer_cfg = config_data.get('optimizer', {})
    loss_cfg = config_data.get('loss', {})
    checkpoint_cfg = config_data.get('checkpoint', {})
    logging_cfg = config_data.get('logging', {})
    callbacks_cfg = config_data.get('callbacks', {})
    data_cfg = config_data.get('data', {})

    data_module = AIAGOESDataModule(
        aia_train_dir=config_data['data']['aia_dir'] + "/train",
        aia_val_dir=config_data['data']['aia_dir'] + "/val",
        aia_test_dir=config_data['data']['aia_dir'] + "/test",
        sxr_train_dir=config_data['data']['sxr_dir'] + "/train",
        sxr_val_dir=config_data['data']['sxr_dir'] + "/val",
        sxr_test_dir=config_data['data']['sxr_dir'] + "/test",
        batch_size=config_data['batch_size'],
        num_workers=data_cfg.get('num_workers', min(8, os.cpu_count() or 1)),
        sxr_norm=sxr_norm,
        wavelengths=wavelengths,
    )
    data_module.setup()

    wandb_logger = WandbLogger(
        entity=config_data['wandb']['entity'],
        project=config_data['wandb']['project'],
        job_type=config_data['wandb']['job_type'],
        tags=config_data['wandb']['tags'],
        name=config_data['wandb']['run_name'],
        notes=config_data['wandb']['notes'],
        config=config_data,
    )

    # Callbacks
    sxr_plot_callback = ImagePredictionLogger_SXR(
        data_module.val_ds, callbacks_cfg.get('sxr_plot_num_samples', 4), sxr_norm)
    patch_size = config_data.get('vit_architecture', {}).get('patch_size', 16)
    attention_callback = AttentionMapCallback(
        patch_size=patch_size,
        use_local_attention=True,
        num_samples=callbacks_cfg.get('attention_num_samples', 4),
        log_every_n_epochs=callbacks_cfg.get('attention_log_every_n_epochs', 1),
    )

    base_weights = (get_base_weights(data_module, sxr_norm)
                    if config_data.get('calculate_base_weights') else loss_cfg.get('base_weights'))
    model = ViTLocal(
        model_kwargs=config_data['vit_architecture'],
        sxr_norm=sxr_norm,
        base_weights=base_weights,
        weight_decay=optimizer_cfg.get('weight_decay', 1e-5),
        scheduler_kwargs=optimizer_cfg.get('scheduler'),
        loss_kwargs={
            'window_size': loss_cfg.get('window_size', 15000),
            'huber_delta': loss_cfg.get('huber_delta', 0.3),
            'adaptive_multipliers': loss_cfg.get('adaptive_multipliers'),
        },
        diagnostic_every_n_steps=loss_cfg.get('diagnostic_every_n_steps', 200),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config_data['data']['checkpoints_dir'],
        monitor=checkpoint_cfg.get('monitor', 'val_total_loss'),
        mode=checkpoint_cfg.get('mode', 'min'),
        save_top_k=checkpoint_cfg.get('save_top_k', 10),
        filename=f"{config_data['wandb']['run_name']}-{{epoch:02d}}-{{val_total_loss:.4f}}",
    )

    accelerator, devices, strategy = resolve_devices(config_data.get('gpu_ids', -1))

    trainer = Trainer(
        default_root_dir=config_data['data']['checkpoints_dir'],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=config_data['epochs'],
        callbacks=[sxr_plot_callback, attention_callback, checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=logging_cfg.get('log_every_n_steps', 10),
        gradient_clip_val=optimizer_cfg.get('gradient_clip_val',None),  # None -> disabled (PyTorch Lightning default)
    )
    trainer.fit(model, data_module)
    wandb.finish()
    print(f"Training complete. Checkpoints saved to {config_data['data']['checkpoints_dir']}")


if __name__ == '__main__':
    main()
