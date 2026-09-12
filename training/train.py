"""
Training script for the AIA-GOES multimodal solar flare forecasting model using PyTorch Lightning.

This script:
1. Loads configuration from a YAML file with variable substitution (e.g., ${base_dir} references).
2. Initializes the AIA-GOES DataModule.
3. Configures logging with Weights & Biases.
4. Builds and trains a Vision Transformer (ViTLocal) model.
5. Computes exact training-split class weights for probabilistic objectives when requested.
6. Optionally computes legacy class weights for the deterministic model.
7. Saves model checkpoints (.ckpt).

Usage:
    python training/train.py --config training/train_config.yaml
    python training/train.py --config training/train_config.yaml \
        --ckpt-path /path/to/checkpoint.ckpt
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
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from training.callbacks import (
    AttentionMapCallback,
    ImagePredictionLogger_SXR,
    PerClassQuantileValidationMetrics,
    PerClassValidationMetrics,
    SpatialGaussianMapCallback,
    SpatialQuantileMapCallback,
)
from forecasting.dataset import AIAGOESDataModule
from forecasting.model import ViTLocal, SXRRegressionDynamicLoss, unnormalize_sxr
from forecasting.model_uncertainty import GaussianNLLViTLocal
from forecasting.model_uncertainty_background_excess import (
    BackgroundExcessGaussianNLLViTLocal,
)
from forecasting.model_uncertainty_og import OGGaussianNLLViTLocal
from forecasting.model_quantile import QuantileViTLocal


FIVE_CLASS_KEYS = (
    'below_b', 'b_class', 'c_class', 'm_class', 'x_class',
)
FOUR_CLASS_KEYS = ('quiet', 'c_class', 'm_class', 'x_class')
GAUSSIAN_MODEL_TYPES = {
    'gaussian_nll', 'gaussian_nll_background_excess', 'gaussian_nll_og',
}


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


def get_five_class_macro_weights(data_module):
    """Count training targets and return exact <B/B/C/M/X macro weights.

    Only the small scalar SXR files are read; AIA images and data transforms
    are not loaded. For class ``k``, ``N / (5 * N_k)`` makes the expected
    weighted sample mean equal the average of the five class-mean NLLs.
    """
    dataset = data_module.train_ds
    counts = dict.fromkeys(FIVE_CLASS_KEYS, 0)
    print("Calculating five-class macro-objective weights from training targets...")

    for index, timestamp in enumerate(dataset.samples, start=1):
        path = dataset.sxr_dir / f"{timestamp}.npy"
        value = np.load(path, allow_pickle=False)
        if value.size != 1:
            raise ValueError(
                f"Expected one SXR value in {path}, found {value.size}"
            )
        flux = float(value.reshape(-1)[0])
        if not np.isfinite(flux):
            raise ValueError(f"Non-finite SXR flux in {path}: {flux!r}")
        if flux < 1e-7:
            class_name = 'below_b'
        elif flux < 1e-6:
            class_name = 'b_class'
        elif flux < 1e-5:
            class_name = 'c_class'
        elif flux < 1e-4:
            class_name = 'm_class'
        else:
            class_name = 'x_class'
        counts[class_name] += 1
        if index % 10_000 == 0:
            print(
                f"Counted {index:,}/{len(dataset.samples):,} "
                "training targets..."
            )

    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(
            "The five-class macro objective requires at least one training example in "
            f"every <B/B/C/M/X bin; empty bins: {empty}"
        )
    total = sum(counts.values())
    weights = {
        name: total / (len(FIVE_CLASS_KEYS) * counts[name])
        for name in FIVE_CLASS_KEYS
    }
    print(f"Five-class training counts: {counts}")
    print(
        "Five-class macro weights: "
        + ", ".join(
            f"{name}={weights[name]:.4f}" for name in FIVE_CLASS_KEYS
        )
    )
    return counts, weights


def get_four_class_macro_weights(data_module, exponent=1.0):
    """Return quiet/C/M/X weights computed from scalar training targets.

    ``exponent=1`` produces inverse-frequency macro weights. ``exponent=0.5``
    produces their square-root relative weighting. Both are normalized to an
    expected sample weight of one, so changing the scheme does not silently
    change the scale of the mean loss relative to uncertainty NLL. Only SXR
    targets are read; loading all AIA images merely to count classes would
    make startup unnecessarily costly.
    """
    if exponent not in {0.5, 1.0}:
        raise ValueError("four-class weighting exponent must be 0.5 or 1.0")
    dataset = data_module.train_ds
    counts = dict.fromkeys(FOUR_CLASS_KEYS, 0)
    thresholds = SXRRegressionDynamicLoss.CLASS_THRESHOLDS
    print("Calculating quiet/C/M/X macro weights from training targets...")

    for index, timestamp in enumerate(dataset.samples, start=1):
        path = dataset.sxr_dir / f"{timestamp}.npy"
        value = np.load(path, allow_pickle=False)
        if value.size != 1:
            raise ValueError(
                f"Expected one SXR value in {path}, found {value.size}"
            )
        flux = float(value.reshape(-1)[0])
        if not np.isfinite(flux):
            raise ValueError(f"Non-finite SXR flux in {path}: {flux!r}")
        if flux < thresholds['c']:
            class_name = 'quiet'
        elif flux < thresholds['m']:
            class_name = 'c_class'
        elif flux < thresholds['x']:
            class_name = 'm_class'
        else:
            class_name = 'x_class'
        counts[class_name] += 1
        if index % 10_000 == 0:
            print(
                f"Counted {index:,}/{len(dataset.samples):,} "
                "training targets..."
            )

    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(
            "The four-class macro mean objective requires at least one "
            f"training example in every quiet/C/M/X bin; empty bins: {empty}"
        )
    total = sum(counts.values())
    raw_weights = {
        name: (total / counts[name]) ** exponent
        for name in FOUR_CLASS_KEYS
    }
    expected_weight = sum(
        counts[name] * raw_weights[name] for name in FOUR_CLASS_KEYS
    ) / total
    weights = {
        name: raw_weights[name] / expected_weight
        for name in FOUR_CLASS_KEYS
    }
    print(f"Four-class training counts: {counts}")
    print(
        "Four-class macro mean weights: "
        + ", ".join(
            f"{name}={weights[name]:.4f}" for name in FOUR_CLASS_KEYS
        )
    )
    return counts, weights


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


def resolve_resume_checkpoint(checkpoint_config, cli_checkpoint_path=None):
    """Resolve and validate an optional checkpoint used to resume training.

    The command-line value takes precedence over ``checkpoint.resume_from`` so
    a one-off recovery can override a config without editing it. Requiring an
    existing file catches path mistakes before datasets, W&B, or GPUs are
    initialized.
    """
    checkpoint_path = (
        cli_checkpoint_path
        if cli_checkpoint_path is not None
        else checkpoint_config.get('resume_from')
    )
    if checkpoint_path in {None, ''}:
        return None
    if not isinstance(checkpoint_path, (str, os.PathLike)):
        raise TypeError(
            "checkpoint.resume_from/--ckpt-path must be a filesystem path"
        )

    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {checkpoint_path}"
        )
    return str(checkpoint_path.resolve())


def build_checkpoint_callback(config_data, checkpoint_config):
    """Build the checkpoint callback shared by fresh and resumed runs."""
    return ModelCheckpoint(
        dirpath=config_data['data']['checkpoints_dir'],
        monitor=checkpoint_config.get('monitor', 'val_total_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 10),
        save_last=checkpoint_config.get('save_last', False),
        # Selection is controlled by ``monitor`` above. Avoid embedding a
        # different metric in the filename: beta-NLL losses have incomparable
        # scales even when runs are all selected by val/mse.
        filename=(
            f"{config_data['wandb']['run_name']}"
            "-{epoch:02d}-{step:06d}"
        ),
    )


def get_resume_fit_kwargs(resume_checkpoint):
    """Return Lightning fit arguments for a trusted full-state checkpoint."""
    if resume_checkpoint is None:
        return {}
    # Full training recovery needs the optimizer, scheduler, loop, and callback
    # state. PyTorch 2.6+ otherwise defaults local torch.load calls to the
    # restricted weights-only loader, which rejects older Lightning metadata.
    return {
        'ckpt_path': resume_checkpoint,
        'weights_only': False,
    }


def main():
    parser = argparse.ArgumentParser(description='Train the FOXES ViTLocal model.')
    parser.add_argument('--config', type=str, default='training/train_config.yaml', required=True,
                        help='Path to train_config.yaml')
    parser.add_argument(
        '--ckpt-path',
        type=str,
        default=None,
        help=(
            'Checkpoint to resume from. Overrides checkpoint.resume_from in '
            'the YAML config.'
        ),
    )
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)
    config_data: dict = resolve_config_variables(config_data)
    seed_everything(config_data.get('seed', 42), workers=True)

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
    resume_checkpoint = resolve_resume_checkpoint(
        checkpoint_cfg, args.ckpt_path
    )
    if resume_checkpoint is not None:
        print(f"Resuming training state from: {resume_checkpoint}")

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
        aia_norm_path=data_cfg.get('aia_norm_path'),
        wavelengths=wavelengths,
    )
    data_module.setup()

    model_type = config_data.get('model_type', 'deterministic')
    class_balanced_objective = (
        model_type in GAUSSIAN_MODEL_TYPES
        and config_data.get('uncertainty', {}).get('nll_weighting')
        == 'five_class'
    ) or (
        model_type == 'quantile'
        and config_data.get('quantile', {}).get('loss_weighting')
        == 'five_class'
    )
    if class_balanced_objective:
        five_class_counts, five_class_weights = get_five_class_macro_weights(
            data_module
        )
        objective_key = (
            'quantile' if model_type == 'quantile' else 'uncertainty'
        )
        config_data[objective_key]['five_class_weights'] = five_class_weights
        config_data['computed_five_class_train_counts'] = five_class_counts

    uncertainty_config = config_data.get('uncertainty', {})
    class_weighting = uncertainty_config.get(
        'class_weighting',
        {
            'four_class_macro': 'inverse_frequency',
            'sqrt_four_class_macro': 'sqrt_inverse_frequency',
        }.get(uncertainty_config.get('mean_loss_weighting')),
    )
    mean_class_balanced = (
        model_type in GAUSSIAN_MODEL_TYPES
        and class_weighting
        in {'inverse_frequency', 'sqrt_inverse_frequency'}
    )
    if mean_class_balanced:
        four_class_counts, four_class_weights = get_four_class_macro_weights(
            data_module,
            exponent=(
                0.5 if class_weighting == 'sqrt_inverse_frequency' else 1.0
            ),
        )
        config_data['uncertainty']['class_weights'] = four_class_weights
        config_data['computed_four_class_train_counts'] = four_class_counts

    wandb_logger = WandbLogger(
        entity=config_data['wandb']['entity'],
        project=config_data['wandb']['project'],
        job_type=config_data['wandb']['job_type'],
        tags=config_data['wandb']['tags'],
        name=config_data['wandb']['run_name'],
        notes=config_data['wandb']['notes'],
        config=config_data,
    )

    # Callbacks. Expensive visualizations can be disabled for short parameter
    # screening runs while remaining enabled by default for normal training.
    callbacks = []
    if callbacks_cfg.get('per_class_metrics_enabled', True):
        callbacks.append(
            PerClassQuantileValidationMetrics()
            if model_type == 'quantile'
            else PerClassValidationMetrics()
        )
    if callbacks_cfg.get('sxr_plot_enabled', True):
        callbacks.append(ImagePredictionLogger_SXR(
            data_module.val_ds,
            callbacks_cfg.get('sxr_plot_num_samples', 4),
            sxr_norm,
        ))
    patch_size = config_data.get('vit_architecture', {}).get('patch_size', 16)
    if (
        model_type == 'quantile'
        and callbacks_cfg.get('spatial_quantile_enabled', False)
    ):
        callbacks.append(SpatialQuantileMapCallback(
            patch_size=patch_size,
            num_samples=callbacks_cfg.get(
                'spatial_quantile_num_samples', 5
            ),
            log_every_n_epochs=callbacks_cfg.get(
                'spatial_quantile_log_every_n_epochs', 1
            ),
        ))
    if (
        model_type in GAUSSIAN_MODEL_TYPES
        and callbacks_cfg.get('spatial_gaussian_enabled', False)
    ):
        callbacks.append(SpatialGaussianMapCallback(
            patch_size=patch_size,
            num_samples=callbacks_cfg.get(
                'spatial_gaussian_num_samples', 5
            ),
            log_every_n_epochs=callbacks_cfg.get(
                'spatial_gaussian_log_every_n_epochs', 1
            ),
        ))
    if callbacks_cfg.get('attention_enabled', True):
        callbacks.append(AttentionMapCallback(
            patch_size=patch_size,
            use_local_attention=True,
            num_samples=callbacks_cfg.get('attention_num_samples', 4),
            log_every_n_epochs=callbacks_cfg.get('attention_log_every_n_epochs', 1),
        ))

    base_weights = (get_base_weights(data_module, sxr_norm)
                    if config_data.get('calculate_base_weights') else loss_cfg.get('base_weights'))
    common_model_kwargs = dict(
        model_kwargs=config_data['vit_architecture'],
        sxr_norm=sxr_norm,
        base_weights=base_weights,
        weight_decay=optimizer_cfg.get('weight_decay', 1e-5),
        scheduler_kwargs=optimizer_cfg.get('scheduler'),
    )
    if model_type == 'gaussian_nll':
        model = GaussianNLLViTLocal(
            **common_model_kwargs,
            uncertainty_kwargs=config_data.get('uncertainty', {}),
        )
    elif model_type == 'gaussian_nll_background_excess':
        model = BackgroundExcessGaussianNLLViTLocal(
            **common_model_kwargs,
            uncertainty_kwargs=config_data.get('uncertainty', {}),
        )
    elif model_type == 'gaussian_nll_og':
        model = OGGaussianNLLViTLocal(
            **common_model_kwargs,
            uncertainty_kwargs=config_data.get('uncertainty', {}),
        )
    elif model_type == 'quantile':
        model = QuantileViTLocal(
            **common_model_kwargs,
            quantile_kwargs=config_data.get('quantile', {}),
        )
    elif model_type == 'deterministic':
        model = ViTLocal(
            **common_model_kwargs,
            diagnostic_every_n_steps=loss_cfg.get('diagnostic_every_n_steps', 200),
            loss_kwargs={
                'window_size': loss_cfg.get('window_size', 15000),
                'huber_delta': loss_cfg.get('huber_delta', 0.3),
                'adaptive_multipliers': loss_cfg.get('adaptive_multipliers'),
            },
        )
    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}; expected 'deterministic', "
            "'gaussian_nll', 'gaussian_nll_background_excess', "
            "'gaussian_nll_og', or 'quantile'"
        )

    checkpoint_callback = build_checkpoint_callback(
        config_data, checkpoint_cfg
    )
    callbacks.append(checkpoint_callback)

    early_stopping_cfg = config_data.get('early_stopping', {})
    if early_stopping_cfg.get('enabled', False):
        callbacks.append(EarlyStopping(
            monitor=early_stopping_cfg.get(
                'monitor', checkpoint_cfg.get('monitor', 'val_total_loss')
            ),
            mode=early_stopping_cfg.get(
                'mode', checkpoint_cfg.get('mode', 'min')
            ),
            patience=early_stopping_cfg.get('patience', 8),
            min_delta=early_stopping_cfg.get('min_delta', 0.0),
            check_finite=early_stopping_cfg.get('check_finite', True),
            verbose=early_stopping_cfg.get('verbose', True),
        ))

    accelerator, devices, strategy = resolve_devices(config_data.get('gpu_ids', -1))

    trainer = Trainer(
        default_root_dir=config_data['data']['checkpoints_dir'],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=config_data['epochs'],
        max_steps=config_data.get('max_steps', -1),
        accumulate_grad_batches=config_data.get(
            'accumulate_grad_batches', 1
        ),
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=logging_cfg.get('log_every_n_steps', 10),
        gradient_clip_val=optimizer_cfg.get('gradient_clip_val',None),  # None -> disabled (PyTorch Lightning default)
    )
    trainer.fit(
        model,
        data_module,
        **get_resume_fit_kwargs(resume_checkpoint),
    )
    wandb.finish()
    print(f"Training complete. Checkpoints saved to {config_data['data']['checkpoints_dir']}")


if __name__ == '__main__':
    main()
