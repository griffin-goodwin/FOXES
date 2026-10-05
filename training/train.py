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

For fine-tuning with a fresh optimizer, set checkpoint.initialize_from in the
YAML instead of --ckpt-path/checkpoint.resume_from. epochs then counts NEW epochs.
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
    PerClassValidationMetrics,
    SpatialUncertaintyMapCallback,
)
from forecasting.dataset import AIAGOESDataModule
from forecasting.model import ViTLocal, SXRRegressionDynamicLoss, unnormalize_sxr
from forecasting.uncertainty_model import GaussianNLLViTLocal
from training.strategies import TrainingStreamDDPStrategy


FOUR_CLASS_KEYS = ('quiet', 'c_class', 'm_class', 'x_class')


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


def _deep_merge_config(base, overrides):
    """Recursively merge YAML overrides without mutating either input."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_training_config(config_path):
    """Load a training YAML, optionally inheriting from ``base_config``."""
    config_path = Path(config_path).expanduser().resolve()

    def load_one(path, ancestors):
        if path in ancestors:
            chain = ' -> '.join(str(item) for item in (*ancestors, path))
            raise ValueError(f"Circular base_config chain: {chain}")
        with path.open('r') as stream:
            current = yaml.safe_load(stream) or {}
        if not isinstance(current, dict):
            raise ValueError(f"Training config must be a mapping: {path}")
        base_reference = current.pop('base_config', None)
        if base_reference is None:
            return current
        base_path = Path(base_reference).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base = load_one(base_path.resolve(), (*ancestors, path))
        return _deep_merge_config(base, current)

    return resolve_config_variables(load_one(config_path, ()))


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


def get_four_class_macro_weights(data_module, exponent=1.0):
    """Return quiet/C/M/X weights computed from scalar training targets.

    ``exponent=1`` produces inverse-frequency macro weights. ``exponent=0.5``
    produces their square-root relative weighting. Any exponent in (0, 1]
    is supported. Weights are normalized to an
    expected sample weight of one, so changing the scheme does not silently
    change the scale of the mean loss relative to uncertainty NLL. Only SXR
    targets are read; loading all AIA images merely to count classes would
    make startup unnecessarily costly.
    """
    if (isinstance(exponent, bool) or not isinstance(exponent, (int, float))
            or not np.isfinite(exponent) or not 0 < exponent <= 1):
        raise ValueError("four-class weighting exponent must be in (0, 1]")
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
        device_count = torch.cuda.device_count()
        print(f"Using all available GPUs ({device_count} GPUs)")
        strategy = TrainingStreamDDPStrategy() if device_count > 1 else "auto"
        return "gpu", -1, strategy
    if isinstance(gpu_config, list):
        print(f"Using GPUs: {gpu_config}")
        strategy = TrainingStreamDDPStrategy() if len(gpu_config) > 1 else "auto"
        return "gpu", gpu_config, strategy
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
    monitor = checkpoint_config.get('monitor', 'val_total_loss')
    filename = f"{config_data['wandb']['run_name']}-epoch={{epoch:02d}}"
    if checkpoint_config.get('include_step_in_filename', True):
        filename += "-step={step:06d}"
    if checkpoint_config.get('include_monitor_in_filename', False):
        metric_label = monitor.replace('/', '_')
        filename += f"-{metric_label}={{{monitor}:.4f}}"
    return ModelCheckpoint(
        dirpath=config_data['data']['checkpoints_dir'],
        monitor=monitor,
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 10),
        save_last=checkpoint_config.get('save_last', False),
        filename=filename,
        auto_insert_metric_name=False,
    )


def resolve_initialization_checkpoint(checkpoint_config, resume_checkpoint=None):
    """Resolve weights-only initialization, mutually exclusive with full resume."""
    path = checkpoint_config.get('initialize_from')
    if path in {None, ''}:
        return None
    if resume_checkpoint is not None:
        raise ValueError(
            'checkpoint.initialize_from cannot be combined with '
            'checkpoint.resume_from or --ckpt-path'
        )
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Initialization checkpoint does not exist: {path}')
    return str(path.resolve())


def initialize_model_weights(model, checkpoint_path):
    """Load a trusted local Lightning checkpoint without restoring training state.

    The new model retains the configured objective, LR and scheduler. Strict
    loading includes the uncertainty head. A patch-specific global module may
    be new, and its replacement of the old shared-global module may leave old
    checkpoint weights unused. No other mismatches are allowed. Normalization
    and attention masks must match.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    current = model.state_dict()
    for name, value in state.items():
        if name == 'sxr_norm' or name.endswith('.attention_mask'):
            if name in current and not torch.equal(value, current[name]):
                raise ValueError(f'Initialization checkpoint differs in {name}')
    saved_mean = checkpoint.get('hyper_parameters', {}).get('mean_parameterization')
    if saved_mean is not None and saved_mean != model.hparams.get('mean_parameterization'):
        raise ValueError('Initialization checkpoint differs in mean_parameterization')
    result = model.load_state_dict(state, strict=False)
    allowed_missing = {
        name for name in current
        if name.startswith('model.patch_global_context.')
    }
    replacing_old_global = (
        getattr(model.model, 'patch_global_context', None) is not None
        and not model.model.global_patch_adjustment
    )
    old_global_prefixes = (
        'model.global_raw_patch_projection.',
        'model.global_attention.',
        'model.global_patch_head.',
    )
    allowed_unexpected = {
        name for name in result.unexpected_keys
        if replacing_old_global and (
            name == 'model.global_query'
            or name.startswith(old_global_prefixes)
        )
    }
    if (set(result.unexpected_keys) - allowed_unexpected
            or set(result.missing_keys) - allowed_missing):
        raise RuntimeError(
            'Initialization checkpoint is incompatible: '
            f'missing={result.missing_keys}, unexpected={result.unexpected_keys}'
        )
    metadata = {
        'source_epoch': checkpoint.get('epoch'),
        'source_global_step': checkpoint.get('global_step'),
    }
    if result.missing_keys:
        metadata['new_parameter_keys'] = list(result.missing_keys)
    if result.unexpected_keys:
        metadata['skipped_parameter_keys'] = list(result.unexpected_keys)
    return metadata


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


def build_model_callbacks(config_data, data_module, sxr_norm, model_type):
    """Build the callback set shared by both trainable model types.

    Per-class metrics, prediction plots, and attention plots use adaptive
    callback interfaces and are identical for the deterministic and
    uncertainty models. Only the spatial uncertainty map is model-specific.
    """
    callbacks_cfg = config_data.get('callbacks', {})
    callbacks = []
    if callbacks_cfg.get('per_class_metrics_enabled', True):
        callbacks.append(PerClassValidationMetrics())
    if callbacks_cfg.get('sxr_plot_enabled', True):
        callbacks.append(ImagePredictionLogger_SXR(
            data_module.val_ds,
            callbacks_cfg.get('sxr_plot_num_samples', 4),
            sxr_norm,
        ))

    patch_size = config_data.get('vit_architecture', {}).get(
        'patch_size', 16
    )
    if (
        model_type == 'uncertainty'
        and config_data.get('uncertainty', {}).get('enabled', True)
        and callbacks_cfg.get('spatial_uncertainty_enabled', False)
    ):
        callbacks.append(SpatialUncertaintyMapCallback(
            patch_size=patch_size,
            num_samples=callbacks_cfg.get(
                'spatial_uncertainty_num_samples', 5
            ),
            log_every_n_epochs=callbacks_cfg.get(
                'spatial_uncertainty_log_every_n_epochs', 1
            ),
        ))
    if callbacks_cfg.get('attention_enabled', True):
        callbacks.append(AttentionMapCallback(
            patch_size=patch_size,
            use_local_attention=True,
            num_samples=callbacks_cfg.get('attention_num_samples', 4),
            log_every_n_epochs=callbacks_cfg.get(
                'attention_log_every_n_epochs', 1
            ),
        ))
    return callbacks


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

    config_data: dict = load_training_config(args.config)
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
    data_cfg = config_data.get('data', {})
    resume_checkpoint = resolve_resume_checkpoint(
        checkpoint_cfg, args.ckpt_path
    )
    initialization_checkpoint = resolve_initialization_checkpoint(
        checkpoint_cfg, resume_checkpoint
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
    mean_parameterization = str(
        config_data.get('mean_parameterization', 'multiplier')
    ).lower()
    if model_type not in {'deterministic', 'uncertainty'}:
        raise ValueError(
            f"Unknown model_type {model_type!r}; expected 'deterministic' "
            "or 'uncertainty'"
        )
    if (
        model_type == 'uncertainty'
        and mean_parameterization not in
        GaussianNLLViTLocal.NETWORK_CLASS.MEAN_PARAMETERIZATIONS
    ):
        raise ValueError(
            'mean_parameterization must be one of '
            f'{GaussianNLLViTLocal.NETWORK_CLASS.MEAN_PARAMETERIZATIONS}'
        )
    if model_type == 'uncertainty':
        initialization = (
            GaussianNLLViTLocal.NETWORK_CLASS.MEAN_HEAD_INITIALIZATIONS[
                mean_parameterization
            ]
        )
        config_data.setdefault('experiment_metadata', {}).update({
            'mean_parameterization': mean_parameterization,
            'mean_head_initialization': initialization,
            'uncertainty_parameterization': (
                'logit_conditioned_relative_log10_patch_std'
                if config_data.get('uncertainty', {}).get('enabled', True)
                else None
            ),
            'uncertainty_likelihood_space': 'normalized_log10_sxr',
            'contrast_prior': (
                'hot_channel_spatial_cosine'
                if config_data.get('contrast_prior', {}).get('enabled', False)
                else None
            ),
        })
    uncertainty_config = config_data.get('uncertainty', {})
    class_weighting = uncertainty_config.get(
        'class_weighting',
        {
            'four_class_macro': 'inverse_frequency',
            'sqrt_four_class_macro': 'sqrt_inverse_frequency',
        }.get(uncertainty_config.get('mean_loss_weighting')),
    )
    mean_class_balanced = (
        model_type == 'uncertainty'
        and class_weighting
        in {'inverse_frequency', 'sqrt_inverse_frequency',
            'power_inverse_frequency'}
    )
    if mean_class_balanced:
        exponent = {
            'inverse_frequency': 1.0,
            'sqrt_inverse_frequency': 0.5,
        }.get(class_weighting)
        if exponent is None:
            exponent = uncertainty_config.get('class_weight_exponent')
        four_class_counts, four_class_weights = get_four_class_macro_weights(
            data_module, exponent=exponent,
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

    callbacks = build_model_callbacks(
        config_data, data_module, sxr_norm, model_type
    )
    print(
        "Visualization callbacks: "
        + (
            ", ".join(type(callback).__name__ for callback in callbacks)
            if callbacks else "none"
        )
    )

    base_weights = (get_base_weights(data_module, sxr_norm)
                    if config_data.get('calculate_base_weights') else loss_cfg.get('base_weights'))
    common_model_kwargs = dict(
        model_kwargs=config_data['vit_architecture'],
        sxr_norm=sxr_norm,
        base_weights=base_weights,
        weight_decay=optimizer_cfg.get('weight_decay', 1e-5),
        scheduler_kwargs=optimizer_cfg.get('scheduler'),
    )
    if model_type == 'uncertainty':
        contrast_prior_config = dict(
            config_data.get('contrast_prior', {})
        )
        contrast_wavelengths = contrast_prior_config.pop(
            'wavelengths', []
        )
        missing_contrast_wavelengths = [
            wavelength for wavelength in contrast_wavelengths
            if wavelength not in wavelengths
        ]
        if missing_contrast_wavelengths:
            raise ValueError(
                "contrast_prior wavelengths are absent from model input: "
                f"{missing_contrast_wavelengths}"
            )
        contrast_prior_config['channel_indices'] = [
            wavelengths.index(wavelength)
            for wavelength in contrast_wavelengths
        ]
        model = GaussianNLLViTLocal(
            **common_model_kwargs,
            uncertainty_kwargs=config_data.get('uncertainty', {}),
            mean_parameterization=mean_parameterization,
            contrast_prior_kwargs=contrast_prior_config,
        )
    else:
        model = ViTLocal(
            **common_model_kwargs,
            diagnostic_every_n_steps=loss_cfg.get('diagnostic_every_n_steps', 200),
            loss_kwargs={
                'window_size': loss_cfg.get('window_size', 15000),
                'huber_delta': loss_cfg.get('huber_delta', 0.3),
                'adaptive_multipliers': loss_cfg.get('adaptive_multipliers'),
            },
        )
    if initialization_checkpoint is not None:
        source = initialize_model_weights(model, initialization_checkpoint)
        wandb_logger.log_hyperparams({'initialization': {
            'checkpoint': initialization_checkpoint, **source,
            'restore_optimizer': False, 'restore_scheduler': False,
        }})
        print(
            f'Initialized model weights from: {initialization_checkpoint}; '
            'starting at epoch 0 with a fresh optimizer and scheduler'
        )

    trainable_modules = config_data.get('finetune', {}).get('trainable_modules')
    if trainable_modules is not None:
        if model_type != 'uncertainty':
            raise ValueError('finetune.trainable_modules requires uncertainty model')
        model.freeze_except(trainable_modules)
        print(f'Training only modules: {trainable_modules}')

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
