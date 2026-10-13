"""Train the released deterministic or coupled Gaussian FOXES model."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.dataset import AIAGOESDataModule
from forecasting.model import (
    CoupledGaussianViTLocal, ViTLocal, SXRRegressionDynamicLoss,
)
from training.callbacks import (
    AttentionMapCallback, ImagePredictionLogger_SXR,
    PerClassValidationMetrics, SpatialUncertaintyMapCallback,
)
from training.strategies import TrainingStreamDDPStrategy

COUPLED_MODEL_TYPES = frozenset({
    'coupled_uncertainty', 'og_uncertainty', 'uncertainty_coupled_single_query',
})
FOUR_CLASS_KEYS = ('quiet', 'c_class', 'm_class', 'x_class')


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

    return load_one(config_path, ())


def get_four_class_macro_weights(data_module, exponent=1.0,
                                 require_all_classes=True):
    """Return quiet/C/M/X weights computed from scalar training targets.

    ``exponent=1`` produces inverse-frequency macro weights. ``exponent=0.5``
    produces their square-root relative weighting. Any exponent in (0, 1]
    is supported.
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
    if empty and require_all_classes:
        raise ValueError(
            "The four-class macro mean objective requires at least one "
            f"training example in every quiet/C/M/X bin; empty bins: {empty}"
        )
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Cannot compute class weights from an empty training set")
    raw_weights = {
        name: (total / max(counts[name], 1)) ** exponent
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


def configure_class_weights(config, data_module, model_type):
    """Compute mean-loss class weights once for either maintained model."""
    coupled = model_type in COUPLED_MODEL_TYPES
    mean_loss = config.get('mean_loss', {})
    uncertainty = dict(config.get('uncertainty', {}))
    loss = config.get('loss', {})
    weighting = mean_loss.get('class_weighting')
    if weighting is None:
        if coupled:
            weighting = uncertainty.get('class_weighting', 'none')
        else:
            weighting = (
                'inverse_frequency' if config.get('calculate_base_weights')
                else 'none'
            )
    if weighting not in {
        'none', 'inverse_frequency', 'sqrt_inverse_frequency',
        'power_inverse_frequency',
    }:
        raise ValueError(f'Unsupported mean_loss.class_weighting: {weighting!r}')
    if coupled:
        uncertainty['class_weighting'] = weighting
        if 'huber_delta' in mean_loss:
            uncertainty['huber_delta'] = mean_loss['huber_delta']

    base_weights = loss.get('base_weights') if not coupled else None
    if weighting != 'none':
        exponent = {
            'inverse_frequency': 1.0,
            'sqrt_inverse_frequency': 0.5,
        }.get(
            weighting,
            mean_loss.get('class_weight_exponent',
                          uncertainty.get('class_weight_exponent')),
        )
        counts, weights = get_four_class_macro_weights(
            data_module, exponent=exponent,
            require_all_classes=coupled,
        )
        config['computed_four_class_train_counts'] = counts
        config['computed_four_class_train_weights'] = weights
        if coupled:
            uncertainty['class_weights'] = weights
        else:
            base_weights = weights
    return base_weights, uncertainty


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


def build_checkpoint_callback(config_data, checkpoint_config):
    """Build the checkpoint callback for training."""
    default_monitor = 'val/total_loss' if config_data.get('model_type') in COUPLED_MODEL_TYPES else 'val_total_loss'
    monitor = checkpoint_config.get('monitor', default_monitor)
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
        'patch_size', 8
    )
    if (
        model_type in COUPLED_MODEL_TYPES
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Training YAML')
    args = parser.parse_args()

    config = load_training_config(args.config)
    model_type = config.get('model_type', 'deterministic')
    if model_type not in COUPLED_MODEL_TYPES | {'deterministic'}:
        raise ValueError(
            f'Unsupported model_type {model_type!r}; use deterministic or '
            'coupled_uncertainty. Older experiments use '
            'python -m training.experimental.train.'
        )
    checkpoint_cfg = config.get('checkpoint', {})
    supported_checkpoint_keys = {
        'monitor', 'mode', 'save_top_k', 'save_last',
        'include_step_in_filename', 'include_monitor_in_filename',
    }
    unknown_checkpoint_keys = set(checkpoint_cfg) - supported_checkpoint_keys
    if unknown_checkpoint_keys:
        raise ValueError(
            f'Unsupported checkpoint settings: {sorted(unknown_checkpoint_keys)}'
        )
    seed_everything(config.get('seed', 42), workers=True)

    data_cfg = config['data']
    print(f"AIA dir: {data_cfg['aia_dir']}")
    print(f"SXR dir: {data_cfg['sxr_dir']}")
    print(f"Checkpoints dir: {data_cfg['checkpoints_dir']}")
    sxr_norm = np.load(data_cfg['sxr_norm_path'])
    optimizer_cfg = config.get('optimizer', {})
    loss_cfg = config.get('loss', {})

    data_module = AIAGOESDataModule(
        aia_train_dir=data_cfg['aia_dir'] + '/train',
        aia_val_dir=data_cfg['aia_dir'] + '/val',
        aia_test_dir=data_cfg['aia_dir'] + '/test',
        sxr_train_dir=data_cfg['sxr_dir'] + '/train',
        sxr_val_dir=data_cfg['sxr_dir'] + '/val',
        sxr_test_dir=data_cfg['sxr_dir'] + '/test',
        batch_size=config['batch_size'],
        num_workers=data_cfg.get('num_workers', min(8, os.cpu_count() or 1)),
        sxr_norm=sxr_norm,
        aia_norm_path=data_cfg.get('aia_norm_path'),
        wavelengths=config['wavelengths'],
    )
    data_module.setup()

    base_weights, uncertainty_cfg = configure_class_weights(
        config, data_module, model_type,
    )
    common_kwargs = dict(
        model_kwargs=config['vit_architecture'],
        sxr_norm=sxr_norm,
        base_weights=base_weights,
        weight_decay=optimizer_cfg.get('weight_decay', 1e-5),
        scheduler_kwargs=optimizer_cfg.get('scheduler'),
    )
    if model_type in COUPLED_MODEL_TYPES:
        model = CoupledGaussianViTLocal(
            **common_kwargs,
            uncertainty_kwargs=uncertainty_cfg,
        )
    else:
        model = ViTLocal(
            **common_kwargs,
            loss_kwargs={
                'window_size': loss_cfg.get('window_size', 15000),
                'huber_delta': loss_cfg.get('huber_delta', 0.3),
                'adaptive_multipliers': loss_cfg.get('adaptive_multipliers'),
            },
            diagnostic_every_n_steps=loss_cfg.get('diagnostic_every_n_steps', 200),
        )

    logger = WandbLogger(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        job_type=config['wandb']['job_type'],
        tags=config['wandb']['tags'],
        name=config['wandb']['run_name'],
        notes=config['wandb']['notes'],
        config=config,
    )
    callbacks = build_model_callbacks(config, data_module, sxr_norm, model_type)
    callbacks.append(build_checkpoint_callback(config, checkpoint_cfg))
    early_stopping = config.get('early_stopping', {})
    if early_stopping.get('enabled', False):
        callbacks.append(EarlyStopping(
            monitor=early_stopping.get('monitor', checkpoint_cfg.get(
                'monitor', 'val/total_loss' if model_type in COUPLED_MODEL_TYPES
                else 'val_total_loss')),
            mode=early_stopping.get('mode', 'min'),
            patience=early_stopping.get('patience', 8),
            min_delta=early_stopping.get('min_delta', 0.0),
            check_finite=early_stopping.get('check_finite', True),
            verbose=early_stopping.get('verbose', True),
        ))
    accelerator, devices, strategy = resolve_devices(config.get('gpu_ids', -1))
    trainer = Trainer(
        default_root_dir=data_cfg['checkpoints_dir'],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=config['epochs'],
        max_steps=config.get('max_steps', -1),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.get('logging', {}).get('log_every_n_steps', 10),
        gradient_clip_val=optimizer_cfg.get('gradient_clip_val'),
    )
    trainer.fit(model, data_module)
    wandb.finish()
    print(f"Training complete. Checkpoints saved to {data_cfg['checkpoints_dir']}")


if __name__ == '__main__':
    main()
