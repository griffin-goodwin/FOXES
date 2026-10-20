from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from forecasting.model import CoupledGaussianViTLocal, CoupledPinballViTLocal, normalize_sxr
from training import callbacks
from training.train import build_model_callbacks, configure_class_weights, load_training_config


@pytest.fixture
def model():
    return CoupledPinballViTLocal(
        dict(embed_dim=8, hidden_dim=16, num_channels=1, num_heads=1,
             num_layers=1, patch_size=1, num_patches=4, dropout=0.0,
             mask_mode='local', local_window=3, global_patch_adjustment=True),
        np.array([-6.0, 1.0], dtype=np.float32),
    )


def test_full_validation_quantile_metrics_and_point_metrics(model):
    logged = {}
    model.log_dict = lambda metrics, **kwargs: logged.update(metrics)
    trainer = SimpleNamespace(sanity_checking=False)
    outputs = {
        'quantiles_norm': torch.tensor([[-1., -.5, 0., .5, 1.], [0., .5, 1., 1.5, 2.]]),
        'target_norm': torch.tensor([0., 1.]),
        'mean_norm': torch.tensor([0., 1.]),
        'target_raw': torch.tensor([5e-7, 2e-5]),
        'prediction_raw': torch.tensor([5e-7, 2e-5]),
    }
    metrics = callbacks.QuantileValidationMetrics()
    metrics.on_validation_epoch_start(trainer, model)
    # Multiple batches must be aggregated by sample count.
    for i in range(2):
        one = {key: value[i:i + 1] for key, value in outputs.items()}
        metrics.on_validation_batch_end(trainer, model, one, None, i)
    metrics.on_validation_epoch_end(trainer, model)
    assert logged['val_quantile/count'] == 2
    assert logged['val_quantile/q0.05/coverage'] == 0
    assert logged['val_quantile/q0.95/coverage'] == 1
    assert logged['val_quantile/q0.05/pinball_loss'] == pytest.approx(0.05)
    assert logged['val_quantile/interval_0.05_0.95/coverage'] == 1
    assert logged['val_quantile/crossing_rate'] == 0
    assert logged['val_quantile_class/M/count'] == 1

    point_metrics = callbacks.PerClassValidationMetrics()
    point_metrics.on_validation_epoch_start(trainer, model)
    point_metrics.on_validation_batch_end(trainer, model, outputs, None, 0)
    point_metrics.on_validation_epoch_end(trainer, model)
    assert logged['val_class/macro_mae_dex'] == 0
    assert not any(key.endswith('/nll') for key in logged)


def test_global_crossing_metrics_preserve_original_predictions(model):
    predictions = torch.tensor([[-1., .5, 0., 1., 2.], [-1., -.5, 0., .5, 1.]])
    original = predictions.clone()
    outputs = {'quantiles_norm': predictions, 'target_norm': torch.zeros(2),
               'target_raw': torch.tensor([5e-7, 2e-5])}
    logged = {}
    model.log_dict = lambda metrics, **kwargs: logged.update(metrics)
    trainer = SimpleNamespace(sanity_checking=False)
    metrics = callbacks.QuantileValidationMetrics()
    metrics.on_validation_epoch_start(trainer, model)
    metrics.on_validation_batch_end(trainer, model, outputs, None, 0)
    metrics.on_validation_epoch_end(trainer, model)
    assert logged['val_quantile/crossing_rate'] == 0.5
    assert logged['val_quantile/mean_adjacent_crossing_dex'] == pytest.approx(.5 / 8)
    assert logged['val_quantile_class/B/crossing_rate'] == 1
    assert logged['val_quantile_class/M/crossing_rate'] == 0
    torch.testing.assert_close(predictions, original)


def test_epoch_pinball_macro_uses_quiet_c_m_x_bins(model):
    logged = {}
    model.log_dict = lambda metrics, **kwargs: logged.update(metrics)
    trainer = SimpleNamespace(sanity_checking=False)
    metrics = callbacks.QuantileValidationMetrics()
    metrics.on_validation_epoch_start(trainer, model)
    for i, (target, raw) in enumerate((
        (torch.ones(3), torch.full((3,), 1e-7)),
        (torch.tensor([3.]), torch.tensor([1e-4])),
    )):
        metrics.on_validation_batch_end(trainer, model, {
            'quantiles_norm': torch.zeros(len(target), 5),
            'target_norm': target, 'target_raw': raw,
        }, None, i)
    metrics.on_validation_epoch_end(trainer, model)
    assert logged['val_quantile/pinball_macro_loss'] == pytest.approx(1.0)
    assert logged['val_quantile/pinball_class_sum'] == pytest.approx(2.0)
    assert logged['val_quantile/pinball_sample_mean'] == pytest.approx(.75)
    assert logged['val_quantile_class/quiet/count'] == 3


def test_quantile_callback_logs_every_head_and_restores_training(model, tmp_path, monkeypatch):
    flux = torch.tensor([5e-8, 5e-7, 2e-6, 2e-5, 2e-4])
    dataset = TensorDataset(torch.rand(5, 2, 2, 1), normalize_sxr(flux, model.sxr_norm))
    dataset.sxr_dir = tmp_path
    dataset.samples = [str(i) for i in range(5)]
    for name, value in zip(dataset.samples, flux):
        np.save(tmp_path / f'{name}.npy', value.numpy())
    figures = []

    def capture(figure):
        figures.append(figure)
        return 'image'

    monkeypatch.setattr(callbacks.wandb, 'Image', capture)
    monkeypatch.setattr(callbacks.wandb, 'Table', lambda **kwargs: kwargs)
    experiment = Mock()
    trainer = SimpleNamespace(sanity_checking=False, is_global_zero=True,
                              current_epoch=0, logger=SimpleNamespace(experiment=experiment))
    callback = callbacks.QuantilePredictionCallback(dataset, num_samples=5,
                                                   map_num_samples=5, patch_size=1)
    model.train()
    callback.on_validation_epoch_end(trainer, model)
    assert model.training
    assert len(figures) == 6  # Five spatial examples and the five-panel parity plot.
    parity = figures[-1]
    assert len(parity.axes) == 5
    assert all(len(axis.collections[0].get_offsets()) == 5 for axis in parity.axes)
    spatial = figures[0]
    images = [axis.images[0] for axis in spatial.axes if axis.images and axis.get_title() != 'AIA input']
    assert len(images) == 10  # Local and corrected contributions for every quantile.
    assert all(image.norm is images[0].norm for image in images)
    table = experiment.log.call_args.args[0]['Quantile prediction values']
    assert len(table['columns']) == 7
    assert len(table['data']) == 5
    assert all(not plt.fignum_exists(figure.number) for figure in figures)

    # Optional attention plots must unpack quantile outputs and use the median map.
    trainer.datamodule = SimpleNamespace(val_ds=dataset)
    callbacks.AttentionMapCallback(
        patch_size=1, num_samples=5, use_local_attention=True,
    )._visualize_attention(
        trainer, model,
    )
    assert 'Attention plots' in experiment.log.call_args.args[0]


def test_gaussian_per_class_uncertainty_metrics_still_include_nll(model):
    gaussian = CoupledGaussianViTLocal(model.hparams.model_kwargs, model.sxr_norm)
    gaussian.log = Mock()
    logged = {}
    gaussian.log_dict = lambda metrics, **kwargs: logged.update(metrics)
    outputs = gaussian.validation_step((torch.rand(2, 2, 2, 1), torch.zeros(2)), 0)
    trainer = SimpleNamespace(sanity_checking=False)
    metrics = callbacks.PerClassValidationMetrics()
    metrics.on_validation_epoch_start(trainer, gaussian)
    metrics.on_validation_batch_end(trainer, gaussian, outputs, None, 0)
    metrics.on_validation_epoch_end(trainer, gaussian)
    assert 'val_class/macro_nll' in logged


def test_pinball_training_config_wires_quantile_callbacks():
    config = load_training_config('training/train_coupled_pinball.yaml')
    assert config['pinball']['class_reduction'] == 'sum'
    data = SimpleNamespace(val_ds=[])
    _, loss_options = configure_class_weights(config, data, 'coupled_pinball')
    assert loss_options == {'bias_penalty_weight': 0.0}
    selected = build_model_callbacks(config, data, np.array([-6., 1.]), 'coupled_pinball')
    assert any(isinstance(item, callbacks.QuantilePredictionCallback) for item in selected)
    assert any(isinstance(item, callbacks.QuantileValidationMetrics) for item in selected)
    assert not any(isinstance(item, callbacks.SpatialUncertaintyMapCallback) for item in selected)
    assert not any(isinstance(item, callbacks.ImagePredictionLogger_SXR) for item in selected)


def test_lightning_training_and_validation_with_quantile_metrics(model, tmp_path):
    dataset = TensorDataset(torch.rand(4, 2, 2, 1), torch.tensor([-.5, 0., .5, 1.]))
    loader = DataLoader(dataset, batch_size=2)
    trainer = Trainer(
        max_epochs=1, limit_train_batches=1, limit_val_batches=1,
        num_sanity_val_steps=0, logger=CSVLogger(tmp_path), enable_checkpointing=False,
        log_every_n_steps=1,
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[callbacks.PerClassValidationMetrics(), callbacks.QuantileValidationMetrics()],
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert 'val_quantile/q0.05/coverage' in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics['val_total_loss'])
    assert 'val/mse' in trainer.callback_metrics
    assert 'val/rmse' in trainer.callback_metrics
    torch.testing.assert_close(
        trainer.callback_metrics['val/rmse'].square(), trainer.callback_metrics['val/mse'],
    )
