import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from forecasting.dataset import AIANormTransform
from forecasting.uncertainty_model import GaussianNLLViTLocal
from training.callbacks import (
    AttentionMapCallback,
    ImagePredictionLogger_SXR,
    PerClassValidationMetrics,
    SpatialUncertaintyMapCallback,
    stratified_validation_indices,
)
from training.train import build_model_callbacks


def test_model_types_share_all_compatible_callbacks():
    class DataModule:
        val_ds = [object()]

    config = {
        'vit_architecture': {'patch_size': 8},
        'uncertainty': {'enabled': True},
        'callbacks': {
            'per_class_metrics_enabled': True,
            'sxr_plot_enabled': True,
            'spatial_uncertainty_enabled': True,
            'attention_enabled': True,
        },
    }
    deterministic = build_model_callbacks(
        config, DataModule(), np.array([-6.0, 1.0]), 'deterministic'
    )
    uncertain = build_model_callbacks(
        config, DataModule(), np.array([-6.0, 1.0]), 'uncertainty'
    )

    common_types = (
        PerClassValidationMetrics,
        ImagePredictionLogger_SXR,
        AttentionMapCallback,
    )
    assert tuple(type(callback) for callback in deterministic) == common_types
    assert tuple(
        type(callback) for callback in uncertain
        if not isinstance(callback, SpatialUncertaintyMapCallback)
    ) == common_types
    assert sum(
        isinstance(callback, SpatialUncertaintyMapCallback)
        for callback in uncertain
    ) == 1


def test_mean_only_uncertainty_arm_uses_exact_common_callback_set():
    class DataModule:
        val_ds = [object()]

    config = {
        'vit_architecture': {'patch_size': 8},
        'uncertainty': {'enabled': False},
        'callbacks': {
            'per_class_metrics_enabled': True,
            'sxr_plot_enabled': True,
            'spatial_uncertainty_enabled': True,
            'attention_enabled': True,
        },
    }
    deterministic = build_model_callbacks(
        config, DataModule(), np.array([-6.0, 1.0]), 'deterministic'
    )
    mean_only = build_model_callbacks(
        config, DataModule(), np.array([-6.0, 1.0]), 'uncertainty'
    )

    assert tuple(map(type, deterministic)) == tuple(map(type, mean_only))


def test_display_image_inverts_aia_normalization():
    transform = AIANormTransform(
        wavelengths=[94, 131, 171, 193, 211, 304, 335],
        q90=np.ones(7),
        clip_q99999=np.full(7, 10.0),
        asinh_mean=np.full(7, 0.5),
        asinh_std=np.full(7, 2.0),
    )
    raw = torch.tensor([0.0, 5.0, 10.0]).reshape(1, 3).expand(7, 1, 3)
    normalized = transform(raw).permute(1, 2, 0)

    displayed = AttentionMapCallback._display_image(normalized, transform)

    expected = np.arcsinh([0.0, 5.0, 10.0]) / np.arcsinh(10.0)
    np.testing.assert_allclose(displayed[0, :, 0], expected, rtol=1e-6)
    np.testing.assert_allclose(displayed[0, :, 1], expected, rtol=1e-6)
    np.testing.assert_allclose(displayed[0, :, 2], expected, rtol=1e-6)


def test_display_image_handles_nonfinite_values():
    image = torch.tensor([[[float("nan"), float("inf"), float("-inf")]]])
    displayed = AttentionMapCallback._display_image(image)

    assert displayed.shape == (1, 1, 3)
    assert np.isfinite(displayed).all()


def test_patch_flux_map_uses_log_scale_for_positive_dynamic_range():
    callback = AttentionMapCallback(use_local_attention=True, patch_size=1)
    image = torch.zeros(2, 2, 7)
    attention = [torch.full((1, 1, 4, 4), 0.25)]
    patch_flux = torch.tensor([1e-10, 1e-9, 1e-8, 1e-7])

    fig = callback._plot_attention_map(
        image, attention, sample_idx=0, epoch=0, patch_size=1,
        patch_flux=patch_flux,
    )

    patch_axis = fig.axes[4]
    assert isinstance(patch_axis.images[0].norm, LogNorm)
    assert "Learned Patch Contribution" in patch_axis.get_title()
    plt.close(fig)


def test_global_patch_plot_compares_local_and_adjusted_flux_on_one_scale():
    corrected = torch.tensor([1e-9, 2e-9, 3e-9, 4e-9])
    adjustment = torch.tensor([0.0, 0.1, -0.1, 0.2])
    fig = SpatialUncertaintyMapCallback._plot_global_patch_adjustment(
        torch.zeros(2, 2, 7), corrected, adjustment,
        patch_size=1, max_abs_adjustment=0.3,
    )

    local_image = fig.axes[1].images[0]
    adjusted_image = fig.axes[2].images[0]
    assert local_image.norm is adjusted_image.norm
    assert isinstance(local_image.norm, LogNorm)
    np.testing.assert_allclose(
        local_image.get_array(),
        (corrected / torch.pow(10.0, adjustment)).reshape(2, 2),
    )
    np.testing.assert_allclose(
        adjusted_image.get_array(), corrected.reshape(2, 2),
    )
    plt.close(fig)


def test_attention_callback_forwards_visualization_samples_one_at_a_time(
    monkeypatch,
):
    class Dataset:
        _callback_flare_class_indices = {
            'Below-B': [0, 1, 2], 'B': [], 'C': [], 'M': [], 'X': [],
        }

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.zeros(2, 2, 7), torch.tensor(float(index))

    class Experiment:
        def __init__(self):
            self.logged = []

        def log(self, value):
            self.logged.append(value)

    class Module(torch.nn.Module):
        predicts_uncertainty = True
        patch_uncertainty_semantics = 'variance_attribution'

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.batch_sizes = []

        @property
        def device(self):
            return self.anchor.device

        def forward(self, images, return_attention=False):
            assert return_attention is True
            batch_size = images.shape[0]
            self.batch_sizes.append(batch_size)
            patch_flux = torch.ones(batch_size, 4)
            patch_variance = torch.ones(batch_size, 4)
            attention = [torch.ones(batch_size, 1, 4, 4)]
            return (
                patch_flux.sum(dim=1, keepdim=True),
                patch_variance.sum(dim=1),
                attention,
                patch_flux,
                patch_variance,
            )

    experiment = Experiment()
    trainer = type('Trainer', (), {
        'current_epoch': 0,
        'datamodule': type('DataModule', (), {'val_ds': Dataset()})(),
        'logger': type('Logger', (), {'experiment': experiment})(),
    })()
    model = Module()
    model.train()
    callback = AttentionMapCallback(
        num_samples=3, patch_size=1, use_local_attention=True
    )
    monkeypatch.setattr(
        callback, '_plot_attention_map', lambda *args, **kwargs: plt.figure()
    )
    monkeypatch.setattr('training.callbacks.wandb.Image', lambda figure: figure)

    callback._visualize_attention(trainer, model)

    assert model.batch_sizes == [1, 1, 1]
    assert len(experiment.logged) == 3
    assert model.training is True


def test_validation_sampling_covers_and_balances_available_flare_classes():
    class Dataset:
        samples = list(range(107))
        _callback_flare_class_indices = {
            "Below-B": list(range(100)),
            "B": [100, 101, 102],
            "C": [103, 104],
            "M": [105],
            "X": [106],
        }

        def __len__(self):
            return len(self.samples)

    indices = stratified_validation_indices(Dataset(), num_samples=8)
    groups = [
        "Below-B" if index < 100 else
        "B" if index < 103 else
        "C" if index < 105 else
        "M" if index < 106 else "X"
        for index in indices
    ]

    assert len(indices) == 8
    assert set(groups) == {"Below-B", "B", "C", "M", "X"}
    assert groups.count("Below-B") == 2
    assert groups.count("B") == 2
    assert groups.count("C") == 2
    assert groups.count("M") == 1
    assert groups.count("X") == 1


def test_per_class_validation_logs_five_class_and_macro_nll():
    class Trainer:
        sanity_checking = False
        is_global_zero = True
        global_step = 12

    class Module:
        device = torch.device("cpu")
        sxr_norm = torch.tensor([-6.0, 1.0])

        def log_dict(self, metrics, **kwargs):
            self.metrics = metrics
            self.log_kwargs = kwargs

    callback = PerClassValidationMetrics()
    trainer = Trainer()
    module = Module()
    target_raw = torch.tensor([5e-8, 5e-7, 5e-6, 5e-5, 5e-4])
    outputs = {
        "target_raw": target_raw,
        "target_norm": torch.zeros(5),
        "mean_norm": torch.zeros(5),
        "variance_norm": torch.ones(5),
        "prediction_raw": target_raw,
    }

    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(trainer, module, outputs, None, 0)
    callback.on_validation_epoch_end(trainer, module)

    expected_nll = 0.5 * np.log(2 * np.pi)
    for name in ("Below-B", "B", "C", "M", "X"):
        np.testing.assert_allclose(
            module.metrics[f"val_class/{name}/nll"], expected_nll
        )
        np.testing.assert_allclose(
            module.metrics[f"val_class/{name}/mean_sigma_dex"], 1.0
        )
        np.testing.assert_allclose(
            module.metrics[f"val_class/{name}/mse_dex2"], 0.0
        )
        np.testing.assert_allclose(
            module.metrics[f"val_class/{name}/coverage_68"], 1.0
        )
        np.testing.assert_allclose(
            module.metrics[f"val_class/{name}/coverage_95"], 1.0
        )
    np.testing.assert_allclose(
        module.metrics["val_class/macro_nll"], expected_nll
    )
    np.testing.assert_allclose(
        module.metrics["val_class/macro_mse_dex2"], 0.0
    )
    np.testing.assert_allclose(
        module.metrics["val_class/macro_coverage_68"], 1.0
    )
    np.testing.assert_allclose(
        module.metrics["val_class/macro_coverage_95"], 1.0
    )
    assert module.log_kwargs["on_epoch"] is True


def test_per_class_validation_logs_mean_metrics_without_uncertainty():
    class Trainer:
        sanity_checking = False

    class Module:
        device = torch.device("cpu")
        sxr_norm = torch.tensor([-6.0, 1.0])
        predicts_uncertainty = False

        def log_dict(self, metrics, **kwargs):
            self.metrics = metrics

    callback = PerClassValidationMetrics()
    module = Module()
    outputs = {
        "target_raw": torch.tensor([5e-8, 5e-7, 5e-6, 5e-5, 5e-4]),
        "target_norm": torch.zeros(5),
        "mean_norm": torch.zeros(5),
        "prediction_raw": torch.tensor([5e-8, 5e-7, 5e-6, 5e-5, 5e-4]),
    }

    callback.on_validation_epoch_start(Trainer(), module)
    callback.on_validation_batch_end(Trainer(), module, outputs, None, 0)
    callback.on_validation_epoch_end(Trainer(), module)

    assert module.metrics["val_class/macro_mse_dex2"] == 0.0
    assert "val_class/macro_nll" not in module.metrics


def test_spatial_uncertainty_callback_logs_each_class_without_attention():
    class Dataset:
        _callback_flare_class_indices = {
            'Below-B': [0], 'B': [1], 'C': [2], 'M': [3], 'X': [4],
        }

        def __len__(self):
            return 5

        def __getitem__(self, index):
            return torch.zeros(2, 2, 7), torch.tensor(float(index))

    class Experiment:
        def __init__(self):
            self.logged = []

        def log(self, value):
            self.logged.append(value)

    class Module(torch.nn.Module):
        uncertainty_kind = 'gaussian'

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.return_attention_values = []

        @property
        def device(self):
            return self.anchor.device

        def forward(self, images, return_attention=False):
            self.return_attention_values.append(return_attention)
            batch_size = images.shape[0]
            patch_flux = torch.tensor(
                [1e-9, 2e-9, 3e-9, 4e-9]
            ).expand(batch_size, 4)
            patch_variance = patch_flux.square()
            prediction = patch_flux.sum(dim=1, keepdim=True)
            variance = patch_variance.sum(dim=1)
            return prediction, variance, patch_flux, patch_variance

    experiment = Experiment()
    trainer = type('Trainer', (), {
        'is_global_zero': True,
        'current_epoch': 0,
        'datamodule': type('DataModule', (), {'val_ds': Dataset()})(),
        'logger': type('Logger', (), {'experiment': experiment})(),
    })()
    model = Module()
    model.train()
    callback = SpatialUncertaintyMapCallback(
        num_samples=5, patch_size=1, log_every_n_epochs=1
    )

    callback.on_validation_epoch_end(trainer, model)

    assert model.return_attention_values == [False]
    assert len(experiment.logged) == 5
    assert model.training is True


@pytest.mark.parametrize('global_enabled', [False, True])
def test_spatial_callback_uses_global_head_adjustment_limit(monkeypatch, global_enabled):
    from types import SimpleNamespace

    model = GaussianNLLViTLocal(
        model_kwargs=dict(
            embed_dim=8, hidden_dim=16, num_channels=7, num_heads=1,
            num_layers=1, patch_size=1, num_patches=4, dropout=0.0,
            patch_global_context=dict(
                enabled=global_enabled, grid_size=2, attention_dim=8,
                num_heads=2, max_abs_adjustment=0.7,
            ),
        ),
        sxr_norm=[-6.0, 1.0],
    )
    logged = []
    trainer = SimpleNamespace(
        is_global_zero=True, current_epoch=0,
        datamodule=SimpleNamespace(
            val_ds=[(torch.zeros(2, 2, 7), torch.tensor(0.0))],
        ),
        logger=SimpleNamespace(experiment=SimpleNamespace(log=logged.append)),
    )
    callback = SpatialUncertaintyMapCallback(
        num_samples=1, patch_size=1, log_every_n_epochs=1,
    )
    adjustment_limits = []
    plot_adjustment = callback._plot_global_patch_adjustment

    def capture_adjustment(*args, **kwargs):
        figure = plot_adjustment(*args, **kwargs)
        adjustment_limits.append(figure.axes[3].images[0].get_clim())
        return figure

    monkeypatch.setattr(callback, '_plot_global_patch_adjustment', capture_adjustment)
    callback.on_validation_epoch_end(trainer, model)

    assert adjustment_limits == ([(-0.7, 0.7)] if global_enabled else [])
    assert len(logged) == (2 if global_enabled else 1)
    assert model.training is True
