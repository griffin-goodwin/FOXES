import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from forecasting.dataset import AIANormTransform
from training.callbacks import (
    AttentionMapCallback,
    PerClassQuantileValidationMetrics,
    PerClassValidationMetrics,
    SpatialGaussianMapCallback,
    SpatialQuantileMapCallback,
    stratified_validation_indices,
)


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


def test_background_excess_plot_distinguishes_learned_and_accounting_maps():
    image = torch.zeros(2, 2, 7)
    accounting_flux = torch.tensor([3e-7, 4e-7, 5e-7, 6e-7])
    variance_attribution = accounting_flux.square()
    component_data = {
        'background_flux_raw': torch.tensor(1e-6),
        'background_variance_raw': torch.tensor(1e-14),
        'excess_patch_flux_raw': torch.tensor([5e-8, 1.5e-7, 2.5e-7, 3.5e-7]),
        'excess_variance_contribution_raw': torch.full((4,), 1e-15),
    }

    fig = SpatialGaussianMapCallback._plot_spatial_gaussian_maps(
        image,
        accounting_flux,
        variance_attribution,
        patch_size=1,
        component_data=component_data,
    )

    titles = [axis.get_title() for axis in fig.axes[:5]]
    assert any('Learned Local Excess' in title for title in titles)
    assert any('Total Accounting Map' in title for title in titles)
    assert 'whole-image B=' in fig._suptitle.get_text()
    assert all('Background Map' not in title for title in titles)
    plt.close(fig)


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


def test_per_class_quantile_validation_logs_coverage_and_pinball():
    class Trainer:
        sanity_checking = False

    class Module:
        device = torch.device("cpu")
        sxr_norm = torch.tensor([-6.0, 1.0])

        def log_dict(self, metrics, **kwargs):
            self.metrics = metrics
            self.log_kwargs = kwargs

    callback = PerClassQuantileValidationMetrics()
    trainer = Trainer()
    module = Module()
    target_raw = torch.tensor([5e-8, 5e-7, 5e-6, 5e-5, 5e-4])
    quantiles = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0, 2.0]] * 5
    )
    outputs = {
        "target_raw": target_raw,
        "target_norm": torch.zeros(5),
        "mean_norm": torch.zeros(5),
        "prediction_raw": target_raw,
        "quantiles_norm": quantiles,
        "pinball_per_sample": torch.full((5,), 0.084),
    }

    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(trainer, module, outputs, None, 0)
    callback.on_validation_epoch_end(trainer, module)

    for name in ("Below-B", "B", "C", "M", "X"):
        np.testing.assert_allclose(
            module.metrics[f"val_class/{name}/pinball"], 0.084
        )
        np.testing.assert_allclose(
            module.metrics[f"val_class/{name}/coverage_68"], 1.0
        )
        np.testing.assert_allclose(
            module.metrics[f"val_class/{name}/coverage_95"], 1.0
        )
        np.testing.assert_allclose(
            module.metrics[f"val_class/{name}/mean_width_68_dex"], 2.0
        )
        np.testing.assert_allclose(
            module.metrics[f"val_class/{name}/mean_width_95_dex"], 4.0
        )
    np.testing.assert_allclose(
        module.metrics["val_class/macro_pinball"], 0.084
    )
    assert module.log_kwargs["on_epoch"] is True


def test_spatial_quantile_plot_contains_median_and_interval_width_maps():
    image = torch.zeros(2, 2, 7)
    patch_quantiles = torch.tensor([
        [1e-9, 2e-9, 3e-9, 4e-9, 5e-9],
        [2e-9, 3e-9, 4e-9, 5e-9, 6e-9],
        [3e-9, 4e-9, 5e-9, 6e-9, 7e-9],
        [4e-9, 5e-9, 6e-9, 7e-9, 8e-9],
    ])

    fig = AttentionMapCallback._plot_spatial_quantile_maps(
        image, patch_quantiles, patch_size=1
    )

    titles = {axis.get_title() for axis in fig.axes}
    assert 'q50 Patch Flux' in titles
    assert 'Patch 68% Interval Width' in titles
    assert 'Patch 95% Interval Width' in titles
    plt.close(fig)


def test_spatial_quantile_callback_does_not_request_attention():
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
        uncertainty_kind = 'quantile'

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
            patch_q50 = torch.full((batch_size, 4), 3e-9)
            offsets = torch.tensor([1e-9, 2e-9, 3e-9, 4e-9, 5e-9])
            patch_quantiles = offsets.reshape(1, 1, 5).expand(
                batch_size, 4, 5
            )
            global_quantiles = patch_quantiles.sum(dim=1)
            return (
                global_quantiles[:, 2:3], global_quantiles,
                patch_q50, patch_quantiles,
            )

    experiment = Experiment()
    trainer = type('Trainer', (), {
        'is_global_zero': True,
        'current_epoch': 0,
        'datamodule': type('DataModule', (), {'val_ds': Dataset()})(),
        'logger': type('Logger', (), {'experiment': experiment})(),
    })()
    model = Module()
    model.train()
    callback = SpatialQuantileMapCallback(
        num_samples=5, patch_size=1, log_every_n_epochs=1
    )

    callback.on_validation_epoch_end(trainer, model)

    assert model.return_attention_values == [False]
    assert len(experiment.logged) == 5
    assert model.training is True


def test_spatial_gaussian_callback_logs_each_class_without_attention():
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
    callback = SpatialGaussianMapCallback(
        num_samples=5, patch_size=1, log_every_n_epochs=1
    )

    callback.on_validation_epoch_end(trainer, model)

    assert model.return_attention_values == [False]
    assert len(experiment.logged) == 5
    assert model.training is True
