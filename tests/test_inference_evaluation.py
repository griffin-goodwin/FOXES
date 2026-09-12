import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

import forecasting.inference as inference_module
from forecasting.evaluation import FOXESEvaluator
from forecasting.inference import (
    _AutocastForwardWrapper,
    _auto_checkpoint_model_class,
    _resolve_amp_dtype,
    evaluate_model_on_dataset,
    quantile_uncertainty_from_prediction,
    uncertainty_from_prediction,
)
from forecasting.model_quantile import QuantileViTLocal
from forecasting.model_uncertainty import GaussianNLLViTLocal
from forecasting.model_uncertainty_background_excess import (
    BackgroundExcessGaussianNLLViTLocal,
)


def make_uncertainty_model(uncertainty_kwargs=None):
    return GaussianNLLViTLocal(
        model_kwargs={
            'embed_dim': 8,
            'hidden_dim': 16,
            'num_channels': 7,
            'num_heads': 1,
            'num_layers': 1,
            'patch_size': 1,
            'num_patches': 4,
            'dropout': 0.0,
            'mask_mode': 'none',
        },
        sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
        uncertainty_kwargs=uncertainty_kwargs,
    )


def make_background_excess_model():
    return BackgroundExcessGaussianNLLViTLocal(
        model_kwargs={
            'embed_dim': 8,
            'hidden_dim': 16,
            'num_channels': 7,
            'num_heads': 1,
            'num_layers': 1,
            'patch_size': 1,
            'num_patches': 4,
            'dropout': 0.0,
            'mask_mode': 'local',
            'local_window': 3,
        },
        sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
        uncertainty_kwargs={
            'background_flux_max': 5e-6,
            'background_initial_cap_fraction': 0.5,
            'excess_initial_fraction': 0.2,
            'solar_disk_radius_fraction': 0.48,
            'background_scale_floor_fraction': 0.05,
            'background_uncertainty_initial_raw': -3.0,
        },
    )


def test_auto_checkpoint_detection_selects_background_excess_gaussian():
    checkpoint = {
        'state_dict': {
            'model.mean_parameterization_version': torch.tensor(4),
            'model.patch_uncertainty_head.1.weight': torch.zeros(1, 8),
            'model.background_head.1.weight': torch.zeros(1, 8),
        },
    }

    assert (
        _auto_checkpoint_model_class(checkpoint)
        is BackgroundExcessGaussianNLLViTLocal
    )


def make_quantile_model():
    return QuantileViTLocal(
        model_kwargs={
            'embed_dim': 8,
            'hidden_dim': 16,
            'num_channels': 7,
            'num_heads': 1,
            'num_layers': 1,
            'patch_size': 1,
            'num_patches': 4,
            'dropout': 0.0,
            'mask_mode': 'none',
        },
        sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
        quantile_kwargs={'loss_weighting': 'unweighted'},
    )


def test_uncertainty_conversion_produces_asymmetric_physical_intervals():
    sxr_norm = torch.tensor([-6.0, 1.0])
    prediction = torch.tensor([[9.9e-7]])
    variance = torch.tensor([1.0])

    result = uncertainty_from_prediction(prediction, variance, sxr_norm)

    torch.testing.assert_close(
        result['prediction_normalized'], torch.tensor([0.0])
    )
    torch.testing.assert_close(result['sigma_normalized'], torch.tensor([1.0]))
    torch.testing.assert_close(result['lower_68'], torch.tensor([9e-8]))
    torch.testing.assert_close(result['upper_68'], torch.tensor([9.99e-6]))
    assert (
        prediction.item() - result['lower_68'].item()
        != result['upper_68'].item() - prediction.item()
    )


def test_quantile_conversion_preserves_direct_intervals():
    sxr_norm = torch.tensor([-6.0, 1.0])
    global_quantiles = torch.tensor([[
        9e-8, 3.0623e-7, 9.9e-7, 3.1523e-6, 9.99e-6,
    ]])

    result = quantile_uncertainty_from_prediction(
        global_quantiles, sxr_norm
    )

    torch.testing.assert_close(result['lower_68'], global_quantiles[:, 1])
    torch.testing.assert_close(result['upper_68'], global_quantiles[:, 3])
    torch.testing.assert_close(result['lower_95'], global_quantiles[:, 0])
    torch.testing.assert_close(result['upper_95'], global_quantiles[:, 4])
    assert result['q025_normalized'] < result['q16_normalized']
    assert result['q84_normalized'] < result['q975_normalized']


def test_inference_parses_gaussian_outputs_without_confusing_variance_and_maps():
    model = make_uncertainty_model()
    images = torch.zeros(2, 2, 2, 7)
    raw_targets = torch.tensor([1e-7, 1e-5])
    dataset = TensorDataset(images, raw_targets)

    results = list(evaluate_model_on_dataset(
        model,
        dataset,
        batch_size=2,
        times=['sample-0', 'sample-1'],
        config_data={'num_workers': 0, 'pin_memory': False, 'use_amp': False},
        save_weights=False,
        input_size=2,
        patch_size=1,
        save_flux=False,
        save_patch_uncertainty=False,
    ))

    assert len(results) == 2
    assert all(len(result) == 6 for result in results)
    assert all(result[-1] is not None for result in results)
    assert all(result[2] is None and result[3] is None for result in results)
    assert all(result[-1]['variance_normalized'] > 0 for result in results)


def test_gaussian_inference_saves_patch_standard_deviation(tmp_path):
    model = make_uncertainty_model({
        'relative_std_floor': 0.0025,
        'relative_std_max': 20.0,
        'patch_scale_floor_fraction': 1.0,
    })
    images = torch.zeros(1, 2, 2, 7)
    dataset = TensorDataset(images, torch.tensor([1e-6]))
    map_dir = tmp_path / 'patch_uncertainty'

    results = list(evaluate_model_on_dataset(
        model,
        dataset,
        batch_size=1,
        times=['sample-0'],
        config_data={
            'num_workers': 0,
            'pin_memory': False,
            'use_amp': False,
            'patch_uncertainty_path': str(map_dir),
        },
        save_weights=False,
        input_size=2,
        patch_size=1,
        save_flux=False,
        save_patch_uncertainty=True,
    ))

    assert len(results) == 1
    assert results[0][-1]['variance_normalized'] > 0
    saved = np.load(map_dir / 'sample-0.npy')
    assert saved.shape == (2, 2)
    assert np.all(saved > 0)


def test_background_excess_inference_saves_named_components(tmp_path):
    model = make_background_excess_model()
    images = torch.zeros(1, 2, 2, 7)
    dataset = TensorDataset(images, torch.tensor([1e-6]))
    flux_dir = tmp_path / 'accounting_flux'
    component_dir = tmp_path / 'components'

    results = list(evaluate_model_on_dataset(
        model,
        dataset,
        batch_size=1,
        times=['sample-0'],
        config_data={
            'num_workers': 0,
            'pin_memory': False,
            'use_amp': False,
            'flux_path': str(flux_dir),
            'component_flux_path': str(component_dir),
        },
        save_weights=False,
        input_size=2,
        patch_size=1,
        save_flux=True,
        save_patch_uncertainty=False,
    ))

    assert len(results) == 1
    assert len(results[0]) == 6
    diagnostics = results[0][-1]
    assert {
        'background_flux_raw', 'excess_flux_raw', 'background_fraction',
        'background_cap_fraction', 'background_std_raw',
        'summed_excess_variance_raw', 'background_variance_fraction',
    }.issubset(diagnostics)
    np.testing.assert_allclose(
        diagnostics['background_flux_raw'] + diagnostics['excess_flux_raw'],
        float(results[0][0].item()),
        rtol=1e-6,
    )
    assert 0 < diagnostics['background_fraction'] < 1
    assert 0 < diagnostics['background_variance_fraction'] < 1

    accounting = np.load(flux_dir / 'sample-0.npy')
    with np.load(component_dir / 'sample-0.npz') as components:
        assert set(components.files) == {
            'background_flux_raw',
            'background_cap_fraction',
            'excess_patch_flux_raw',
            'background_allocation_raw',
            'total_accounting_patch_flux_raw',
            'background_variance_raw',
            'excess_variance_contribution_raw',
            'total_variance_attribution_raw',
            'global_variance_raw',
        }
        assert components['excess_patch_flux_raw'].shape == (2, 2)
        np.testing.assert_allclose(
            accounting,
            components['total_accounting_patch_flux_raw'],
        )
        np.testing.assert_allclose(
            components['background_allocation_raw'].sum(),
            components['background_flux_raw'],
        )
        np.testing.assert_allclose(
            components['total_accounting_patch_flux_raw'],
            components['background_allocation_raw']
            + components['excess_patch_flux_raw'],
        )
        np.testing.assert_allclose(
            components['total_variance_attribution_raw'].sum(),
            components['global_variance_raw'],
            rtol=1e-6,
        )


def test_quantile_inference_saves_spatial_interval_maps(tmp_path):
    model = make_quantile_model()
    images = torch.zeros(2, 2, 2, 7)
    dataset = TensorDataset(images, torch.tensor([1e-7, 1e-5]))
    map_dir = tmp_path / 'spatial_quantiles'

    results = list(evaluate_model_on_dataset(
        model,
        dataset,
        batch_size=2,
        times=['sample-0', 'sample-1'],
        config_data={
            'num_workers': 0,
            'pin_memory': False,
            'use_amp': False,
            'patch_uncertainty_path': str(map_dir),
        },
        save_weights=False,
        input_size=2,
        patch_size=1,
        save_flux=False,
        save_patch_uncertainty=True,
    ))

    assert len(results) == 2
    assert all(len(result) == 6 for result in results)
    assert all(result[-1]['q025_normalized'] < result[-1]['q975_normalized']
               for result in results)
    saved = np.load(map_dir / 'sample-0.npz')
    assert set(saved.files) == {
        'q025', 'q16', 'q50', 'q84', 'q975', 'width_68', 'width_95',
    }
    assert saved['q50'].shape == (2, 2)
    assert np.all(saved['width_68'] > 0)
    assert np.all(saved['width_95'] > saved['width_68'])


def test_deterministic_inference_retains_five_item_result_tuple():
    class DeterministicModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, images, return_attention=False):
            predictions = torch.full(
                (images.shape[0], 1), 1e-6, device=images.device
            ) + self.anchor * 0
            patch_flux = torch.full(
                (images.shape[0], 4), 2.5e-7, device=images.device
            )
            return predictions, patch_flux

    dataset = TensorDataset(
        torch.zeros(1, 2, 2, 7), torch.tensor([1e-6])
    )
    result = next(evaluate_model_on_dataset(
        DeterministicModel(), dataset, batch_size=1, times=['sample'],
        config_data={'num_workers': 0, 'pin_memory': False, 'use_amp': False},
        save_weights=False, input_size=2, patch_size=1, save_flux=False,
    ))

    assert len(result) == 5


def test_inference_mode_is_enabled_and_gc_runs_only_at_shutdown(monkeypatch):
    inference_mode_seen = []
    collect_calls = []

    class InferenceModeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, images, return_attention=False):
            inference_mode_seen.append(torch.is_inference_mode_enabled())
            predictions = torch.full(
                (images.shape[0], 1), 1e-6, device=images.device
            ) + self.anchor * 0
            patch_flux = torch.full(
                (images.shape[0], 4), 2.5e-7, device=images.device
            )
            return predictions, patch_flux

    monkeypatch.setattr(
        inference_module.gc, 'collect', lambda: collect_calls.append(True)
    )
    dataset = TensorDataset(
        torch.zeros(4, 2, 2, 7), torch.full((4,), 1e-6)
    )

    results = list(evaluate_model_on_dataset(
        InferenceModeModel(), dataset, batch_size=2,
        times=[f'sample-{index}' for index in range(4)],
        config_data={'num_workers': 0, 'pin_memory': False},
        save_weights=False, input_size=2, patch_size=1, save_flux=False,
    ))

    assert len(results) == 4
    assert inference_mode_seen == [True, True]
    assert collect_calls == [True]


def test_amp_dtype_is_validated_and_wrapper_enters_requested_dtype():
    assert _resolve_amp_dtype(
        {'amp_dtype': 'bf16'}, torch.device('cpu')
    ) == torch.bfloat16
    with pytest.raises(ValueError, match='amp_dtype'):
        _resolve_amp_dtype(
            {'amp_dtype': 'not-a-dtype'}, torch.device('cpu')
        )

    class MatmulProbe(torch.nn.Module):
        marker = 'forwarded'

        def forward(self, tensor):
            return tensor @ tensor

    wrapped = _AutocastForwardWrapper(
        MatmulProbe(), 'cpu', torch.bfloat16
    )
    assert wrapped(torch.eye(2)).dtype == torch.bfloat16
    assert wrapped.marker == 'forwarded'


def test_evaluator_reports_nll_coverage_and_class_breakdown(tmp_path):
    raw = np.array([5e-8, 5e-7, 2e-6, 2e-5, 2e-4])
    normalized = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    variance = np.full(5, 0.25)
    csv_path = tmp_path / 'predictions.csv'
    pd.DataFrame({
        'timestamp': [f'sample-{index}' for index in range(5)],
        'predictions': raw,
        'groundtruth': raw,
        'prediction_normalized': normalized,
        'groundtruth_normalized': normalized,
        'variance_normalized': variance,
        'sigma_normalized': np.full(5, 0.5),
        'sigma_dex': np.full(5, 0.5),
        'lower_68': raw * 0.5,
        'upper_68': raw * 2.0,
        'lower_95': raw * 0.25,
        'upper_95': raw * 4.0,
    }).to_csv(csv_path, index=False)

    evaluator = FOXESEvaluator(csv_path, output_dir=tmp_path / 'evaluation')
    evaluator.load_data()
    metrics = evaluator.calculate_uncertainty_metrics()

    assert metrics['Group'].tolist() == [
        'Overall', 'Below-B', 'B', 'C', 'M', 'X'
    ]
    overall = metrics.iloc[0]
    expected_nll = 0.5 * np.log(2.0 * np.pi * 0.25)
    np.testing.assert_allclose(
        overall['Gaussian_NLL_Normalized'], expected_nll
    )
    assert overall['Coverage_68'] == 1.0
    assert overall['Coverage_95'] == 1.0
    assert overall['Mean_Absolute_Z'] == 0.0
    assert (tmp_path / 'evaluation/metrics/uncertainty_metrics.csv').exists()
    assert (tmp_path / 'evaluation/plots/uncertainty_calibration.png').exists()


def test_evaluator_reports_quantile_coverage_and_pinball(tmp_path):
    raw = np.array([5e-8, 5e-7, 2e-6, 2e-5, 2e-4])
    target = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    csv_path = tmp_path / 'quantile_predictions.csv'
    pd.DataFrame({
        'timestamp': [f'sample-{index}' for index in range(5)],
        'predictions': raw,
        'groundtruth': raw,
        'prediction_normalized': target,
        'groundtruth_normalized': target,
        'q025_normalized': target - 1.0,
        'q16_normalized': target - 0.5,
        'q50_normalized': target,
        'q84_normalized': target + 0.5,
        'q975_normalized': target + 1.0,
        'lower_68': raw * 0.5,
        'upper_68': raw * 2.0,
        'lower_95': raw * 0.25,
        'upper_95': raw * 4.0,
    }).to_csv(csv_path, index=False)

    evaluator = FOXESEvaluator(
        csv_path, output_dir=tmp_path / 'quantile_evaluation'
    )
    evaluator.load_data()
    metrics = evaluator.calculate_uncertainty_metrics()

    assert evaluator.uncertainty_kind == 'quantile'
    assert metrics['Group'].tolist() == [
        'Overall', 'Below-B', 'B', 'C', 'M', 'X'
    ]
    overall = metrics.iloc[0]
    np.testing.assert_allclose(overall['Mean_Pinball_Normalized'], 0.042)
    assert overall['Coverage_68'] == 1.0
    assert overall['Coverage_95'] == 1.0
    assert (
        tmp_path
        / 'quantile_evaluation/metrics/uncertainty_metrics.csv'
    ).exists()
    assert (
        tmp_path
        / 'quantile_evaluation/plots/uncertainty_calibration.png'
    ).exists()
