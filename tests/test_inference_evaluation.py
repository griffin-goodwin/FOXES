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
    uncertainty_from_prediction,
)
from forecasting.model import ViTLocal
from forecasting.uncertainty_model import GaussianNLLViTLocal


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
            'mask_mode': 'local',
        },
        sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
        uncertainty_kwargs=uncertainty_kwargs,
    )


def test_auto_checkpoint_detection_has_only_two_model_paths():
    deterministic = {
        'state_dict': {'model.mlp_head.1.weight': torch.zeros(1, 8)}
    }
    uncertain = {
        'state_dict': {
            'model.patch_uncertainty_head.1.weight': torch.zeros(1, 8),
        },
        'hyper_parameters': {'mean_parameterization': 'multiplier'},
    }

    assert _auto_checkpoint_model_class(deterministic) is ViTLocal
    assert _auto_checkpoint_model_class(uncertain) is GaussianNLLViTLocal


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


def test_inference_parses_uncertainty_outputs():
    model = make_uncertainty_model()
    dataset = TensorDataset(
        torch.zeros(2, 2, 2, 7), torch.tensor([1e-7, 1e-5])
    )

    results = list(evaluate_model_on_dataset(
        model, dataset, batch_size=2, times=['sample-0', 'sample-1'],
        config_data={'num_workers': 0, 'pin_memory': False, 'use_amp': False},
        save_weights=False, input_size=2, patch_size=1, save_flux=False,
        save_patch_uncertainty=False,
    ))

    assert len(results) == 2
    assert all(len(result) == 6 for result in results)
    assert all(result[-1]['variance_normalized'] > 0 for result in results)
    assert all(result[2] is None and result[3] is None for result in results)


def test_uncertainty_inference_saves_patch_standard_deviation(tmp_path):
    model = make_uncertainty_model({
        'initial_relative_std': 0.75,
        'min_log10_relative_std': -4.0,
        'max_log10_relative_std': 4.0,
    })
    dataset = TensorDataset(
        torch.zeros(1, 2, 2, 7), torch.tensor([1e-6])
    )
    map_dir = tmp_path / 'patch_uncertainty'

    results = list(evaluate_model_on_dataset(
        model, dataset, batch_size=1, times=['sample-0'],
        config_data={
            'num_workers': 0,
            'pin_memory': False,
            'use_amp': False,
            'patch_uncertainty_path': str(map_dir),
        },
        save_weights=False, input_size=2, patch_size=1, save_flux=False,
        save_patch_uncertainty=True,
    ))

    assert results[0][-1]['variance_normalized'] > 0
    saved = np.load(map_dir / 'sample-0.npy')
    assert saved.shape == (2, 2)
    assert np.all(saved > 0)


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
    csv_path = tmp_path / 'predictions.csv'
    pd.DataFrame({
        'timestamp': [f'sample-{index}' for index in range(5)],
        'predictions': raw,
        'groundtruth': raw,
        'prediction_normalized': normalized,
        'groundtruth_normalized': normalized,
        'variance_normalized': np.full(5, 0.25),
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
    np.testing.assert_allclose(
        metrics.iloc[0]['Gaussian_NLL_Normalized'],
        0.5 * np.log(2.0 * np.pi * 0.25),
    )
    assert metrics.iloc[0]['Coverage_68'] == 1.0
    assert metrics.iloc[0]['Coverage_95'] == 1.0
