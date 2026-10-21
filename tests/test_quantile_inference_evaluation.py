"""Quantile checkpoints retain their heads from inference through evaluation."""

import sys

import numpy as np
import pandas as pd
import pytest
import torch
import pytorch_lightning as pl
import yaml
from matplotlib.axes import Axes
from torch.utils.data import TensorDataset

from forecasting import inference
from forecasting.evaluation import FOXESEvaluator
from forecasting.model import CoupledPinballViTLocal, normalize_sxr
from forecasting.quantiles import quantile_columns, quantile_label


LEVELS = (0.1, 0.5, 0.7, 0.85, 0.98)  # median deliberately not the middle head
ARCHITECTURE = dict(
    embed_dim=8, hidden_dim=16, num_channels=7, num_heads=2, num_layers=1,
    patch_size=1, num_patches=4, dropout=0.0, mask_mode='local',
    local_window=3, global_patch_adjustment=True,
)


def checkpoint(tmp_path):
    model = CoupledPinballViTLocal(ARCHITECTURE, np.array([-6., 2.], np.float32), quantiles=LEVELS)
    # Positive, crossed global heads make accidental sorting detectable.
    with torch.no_grad():
        model.model.mlp_head[-1].weight.zero_()
        model.model.mlp_head[-1].bias.copy_(torch.tensor([0., 3., 1., 4., 2.]))
    path = tmp_path / 'pinball.ckpt'
    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams),
                'pytorch-lightning_version': pl.__version__}, path)
    return model.eval(), path


@pytest.mark.parametrize('attention', [False, True])
def test_checkpoint_inference_saves_original_heads_and_maps(tmp_path, attention):
    model, path = checkpoint(tmp_path)
    loaded = inference.load_model_from_config({'data': {'checkpoint_path': str(path)}, 'device': 'cpu'})
    assert isinstance(loaded, CoupledPinballViTLocal)
    assert loaded.quantiles == LEVELS
    images = torch.zeros(3, 2, 2, 7)
    targets = torch.tensor([2e-7, 2e-6, 2e-4])
    config = {'num_workers': 0, 'pin_memory': False, 'flux_path': str(tmp_path / 'flux'),
              'patch_uncertainty_path': str(tmp_path / 'patch_uncertainty')}
    rows = list(inference.evaluate_model_on_dataset(
        loaded, TensorDataset(images, targets), batch_size=2, times=['a', 'b', 'c'],
        config_data=config, save_weights=attention, input_size=2, patch_size=1,
        save_flux=True, save_patch_uncertainty=True,
    ))
    with torch.no_grad():
        raw, normalized, patches = model(images)
    for i, (point, target, weights, flux, index, extra) in enumerate(rows):
        assert index == i
        assert point.item() == pytest.approx(raw[i, 1].item())
        assert target == pytest.approx(targets[i].item())
        assert (weights is not None) == attention
        assert extra['quantile_crossing'] == 1
        assert 'variance_normalized' not in extra
        np.testing.assert_allclose(flux, patches[i, :, 1].reshape(2, 2))
        for head, level in enumerate(LEVELS):
            raw_column, norm_column = quantile_columns(level)
            assert extra[raw_column] == pytest.approx(raw[i, head].item())
            assert extra[norm_column] == pytest.approx(normalized[i, head].item())
            saved = np.load(tmp_path / 'quantile_flux' / quantile_label(level) / f'{"abc"[i]}.npy')
            np.testing.assert_allclose(saved, patches[i, :, head].reshape(2, 2))
            assert saved.sum() == pytest.approx(extra[raw_column])
    assert not list((tmp_path / 'patch_uncertainty').glob('*.npy'))


@pytest.mark.parametrize('prediction_only', [False, True])
def test_inference_main_writes_quantile_csv(tmp_path, monkeypatch, prediction_only):
    _, path = checkpoint(tmp_path)
    dataset = TensorDataset(torch.zeros(2, 2, 2, 7), torch.tensor([2e-7, 3e-5]))
    dataset.samples = ['a', 'b']
    monkeypatch.setattr(inference, 'AIAGOESDataset', lambda **kwargs: dataset)
    config = {
        'data': {'checkpoint_path': str(path), 'aia_dir': 'unused', 'sxr_dir': 'unused'},
        'device': 'cpu', 'wavelengths': [94, 131, 171, 193, 211, 304, 335],
        'num_workers': 0, 'pin_memory': False, 'prediction_only': prediction_only,
        'model_params': {'input_size': 2, 'patch_size': 1, 'batch_size': 2, 'no_weights': True},
        'output_path': str(tmp_path / 'predictions.csv'),
    }
    config_path = tmp_path / 'inference.yaml'
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, 'argv', ['inference.py', '--config', str(config_path)])
    inference.main()
    frame = pd.read_csv(config['output_path'])
    np.testing.assert_allclose(frame.predictions, frame['prediction_q0.5'])
    assert frame.quantile_crossing.eq(1).all()
    if prediction_only:
        assert frame.groundtruth.isna().all()
        assert frame.groundtruth_normalized.isna().all()
    else:
        expected = normalize_sxr(dataset.tensors[1], torch.tensor([-6., 2.])).numpy()
        np.testing.assert_allclose(frame.groundtruth_normalized, expected)
        evaluator = FOXESEvaluator(config['output_path'], tmp_path / 'evaluation')
        evaluator.load_data()
        assert evaluator.quantiles == LEVELS


def quantile_csv(tmp_path):
    levels = (0.05, 0.16, 0.5, 0.84, 0.95)
    truth = np.array([2e-7, 4e-7, 3e-6, 4e-5, 5e-4])
    normalized = (np.log10(truth + 1e-8) + 6) / 2
    offsets = np.array([[-.5, -.2, .1, .2, .5], [-.4, -.1, -.1, .1, .4],
                        [-.3, -.2, .2, .1, .3], [.5, .2, -.1, -.2, -.5],
                        [-.5, -.2, .05, .2, .5]])
    norm = normalized[:, None] + offsets
    raw = 10 ** (norm * 2 - 6) - 1e-8
    frame = pd.DataFrame({'timestamp': list('abcde'), 'groundtruth': truth,
                          'predictions': raw[:, 2], 'prediction_normalized': norm[:, 2],
                          'groundtruth_normalized': normalized})
    for i, q in enumerate(levels):
        raw_name, norm_name = quantile_columns(q)
        frame[raw_name], frame[norm_name] = raw[:, i], norm[:, i]
    path = tmp_path / 'predictions.csv'
    frame.to_csv(path, index=False)
    return path, levels, offsets, raw


def test_evaluation_scores_class_sums_median_coverage_crossings_and_plots(tmp_path, monkeypatch):
    path, levels, offsets, raw = quantile_csv(tmp_path)
    evaluator = FOXESEvaluator(path, tmp_path / 'evaluation')
    evaluator.load_data()
    summary = evaluator.calculate_uncertainty_metrics().set_index('Group')
    pinball = np.maximum(-offsets * levels, -offsets * (np.array(levels) - 1)).mean(axis=1)
    expected_class_sum = pinball[:2].mean() + pinball[2:].sum()
    assert summary.loc['Overall', 'Pinball_Class_Sum_Normalized'] == pytest.approx(expected_class_sum)
    assert summary.loc['Overall', 'Pinball_Class_Macro_Normalized'] == pytest.approx(expected_class_sum / 4)
    assert summary.loc['Overall', 'Pinball_Normalized'] == pytest.approx(pinball.mean())
    assert summary.loc['Overall', 'Median_MSE_Normalized'] == pytest.approx(np.mean(offsets[:, 2] ** 2))
    assert summary.loc['Overall', 'Median_RMSE_Normalized'] == pytest.approx(np.sqrt(np.mean(offsets[:, 2] ** 2)))
    assert summary.loc['Overall', 'Crossing_Rate'] == pytest.approx(2 / 5)
    assert summary.loc['Quiet', 'Sample_Count'] == 2
    head = pd.read_csv(tmp_path / 'evaluation/metrics/quantile_metrics.csv')
    q50 = head[(head.Group == 'Overall') & (head.Quantile == .5)].iloc[0]
    assert q50.Coverage == pytest.approx(3 / 5)
    intervals = pd.read_csv(tmp_path / 'evaluation/metrics/quantile_interval_metrics.csv')
    outer = intervals[(intervals.Group == 'Overall') & (intervals.Lower_Quantile == .05)].iloc[0]
    assert outer.Nominal_Coverage == pytest.approx(.9)
    assert outer.Interval_Crossing_Rate == pytest.approx(1 / 5)
    assert outer.Coverage == pytest.approx(4 / 5)
    assert outer.Ordered_Interval_Count == 4
    monkeypatch.setattr(evaluator, '_plot_regression', lambda: None)
    point_metrics = evaluator.calculate_metrics()
    expected_mse = np.mean((np.log10(evaluator.y_true) - np.log10(raw[:, 2])) ** 2)
    assert point_metrics.iloc[0].MSE == pytest.approx(expected_mse)
    assert point_metrics.iloc[0].RMSE == pytest.approx(np.sqrt(expected_mse))
    for plot in ('quantile_predictions.png', 'quantile_calibration.png'):
        assert (tmp_path / 'evaluation/plots' / plot).stat().st_size > 0
    assert 'Gaussian_NLL_Normalized' not in summary.columns


def test_quantile_patch_panels_use_all_original_heads_one_scale(tmp_path, monkeypatch):
    path, levels, _, _ = quantile_csv(tmp_path)
    for i, q in enumerate(levels):
        directory = tmp_path / 'patches/quantile_flux' / quantile_label(q)
        directory.mkdir(parents=True)
        np.save(directory / 'a.npy', np.ones((2, 2)) * (i + 1) * 1e-7)
    evaluator = FOXESEvaluator(path, tmp_path / 'evaluation', inference_patch_dir=tmp_path / 'patches')
    evaluator.load_data()
    scales = []
    original = Axes.imshow
    def capture(ax, array, **kwargs):
        scales.append(kwargs['norm'])
        return original(ax, array, **kwargs)
    monkeypatch.setattr(Axes, 'imshow', capture)
    assert evaluator.plot_patch_maps() == 1
    assert len(scales) == 5
    assert all(scale.vmin == pytest.approx(1e-7) and scale.vmax == pytest.approx(5e-7) for scale in scales)
    assert evaluator.plot_patch_maps() == 0  # existing figures resume as before


@pytest.mark.parametrize('damage', ['missing_raw', 'missing_norm', 'median', 'variance'])
def test_evaluation_rejects_incomplete_or_ambiguous_quantile_csv(tmp_path, damage):
    path, _, _, _ = quantile_csv(tmp_path)
    frame = pd.read_csv(path)
    if damage == 'missing_raw':
        frame.drop(columns='prediction_q0.16', inplace=True)
    elif damage == 'missing_norm':
        frame.drop(columns='prediction_q0.16_normalized', inplace=True)
    elif damage == 'median':
        frame['predictions'] *= 2
    else:
        frame['variance_normalized'] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        FOXESEvaluator(path, tmp_path / 'evaluation').load_data()


def test_public_predictor_handles_quantile_checkpoint(tmp_path):
    import helio_foxes
    _, path = checkpoint(tmp_path)
    predictor = helio_foxes.load_model(path, device='cpu')
    result = helio_foxes.predict(np.zeros((7, 2, 2), np.float32), model=predictor)
    assert tuple(result.quantiles) == LEVELS
    assert result.flux == result.quantiles[.5]
    assert result.patch_uncertainty_map is None
    assert result.intervals['88'] == (result.quantiles[.1], result.quantiles[.98])
    for q in LEVELS:
        assert result.quantile_flux_maps[q].sum() == pytest.approx(result.quantiles[q])
    np.testing.assert_array_equal(result.flux_map, result.quantile_flux_maps[.5])
