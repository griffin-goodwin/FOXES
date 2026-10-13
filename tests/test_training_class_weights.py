from types import SimpleNamespace

import numpy as np

from training.train import (
    configure_class_weights, get_four_class_macro_weights,
    load_training_config,
)


def test_four_class_macro_weights_are_derived_from_training_targets(tmp_path):
    fluxes = [5e-8, 5e-7, 2e-6, 2e-5, 2e-4]
    samples = []
    for index, flux in enumerate(fluxes):
        timestamp = f"sample_{index}"
        samples.append(timestamp)
        np.save(tmp_path / f"{timestamp}.npy", flux)
    data_module = SimpleNamespace(
        train_ds=SimpleNamespace(samples=samples, sxr_dir=tmp_path)
    )

    counts, weights = get_four_class_macro_weights(data_module)

    assert counts == {
        "quiet": 2,
        "c_class": 1,
        "m_class": 1,
        "x_class": 1,
    }
    assert weights == {
        "quiet": 0.625,
        "c_class": 1.25,
        "m_class": 1.25,
        "x_class": 1.25,
    }

    _, sqrt_weights = get_four_class_macro_weights(
        data_module, exponent=0.5,
    )
    expected_sample_weight = (
        2 * sqrt_weights["quiet"]
        + sqrt_weights["c_class"]
        + sqrt_weights["m_class"]
        + sqrt_weights["x_class"]
    ) / 5
    assert np.isclose(expected_sample_weight, 1.0)
    assert np.isclose(
        sqrt_weights["x_class"] / sqrt_weights["quiet"],
        np.sqrt(2.0),
    )


def test_legacy_base_weight_request_uses_macro_counts_without_a_loader(tmp_path):
    fluxes = [5e-8, 5e-8, 2e-6, 2e-5]
    samples = [f"sample_{index}" for index in range(len(fluxes))]
    for timestamp, flux in zip(samples, fluxes):
        np.save(tmp_path / f"{timestamp}.npy", flux)
    data_module = SimpleNamespace(
        train_ds=SimpleNamespace(samples=samples, sxr_dir=tmp_path)
    )
    config = {'calculate_base_weights': True}

    weights, _ = configure_class_weights(config, data_module, 'deterministic')

    assert config['computed_four_class_train_counts'] == {
        'quiet': 2, 'c_class': 1, 'm_class': 1, 'x_class': 0,
    }
    assert np.isclose(weights['c_class'] / weights['quiet'], 2.0)
    assert np.isclose(weights['m_class'] / weights['quiet'], 2.0)
    assert weights['x_class'] > 0


def test_mean_loss_weights_are_passed_to_coupled_model(tmp_path):
    fluxes = [5e-8, 5e-8, 2e-6, 2e-5, 2e-4]
    samples = [f"sample_{index}" for index in range(len(fluxes))]
    for timestamp, flux in zip(samples, fluxes):
        np.save(tmp_path / f"{timestamp}.npy", flux)
    data_module = SimpleNamespace(
        train_ds=SimpleNamespace(samples=samples, sxr_dir=tmp_path)
    )
    config = {
        'mean_loss': {
            'class_weighting': 'inverse_frequency', 'huber_delta': 0.3,
            'bias_penalty_weight': 0.2,
        },
        'uncertainty': {'enabled': True},
    }

    base_weights, uncertainty = configure_class_weights(
        config, data_module, 'coupled_uncertainty',
    )

    assert base_weights is None
    assert uncertainty['class_weighting'] == 'inverse_frequency'
    assert uncertainty['huber_delta'] == 0.3
    assert uncertainty['bias_penalty_weight'] == 0.2
    assert uncertainty['class_weights'] == config['computed_four_class_train_weights']


def test_maintained_training_config_resolves_mean_loss():
    config = load_training_config('training/train_coupled_gaussian.yaml')
    assert config['mean_loss'] == {
        'class_weighting': 'inverse_frequency', 'huber_delta': 0.3,
        'bias_penalty_weight': 0.0,
    }
