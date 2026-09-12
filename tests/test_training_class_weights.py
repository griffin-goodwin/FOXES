from types import SimpleNamespace

import numpy as np

from training.train import get_four_class_macro_weights


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
