import numpy as np
import pandas as pd

from analysis.sxr_response_calibration import (
    AIA_WAVELENGTHS,
    fit_empirical_weights,
    fit_global_scale,
    fit_integrated_weights,
    fit_response_weights,
    normalized_weights,
    patch_proxy_map,
    proxy_values,
)


def test_response_fit_recovers_nonnegative_mixture():
    logt = np.linspace(6.0, 8.0, 41)
    aia = np.column_stack(
        [
            np.exp(-0.5 * ((logt - 6.4) / 0.25) ** 2),
            np.exp(-0.5 * ((logt - 7.1) / 0.30) ** 2),
        ]
    )
    expected = np.array([2.0, 0.5])
    goes = aia @ expected

    weights, curves = fit_response_weights(
        [94, 131], logt, aia, logt, goes, goes_weight_power=0.0
    )

    np.testing.assert_allclose(weights.to_numpy(), expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(curves["aia_weighted_sum"], goes, rtol=1e-8)


def test_empirical_weights_and_global_scale_are_nonnegative():
    rng = np.random.default_rng(7)
    features = rng.uniform(0.1, 10.0, size=(100, 2))
    target = features @ np.array([3e-8, 7e-8])
    frame = pd.DataFrame(
        {
            "aia_94_mean_dn_s": features[:, 0],
            "aia_131_mean_dn_s": features[:, 1],
            "goes_xrsb_w_m2": target,
        }
    )

    weights = fit_empirical_weights(frame, channels=[94, 131])
    scale = fit_global_scale(frame, weights)

    assert (weights >= 0).all()
    np.testing.assert_allclose(proxy_values(frame, weights) * scale, target, rtol=1e-10)


def test_integrated_l1_regression_recovers_exact_positive_mapping():
    rng = np.random.default_rng(19)
    features = rng.uniform(1.0, 20.0, size=(40, 2))
    target = features @ np.array([2e-8, 5e-8])
    frame = pd.DataFrame(
        {
            "aia_94_sum_dn_s": features[:, 0],
            "aia_131_sum_dn_s": features[:, 1],
            "goes_xrsb_w_m2": target,
        }
    )

    weights = fit_integrated_weights(frame, channels=[94, 131], loss="l1")

    assert (weights >= 0).all()
    np.testing.assert_allclose(proxy_values(frame, weights), target, rtol=1e-9)


def test_patch_proxy_map_is_nonnegative_and_normalized():
    image = np.zeros((len(AIA_WAVELENGTHS), 8, 8), dtype=np.float32)
    image[AIA_WAVELENGTHS.index(94), :4, :4] = 2.0
    image[AIA_WAVELENGTHS.index(131), 4:, 4:] = 1.0

    result = patch_proxy_map(image, {94: 1.0, 131: 2.0}, patch_size=4)

    assert result.shape == (2, 2)
    assert np.all(result >= 0)
    np.testing.assert_allclose(result.sum(), 1.0)
    np.testing.assert_allclose(normalized_weights({94: 1.0, 131: 3.0}).sum(), 1.0)
