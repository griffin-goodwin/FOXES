"""Raw-flux intervals are exported before plotting, with gaps preserved."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import helio_foxes as foxes
from helio_foxes.api import _regional_flux_intervals


BOUND_COLUMNS = ["lower_68", "upper_68", "lower_95", "upper_95"]


def test_regional_intervals_preserve_median_variance_and_nominal_levels():
    flux = np.array([2e-7, 3e-6, 2e-4])
    variance = np.array([1e-14, 4e-12, 4e-8])
    bounds = _regional_flux_intervals(flux, variance)
    median = flux
    lower, upper = bounds["lower_68"], bounds["upper_68"]
    np.testing.assert_allclose(lower * upper, median**2, rtol=1e-12)
    log_variance = np.log(upper / median)**2
    recovered_variance = median**2 * np.exp(log_variance) * np.expm1(log_variance)
    np.testing.assert_allclose(recovered_variance, variance, rtol=1e-12)
    np.testing.assert_allclose(
        np.log(bounds["upper_95"] / median),
        1.96 * np.log(upper / median), rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.log(bounds["lower_95"] / median),
        1.96 * np.log(lower / median), rtol=1e-12,
    )


def test_raw_intervals_preserve_gaps_and_zero_variance():
    with np.errstate(all="raise"):
        bounds = _regional_flux_intervals(
            [1e-6, 0, np.nan, 1e-6, -1e-6, 1e-6],
            [0, 1e-12, 1e-12, np.nan, 1e-12, -1e-12],
        )
    for values in bounds.values():
        assert values[0] == 1e-6
        assert np.isnan(values[1:]).all()


def make_series():
    predictions = []
    for minute in range(3):
        predictions.append(foxes.Prediction(
            metadata=foxes.api.ObservationMetadata(f"2023-08-01 00:0{minute}", (7, 1, 2)),
            total_flux=4e-6, flux_map=np.array([[1e-6, 3e-6]]),
            flux_std_map=np.array([[1e-6, 2e-6]]) if minute == 0 else None,
            prediction_intervals={"68": (0.5e-6, 9e-6), "95": (0.1e-6, 12e-6)} if minute == 0 else None,
            harp_contributions=[{
                "HARPNUM": 100, "top_pct": 0, "left_pct": 0,
                "width_pct": 50, "height_pct": 100,
            }] if minute != 2 else [],
        ))
    return foxes.PredictionTimeSeries(tuple(predictions))


def test_light_curve_tables_export_raw_bounds_and_preserve_missing_values():
    series = make_series()
    curve = series.harp_light_curves()[100]
    assert list(curve.columns) == ["total_flux", "flux_std", *BOUND_COLUMNS]
    assert curve.total_flux.iloc[0] == 1e-6
    assert curve.flux_std.iloc[0] == 1e-6
    assert np.isfinite(curve.iloc[0][BOUND_COLUMNS]).all()
    assert curve.iloc[1][BOUND_COLUMNS].isna().all()  # Deterministic prediction.
    assert curve.iloc[2].isna().all()  # Region absent.
    assert str(curve.index.tz) == "UTC"

    totals = series.full_disk_light_curve()
    assert list(totals.columns) == ["total_flux", "ground_truth", *BOUND_COLUMNS]
    for level in ("68", "95"):
        assert tuple(totals.iloc[0][[f"lower_{level}", f"upper_{level}"]]) == series.predictions[0].prediction_intervals[level]
    assert totals.iloc[1:][BOUND_COLUMNS].isna().all().all()


def test_regional_plot_uses_table_bounds_without_recalculating(monkeypatch):
    series = make_series()
    curve = pd.DataFrame({
        "total_flux": [1e-6, 1e-6, 1e-6], "flux_std": [np.nan] * 3,
        "lower_68": [2e-7, 3e-7, 4e-7], "upper_68": [2e-6, 3e-6, 4e-6],
    }, index=series.full_disk_light_curve().index)
    monkeypatch.setattr(foxes.PredictionTimeSeries, "harp_light_curves", lambda self: {100: curve})
    monkeypatch.setattr(
        "helio_foxes.api._regional_flux_intervals",
        lambda *args, **kwargs: pytest.fail("Plotting must read precomputed bounds"),
    )
    ax = series.plot_light_curves()
    assert len(ax.collections) == 1
    vertices = ax.collections[0].get_paths()[0].vertices[:, 1]
    assert vertices.min() == 2e-7
    assert vertices.max() == 4e-6
    plt.close(ax.figure)


def test_full_disk_plot_uses_saved_intervals_and_respects_toggle():
    series = make_series()
    ax = series.plot_light_curves(harpnums=[], show_full_disk=True)
    assert len(ax.collections) == 1
    vertices = ax.collections[0].get_paths()[0].vertices
    assert vertices[:, 1].min() == 0.5e-6
    assert vertices[:, 1].max() == 9e-6
    assert np.ptp(vertices[:, 0]) == 0  # Unknown intervals leave gaps.
    plt.close(ax.figure)
    ax = series.plot_light_curves(harpnums=[], show_full_disk=True, show_uncertainty=False)
    assert not ax.collections
    plt.close(ax.figure)
