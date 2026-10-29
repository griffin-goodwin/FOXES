"""Log-safe curves, series-wide peak filtering, and unobtrusive legends."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import helio_foxes as foxes


def make_series():
    predictions = []
    for minute, values in enumerate(([0, 2e-7], [1e-6, 2e-7], [0, 2e-7])):
        metadata = foxes.ObservationMetadata(f"2023-08-01 00:0{minute}", (7, 2, 2))
        predictions.append(foxes.Prediction(
            metadata=metadata, flux=[0, 1e-6, 0][minute], flux_map=np.array([values]),
            patch_uncertainty_map=np.array([[2e-7, 3e-7]]),
            sxr={"xrsb_flux": [0, -1e-6, 1e-6][minute]},
            harp_regions=[
                {"HARPNUM": 100, "top_pct": 0, "left_pct": 0, "width_pct": 50, "height_pct": 100},
                {"HARPNUM": 200, "top_pct": 0, "left_pct": 50, "width_pct": 50, "height_pct": 100},
            ],
        ))
    snapshot = foxes.Observation.from_array(np.ones((1, 2, 2)), timestamp="2023-08-01 00:01", wavelengths=(193,))
    return foxes.PredictionSeries(tuple(predictions), snapshot=snapshot)


def test_log_lines_mask_nonpositive_values_and_preserve_data():
    series = make_series()
    ax = series.plot(include_full_disk=True, include_sxr=True)
    assert ax.get_yscale() == "log"
    assert np.diff(ax.get_xlim())[0] == pytest.approx(2 / (24 * 60))
    lines = {line.get_label(): line for line in ax.lines}
    for name in ("HARP 100", "FOXES full disk"):
        np.testing.assert_allclose(lines[name].get_ydata(), [np.nan, 1e-6, np.nan], equal_nan=True)
    np.testing.assert_allclose(lines["Observed GOES XRS-B"].get_ydata(), [np.nan, np.nan, 1e-6], equal_nan=True)
    assert series.harp_light_curves()[100].flux_sum.iloc[0] == 0
    for collection in ax.collections:
        for path in collection.get_paths():
            assert np.all(path.vertices[:, 1] > 0)
    # HARP 200's lower uncertainty bound is negative at every time: no band.
    assert len(ax.collections) == 1
    plt.close(ax.figure)


def test_peak_threshold_filters_curves_and_snapshot_without_changing_colors():
    series = make_series()
    original = series.plot(show_uncertainty=False)
    color = original.lines[0].get_color()
    ax = series.plot(show_snapshot=True, min_peak_flux=5e-7)
    assert [line.get_label() for line in ax.lines] == ["HARP 100", "AIA snapshot"]
    assert ax.lines[0].get_color() == color
    assert len(ax.figure.axes[1].patches) == 1
    assert ax.figure.axes[1].texts[0].get_text().startswith("#100")
    assert set(series.harp_light_curves()) == {100, 200}
    assert [text.get_text() for text in ax.get_legend().texts] == ["HARP regions (1); IDs on flux map", "AIA snapshot"]
    assert ax.get_legend().get_bbox_to_anchor().y0 < ax.get_window_extent().y0
    plt.close(original.figure)
    plt.close(ax.figure)


def test_peak_threshold_is_over_entire_series_and_can_hide_all():
    series = make_series()
    # The last snapshot has zero HARP-100 flux, but its earlier peak qualifies.
    last_snapshot = foxes.Observation.from_array(np.ones((1, 2, 2)), timestamp="2023-08-01 00:02", wavelengths=(193,))
    series = foxes.PredictionSeries(series.predictions, snapshot=last_snapshot)
    ax = series.plot(show_snapshot=True, min_peak_flux=1e-6)
    assert any(line.get_label() == "HARP 100" for line in ax.lines)
    assert not ax.figure.axes[1].patches  # Zero instantaneous flux is not labeled.
    plt.close(ax.figure)
    ax = series.plot(min_peak_flux=2e-6)
    assert not ax.lines and ax.get_legend() is None
    plt.close(ax.figure)


def test_legend_overrides():
    series = make_series()
    ax = series.plot(show_snapshot=True, legend="full")
    assert "HARP 100" in [text.get_text() for text in ax.get_legend().texts]
    plt.close(ax.figure)
    ax = series.plot(legend="none")
    assert ax.get_legend() is None
    plt.close(ax.figure)


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, None, "invalid"])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="min_peak_flux"):
        make_series().plot(min_peak_flux=threshold)
