"""Retain the AIA image nearest a target time and reuse HARP overlays beside light curves."""

from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pytest

import helio_foxes as foxes


class SnapshotModel:
    def predict(self, observation):
        # Reverse ID/flux rank order to ensure colors are keyed by identity.
        return foxes.Prediction(
            metadata=observation.metadata, flux=10e-6,
            flux_map=np.array([[1, 4], [2, 3]]) * 1e-6,
            harp_regions=observation.harp_regions,
        )


class UncertainSnapshotModel(SnapshotModel):
    def predict(self, observation):
        result = super().predict(observation)
        return foxes.Prediction(
            metadata=result.metadata, flux=result.flux, flux_map=result.flux_map,
            harp_regions=result.harp_regions, patch_uncertainty_map=np.full((2, 2), 0.2e-6),
        )


def observations(minutes=(0, 4, 10)):
    start = foxes.parse_timestamp("2023-08-01")
    for minute in minutes:
        image = np.arange(32 * 32).reshape(32, 32) + minute
        stack = np.stack([image + channel * 1000 for channel in range(7)])
        yield foxes.Observation.from_array(
            stack, timestamp=start + timedelta(minutes=minute),
            harp_regions=[
                {"HARPNUM": 100, "top_pct": 15, "left_pct": 10, "width_pct": 25, "height_pct": 25},
                {"HARPNUM": 200, "top_pct": 55, "left_pct": 60, "width_pct": 25, "height_pct": 25},
            ],
        )


def make_series(**kwargs):
    return foxes.Predictor.predict_series(SnapshotModel(), observations(), **kwargs)


def test_target_retains_one_channel_without_temporary_files(monkeypatch):
    monkeypatch.setattr("helio_foxes.api.TemporaryDirectory", lambda **kw: pytest.fail("No spool needed"))
    series = make_series(snapshot_time="2023-08-01 00:05")
    snapshot = series.snapshot
    assert snapshot.metadata.timestamp.minute == 4
    assert snapshot.metadata.wavelengths == (193,)
    assert snapshot.data.shape == (1, 32, 32)
    np.testing.assert_array_equal(snapshot.data[0], list(observations())[1].data[3])


def test_target_ties_choose_earlier_and_channel_can_be_changed():
    series = foxes.Predictor.predict_series(
        SnapshotModel(), observations((10, 4, 0)),
        snapshot_time="2023-08-01 00:07", snapshot_wavelength=171,
    )
    assert series.snapshot.metadata.timestamp.minute == 4
    assert series.snapshot.metadata.wavelengths == (171,)
    with pytest.raises(ValueError, match="absent"):
        make_series(snapshot_time="2023-08-01", snapshot_wavelength=999)


def test_no_target_retains_no_image(monkeypatch):
    monkeypatch.setattr("helio_foxes.api.TemporaryDirectory", lambda **kw: pytest.fail("No spool needed"))
    assert make_series().snapshot is None


def test_empty_series_has_no_snapshot():
    series = foxes.Predictor.predict_series(
        SnapshotModel(), iter(()), snapshot_time="2023-08-01",
    )
    assert series.predictions == ()
    assert series.snapshot is None


@pytest.mark.parametrize("target, expected_minute", [
    ("2023-07-31", 0), ("2023-08-02", 10), ("2023-08-01 00:10", 10),
])
def test_target_selects_nearest_available_observation(target, expected_minute):
    assert make_series(snapshot_time=target).snapshot.metadata.timestamp.minute == expected_minute


def test_snapshot_reuses_overlay_and_matches_colors(monkeypatch):
    series = make_series(snapshot_time="2023-08-01 00:05")
    original = foxes.api.plot_flux_map
    calls = []

    def plot(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr("helio_foxes.api.plot_flux_map", plot)
    ax = series.plot(show_snapshot=True)
    assert len(calls) == 1
    snapshot_ax = ax.figure.axes[1]
    assert len(ax.figure.axes) == 4  # Curves, standard flux map, AIA, flux colorbar.
    assert snapshot_ax.get_position().x0 > ax.get_position().x0
    assert snapshot_ax.images[0].origin == "lower"
    from matplotlib.colors import LogNorm
    assert isinstance(snapshot_ax.images[0].norm, LogNorm)
    assert "aia_image" not in calls[0]
    np.testing.assert_array_equal(snapshot_ax.images[0].get_array(), series.predictions[1].flux_map)
    aia_ax = ax.figure.axes[2]
    np.testing.assert_array_equal(aia_ax.images[0].get_array(), series.snapshot.data[0])
    assert not aia_ax.patches and not aia_ax.texts
    assert aia_ax.get_position().x0 > snapshot_ax.get_position().x0
    assert "AIA 193" in aia_ax.get_title()
    assert "00:04:00" in aia_ax.get_title()
    for label, rectangle in zip(snapshot_ax.texts, snapshot_ax.patches):
        harpnum = int(label.get_text().splitlines()[0][1:])
        curve = next(line for line in ax.lines if line.get_label() == f"HARP {harpnum}")
        np.testing.assert_allclose(rectangle.get_edgecolor(), to_rgba(curve.get_color()))
        np.testing.assert_allclose(to_rgba(label.get_color()), to_rgba(curve.get_color()))
    assert len(snapshot_ax.patches) == 2
    assert any(line.get_label() == "AIA snapshot" for line in ax.lines)
    plt.close(ax.figure)


def test_toggle_and_selected_regions_keep_colors_stable():
    series = make_series(snapshot_time="2023-08-01 00:05")
    plain_ax = series.plot()
    assert len(plain_ax.figure.axes) == 1
    color = next(line.get_color() for line in plain_ax.lines if line.get_label() == "HARP 200")
    ax = series.plot(show_snapshot=True, harpnums=[200])
    assert len(ax.figure.axes[1].patches) == 1
    assert ax.lines[0].get_color() == color
    assert ax.figure.axes[1].texts[0].get_text().startswith("#200")
    plt.close(plain_ax.figure)
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="No AIA snapshot"):
        make_series().plot(show_snapshot=True)


def test_existing_axes_and_no_snapshot_harps_do_not_query(monkeypatch):
    series = make_series(snapshot_time="2023-08-01 00:05")
    monkeypatch.setattr(foxes.HARPProcessor, "fetch_date", lambda *args: pytest.fail("No query"))
    fig, (left, right, aia) = plt.subplots(1, 3)
    assert series.plot(ax=left, snapshot_ax=right, aia_ax=aia, show_snapshot=True, harpnums=[]) is left
    assert not right.patches
    with pytest.raises(ValueError, match="snapshot_ax"):
        series.plot(ax=left, show_snapshot=True)
    plt.close(fig)


def test_failed_prediction_raises():
    class FailingModel(SnapshotModel):
        def predict(self, observation):
            if observation.metadata.timestamp.minute == 4:
                raise RuntimeError("model failure")
            return super().predict(observation)

    with pytest.raises(RuntimeError, match="model failure"):
        foxes.Predictor.predict_series(
            FailingModel(), observations(), snapshot_time="2023-08-01 00:05",
        )


def test_snapshot_is_independent_of_source_and_input_order():
    source = list(observations((10, 4, 0)))
    series = foxes.Predictor.predict_series(SnapshotModel(), iter(source), snapshot_time="2023-08-01 00:05")
    expected = source[1].data[3].copy()
    source[1].data[:] = -99
    assert series.snapshot.metadata.timestamp.minute == 4
    np.testing.assert_array_equal(series.snapshot.data[0], expected)
    assert [p.metadata.timestamp.minute for p in series.predictions] == [0, 4, 10]


def test_uncertainties_on_bands_and_snapshot_labels_and_toggle():
    series = foxes.Predictor.predict_series(UncertainSnapshotModel(), observations(), snapshot_time="2023-08-01 00:05")
    ax = series.plot(show_snapshot=True)
    assert len(ax.collections) == 2
    assert r"\sigma" in ax.get_title()
    assert all("±" in label.get_text() for label in ax.figure.axes[1].texts)
    curves = series.harp_light_curves()
    for polygon, harpnum in zip(ax.collections, curves):
        sigma = curves[harpnum].flux_sum_uncertainty.iloc[0]
        flux = curves[harpnum].flux_sum.iloc[0]
        ys = np.concatenate([path.vertices[:, 1] for path in polygon.get_paths()])
        assert min(ys) == pytest.approx(flux - sigma)
        assert max(ys) == pytest.approx(flux + sigma)
    plt.close(ax.figure)
    ax = series.plot(show_snapshot=True, show_uncertainty=False)
    assert not ax.collections
    assert all("±" not in label.get_text() for label in ax.figure.axes[1].texts)
    plt.close(ax.figure)


def test_deterministic_model_never_fabricates_uncertainty():
    series = make_series(snapshot_time="2023-08-01 00:05")
    ax = series.plot(show_snapshot=True, show_uncertainty=True)
    assert not ax.collections
    assert all("±" not in label.get_text() for label in ax.figure.axes[1].texts)
    assert r"\sigma" not in ax.get_title()
    plt.close(ax.figure)


def test_mixed_models_do_not_bridge_unknown_uncertainties():
    images = list(observations())
    uncertain = UncertainSnapshotModel()
    deterministic = SnapshotModel()
    series = foxes.PredictionSeries((uncertain.predict(images[0]), deterministic.predict(images[1]),
                                     uncertain.predict(images[2])))
    ax = series.plot()
    for collection in ax.collections:
        for path in collection.get_paths():
            # Isolated known uncertainties must not shade across the middle sample.
            assert np.ptp(path.vertices[:, 0]) == 0
    plt.close(ax.figure)
