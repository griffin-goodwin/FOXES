"""Checks for the importable UI-facing API."""

import numpy as np
from datetime import datetime, timezone
import sys
import types
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

import foxes
from forecasting.model import CoupledGaussianViTLocal, ViTLocal


def test_aia_processor_uses_temporary_files_and_returns_stack(monkeypatch):
    staged = []

    class Downloader:
        def __init__(self, base_path, **kwargs):
            self.base_path = base_path
            self.wavelengths = kwargs["wavelengths"]
            staged.append(base_path)

        def downloadDate(self, date):
            from pathlib import Path
            for wavelength in self.wavelengths:
                path = Path(self.base_path) / wavelength
                path.mkdir()
                (path / f"{date.isoformat(timespec='seconds')}.fits").touch()

    class Stacker:
        def __init__(self, data, wavelengths, resolution, **kwargs):
            assert all(len(files) == 1 for files in data)
            assert wavelengths == foxes.AIAProcessor.WAVELENGTHS
            self.resolution = resolution

        def __getitem__(self, index):
            assert index == 0
            return np.ones((7, self.resolution, self.resolution), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "download.download_sdo", types.SimpleNamespace(SDODownloader=Downloader))
    monkeypatch.setitem(sys.modules, "data.convert_aia", types.SimpleNamespace(AIAStackDataset=Stacker))
    stack = foxes.AIAProcessor(resolution=2).fetch_date("2023-08-01T00:00:00Z")
    assert stack.shape == (7, 2, 2)
    assert stack.dtype == np.float32
    assert not __import__("os").path.exists(staged[0])
    observation = foxes.ObservationProcessor(resolution=2, include_harps=False, include_sxr=False).fetch_observation(
        "2023-07-31T20:00:00-04:00"
    )
    assert isinstance(observation, foxes.Observation)
    assert observation.data.shape == (7, 2, 2)
    assert observation.metadata.timestamp == datetime(2023, 8, 1, tzinfo=timezone.utc)
    assert observation.metadata.wavelengths == foxes.AIAProcessor.WAVELENGTHS
    assert observation.metadata.image_shape == observation.data.shape
    assert observation.metadata.padding_factor == 0.1
    assert observation.harp_regions is None
    assert not __import__("os").path.exists(staged[1])


def test_aia_processor_rejects_partial_download_and_cleans_up(monkeypatch):
    staged = []

    class Downloader:
        def __init__(self, base_path, **kwargs):
            staged.append(base_path)

        def downloadDate(self, date):
            pass

    monkeypatch.setitem(sys.modules, "download.download_sdo", types.SimpleNamespace(SDODownloader=Downloader))
    monkeypatch.setitem(sys.modules, "data.convert_aia", types.SimpleNamespace(AIAStackDataset=object))
    with pytest.raises(FileNotFoundError, match="wavelengths"):
        foxes.AIAProcessor(resolution=2).fetch_date("2023-08-01T00:00:00")
    assert not __import__("os").path.exists(staged[0])


@pytest.mark.parametrize("model_class", [ViTLocal, CoupledGaussianViTLocal])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("input_kind", ["array", "observation", "timestamped_array"])
def test_predict_matches_model_outputs(model_class, wrapped, input_kind, monkeypatch, tmp_path):
    kwargs = dict(
        embed_dim=8, hidden_dim=16, num_channels=7,
        num_heads=2, num_layers=1, patch_size=1,
        num_patches=4, dropout=0.0, mask_mode="local", local_window=3,
    )
    if model_class is CoupledGaussianViTLocal:
        kwargs["global_patch_adjustment"] = True
    model = model_class(
        model_kwargs=kwargs, sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
        **({"uncertainty_kwargs": {"enabled": True}}
           if model_class is CoupledGaussianViTLocal else {}),
    ).eval()
    loaded_model = torch.nn.DataParallel(model) if wrapped else model
    monkeypatch.setattr(
        "forecasting.inference.load_model_from_config",
        lambda config: loaded_model,
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    predictor = foxes.load_model(checkpoint, device="cpu")
    assert predictor.model_kwargs == kwargs

    image = np.zeros((7, 2, 2), dtype=np.float32)
    timestamp = "2023-08-01T00:00:00Z"
    metadata = None
    regions = [{"HARPNUM": 1234, "top_pct": 0, "left_pct": 0,
                "width_pct": 50, "height_pct": 50}]
    sxr = {"xrsb_flux": 1e-6, "xrsa_flux": 1e-8, "satellite": "GOES-18"}
    if input_kind == "observation":
        metadata = foxes.ObservationMetadata(
            timestamp=timestamp, image_shape=image.shape, padding_factor=0.2,
        )
        image = foxes.Observation(data=image, metadata=metadata, harp_regions=regions, sxr=sxr)
    result = foxes.predict(
        image, model=predictor,
        **({"timestamp": timestamp} if input_kind == "timestamped_array" else {}),
    )
    if input_kind == "observation":
        assert result.metadata is metadata
        assert result.sxr is sxr
        assert result.harp_regions is not regions
        assert result.harp_regions[0]["harpnum"] == 1234
        assert result.harp_regions[0]["flux_sum"] == pytest.approx(float(max(result.flux_map[1, 0], 0)))
        assert result.harp_regions[0]["top_pct"] == regions[0]["top_pct"]
        assert "flux_sum" not in regions[0]
        monkeypatch.setattr(
            foxes.HARPProcessor, "fetch_date",
            lambda *args, **kwargs: pytest.fail("Plot must reuse observation HARPs"),
        )
        with pytest.raises(ValueError, match="timestamp conflicts"):
            foxes.predict(image, model=predictor, timestamp="2023-08-02T00:00:00Z")
        with pytest.raises(ValueError, match="Observation has no AIA data"):
            foxes.predict(foxes.Observation(data=None, metadata=metadata), model=predictor)
    elif input_kind == "timestamped_array":
        assert result.metadata.timestamp == datetime(2023, 8, 1, tzinfo=timezone.utc)
        assert result.metadata.image_shape == image.shape
        assert result.metadata.wavelengths == foxes.AIAProcessor.WAVELENGTHS
    else:
        assert result.metadata is None
    if input_kind != "observation":
        assert result.harp_regions is None
        assert result.sxr is None
    assert result.flux_map.shape == (2, 2)
    assert np.isclose(result.flux_map.sum(), result.flux, rtol=1e-5)
    if model_class is CoupledGaussianViTLocal:
        assert set(result.intervals) == {"68", "95"}
        assert result.patch_uncertainty_map.shape == (2, 2)
        if input_kind == "observation":
            assert result.harp_regions[0]["flux_sum_uncertainty"] == pytest.approx(
                float(result.patch_uncertainty_map[1, 0]),
            )
    else:
        assert result.intervals is None
        assert result.patch_uncertainty_map is None
        if input_kind == "observation":
            assert result.harp_regions[0]["flux_sum_uncertainty"] is None

    ax = foxes.plot_flux_map(result, show_harp_boxes=input_kind == "observation", min_harp_flux=0)
    assert ax.images[0].origin == "lower"
    predicted_title = f"Predicted FOXES: {result.flux:.2e}"
    if result.patch_uncertainty_map is not None:
        full_disk_uncertainty = np.sqrt(np.sum(np.square(result.patch_uncertainty_map)))
        predicted_title += f" ± {full_disk_uncertainty:.2e}"
    assert f"{predicted_title} W/m²" in ax.get_title()
    if input_kind == "observation":
        assert "Observed SXR-B: 1.00e-06 W/m²" in ax.get_title()
    if input_kind == "observation":
        assert len(ax.patches) == 1
        expected_label = f'#1234\n{result.harp_regions[0]["flux_sum"]:.2e}'
        if result.patch_uncertainty_map is not None:
            expected_label += f'\n± {result.harp_regions[0]["flux_sum_uncertainty"]:.2e}'
        assert ax.texts[0].get_text() == expected_label
    plt.close(ax.figure)

    with pytest.raises(ValueError, match="shape"):
        foxes.predict(np.zeros((2, 2, 7)), model=predictor)


@pytest.mark.parametrize("plotter", [foxes.plot_flux_map, foxes.plot_uncertainty_map])
def test_plot_map_color_limits(plotter):
    values = np.array([[1e-9, 1e-8], [0, 1e-7]])
    ax = plotter(values)
    assert ax.images[0].norm.vmin == 1e-9
    assert ax.images[0].norm.vmax == 1e-7
    plt.close(ax.figure)

    ax = plotter(values, vmin=1e-10, vmax=1e-6)
    assert ax.images[0].norm.vmin == 1e-10
    assert ax.images[0].norm.vmax == 1e-6
    plt.close(ax.figure)

    with pytest.raises(ValueError, match="vmin and vmax"):
        plotter(values, vmin=0)


def test_plot_flux_map_harp_toggle_draws_boxes_and_patch_sums():
    flux_map = np.array([[1e-9, 2e-9], [3e-9, 4e-9]])
    regions = [{
        "HARPNUM": 1234, "top_pct": 0, "left_pct": 0,
        "width_pct": 50, "height_pct": 50,
    }]

    ax = foxes.plot_flux_map(flux_map, harp_regions=regions)
    assert not ax.patches and not ax.texts
    assert not any(panel.tables for panel in ax.figure.axes)
    plt.close(ax.figure)

    ax = foxes.plot_flux_map(
        flux_map, show_harp_boxes=True, harp_regions=regions, min_harp_flux=0,
    )
    assert len(ax.patches) == 1
    assert np.allclose(ax.patches[0].get_bbox().bounds, (-0.5, 0.5, 1, 1))
    assert ax.texts[0].get_text() == "#1234\n3.00e-09"
    plt.close(ax.figure)

    with pytest.raises(ValueError, match="timestamp is required"):
        foxes.plot_flux_map(flux_map, show_harp_boxes=True)


def test_plot_flux_map_queries_harps_when_regions_omitted(monkeypatch):
    calls = []
    region = {
        "HARPNUM": 1234, "top_pct": 0, "left_pct": 0,
        "width_pct": 50, "height_pct": 50,
    }

    def fetch(self, timestamp):
        calls.append((self.email, timestamp))
        return [region]

    monkeypatch.setattr(foxes.HARPProcessor, "fetch_date", fetch)
    flux_map = np.array([[1e-9, 2e-9], [3e-9, 4e-9]])
    ax = foxes.plot_flux_map(
        flux_map, show_harp_boxes=True, min_harp_flux=0,
        timestamp="2023-08-01T00:00:00Z", jsoc_email="test@example.com",
    )
    assert calls == [("test@example.com", "2023-08-01T00:00:00Z")]
    assert len(ax.patches) == 1
    assert ax.texts[0].get_text() == "#1234\n3.00e-09"
    plt.close(ax.figure)

    ax = foxes.plot_flux_map(
        flux_map, show_harp_boxes=True, harp_regions=[],
    )
    assert len(ax.patches) == 0
    assert len(calls) == 1
    plt.close(ax.figure)


def test_plot_flux_map_uses_prediction_metadata_for_harp_query(monkeypatch):
    calls = []

    def fetch(self, timestamp):
        calls.append((self.email, timestamp, self.resolution, self.padding_factor))
        return [{"HARPNUM": 1234, "top_pct": 0, "left_pct": 0,
                 "width_pct": 50, "height_pct": 50}]

    monkeypatch.setattr(foxes.HARPProcessor, "fetch_date", fetch)
    metadata = foxes.ObservationMetadata(
        timestamp="2023-07-31T20:00:00-04:00",
        image_shape=(7, 1024, 1024), padding_factor=0.25,
    )
    result = foxes.Prediction(
        flux=1e-8, flux_map=np.array([[1e-9, 2e-9], [3e-9, 4e-9]]),
        metadata=metadata,
    )
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, jsoc_email="test@example.com", min_harp_flux=0)
    assert calls == [("test@example.com", metadata.timestamp, 1024, 0.25)]
    assert ax.texts[0].get_text() == "#1234\n3.00e-09"
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="timestamp conflicts"):
        foxes.plot_flux_map(result, show_harp_boxes=True, timestamp="2024-01-01")
    assert len(calls) == 1


def test_observation_rejects_metadata_with_wrong_image_shape():
    metadata = foxes.ObservationMetadata(timestamp="2023-08-01", image_shape=(7, 2, 2))
    with pytest.raises(ValueError, match="shape does not match"):
        foxes.Observation(data=np.zeros((7, 4, 4)), metadata=metadata)


@pytest.mark.parametrize("include_aia,include_harps", [
    (True, True), (True, False), (False, True), (False, False),
])
@pytest.mark.parametrize("per_call", [False, True])
@pytest.mark.parametrize("include_sxr", [False, True])
def test_observation_processor_source_toggles(monkeypatch, include_aia, include_harps, include_sxr, per_call):
    calls = []
    stack = np.zeros((7, 2, 2), dtype=np.float32)
    regions = [{"HARPNUM": 1234, "top_pct": 0, "left_pct": 0,
                "width_pct": 50, "height_pct": 50}]

    def fetch_aia(self, timestamp):
        calls.append(("aia", timestamp, self.email, self.resolution))
        return stack

    def fetch_harp(self, timestamp):
        calls.append(("harp", timestamp, self.email, self.resolution))
        assert self.padding_factor == 0.1
        return regions

    sxr = {"xrsb_flux": 1e-6}

    def fetch_sxr(self, timestamp):
        calls.append(("sxr", timestamp, self.tolerance, self.satellite))
        return sxr

    monkeypatch.setattr(foxes.AIAProcessor, "fetch_date", fetch_aia)
    monkeypatch.setattr(foxes.HARPProcessor, "fetch_date", fetch_harp)
    monkeypatch.setattr(foxes.SXRProcessor, "fetch_date", fetch_sxr)
    toggles = {"include_aia": include_aia, "include_harps": include_harps, "include_sxr": include_sxr}
    # Per-call values override the opposite configuration without changing it.
    configured = {key: not value for key, value in toggles.items()} if per_call else toggles
    processor = foxes.ObservationProcessor(
        email="test@example.com", resolution=2, sxr_tolerance=90, sxr_satellite=16, **configured,
    )
    observation = processor.fetch_observation(
        "2023-07-31T20:00:00-04:00", **(toggles if per_call else {}),
    )
    timestamp = datetime(2023, 8, 1, tzinfo=timezone.utc)
    expected = []
    if include_aia:
        expected.append(("aia", timestamp, "test@example.com", 2))
    if include_harps:
        expected.append(("harp", timestamp, "test@example.com", 2))
    if include_sxr:
        expected.append(("sxr", timestamp, 90, 16))
    assert calls == expected
    assert observation.data is (stack if include_aia else None)
    assert observation.harp_regions is (regions if include_harps else None)
    assert observation.sxr is (sxr if include_sxr else None)
    assert observation.metadata.timestamp == timestamp
    assert observation.metadata.image_shape == (7, 2, 2)
    assert processor.include_aia == configured["include_aia"]
    assert processor.include_harps == configured["include_harps"]
    assert processor.include_sxr == configured["include_sxr"]


def test_observation_processor_defaults_to_all_and_keeps_aia_when_supplements_missing(monkeypatch):
    calls = []
    stack = np.zeros((7, 2, 2), dtype=np.float32)

    def fetch_aia(self, timestamp):
        calls.append("aia")
        return stack

    def fetch_harp(self, timestamp):
        calls.append("harp")
        return []

    def fetch_sxr(self, timestamp):
        calls.append("sxr")
        return None

    monkeypatch.setattr(foxes.AIAProcessor, "fetch_date", fetch_aia)
    monkeypatch.setattr(foxes.HARPProcessor, "fetch_date", fetch_harp)
    monkeypatch.setattr(foxes.SXRProcessor, "fetch_date", fetch_sxr)
    observation = foxes.ObservationProcessor(resolution=2).fetch_observation("2023-08-01")
    assert calls == ["aia", "harp", "sxr"]
    assert observation.data is stack
    assert observation.harp_regions == []
    assert observation.sxr is None


def test_observation_records_harp_expansion_for_deferred_query(monkeypatch):
    monkeypatch.setattr(
        foxes.AIAProcessor, "fetch_date", lambda self, timestamp: np.zeros((7, 2, 2), dtype=np.float32),
    )
    calls = []

    def fetch(self, timestamp):
        calls.append((self.bbox_expansion, timestamp))
        return []

    monkeypatch.setattr(foxes.HARPProcessor, "fetch_date", fetch)
    processor = foxes.ObservationProcessor(resolution=2, harp_bbox_expansion=0.1, include_sxr=False)
    observation = processor.fetch_observation("2023-08-01", include_harps=False)
    assert calls == []
    assert observation.metadata.harp_bbox_expansion == 0.1
    result = foxes.Prediction(flux=4e-9, flux_map=np.full((2, 2), 1e-9), metadata=observation.metadata)
    ax = foxes.plot_flux_map(result, show_harp_boxes=True)
    assert calls == [(0.1, observation.metadata.timestamp)]
    plt.close(ax.figure)
    processor.fetch_observation("2023-08-01")
    assert calls == [(0.1, observation.metadata.timestamp)] * 2


@pytest.mark.parametrize("regions", [[], [
    {"HARPNUM": 1234, "top_pct": 0, "left_pct": 0, "width_pct": 50, "height_pct": 50},
]])
def test_plot_does_not_repeat_fetched_harp_queries(monkeypatch, regions):
    monkeypatch.setattr(
        foxes.HARPProcessor, "fetch_date",
        lambda *args, **kwargs: pytest.fail("Already-fetched HARPs must not be queried again"),
    )
    result = foxes.Prediction(
        flux=1e-8, flux_map=np.array([[1e-9, 2e-9], [3e-9, 4e-9]]),
        harp_regions=regions,
    )
    for _ in range(2):
        ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=0)
        assert len(ax.patches) == len(regions)
        plt.close(ax.figure)
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, harp_regions=[])
    assert not ax.patches
    plt.close(ax.figure)


def test_prediction_reports_harp_ids_locations_and_flux_sums():
    regions = [
        {"HARPNUM": 1234, "top_pct": 0, "left_pct": 0, "width_pct": 50, "height_pct": 50,
         "LON_MIN": -30.0, "LON_MAX": -10.0, "LAT_MIN": 10.0, "LAT_MAX": 30.0},
        {"harpnum": 5678, "top_pct": 50, "left_pct": 50, "width_pct": 50, "height_pct": 50},
        {"HARPNUM": 9012, "top_pct": 0, "left_pct": 0, "width_pct": 100, "height_pct": 100},
        {"HARPNUM": 3456, "top_pct": 0, "left_pct": 0, "width_pct": 10, "height_pct": 10},
    ]
    result = foxes.Prediction(
        flux=1e-8, flux_map=np.array([[1e-9, 2e-9], [3e-9, 4e-9]]), harp_regions=regions,
    )
    assert [harp["harpnum"] for harp in result.harp_regions] == [1234, 5678, 9012, 3456]
    assert [harp["flux_sum"] for harp in result.harp_regions] == pytest.approx([3e-9, 2e-9, 1e-8, 0])
    assert result.harp_regions[0]["LON_MIN"] == -30.0
    assert result.harp_regions[0]["LON_MAX"] == -10.0
    assert result.harp_regions[0]["LAT_MIN"] == 10.0
    assert result.harp_regions[0]["LAT_MAX"] == 30.0
    assert result.harp_regions[0]["top_pct"] == 0
    assert result.harp_regions[0]["width_pct"] == 50
    assert all("flux_sum" not in region for region in regions)


def test_harp_labels_include_sorted_fluxes_and_match_fixed_rank_colors():
    from matplotlib.colors import to_rgba

    flux_map = np.array([[1e-9, 2e-9], [3e-9, 4e-9]])
    regions = [
        {"HARPNUM": 1001, "top_pct": 50, "left_pct": 0, "width_pct": 50, "height_pct": 50},
        {"HARPNUM": 2002, "top_pct": 0, "left_pct": 50, "width_pct": 50, "height_pct": 50},
        {"HARPNUM": 3003, "top_pct": 0, "left_pct": 0, "width_pct": 50, "height_pct": 50},
    ]
    ax = foxes.plot_flux_map(flux_map, show_harp_boxes=True, harp_regions=regions, min_harp_flux=0)
    assert [text.get_text() for text in ax.texts] == ["#2002\n4.00e-09", "#3003\n3.00e-09", "#1001\n1.00e-09"]
    assert not any(panel.tables for panel in ax.figure.axes)
    for index, color in enumerate(("#56B4E9", "#E69F00", "#009E73")):
        assert np.allclose(ax.patches[index].get_edgecolor(), to_rgba(color))
        assert np.allclose(to_rgba(ax.texts[index].get_color()), to_rgba(color))
    plt.close(ax.figure)

    # Removing the weakest HARP does not change the colors of ranks 1 and 2.
    ax = foxes.plot_flux_map(flux_map, show_harp_boxes=True, harp_regions=regions[1:], min_harp_flux=0)
    assert np.allclose(ax.patches[0].get_edgecolor(), to_rgba("#56B4E9"))
    assert np.allclose(ax.patches[1].get_edgecolor(), to_rgba("#E69F00"))
    plt.close(ax.figure)


def test_harp_labels_use_caller_supplied_axes():
    _, ax = plt.subplots()
    regions = [{"HARPNUM": 1234, "top_pct": 0, "left_pct": 0,
                "width_pct": 50, "height_pct": 50}]
    returned = foxes.plot_flux_map(
        np.array([[1e-9, 2e-9], [3e-9, 4e-9]]), ax=ax,
        show_harp_boxes=True, harp_regions=regions, min_harp_flux=0,
    )
    assert returned is ax
    ax.figure.canvas.draw()
    assert len(ax.figure.axes) == 2  # Map and colorbar only.
    assert not any(panel.tables for panel in ax.figure.axes)
    plt.close(ax.figure)


@pytest.mark.parametrize("supplied_axes", [False, True])
def test_harp_labels_remain_readable_after_resize_and_dpi_changes(supplied_axes):
    regions = [
        {"HARPNUM": 12345, "top_pct": 0, "left_pct": 0, "width_pct": 1, "height_pct": 1},
        {"HARPNUM": 23456, "top_pct": 0, "left_pct": 99, "width_pct": 1, "height_pct": 1},
        {"HARPNUM": 34567, "top_pct": 99, "left_pct": 99, "width_pct": 1, "height_pct": 1},
        {"HARPNUM": 45678, "top_pct": 99, "left_pct": 0, "width_pct": 1, "height_pct": 1},
    ]
    ax = plt.subplots()[1] if supplied_axes else None
    ax = foxes.plot_flux_map(
        np.ones((32, 32)) * 1e-9, ax=ax,
        show_harp_boxes=True, harp_regions=regions, min_harp_flux=0, harp_label_fontsize=10,
    )
    for size, dpi in [((7, 6), 100), ((4.2, 4.2), 150)]:
        ax.figure.set_size_inches(*size)
        ax.figure.set_dpi(dpi)
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        bounds = ax.get_window_extent(renderer)
        assert len(ax.texts) == 4
        for label in ax.texts:
            box = label.get_bbox_patch().get_window_extent(renderer)
            assert box.x0 >= bounds.x0
            assert box.x1 <= bounds.x1
            assert box.y0 >= bounds.y0
            assert box.y1 <= bounds.y1
            assert label.get_fontsize() == 10
            assert not label.arrow_patch.get_visible()
    plt.close(ax.figure)


def test_harp_labels_accept_explicit_offsets_for_crowded_regions():
    regions = [
        {"HARPNUM": harpnum, "top_pct": 48, "left_pct": left, "width_pct": 1, "height_pct": 1}
        for harpnum, left in [(12345, 48), (23456, 50), (34567, 52)]
    ]
    ax = foxes.plot_flux_map(
        np.ones((64, 64)) * 1e-9, show_harp_boxes=True,
        harp_regions=regions, harp_label_fontsize=12, min_harp_flux=0,
        harp_label_offsets={12345: (-100, 30), 23456: (0, -70), 34567: (100, 30)},
    )
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    boxes = [label.get_bbox_patch().get_window_extent(renderer) for label in ax.texts]
    for index, label in enumerate(ax.texts):
        assert label.get_fontsize() == 12
        assert label.arrow_patch.get_visible()
        assert all(not boxes[index].overlaps(other) for other in boxes[:index])
    plt.close(ax.figure)


@pytest.mark.parametrize("threshold,visible", [(0, 3), (3e-9, 2), (4e-9, 1), (5e-9, 0)])
def test_harp_flux_threshold_filters_only_plot(threshold, visible):
    regions = [
        {"HARPNUM": 1001, "top_pct": 50, "left_pct": 0, "width_pct": 50, "height_pct": 50},
        {"HARPNUM": 2002, "top_pct": 0, "left_pct": 50, "width_pct": 50, "height_pct": 50},
        {"HARPNUM": 3003, "top_pct": 0, "left_pct": 0, "width_pct": 50, "height_pct": 50},
    ]
    result = foxes.Prediction(
        flux=1e-8, flux_map=np.array([[1e-9, 2e-9], [3e-9, 4e-9]]), harp_regions=regions,
    )
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=threshold)
    assert len(ax.patches) == visible
    assert len(ax.texts) == visible
    assert len(result.harp_regions) == 3
    assert not any(panel.tables for panel in ax.figure.axes)
    plt.close(ax.figure)


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf])
def test_plot_rejects_invalid_harp_flux_threshold_before_query(monkeypatch, threshold):
    monkeypatch.setattr(foxes.HARPProcessor, "fetch_date", lambda *a, **kw: pytest.fail("Unexpected query"))
    with pytest.raises(ValueError, match="min_harp_flux must be finite and nonnegative"):
        foxes.plot_flux_map(
            np.ones((2, 2)), show_harp_boxes=True,
            timestamp="2023-08-01", min_harp_flux=threshold,
        )
