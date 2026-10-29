"""Time parsing, local inputs, bulk selection, and HARP time series."""

from datetime import date, datetime, timedelta, timezone
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import helio_foxes as foxes


@pytest.mark.parametrize("value", [
    "2023-08-01", "2023/08/01 00:00:00", "2023.08.01_00:00:00_UTC",
    "August 1, 2023", "1 Aug 2023 00:00 UTC", "2023-07-31T20:00:00-04:00",
    date(2023, 8, 1), np.datetime64("2023-08-01"),
])
def test_timestamp_formats(value):
    assert foxes.parse_timestamp(value) == datetime(2023, 8, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("value", [
    "01/08/2023", "August 2023", "12:00", "yesterday", "2023-02-30",
    "2023-08-01 12:00 EST", "", np.datetime64("NaT"), pd.NaT,
])
def test_bad_timestamp(value):
    with pytest.raises(ValueError, match="timestamp"):
        foxes.parse_timestamp(value)


def test_numeric_timestamp_rejected():
    with pytest.raises(TypeError, match="timestamp"):
        foxes.parse_timestamp(1690848000)


def test_local_array_and_npy(tmp_path):
    data = np.ones((7, 2, 2))
    observation = foxes.Observation.from_array(data, timestamp="2023/08/01", harp_regions=[])
    assert observation.data.dtype == np.float32
    data[:] = 2
    assert np.all(observation.data == 1)
    path = tmp_path / "stack.npy"
    np.save(path, observation.data)
    loaded = foxes.Observation.from_npy(path, timestamp="August 1, 2023")
    assert loaded.metadata == observation.metadata
    np.testing.assert_array_equal(loaded.data, observation.data)
    for bad in [np.zeros((2, 2)), np.full((7, 2, 2), np.nan), np.ones((7, 2, 2), dtype=complex)]:
        with pytest.raises(ValueError, match="channel-first"):
            foxes.Observation.from_array(bad, timestamp="2023-08-01")


def test_local_fits_preprocessing_preserves_files(monkeypatch, tmp_path):
    paths = [tmp_path / f"{w}.fits" for w in foxes.AIAProcessor.WAVELENGTHS]
    for path in paths:
        path.touch()

    class Stacker:
        def __init__(self, data, wavelengths, resolution, **kwargs):
            assert data == [[str(path)] for path in paths]
            assert tuple(wavelengths) == foxes.AIAProcessor.WAVELENGTHS

        def __getitem__(self, index):
            return np.ones((7, 2, 2))

    monkeypatch.setitem(sys.modules, "data.convert_aia", types.SimpleNamespace(AIAStackDataset=Stacker))
    processor = foxes.AIAProcessor(resolution=2)
    assert processor.process_files(paths).shape == (7, 2, 2)
    assert all(path.exists() for path in paths)
    with pytest.raises(ValueError, match="seven"):
        processor.process_files(paths[:-1])


def test_range_grid_and_bulk_dispatch(monkeypatch):
    processor = foxes.ObservationProcessor(resolution=2, include_harps=False, include_sxr=False)
    calls = []

    def bulk(times, *, on_error, progress=False):
        calls.append((times, on_error))
        for stamp in times:
            yield stamp, np.zeros((7, 2, 2))

    monkeypatch.setattr(processor.aia_processor, "_iter_range", bulk)
    monkeypatch.setattr(processor.aia_processor, "fetch_date", lambda *a: pytest.fail("single fetch"))
    observations = list(processor.iter_observations("2023/08/01", "2023-08-01T00:25Z", cadence="12min"))
    assert len(calls) == 1
    assert [o.metadata.timestamp.minute for o in observations] == [0, 12, 24]
    assert all(o.data.shape == (7, 2, 2) for o in observations)
    for cadence in [0, -1, "NaT", "nonsense", True, float("inf"), "500ms"]:
        with pytest.raises(ValueError, match="cadence"):
            list(processor.iter_observations("2023-08-01", "2023-08-02", cadence=cadence))
    with pytest.raises(ValueError, match="end"):
        list(processor.iter_observations("2023-08-02", "2023-08-01"))


def test_bulk_query_once_and_bounded_selection(monkeypatch, tmp_path, capsys):
    from download.download_sdo import SDODownloader

    start = datetime(2023, 8, 1)
    queries = []
    # A distant frame must never fill a missing channel in a later sample.
    header = pd.DataFrame({
        "DATE__OBS": ["2023-08-01T00:00:12Z", "2023-08-01T00:12:12Z"],
        "QUALITY": [0, 0], "WAVELNTH": [94, 94],
    })

    class Client:
        def __init__(self, **kwargs):
            pass

        def keys(self, series):
            return ["DATE__OBS", "QUALITY", "WAVELNTH"]

        def query(self, selector, **kwargs):
            queries.append(selector)
            return header, pd.DataFrame({"image": ["/one", "/two"]})

    monkeypatch.setattr("download.download_sdo.drms.Client", Client)
    downloader = SDODownloader(tmp_path, wavelengths=[94], tolerance=30, n_workers=1)
    selected = []
    monkeypatch.setattr(downloader, "download", lambda row: selected.append(row) or "file.fits")
    samples = list(downloader.iter_download_range(
        [start, start + timedelta(minutes=12), start + timedelta(minutes=24)], on_error="skip", progress=True,
    ))
    assert len(queries) == 1
    assert "2023-07-31_23:59:30.000000Z/1512s@12s][94]{image}" in queries[0]
    assert len(samples) == 2
    assert [row[1] for row in selected] == ["/one", "/two"]
    assert "3/3" in capsys.readouterr().err  # The missing final sample also advances tqdm.


def test_bulk_preprocessing_cleanup(monkeypatch):
    staged = []

    class Downloader:
        def __init__(self, base_path, **kwargs):
            staged.append(__import__("pathlib").Path(base_path))

        def iter_download_range(self, dates, **kwargs):
            for i, stamp in enumerate(dates):
                path = staged[0] / f"{i}.fits"
                path.touch()
                yield stamp, [path]

    monkeypatch.setitem(sys.modules, "download.download_sdo", types.SimpleNamespace(SDODownloader=Downloader))
    processor = foxes.AIAProcessor(resolution=2)
    count = []

    def process(paths):
        count.append(paths)
        if len(count) == 1:
            raise ValueError("corrupt input")
        return np.ones((7, 2, 2))

    monkeypatch.setattr(processor, "process_files", process)
    result = list(processor._iter_range([
        foxes.parse_timestamp("2023-08-01"), foxes.parse_timestamp("2023-08-02"),
    ], on_error="skip"))
    assert len(result) == 1
    assert not staged[0].exists()


def test_range_without_aia_never_downloads(monkeypatch):
    processor = foxes.ObservationProcessor(resolution=2, include_harps=False, include_sxr=False)
    monkeypatch.setattr(processor.aia_processor, "_iter_range", lambda *a, **kw: pytest.fail("download"))
    observations = list(processor.iter_observations("2023-08-01", "2023-08-01", include_aia=False))
    assert len(observations) == 1
    assert observations[0].data is None


def test_harp_series_gaps_and_model_reuse():
    observations = [foxes.Observation.from_array(np.ones((7, 2, 2)), timestamp=f"2023-08-0{i}") for i in (1, 2, 3)]
    calls = []
    region = {"HARPNUM": 12, "top_pct": 0, "left_pct": 0, "width_pct": 100, "height_pct": 100}

    class Model:
        def predict(self, observation):
            calls.append(observation.metadata.timestamp)
            return foxes.Prediction(
                metadata=observation.metadata, flux=4e-6, flux_map=np.ones((2, 2)) * 1e-6,
                harp_regions=[region] if len(calls) != 2 else [],
            )

    series = foxes.Predictor.predict_series(Model(), iter(observations))
    assert len(calls) == 3
    curve = series.harp_light_curves()[12]
    np.testing.assert_allclose(curve.flux_sum, [4e-6, np.nan, 4e-6], equal_nan=True)
    assert curve.flux_sum_uncertainty.isna().all()
    assert str(curve.index.tz) == "UTC"
    ax = series.plot(include_full_disk=True)
    assert len(ax.lines) == 2
    assert np.isnan(ax.lines[0].get_ydata()[1])
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="unique"):
        foxes.PredictionSeries((series.predictions[0], series.predictions[0]))
    assert foxes.PredictionSeries(()).to_dataframe().empty
