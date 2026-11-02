"""Sampling, GOES targets, and resumable pair downloads for expand_training_data."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import download.expand_training_data as expansion
from download.expand_training_data import WAVELENGTHS, add_goes_flux, file_times, get_catalog, make_plan

NONE = pd.DatetimeIndex([])


@pytest.fixture
def config(tmp_path):
    result = yaml.safe_load((Path(__file__).parents[1] / "download/expand_training_data.yaml").read_text())
    result.update(date_ranges=[["2020-01-01", "2020-01-03"]], max_samples=None, inter_event_jitter_minutes=0,
                  event_half_window_minutes=30, event_cadence_minutes=10, catalog_workers=1,
                  output_dir=str(tmp_path), jsoc_emails=["a@example.com", "b@example.com"],
                  parallel_timestamps=2, resolution=2)
    return result


def catalog(*flares):
    frame = pd.DataFrame(flares, columns=["start_time", "peak_time", "end_time", "goes_class"])
    for column in ("start_time", "peak_time", "end_time"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def times(plan, kind):
    return set(plan.loc[plan.kind == kind, "timestamp"].dt.strftime("%Y-%m-%dT%H:%M"))


def test_peak_windows_skip_existing_times_and_heldout_days(config):
    flares = catalog(("2020-01-01T11:50", "2020-01-01T12:00", "2020-01-01T12:10", "M1.0"),
                     ("2020-01-02T12:00", "2020-01-02T12:05", "2020-01-02T13:00", "C1.0"))
    plan = make_plan(config, flares, pd.DatetimeIndex(["2020-01-01T12:00"]), pd.DatetimeIndex(["2020-01-02"]))
    assert set(plan.loc[plan.kind == "near_event", "offset_minutes"]) == {-30, -20, -10, 10, 20, 30}
    assert not (plan.timestamp.dt.day == 2).any()
    # heldout_guard_minutes also protects the neighbouring days' edges.
    assert "2020-01-03T00:00" not in times(plan, "inter_event")
    assert {"2020-01-01T21:00", "2020-01-03T03:00"} <= times(plan, "inter_event")


def test_inter_event_grid_avoids_flares_of_every_class(config):
    config["event_classes"] = "MX"
    flares = catalog(("2020-01-01T00:10", "2020-01-01T00:30", "2020-01-01T10:00", "M1.0"),
                     ("2020-01-01T02:00", "2020-01-01T02:30", "2020-01-01T03:00", "C1.0"),
                     ("2020-01-01T17:00", "2020-01-01T17:20", "2020-01-01T18:30", "B2.0"))
    plan = make_plan(config, flares, NONE, NONE)
    first_day = {t for t in times(plan, "inter_event") if t.startswith("2020-01-01")}
    assert first_day == {"2020-01-01T12:00", "2020-01-01T15:00", "2020-01-01T21:00"}
    assert set(plan.loc[plan.kind == "near_event", "goes_class"]) == {"M1.0"}


def test_overlapping_windows_keep_the_nearest_peak(config):
    flares = catalog(("2020-01-01T11:50", "2020-01-01T12:00", "2020-01-01T12:10", "M1.0"),
                     ("2020-01-01T12:10", "2020-01-01T12:20", "2020-01-01T12:30", "C1.0"))
    plan = make_plan(config, flares, NONE, NONE).set_index("timestamp")
    assert plan.index.is_unique
    assert plan.loc["2020-01-01T12:00", "goes_class"] == "M1.0"
    assert plan.loc["2020-01-01T12:30", "goes_class"] == "C1.0"


def test_cap_is_seeded_and_sorted(config):
    config.update(max_samples=8, inter_event_jitter_minutes=40)
    flares = catalog(("2020-01-02T11:50", "2020-01-02T12:00", "2020-01-02T12:10", "X1.0"))
    plan = make_plan(config, flares, NONE, NONE)
    pd.testing.assert_frame_equal(plan, make_plan(config, flares, NONE, NONE))
    assert len(plan) == 8 and plan.timestamp.is_monotonic_increasing


def test_file_times_reads_timestamp_names_and_rejects_missing_folders(tmp_path):
    folder = tmp_path / "aia" / "nested"
    folder.mkdir(parents=True)
    for name in ("2020-01-01T12:00:00.npy", "2020-01-01T13_00_00.npy", "._2020-01-01T14:00:00.npy", "stats.npy"):
        (folder / name).touch()
    assert sorted(file_times([tmp_path / "aia"])) == [pd.Timestamp("2020-01-01T12:00"), pd.Timestamp("2020-01-01T13:00")]
    with pytest.raises(FileNotFoundError):
        file_times([tmp_path / "missing"])


def test_catalog_reads_cached_months_and_widens_inconsistent_boundaries(tmp_path, config):
    cache = tmp_path / "catalog_cache"
    cache.mkdir()
    columns = ["start_time", "peak_time", "end_time", "goes_class"]
    pd.DataFrame([["2019-12-31 23:00:00.000", "2019-12-31 23:10:00.000", "2019-12-31 23:20:00.000", "B1.0"]],
                 columns=columns).to_csv(cache / "goes_2019-12.csv", index=False)
    pd.DataFrame([["2020-01-01 12:30:00.000", "2020-01-01 12:00:00.000", "2020-01-01 12:10:00.000", " c1.0"],
                  ["2020-01-02 00:00:00+00:00", "2020-01-02 01:00:00+00:00", "2020-01-02 02:00:00+00:00", "M2.0"]],
                 columns=columns).to_csv(cache / "goes_2020-01.csv", index=False)
    flares = get_catalog(config, cache)
    assert flares.goes_class.tolist() == ["B1.0", "C1.0", "M2.0"]
    assert flares.start_time[1] == pd.Timestamp("2020-01-01T12:00")
    assert flares.end_time[1] == pd.Timestamp("2020-01-01T12:30")
    assert flares.peak_time[2] == pd.Timestamp("2020-01-02T01:00")


def test_goes_flux_comes_from_the_training_lookup(tmp_path):
    for satellite, flux in ((15, 2e-6), (16, 4e-6)):
        pd.DataFrame({"time": ["2020-01-01 12:00:00", "2020-01-01 12:01:00", "2020-01-01 12:02:00"],
                      "xrsb_flux": [flux, flux, -1.0], "xrsb_flag": [0, 1, 0]}).to_csv(
            tmp_path / f"combined_g{satellite}_avg1m_20200101_20200102.csv", index=False)
    plan = pd.DataFrame({"timestamp": pd.date_range("2020-01-01T12:00", periods=4, freq="min"), "kind": "inter_event"})
    result = add_goes_flux(plan, tmp_path)
    # 12:01 is flagged, 12:02 is negative, and 12:03 has no data.
    assert result.timestamp.tolist() == ["2020-01-01T12:00:00"]
    assert result.xrsb_flux[0] == pytest.approx(3e-6)
    assert result.satellites[0] == "GOES-16,GOES-15"


def test_download_saves_pairs_removes_fits_and_resumes(tmp_path, config, monkeypatch):
    import data.convert_aia
    import download.download_sdo
    import itipy.data.editor

    pd.DataFrame({"timestamp": ["2020-01-01T12:00:00", "2020-01-01T13:00:00", "2020-01-01T14:00:00"],
                  "xrsb_flux": [1e-6, 2e-6, 3e-6]}).to_csv(tmp_path / "samples.csv", index=False)
    emails, requested = [], []

    class Downloader:
        def __init__(self, directory, email, **kwargs):
            self.directory = Path(directory)
            emails.append(email)

        def downloadDate(self, date):
            requested.append(date)
            if date.hour == 14:
                return  # JSOC had no frames within tolerance.
            for wl in WAVELENGTHS:
                path = self.directory / str(wl) / f"{date.isoformat()}.fits"
                path.parent.mkdir(exist_ok=True)
                path.touch()

    class Stack:
        def __init__(self, data, **kwargs):
            assert all(Path(path).exists() for (path,) in data)

        def __getitem__(self, index):
            return np.ones((7, 2, 2))

    monkeypatch.setattr(download.download_sdo, "SDODownloader", Downloader)
    monkeypatch.setattr(data.convert_aia, "AIAStackDataset", Stack)
    monkeypatch.setattr(itipy.data.editor, "get_auto_calibration_table", lambda: None)

    assert expansion.download(config, limit=1) == 0
    emails.clear()
    assert expansion.download(config) == 1
    aia = np.load(tmp_path / "AIA_processed/train/2020-01-01T13:00:00.npy")
    sxr = np.load(tmp_path / "SXR_processed/train/2020-01-01T13:00:00.npy")
    assert aia.shape == (7, 2, 2) and aia.dtype == sxr.dtype == np.float32
    assert sxr.tolist() == [pytest.approx(2e-6)]
    assert not (tmp_path / "AIA_processed/train/2020-01-01T14:00:00.npy").exists()
    assert "no AIA frame" in pd.read_csv(tmp_path / "failed.csv").error[0]
    assert not list((tmp_path / "staging").rglob("*.fits"))
    assert emails == ["a@example.com", "b@example.com"][:len(emails)]

    requested.clear()
    assert expansion.download(config) == 1
    assert requested == [datetime(2020, 1, 1, 14)]
