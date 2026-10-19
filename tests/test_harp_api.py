"""Focused checks for HARP query construction and sweep selection."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from foxes.api import HARPProcessor, Prediction


def test_query_sharp_uses_drms_and_requested_email(monkeypatch):
    seen = {}
    expected = pd.DataFrame({"T_REC": ["2023.08.01_00:00:00_TAI"]})

    class Client:
        def __init__(self, email):
            seen["email"] = email

        def query(self, ds, key):
            seen["ds"] = ds
            seen["key"] = key
            return expected

    monkeypatch.setattr("foxes.api.drms.Client", Client)
    processor = HARPProcessor(email="test@example.com")
    result = processor._query_sharp(datetime(2023, 8, 1, tzinfo=timezone.utc), processor.email)

    assert result is expected
    assert seen["email"] == "test@example.com"
    assert seen["ds"] == "hmi.sharp_720s[][2023.07.31_23:45:00_TAI/1800s]"
    assert "NOAA_ARS" in seen["key"]


def test_query_sharp_falls_back_on_drms_failure(monkeypatch):
    class Client:
        def __init__(self, email):
            raise RuntimeError("primary failed")

    expected = pd.DataFrame({"T_REC": ["2023.08.01_00:00:00_TAI"]})
    processor = HARPProcessor()
    monkeypatch.setattr("foxes.api.drms.Client", Client)
    monkeypatch.setattr(processor, "_query_sharp_via_sunpy", lambda timestamp, email: expected)

    assert processor._query_sharp(datetime(2023, 8, 1), "test@example.com") is expected


def test_nearest_good_sweep_skips_bad_nearest():
    processor = HARPProcessor()
    df = pd.DataFrame({
        "T_REC": ["2023.08.01_00:00:00_TAI", "2023.08.01_00:12:00_TAI"],
        "QUALITY": [65536, 0],
    })

    assert processor._nearest_good_t_rec(df, datetime(2023, 8, 1)) == "2023.08.01_00:12:00_TAI"
    assert processor._nearest_good_t_rec(df.assign(QUALITY=65536), datetime(2023, 8, 1)) is None


def test_fetch_returns_raw_rows_from_selected_good_sweep(monkeypatch):
    processor = HARPProcessor(email="test@example.com")
    df = pd.DataFrame({
        "T_REC": ["2023.08.01_00:00:00_TAI", "2023.08.01_00:12:00_TAI", "2023.08.01_00:12:00_TAI"],
        "QUALITY": [65536, 0, 65536],
        "HARPNUM": [1, 2, 3],
        "NOAA_ARS": ["10001", "10002,10003", "10004"],
    })
    monkeypatch.setattr(processor, "_query_sharp", lambda timestamp, email: df)
    monkeypatch.setattr(processor, "_bbox_to_pixel_fractions", lambda row: {
        "top_pct": 20.0, "left_pct": 30.0, "width_pct": 10.0, "height_pct": 5.0,
    })

    assert processor.fetch_date(datetime(2023, 8, 1)) == [
        {"T_REC": "2023.08.01_00:12:00_TAI", "QUALITY": 0,
         "HARPNUM": 2, "NOAA_ARS": "10002,10003",
         "top_pct": 20.0, "left_pct": 30.0,
         "width_pct": 10.0, "height_pct": 5.0}
    ]


@pytest.mark.parametrize("timestamp", [
    "2023-08-01T00:00:00Z",
    "2023-08-01T00:00:00",
    "2023-07-31T20:00:00-04:00",
    datetime(2023, 7, 31, 20, tzinfo=timezone(timedelta(hours=-4))),
])
def test_fetch_accepts_aia_timestamp_formats(monkeypatch, timestamp):
    processor = HARPProcessor(email="test@example.com")
    seen = []

    def query(normalized, email):
        seen.append((normalized, email))
        return pd.DataFrame({"T_REC": ["2023.08.01_00:00:00_TAI"], "QUALITY": [0], "HARPNUM": [2]})

    monkeypatch.setattr(processor, "_query_sharp", query)
    monkeypatch.setattr(processor, "_bbox_to_pixel_fractions", lambda row: {"top_pct": 20.0})
    assert processor.fetch_date(timestamp)[0]["HARPNUM"] == 2
    assert seen == [(datetime(2023, 8, 1), "test@example.com")]


def test_bbox_projects_visible_region_and_clips_far_crop():
    processor = HARPProcessor()
    row = pd.Series({
        "T_REC": "2023.08.01_00:00:00_TAI", "RSUN_OBS": 945.497437,
        "LON_MIN": -31.169172, "LON_MAX": 14.767216,
        "LAT_MIN": 7.121964, "LAT_MAX": 27.821505,
    })
    box = processor._bbox_to_pixel_fractions(row)
    assert box is not None
    assert all(0 <= box[key] <= 100 for key in ("top_pct", "left_pct", "width_pct", "height_pct"))
    assert box["width_pct"] > 0 and box["height_pct"] > 0
    outside = row.copy()
    outside["RSUN_OBS"] = 10.0
    assert processor._bbox_to_pixel_fractions(outside) is None


def test_bbox_dense_edge_sampling_covers_near_limb_extremum():
    processor = HARPProcessor()
    row = pd.Series({
        "T_REC": "2024.06.06_12:00:00_TAI", "RSUN_OBS": 945.497437,
        "LON_MIN": -89.0, "LON_MAX": -75.0,
        "LAT_MIN": -20.0, "LAT_MAX": 20.0,
    })
    box = processor._bbox_to_pixel_fractions(row)
    assert box is not None
    assert box["left_pct"] < 6.3
    assert box["width_pct"] > 2.82


@pytest.mark.parametrize("expansion,expected,flux_sum", [
    (0, {"top_pct": 25.0, "left_pct": 25.0, "width_pct": 50.0, "height_pct": 50.0}, 30e-9),
    (0.5, {"top_pct": 0.0, "left_pct": 0.0, "width_pct": 100.0, "height_pct": 100.0}, 120e-9),
    (100, {"top_pct": 0.0, "left_pct": 0.0, "width_pct": 100.0, "height_pct": 100.0}, 120e-9),
])
def test_bbox_expansion_clips_to_crop_and_changes_prediction_sums(monkeypatch, expansion, expected, flux_sum):
    projected = SimpleNamespace(
        Tx=SimpleNamespace(to_value=lambda unit: np.array([-500.0, 500.0])),
        Ty=SimpleNamespace(to_value=lambda unit: np.array([-500.0, 500.0])),
    )
    monkeypatch.setattr(
        "foxes.api.frames.HeliographicStonyhurst",
        lambda *args, **kwargs: SimpleNamespace(transform_to=lambda target: projected),
    )
    row = pd.Series({
        "T_REC": "2023.08.01_00:00:00_TAI", "RSUN_OBS": 1000,
        "LON_MIN": -1, "LON_MAX": 1, "LAT_MIN": -1, "LAT_MAX": 1,
    })
    box = HARPProcessor(resolution=4, padding_factor=0, bbox_expansion=expansion)._bbox_to_pixel_fractions(row)
    assert box == expected
    result = Prediction(
        flux=120e-9, flux_map=np.arange(16).reshape(4, 4) * 1e-9,
        harp_regions=[{"HARPNUM": 1234, **box}],
    )
    assert result.harp_regions[0]["flux_sum"] == pytest.approx(flux_sum)
    assert result.harp_regions[0]["width_pct"] == expected["width_pct"]


@pytest.mark.parametrize("expansion", [-0.1, np.nan, np.inf])
def test_invalid_bbox_expansion_is_rejected(expansion):
    with pytest.raises(ValueError, match="bbox_expansion must be finite and nonnegative"):
        HARPProcessor(bbox_expansion=expansion)
