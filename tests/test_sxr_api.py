"""GOES selection, quality, temporary downloads, and actual SunPy parsing."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import h5netcdf
import numpy as np
import pandas as pd
import pytest
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries

import foxes


TARGET = datetime(2023, 8, 1, tzinfo=timezone.utc)


def write_xrs(path, offsets, *, xrsa=None, xrsb=None, a_quality=None, b_quality=None):
    """Write the NOAA science fields consumed by SunPy's real XRS reader."""
    count = len(offsets)
    with h5netcdf.File(path, "w") as dataset:
        dataset.dimensions = {"time": count}
        dataset.attrs.update(id=path.name, summary="GOES XRS irradiance")
        values = {
            "time": offsets,
            "xrsa_flux": xrsa if xrsa is not None else [1e-8] * count,
            "xrsb_flux": xrsb if xrsb is not None else [1e-6] * count,
            "xrsa_flag": a_quality if a_quality is not None else [0] * count,
            "xrsb_flag": b_quality if b_quality is not None else [0] * count,
        }
        for name, data in values.items():
            variable = dataset.create_variable(name, ("time",), dtype="f8")
            variable[:] = data
            if name == "time":
                variable.attrs["units"] = "seconds since 2023-08-01 00:00:00 UTC"


def mock_download(monkeypatch, records, *, errors=(), fail_fetch=False):
    staged = []
    searches = []
    response = SimpleNamespace(file_num=len(records))

    def search(*query):
        searches.append(query)
        return response

    def fetch(found, *, path, progress):
        assert found is response
        assert progress is False
        directory = Path(path).parent
        staged.append(directory)
        if fail_fetch:
            (directory / "partial.nc").touch()
            raise OSError("download interrupted")
        paths = []
        for index, (satellite, values) in enumerate(records):
            filename = f"sci_xrsf-l2-avg1m_g{satellite:02d}_d20230801_v{index}.nc"
            output = directory / filename
            if values is None:
                output.write_text("corrupt file")
            else:
                write_xrs(output, **values)
            paths.append(str(output))

        class Downloads(list):
            pass

        result = Downloads(paths)
        result.errors = errors
        return result

    monkeypatch.setattr(Fido, "search", search)
    monkeypatch.setattr(Fido, "fetch", fetch)
    return staged, searches


def test_sxr_real_netcdf_reader_quality_and_temporary_cleanup(monkeypatch):
    staged, searches = mock_download(monkeypatch, [
        (18, dict(offsets=[-60, 0, 30, 60], b_quality=[0, 1, 1, 0],
                  xrsb=[2e-6, 9e-6, 8e-6, 3e-6])),
    ])
    result = foxes.SXRProcessor().fetch_date("2023-07-31T20:00:00-04:00")
    assert result["timestamp"] == datetime(2023, 7, 31, 23, 59, tzinfo=timezone.utc)
    assert result["time_offset_seconds"] == -60
    assert result["satellites"] == ["GOES-18"]
    assert result["xrsb_flux"] == pytest.approx(2e-6)
    assert result["xrsa_flux"] == pytest.approx(1e-8)
    assert all(not path.exists() for path in staged)
    query = searches[0]
    time = next(attr for attr in query if isinstance(attr, a.Time))
    assert time.start.to_datetime(timezone=timezone.utc) == pd.Timestamp(TARGET) - pd.Timedelta(seconds=120)
    assert time.end.to_datetime(timezone=timezone.utc) == pd.Timestamp(TARGET) + pd.Timedelta(seconds=120)
    assert next(attr for attr in query if isinstance(attr, a.Resolution)).value == "avg1m"


def test_sxr_averages_same_time_satellites_once_and_skips_bad_files(monkeypatch):
    staged, _ = mock_download(monkeypatch, [
        (19, dict(offsets=[90])), (16, dict(offsets=[0])),
        (18, dict(offsets=[0], xrsb=[4e-6])), (17, None),
        (16, dict(offsets=[0])),
    ], errors=["another file unavailable"])
    result = foxes.SXRProcessor().fetch_date(TARGET)
    assert result["satellites"] == ["GOES-18", "GOES-16"]
    assert result["xrsb_flux"] == pytest.approx(2.5e-6)
    assert [row["xrsb_flux"] for row in result["measurements"]] == [4e-6, 1e-6]
    # Compare directly with the target generation code, in physical flux space.
    from data.align_aia_sxr import create_combined_lookup_table
    training = create_combined_lookup_table({
        16: pd.DataFrame({"xrsb_flux": [1e-6]}, index=[pd.Timestamp(TARGET)]),
        18: pd.DataFrame({"xrsb_flux": [4e-6]}, index=[pd.Timestamp(TARGET)]),
    }, [TARGET])
    assert result["xrsb_flux"] == training[0]["sxr_b"]
    assert all(not path.exists() for path in staged)


def test_sxr_explicit_satellite_and_no_extra_legacy_scaling(monkeypatch):
    _, searches = mock_download(monkeypatch, [(15, dict(offsets=[0])), (18, dict(offsets=[0]))])
    result = foxes.SXRProcessor(satellite=15).fetch_date("2023-08-01")
    assert result["satellites"] == ["GOES-15"]
    assert result["xrsa_flux"] == pytest.approx(1e-8)
    assert result["xrsb_flux"] == pytest.approx(1e-6)
    assert next(attr for attr in searches[0] if isinstance(attr, a.goes.SatelliteNumber)).value == 15


@pytest.mark.parametrize("values", [
    dict(offsets=[-121, 121]), dict(offsets=[0], b_quality=[1]),
    dict(offsets=[0], xrsb=[np.nan]), dict(offsets=[0], xrsb=[0]),
    dict(offsets=[0], xrsb=[-9999]),
])
def test_sxr_returns_none_for_gaps_bad_quality_or_invalid_flux(monkeypatch, values):
    staged, _ = mock_download(monkeypatch, [(18, values)])
    assert foxes.SXRProcessor().fetch_date(TARGET) is None
    assert all(not path.exists() for path in staged)


def test_sxr_accepts_tolerance_boundary_and_converts_units():
    series = TimeSeries(
        pd.DataFrame({"xrsa": [1e-5], "xrsb": [1e-3], "xrsa_quality": [0], "xrsb_quality": [0]},
                     index=[pd.Timestamp(TARGET) + pd.Timedelta(seconds=120)]),
        {"TELESCOP": "GOES 18"},
        {"xrsa": u.erg / u.cm**2 / u.s, "xrsb": u.erg / u.cm**2 / u.s,
         "xrsa_quality": u.one, "xrsb_quality": u.one}, source="XRS",
    )
    result, = foxes.SXRProcessor()._samples(series, TARGET)
    assert result["time_offset_seconds"] == 120
    assert result["xrsa_flux"] == pytest.approx(1e-8)
    assert result["xrsb_flux"] == pytest.approx(1e-6)


@pytest.mark.parametrize("short_values", [{"a_quality": [1]}, {"xrsa": [np.inf]}, {"xrsa": [-9999]}])
def test_sxr_bad_short_band_does_not_remove_valid_training_target(monkeypatch, short_values):
    mock_download(monkeypatch, [
        (16, dict(offsets=[0], xrsb=[1e-6], **short_values)),
        (18, dict(offsets=[0], xrsb=[4e-6], xrsa=[2e-8])),
    ])
    result = foxes.SXRProcessor().fetch_date(TARGET)
    assert result["xrsb_flux"] == pytest.approx(2.5e-6)
    assert result["xrsa_flux"] == pytest.approx(2e-8)
    assert result["measurements"][1]["xrsa_flux"] is None


def test_sxr_all_short_band_data_bad_still_returns_long_band(monkeypatch):
    mock_download(monkeypatch, [(18, dict(offsets=[0], a_quality=[1]))])
    result = foxes.SXRProcessor().fetch_date(TARGET)
    assert result["xrsa_flux"] is None
    assert result["xrsb_flux"] == pytest.approx(1e-6)


def test_sxr_never_averages_different_timestamps(monkeypatch):
    mock_download(monkeypatch, [
        (16, dict(offsets=[0], xrsb=[1e-6])),
        (18, dict(offsets=[-30, 30], xrsb=[4e-6, 6e-6])),
    ])
    result = foxes.SXRProcessor().fetch_date(TARGET)
    assert result["satellites"] == ["GOES-16"]
    assert result["xrsb_flux"] == pytest.approx(1e-6)
    assert result["time_offset_seconds"] == 0


def test_sxr_no_search_results_does_not_download(monkeypatch):
    staged, _ = mock_download(monkeypatch, [])
    assert foxes.SXRProcessor().fetch_date(TARGET) is None
    assert staged == []


def test_sxr_cleans_up_interrupted_download(monkeypatch):
    staged, _ = mock_download(monkeypatch, [(18, {})], fail_fetch=True)
    assert foxes.SXRProcessor().fetch_date(TARGET) is None
    assert len(staged) == 1 and not staged[0].exists()


@pytest.mark.parametrize("stage", ["import", "read"])
def test_sxr_missing_dependency_explains_active_environment_and_cleans_up(monkeypatch, caplog, stage):
    import builtins
    import sys

    staged, _ = mock_download(monkeypatch, [(18, dict(offsets=[0]))])

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'h5netcdf'", name="h5netcdf")

    if stage == "import":
        original_import = builtins.__import__

        def import_module(name, *args, **kwargs):
            if name == "sunpy.timeseries":
                return missing()
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_module)
    else:
        monkeypatch.setattr("sunpy.timeseries.TimeSeries", missing)

    assert foxes.SXRProcessor().fetch_date(TARGET) is None
    assert sys.executable in caplog.text
    assert "%pip install h5netcdf h5py" in caplog.text
    assert all(not directory.exists() for directory in staged)


@pytest.mark.parametrize("kwargs", [
    {"tolerance": 0}, {"tolerance": -1}, {"tolerance": np.nan},
    {"satellite": 0}, {"satellite": 18.5}, {"satellite": True},
])
def test_sxr_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        foxes.SXRProcessor(**kwargs)


def test_sxr_bad_timestamp_is_caller_error():
    with pytest.raises(ValueError):
        foxes.SXRProcessor().fetch_date("invalid datetime")
