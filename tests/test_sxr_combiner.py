"""Preserve science flux precision through combining, alignment, and NPY export."""

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray", reason="Requires repository data-processing dependencies")
pytest.importorskip("pyarrow", reason="Requires lossless Parquet processing")

from data.combine_sxr import SXRDataProcessor, prepare_sxr_frame
from data.align_aia_sxr import (
    create_combined_lookup_table, load_and_prepare_goes_data, process_batch,
)


def test_combiner_preserves_processed_values_and_dtypes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "combined"
    times = pd.date_range("2018-01-01", periods=4, freq="min")
    dataset = xr.Dataset({
        "xrsb_flux": ("time", np.array([7e-6, 0, 1.4e-5, 2.1e-5], dtype=np.float32)),
        "xrsa_flux": ("time", np.array([1e-8, 0, 2e-8, 3e-8], dtype=np.float32)),
        "xrsb_flag": ("time", [0, 0, 1, 0]),
    }, coords={"time": times})
    dataset.to_netcdf(raw / "sci_xrsf_g15_20180101.nc")
    expected = prepare_sxr_frame(dataset)

    SXRDataProcessor(raw, output).combine_goes_data()

    files = list(output.glob("*.parquet"))
    assert len(files) == 1
    assert not list(output.glob("*.tmp"))
    restored = load_and_prepare_goes_data(output)[15]
    pd.testing.assert_frame_equal(restored, expected, check_exact=True)


def test_binary_alignment_preserves_halfway_rounding_and_prefers_parquet(tmp_path):
    timestamp = pd.Timestamp("2018-01-01")
    first = np.float32(7e-6)
    second = np.nextafter(first, np.float32(np.inf))
    for satellite, value in ((15, first), (16, second)):
        frame = pd.DataFrame({
            "xrsb_flux": np.array([value], dtype=np.float32), "xrsb_flag": [0],
        }, index=pd.DatetimeIndex([timestamp], name="time"))
        frame.to_parquet(tmp_path / f"combined_g{satellite}_avg1m_20180101_20180101.parquet")
    # A stale CSV must not replace the lossless value for the same satellite.
    frame.assign(xrsb_flux=1e-3).to_csv(tmp_path / "combined_g15_avg1m_20170101_20170101.csv")

    loaded = load_and_prepare_goes_data(tmp_path)
    assert loaded[15].xrsb_flux.dtype == np.float32
    assert loaded[15].xrsb_flux.iloc[0] == first
    lookup = create_combined_lookup_table(loaded, [timestamp])
    output = tmp_path / "targets"
    output.mkdir()
    successful, failed = process_batch(lookup, output)

    expected = np.array([np.mean(np.array([first, second], dtype=np.float64))], dtype=np.float32)
    saved = np.load(output / "2018-01-01T00:00:00.npy")
    assert (successful, failed) == (1, 0)
    assert saved.dtype == np.float32
    assert saved.tobytes() == expected.tobytes()


def test_historical_csv_only_directory_remains_readable(tmp_path):
    frame = pd.DataFrame({
        "time": ["2018-01-01 00:00:00", "2018-01-01 00:01:00"],
        "xrsb_flux": [1e-6, 2e-6], "xrsb_flag": [0, 1],
    })
    frame.to_csv(tmp_path / "combined_g16_avg1m_20180101_20180101.csv", index=False)
    loaded = load_and_prepare_goes_data(tmp_path)
    assert list(loaded) == [16]
    assert loaded[16].index.tolist() == [pd.Timestamp("2018-01-01")]
    assert loaded[16].xrsb_flux.iloc[0] == 1e-6


def test_multiple_binary_files_for_one_satellite_require_a_choice(tmp_path):
    for day in ("20180101", "20180102"):
        (tmp_path / f"combined_g16_avg1m_{day}_{day}.parquet").touch()
    with pytest.raises(ValueError, match="Multiple combined files for GOES-16"):
        load_and_prepare_goes_data(tmp_path)


@pytest.mark.parametrize("failure", ["unreadable", "write_error"])
def test_failed_satellite_is_not_reported_as_success(tmp_path, monkeypatch, capsys, failure):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "sci_xrsf_g16_20180101.nc"
    if failure == "unreadable":
        source.write_bytes(b"not a netCDF file")
    else:
        xr.Dataset({
            "xrsb_flux": ("time", [1e-6]),
            "xrsa_flux": ("time", [1e-8]),
            "xrsb_flag": ("time", [0]),
        }, coords={"time": pd.date_range("2018-01-01", periods=1)}).to_netcdf(source)

        def failed_save(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(pd.DataFrame, "to_parquet", failed_save)
    output = tmp_path / "combined"
    SXRDataProcessor(raw, output).combine_goes_data()
    stdout = capsys.readouterr().out
    assert "Successfully processed: 0 satellites" in stdout
    assert "Failed: 1 satellites" in stdout
    assert not list(output.glob("*.parquet"))
