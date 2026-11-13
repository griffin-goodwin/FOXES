"""Exercise the packaged calibration through ITI with no network or home cache."""

import hashlib
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data import aia_calibration
from data.convert_aia import AIAStackDataset
from itipy.data import editor as iti_editor


@pytest.fixture
def offline_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def no_download(*args, **kwargs):
        pytest.fail("Calibration must not attempt a download")

    monkeypatch.setattr(iti_editor.request, "urlretrieve", no_download)
    return tmp_path


def test_bundled_snapshot_integrity(offline_home):
    resource = files("data").joinpath("calibration").joinpath("sdo_autocal_table.csv")
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == (
        "7467f9fb65c2448237312d3973583351774bd2f2ac1da9eb564b0cfc75993fa3"
    )
    table = aia_calibration.get_auto_calibration_table()
    assert table.shape == (589, 8)
    assert table.DATE.min() == pd.Timestamp("2010-05-15 00:01:06.184")
    assert table.DATE.max() == pd.Timestamp("2020-07-08 00:01:09.184")
    assert not (offline_home / ".iti").exists()


def test_existing_cache_is_preferred_and_left_untouched(offline_home):
    expected = aia_calibration.get_auto_calibration_table().head(2).copy()
    expected["0094"] *= 2
    cache = offline_home / ".iti" / "sdo_autocal_table.csv"
    cache.parent.mkdir()
    expected.to_csv(cache)
    original = cache.read_bytes()

    pd.testing.assert_frame_equal(aia_calibration.get_auto_calibration_table(), expected)
    assert cache.read_bytes() == original


@pytest.mark.parametrize("invalid", ["empty", "html", "channel", "date", "zero", "nan", "index", "float_index"])
def test_invalid_cache_falls_back_without_overwriting(offline_home, caplog, invalid):
    expected = aia_calibration.get_auto_calibration_table()
    cache = offline_home / ".iti" / "sdo_autocal_table.csv"
    cache.parent.mkdir()
    broken = expected.head(2).copy()
    if invalid == "empty":
        cache.touch()
    elif invalid == "html":
        cache.write_text("<html>Server unavailable</html>")
    else:
        if invalid == "channel":
            broken = broken.drop(columns="0304")
        elif invalid == "date":
            broken["DATE"] = "invalid date"
        elif invalid == "zero":
            broken.loc[0, "0094"] = 0
        elif invalid == "nan":
            broken.loc[0, "0094"] = np.nan
        elif invalid == "index":
            broken.index = [3, 4]
        elif invalid == "float_index":
            broken.index = [0.0, 1.0]
        broken.to_csv(cache)
    original = cache.read_bytes()

    pd.testing.assert_frame_equal(aia_calibration.get_auto_calibration_table(), expected)
    assert cache.read_bytes() == original
    assert "using packaged table" in caplog.text


def test_unreadable_cache_uses_bundled_table(offline_home):
    # A directory at the cache filename also raises OSError when running as root.
    (offline_home / ".iti" / "sdo_autocal_table.csv").mkdir(parents=True)
    assert len(aia_calibration.get_auto_calibration_table()) == 589


def test_stacker_uses_packaged_table_offline(offline_home, monkeypatch):
    wavelengths = [94, 131, 171, 193, 211, 304, 335]
    stack = AIAStackDataset(data=[[] for _ in wavelengths], wavelengths=wavelengths, resolution=8)
    expected = aia_calibration.get_auto_calibration_table()
    corrected = []
    monkeypatch.setattr(iti_editor, "Map", lambda data, meta: SimpleNamespace(data=data, meta=meta))
    for dataset, wavelength in zip(stack.data_sets, wavelengths):
        prep = next(editor for editor in dataset.editors if isinstance(editor, iti_editor.AIAPrepEditor))
        pd.testing.assert_frame_equal(prep.table, expected)
        # ITI must retain nearest-endpoint behavior for data after the snapshot.
        s_map = SimpleNamespace(data=np.full((2, 2), 100.0),
                                date=SimpleNamespace(datetime=pd.Timestamp("2023-08-01").to_pydatetime()),
                                meta={"wavelnth": wavelength, "exptime": 2.0})
        corrected.append(prep.call(s_map).data)
        np.testing.assert_allclose(corrected[-1], 50.0 / expected.iloc[-1][f"{wavelength:04}"])
    assert len(corrected) == 7
    assert not (offline_home / ".iti").exists()
