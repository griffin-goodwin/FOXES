"""Complete NumPy observation archives and iterator saving, without network access."""

from datetime import datetime, timezone
import json

import numpy as np
import pytest

import helio_foxes as foxes


@pytest.mark.parametrize("regions", [None, [], [{"HARPNUM": np.int64(12), "QUALITY": np.int32(0),
    "top_pct": 0.0, "left_pct": 0.0, "width_pct": 100.0, "height_pct": 100.0,
    "missing_keyword": np.nan}]])
@pytest.mark.parametrize("include_data", [False, True])
def test_full_observation_roundtrip(tmp_path, regions, include_data):
    timestamp = datetime(2023, 8, 1, tzinfo=timezone.utc)
    metadata = foxes.ObservationMetadata(timestamp, (7, 2, 2), padding_factor=0.2, harp_bbox_expansion=0.15)
    data = np.arange(28, dtype=np.float64).reshape(7, 2, 2) if include_data else None
    sxr = {"timestamp": timestamp, "xrsb_flux": np.float64(1e-6), "satellites": ["GOES-18"],
           "measurements": [{"timestamp": timestamp, "xrsa_flux": None, "satellite": "GOES-18"}]}
    observation = foxes.Observation(metadata=metadata, data=data, harp_regions=regions, sxr=sxr)
    path = observation.save(tmp_path / "nested" / "observation.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert archive["metadata"].dtype.kind == "U"
        if include_data:
            np.testing.assert_array_equal(archive["data"], data)
    restored = foxes.Observation.load(path)
    assert restored.metadata == metadata
    assert restored.sxr == sxr
    if include_data:
        np.testing.assert_array_equal(restored.data, data)
        assert restored.data.dtype == data.dtype
    else:
        assert restored.data is None
    if regions:
        assert restored.harp_regions[0]["HARPNUM"] == 12
        assert np.isnan(restored.harp_regions[0]["missing_keyword"])
    else:
        assert restored.harp_regions == regions


def test_overwrite_is_explicit_and_atomic(tmp_path, monkeypatch):
    observation = foxes.Observation.from_array(np.ones((7, 2, 2)), timestamp="2023-08-01")
    path = observation.save(tmp_path / "observation.npz")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        observation.save(path)
    assert path.read_bytes() == original

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    with monkeypatch.context() as patch:
        patch.setattr("helio_foxes.api.np.savez_compressed", fail)
        with pytest.raises(OSError, match="disk failure"):
            observation.save(path, overwrite=True)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".foxes-*.tmp"))
    observation.save(path, overwrite=True)
    assert foxes.Observation.load(path).metadata == observation.metadata


def test_bad_archives_and_invalid_input(tmp_path):
    observation = foxes.Observation.from_array(np.ones((7, 2, 2)), timestamp="2023-08-01")
    with pytest.raises(ValueError, match=".npz"):
        observation.save(tmp_path / "array.npy")
    path = tmp_path / "bad.npz"
    np.savez(path, metadata=json.dumps({"schema_version": 999}))
    with pytest.raises(ValueError, match="schema_version"):
        foxes.Observation.load(path)
    unsafe = tmp_path / "pickle.npz"
    np.savez(unsafe, metadata=np.array({"key": "value"}, dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        foxes.Observation.load(unsafe)
    with pytest.raises(FileNotFoundError):
        list(foxes.load_observations(tmp_path / "nonexistent"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No observation"):
        list(foxes.load_observations(empty))


def test_iterator_saves_before_yield_and_reload_has_no_queries(tmp_path, monkeypatch):
    processor = foxes.ObservationProcessor(resolution=2, include_harps=False, include_sxr=False)
    progress_options = []

    def samples(times, *, on_error, progress):
        progress_options.append(progress)
        for stamp in times:
            yield stamp, np.full((7, 2, 2), stamp.hour, dtype=np.float32)

    monkeypatch.setattr(processor.aia_processor, "_iter_range", samples)
    observations = processor.iter_observations("2023-08-01", "2023-08-01 02:00", save_dir=tmp_path, progress=True)
    first = next(observations)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    remainder = list(observations)
    assert progress_options == [True]
    assert len(remainder) == 2
    assert len(list(tmp_path.glob("*.npz"))) == 3
    monkeypatch.setattr(processor, "fetch_observation", lambda *a, **kw: pytest.fail("no network expected"))
    loaded = list(foxes.load_observations(tmp_path))
    assert [o.metadata.timestamp.hour for o in loaded] == [0, 1, 2]
    np.testing.assert_array_equal(loaded[0].data, first.data)
    for i, restored in enumerate(loaded):
        assert restored.data[0, 0, 0] == i


def test_existing_archive_skips_download_and_preprocessing(tmp_path, monkeypatch):
    processor = foxes.ObservationProcessor(resolution=2, include_harps=False, include_sxr=False)
    closed = []

    def samples(times, **kwargs):
        try:
            for stamp in times:
                yield stamp, np.ones((7, 2, 2))
        finally:
            closed.append(True)

    monkeypatch.setattr(processor.aia_processor, "_iter_range", samples)
    list(processor.iter_observations("2023-08-01", "2023-08-01", save_dir=tmp_path))
    assert closed == [True]
    with monkeypatch.context() as patch:
        patch.setattr(processor.aia_processor, "_iter_range", lambda *a, **kw: pytest.fail("No AIA query/preprocessing"))
        patch.setattr(processor, "fetch_observation", lambda *a, **kw: pytest.fail("No supplementary query"))
        restored = list(processor.iter_observations("2023-08-01", "2023-08-01", save_dir=tmp_path, on_error="skip", progress=True))
    assert len(restored) == 1
    assert restored[0].data.shape == (7, 2, 2)
    list(processor.iter_observations("2023-08-01", "2023-08-01", save_dir=tmp_path, overwrite=True))
    assert closed == [True, True]


def test_partial_cache_only_fetches_missing_and_preserves_order_with_skips(tmp_path, monkeypatch):
    processor = foxes.ObservationProcessor(resolution=2, include_harps=False, include_sxr=False)
    for hour in (1, 3):
        observation = foxes.Observation.from_array(np.full((7, 2, 2), hour), timestamp=f"2023-08-01 0{hour}:00")
        observation.save(tmp_path / f"observation_20230801T0{hour}0000.000000Z.npz")
    requested = []

    def samples(times, **kwargs):
        requested.extend(times)
        for stamp in times:
            if stamp.hour:  # Simulate missing data at 00:00 with on_error='skip'.
                yield stamp, np.full((7, 2, 2), stamp.hour)

    monkeypatch.setattr(processor.aia_processor, "_iter_range", samples)
    observations = list(processor.iter_observations("2023-08-01", "2023-08-01 04:00", save_dir=tmp_path, on_error="skip"))
    assert [stamp.hour for stamp in requested] == [0, 2, 4]
    assert [o.metadata.timestamp.hour for o in observations] == [1, 2, 3, 4]
    assert [o.data[0, 0, 0] for o in observations] == [1, 2, 3, 4]
    assert len(list(tmp_path.glob("*.npz"))) == 4


def test_corrupt_cache_does_not_trigger_download(tmp_path, monkeypatch):
    processor = foxes.ObservationProcessor(resolution=2, include_harps=False, include_sxr=False)
    (tmp_path / "observation_20230801T000000.000000Z.npz").write_bytes(b"broken")
    monkeypatch.setattr(processor.aia_processor, "_iter_range", lambda *a, **kw: pytest.fail("No download"))
    with pytest.raises(ValueError):
        list(processor.iter_observations("2023-08-01", "2023-08-01", save_dir=tmp_path, on_error="skip"))


def test_cached_observation_honors_disabled_sources(tmp_path, monkeypatch):
    processor = foxes.ObservationProcessor(resolution=2, include_harps=False, include_sxr=False)
    observation = foxes.Observation.from_array(np.ones((7, 2, 2)), timestamp="2023-08-01", harp_regions=[], sxr={"xrsb_flux": 1e-6})
    observation.save(tmp_path / "observation_20230801T000000.000000Z.npz")
    monkeypatch.setattr(processor, "fetch_observation", lambda *a, **kw: pytest.fail("No query"))
    restored = list(processor.iter_observations("2023-08-01", "2023-08-01", save_dir=tmp_path, include_aia=False))[0]
    assert restored.data is None and restored.harp_regions is None and restored.sxr is None


def test_cache_timestamp_must_match_requested_time(tmp_path, monkeypatch):
    processor = foxes.ObservationProcessor(resolution=2, include_harps=False, include_sxr=False)
    observation = foxes.Observation.from_array(np.ones((7, 2, 2)), timestamp="2023-08-02")
    observation.save(tmp_path / "observation_20230801T000000.000000Z.npz")
    monkeypatch.setattr(processor.aia_processor, "_iter_range", lambda *a, **kw: pytest.fail("No query"))
    with pytest.raises(ValueError, match="timestamp does not match"):
        list(processor.iter_observations("2023-08-01", "2023-08-01", save_dir=tmp_path))
