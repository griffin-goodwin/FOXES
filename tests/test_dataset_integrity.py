from pathlib import Path

import numpy as np

from data.check_dataset_integrity import (
    SampleResult,
    inspect_sample,
    median_solar_disk_contrast,
    quarantine_invalid_pairs,
    read_invalid_csv,
    validate_quarantine_path,
)


def test_disk_contrast_separates_solar_product_from_gaussian_noise(tmp_path):
    size = 64
    y, x = np.ogrid[:size, :size]
    disk = (y - 31.5) ** 2 + (x - 31.5) ** 2 <= 22 ** 2
    solar = np.ones((7, size, size), dtype=np.float32)
    solar[:, disk] = 10.0
    noise = np.random.default_rng(42).normal(
        100.0, 10.0, size=(7, size, size)
    ).astype(np.float32)

    assert median_solar_disk_contrast(solar, downsample=2) > 5.0
    assert median_solar_disk_contrast(noise, downsample=2) < 1.1

    aia_path = tmp_path / 'noise.npy'
    sxr_path = tmp_path / 'sxr.npy'
    np.save(aia_path, noise)
    np.save(sxr_path, np.array(1e-7, dtype=np.float32))
    result = inspect_sample(
        'test', 'noise', aia_path, sxr_path, (7, size, size),
        min_disk_contrast=2.0, noise_check_downsample=2,
    )

    assert not result.valid
    assert 'noise_like_aia' in result.errors
    assert result.aia_disk_contrast < 1.1


def test_quarantine_moves_both_members_of_invalid_pair(tmp_path):
    aia_dir = tmp_path / "aia"
    sxr_dir = tmp_path / "sxr"
    quarantine = tmp_path / "quarantine"
    (aia_dir / "val").mkdir(parents=True)
    (sxr_dir / "val").mkdir(parents=True)
    aia_path = aia_dir / "val" / "timestamp.npy"
    sxr_path = sxr_dir / "val" / "timestamp.npy"
    np.save(aia_path, np.zeros((7, 2, 2), dtype=np.float32))
    np.save(sxr_path, np.array(1e-7, dtype=np.float32))

    destination = validate_quarantine_path(quarantine, (aia_dir, sxr_dir))
    moves = quarantine_invalid_pairs(
        [SampleResult("val", "timestamp", False, errors="blank")],
        aia_dir,
        sxr_dir,
        destination,
    )

    assert not aia_path.exists()
    assert not sxr_path.exists()
    assert (quarantine / "aia" / "val" / "timestamp.npy").is_file()
    assert (quarantine / "sxr" / "val" / "timestamp.npy").is_file()
    assert [move["status"] for move in moves] == ["moved", "moved"]


def test_quarantine_does_not_overwrite_existing_file(tmp_path):
    aia_dir = tmp_path / "aia"
    sxr_dir = tmp_path / "sxr"
    quarantine = tmp_path / "quarantine"
    (aia_dir / "train").mkdir(parents=True)
    (sxr_dir / "train").mkdir(parents=True)
    source = aia_dir / "train" / "timestamp.npy"
    sxr_source = sxr_dir / "train" / "timestamp.npy"
    destination = quarantine / "aia" / "train" / "timestamp.npy"
    destination.parent.mkdir(parents=True)
    np.save(source, np.array([1]))
    np.save(sxr_source, np.array([1e-7]))
    np.save(destination, np.array([2]))

    moves = quarantine_invalid_pairs(
        [SampleResult("train", "timestamp", False, errors="invalid")],
        aia_dir,
        sxr_dir,
        quarantine,
    )

    assert source.is_file()
    assert np.load(destination).item() == 2
    assert moves[0]["status"] == "not_moved"
    assert moves[0]["error"].startswith("destination_exists:")


def test_existing_invalid_csv_can_drive_later_quarantine(tmp_path):
    csv_path = tmp_path / "integrity_invalid.csv"
    csv_path.write_text(
        "split,timestamp,errors\n"
        "val,2020-01-01T00_00_00,blank_aia_channel_0\n"
    )

    results = read_invalid_csv(csv_path)

    assert len(results) == 1
    assert results[0].split == "val"
    assert results[0].timestamp == "2020-01-01T00_00_00"
    assert results[0].errors == "blank_aia_channel_0"
