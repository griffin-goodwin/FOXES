"""Dataset failures must preserve sample and timestamp alignment."""

import numpy as np
import pytest

from forecasting.dataset import AIAGOESDataset


@pytest.fixture
def paired_data(tmp_path):
    aia_dir, sxr_dir = tmp_path / 'aia', tmp_path / 'sxr'
    aia_dir.mkdir()
    sxr_dir.mkdir()
    timestamps = ['2020-01-01T00_00_00', '2020-01-01T00_01_00']
    for timestamp in timestamps:
        np.save(aia_dir / f'{timestamp}.npy', np.ones((7, 2, 2), dtype=np.float32))
        np.save(sxr_dir / f'{timestamp}.npy', np.array(1e-7))
    return aia_dir, sxr_dir, timestamps


@pytest.mark.parametrize('failure', ['nonfinite', 'wrong_shape', 'corrupt', 'missing'])
def test_bad_image_raises_for_its_own_timestamp(paired_data, failure):
    aia_dir, sxr_dir, timestamps = paired_data
    dataset = AIAGOESDataset(aia_dir, sxr_dir, target_size=(2, 2))
    path = aia_dir / f'{timestamps[0]}.npy'
    if failure == 'nonfinite':
        image = np.ones((7, 2, 2), dtype=np.float32)
        image[0, 0, 0] = np.nan
        np.save(path, image)
    elif failure == 'wrong_shape':
        np.save(path, np.zeros((3, 2, 2)))
    elif failure == 'corrupt':
        path.write_bytes(b'not a numpy file')
    else:
        path.unlink()

    with pytest.raises(ValueError, match=timestamps[0]):
        dataset[0]
    image, target = dataset[1]
    assert image.shape == (2, 2, 7)
    assert target.item() == pytest.approx(1e-7)
    assert dataset.samples == timestamps


def test_unknown_channels_are_rejected(paired_data):
    aia_dir, sxr_dir, _ = paired_data
    with pytest.raises(ValueError, match='Unsupported AIA wavelengths'):
        AIAGOESDataset(aia_dir, sxr_dir, wavelengths=[94, 999])
