"""Local conversion must observe write failures before removing its source."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from download import hugging_face_data_download as download


@pytest.mark.parametrize('fail_write', [False, True])
def test_source_deleted_only_after_successful_numpy_writes(tmp_path, monkeypatch, fail_write):
    monkeypatch.setattr(download, 'AIA_SHAPE', (7, 2, 2))
    monkeypatch.setattr(download, 'AIA_SIZE', 28)
    source = tmp_path / 'source'
    source.mkdir()
    parquet = source / 'part.parquet'
    image = np.arange(28, dtype=np.float32)
    pq.write_table(pa.table({
        'filename': ['sample.npy'],
        'aia_stack': pa.FixedSizeListArray.from_arrays(pa.array(image), 28),
        'sxr_value': [1e-6],
    }), parquet)
    config = {
        'aia_dir': str(tmp_path / 'aia'), 'sxr_dir': str(tmp_path / 'sxr'),
        'num_workers': 1, 'print_every': 100, 'delete_parquet_after': True,
    }
    if fail_write:
        def failed_save(*args):
            raise OSError('disk full')
        monkeypatch.setattr(download, '_write_arrays', failed_save)
        with pytest.raises(OSError, match='disk full'):
            download.convert_local_parquet_split(source, 'train', config)
        assert parquet.is_file()
    else:
        download.convert_local_parquet_split(source, 'train', config)
        assert not parquet.exists()
        np.testing.assert_array_equal(
            np.load(tmp_path / 'aia/train/sample.npy'), image.reshape(7, 2, 2),
        )
        assert np.load(tmp_path / 'sxr/train/sample.npy').item() == pytest.approx(1e-6)
