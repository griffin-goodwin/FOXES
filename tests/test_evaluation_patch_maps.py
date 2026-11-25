"""Patch-map plotting uses inference timestamps and tolerates partial outputs."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from forecasting.evaluation import FOXESEvaluator
from data import _timestamps


@pytest.mark.parametrize('windows', [False, True])
def test_patch_maps_match_csv_timestamps_with_optional_aia(tmp_path, monkeypatch, windows):
    if _timestamps._WINDOWS and not windows:
        pytest.skip('Windows cannot create legacy colon-separated filenames')
    monkeypatch.setattr(_timestamps, '_WINDOWS', windows)
    patches = tmp_path / 'inference_patch'
    flux_dir = patches / 'flux'
    uncertainty_dir = patches / 'patch_uncertainty'
    aia_dir = tmp_path / 'aia'
    for directory in (flux_dir, uncertainty_dir, aia_dir):
        directory.mkdir(parents=True)

    first = '2023-08-04T15:30:00'
    second = '2023-08-06T05:46:00'
    first_name = '2023-08-04T153000.npy' if windows else f'{first}.npy'
    second_name = '2023-08-06T054600.npy' if windows else f'{second}.npy'
    np.save(flux_dir / first_name, np.ones((2, 2)) * 1e-7)
    np.save(uncertainty_dir / first_name, np.ones((2, 2)) * 1e-8)
    np.save(aia_dir / first_name, np.zeros((7, 16, 16)))
    np.save(flux_dir / second_name, np.ones((2, 2)) * 2e-3)
    np.save(flux_dir / 'unrelated.npy', np.ones((2, 2)))

    csv_path = tmp_path / 'predictions.csv'
    pd.DataFrame({
        'timestamp': [first, second],
        'predictions': [4e-7, 8e-7],
        'groundtruth': [5e-7, 9e-7],
    }).to_csv(csv_path, index=False)
    evaluator = FOXESEvaluator(
        csv_path, tmp_path / 'evaluation',
        inference_patch_dir=patches, aia_dir=aia_dir,
    )
    evaluator.load_data()

    plotted = []
    original_imshow = Axes.imshow

    def record_imshow(self, image, *args, **kwargs):
        plotted.append(kwargs)
        return original_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(Axes, 'imshow', record_imshow)

    assert evaluator.plot_patch_maps() == 2
    assert all(call['origin'] == 'lower' for call in plotted)
    patch_norms = [call['norm'] for call in plotted if 'norm' in call]
    assert len(patch_norms) == 3
    assert all(call['cmap'] == 'inferno' for call in plotted if 'norm' in call)
    assert all(isinstance(norm, LogNorm) for norm in patch_norms)
    assert all(np.isclose(norm.vmin, 1e-8, rtol=1e-12, atol=0)
               and np.isclose(norm.vmax, 2e-3, rtol=1e-12, atol=0)
               for norm in patch_norms)
    figures = sorted((tmp_path / 'evaluation' / 'plots' / 'patch_maps').glob('*.png'))
    assert [path.name for path in figures] == [
        '2023-08-04T15_30_00.png', '2023-08-06T05_46_00.png',
    ]
    assert all(path.stat().st_size > 0 for path in figures)
