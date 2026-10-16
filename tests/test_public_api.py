"""Checks for the importable UI-facing API."""

import numpy as np
import sys
import types
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

import foxes
from forecasting.model import CoupledGaussianViTLocal, ViTLocal


def test_data_processor_uses_temporary_files_and_returns_stack(monkeypatch):
    staged = []

    class Downloader:
        def __init__(self, base_path, **kwargs):
            self.base_path = base_path
            self.wavelengths = kwargs["wavelengths"]
            staged.append(base_path)

        def downloadDate(self, date):
            from pathlib import Path
            for wavelength in self.wavelengths:
                path = Path(self.base_path) / wavelength
                path.mkdir()
                (path / f"{date.isoformat(timespec='seconds')}.fits").touch()

    class Stacker:
        def __init__(self, data, wavelengths, resolution, **kwargs):
            assert all(len(files) == 1 for files in data)
            assert wavelengths == foxes.DataProcessor.WAVELENGTHS
            self.resolution = resolution

        def __getitem__(self, index):
            assert index == 0
            return np.ones((7, self.resolution, self.resolution), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "download.download_sdo", types.SimpleNamespace(SDODownloader=Downloader))
    monkeypatch.setitem(sys.modules, "data.convert_aia", types.SimpleNamespace(AIAStackDataset=Stacker))
    stack = foxes.DataProcessor(resolution=2).fetch_date("2023-08-01T00:00:00Z")
    assert stack.shape == (7, 2, 2)
    assert stack.dtype == np.float32
    assert not __import__("os").path.exists(staged[0])


def test_data_processor_rejects_partial_download_and_cleans_up(monkeypatch):
    staged = []

    class Downloader:
        def __init__(self, base_path, **kwargs):
            staged.append(base_path)

        def downloadDate(self, date):
            pass

    monkeypatch.setitem(sys.modules, "download.download_sdo", types.SimpleNamespace(SDODownloader=Downloader))
    monkeypatch.setitem(sys.modules, "data.convert_aia", types.SimpleNamespace(AIAStackDataset=object))
    with pytest.raises(FileNotFoundError, match="wavelengths"):
        foxes.DataProcessor(resolution=2).fetch_date("2023-08-01T00:00:00")
    assert not __import__("os").path.exists(staged[0])


@pytest.mark.parametrize("model_class", [ViTLocal, CoupledGaussianViTLocal])
@pytest.mark.parametrize("wrapped", [False, True])
def test_predict_matches_model_outputs(model_class, wrapped, monkeypatch, tmp_path):
    kwargs = dict(
        embed_dim=8, hidden_dim=16, num_channels=7,
        num_heads=2, num_layers=1, patch_size=1,
        num_patches=4, dropout=0.0, mask_mode="local", local_window=3,
    )
    if model_class is CoupledGaussianViTLocal:
        kwargs["global_patch_adjustment"] = True
    model = model_class(
        model_kwargs=kwargs, sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
        **({"uncertainty_kwargs": {"enabled": True}}
           if model_class is CoupledGaussianViTLocal else {}),
    ).eval()
    loaded_model = torch.nn.DataParallel(model) if wrapped else model
    monkeypatch.setattr(
        "forecasting.inference.load_model_from_config",
        lambda config: loaded_model,
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    predictor = foxes.load_model(checkpoint, device="cpu")
    assert predictor.model_kwargs == kwargs

    result = foxes.predict(np.zeros((7, 2, 2), dtype=np.float32), model=predictor)
    assert result.flux_map.shape == (2, 2)
    assert np.isclose(result.flux_map.sum(), result.flux, rtol=1e-5)
    if model_class is CoupledGaussianViTLocal:
        assert set(result.intervals) == {"68", "95"}
        assert result.patch_uncertainty_map.shape == (2, 2)
    else:
        assert result.intervals is None
        assert result.patch_uncertainty_map is None

    ax = foxes.plot_flux_map(result)
    assert ax.images[0].origin == "lower"
    plt.close(ax.figure)

    with pytest.raises(ValueError, match="shape"):
        foxes.predict(np.zeros((2, 2, 7)), model=predictor)


@pytest.mark.parametrize("plotter", [foxes.plot_flux_map, foxes.plot_uncertainty_map])
def test_plot_map_color_limits(plotter):
    values = np.array([[1e-9, 1e-8], [0, 1e-7]])
    ax = plotter(values)
    assert ax.images[0].norm.vmin == 1e-9
    assert ax.images[0].norm.vmax == 1e-7
    plt.close(ax.figure)

    ax = plotter(values, vmin=1e-10, vmax=1e-6)
    assert ax.images[0].norm.vmin == 1e-10
    assert ax.images[0].norm.vmax == 1e-6
    plt.close(ax.figure)

    with pytest.raises(ValueError, match="vmin and vmax"):
        plotter(values, vmin=0)
