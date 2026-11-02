"""Deterministic legacy scaling preserves additive flux and UTC alignment."""

from datetime import datetime, timezone
import sys

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset
import yaml

import helio_foxes
from forecasting.calibration import LegacyCalibration
from forecasting.inference import evaluate_model_on_dataset
from forecasting.model import CoupledGaussianViTLocal, ViTLocal


@pytest.fixture
def model():
    torch.manual_seed(42)
    # Each patch has 64 pixels: the legacy factor must still be 0.7, not 0.7/64.
    kwargs = dict(embed_dim=8, hidden_dim=16, num_channels=7, num_heads=2,
                  num_layers=1, patch_size=8, num_patches=4, dropout=0.0,
                  mask_mode="local", local_window=1)
    return ViTLocal(
        model_kwargs=kwargs, sxr_norm=np.array([-6, .7], dtype=np.float32),
    ).eval()


def predictors(model, monkeypatch, tmp_path, **options):
    monkeypatch.setattr("forecasting.inference.load_model_from_config", lambda config: model)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    return (
        helio_foxes.FOXESModel(checkpoint, device="cpu"),
        helio_foxes.FOXESModel(checkpoint, device="cpu", legacy_correction_enabled=True, **options),
    )


@pytest.mark.parametrize("timestamp,scale", [
    ("2016-11-18T23:59:59.999999Z", .7),
    ("2016-11-19T00:00:00Z", 1),
    ("2016-11-18T19:00:00-05:00", 1),
    ("2017-09-06T12:02:00Z", 1),
    ("2023-07-11T01:30:00Z", 1),
])
def test_public_prediction_scales_all_physical_outputs(model, monkeypatch, tmp_path, timestamp, scale):
    native, legacy = predictors(model, monkeypatch, tmp_path)
    norm = np.asarray(model.sxr_norm).copy()
    observation = helio_foxes.Observation.from_array(
        np.ones((7, 16, 16), dtype=np.float32), timestamp=timestamp,
        harp_regions=[{"HARPNUM": 1, "top_pct": 0, "left_pct": 0,
                       "width_pct": 100, "height_pct": 100}],
        sxr={"xrsb_flux": 1e-6},
    )
    original = native.predict(observation)
    corrected = legacy.predict(observation)
    repeated = legacy.predict(observation)
    assert corrected.legacy_correction_enabled is True
    assert corrected.flux_scale_factor == scale
    assert corrected.metadata is observation.metadata
    assert corrected.ground_truth == observation.sxr["xrsb_flux"] == 1e-6
    assert corrected.total_flux == pytest.approx(original.total_flux * scale)
    np.testing.assert_allclose(corrected.flux_map, original.flux_map * scale)
    assert corrected.flux_map.sum() == pytest.approx(corrected.total_flux)
    assert corrected.harp_contributions[0]["total_flux"] == pytest.approx(original.total_flux * scale)
    np.testing.assert_array_equal(repeated.flux_map, corrected.flux_map)
    np.testing.assert_array_equal(np.asarray(model.sxr_norm), norm)
    assert corrected.prediction_intervals is None
    assert corrected.flux_std_map is None
    frame = helio_foxes.PredictionTimeSeries((corrected,)).full_disk_light_curve()
    assert frame.iloc[0].legacy_correction_enabled
    assert frame.iloc[0].flux_scale_factor == scale
    assert frame.iloc[0].ground_truth == 1e-6
    assert frame.iloc[0].total_flux == corrected.total_flux


def test_custom_cutoff(model, monkeypatch, tmp_path):
    native, legacy = predictors(model, monkeypatch, tmp_path, legacy_cutoff="2019-01-01")
    image = np.ones((7, 16, 16), dtype=np.float32)

    def at(timestamp):
        return helio_foxes.Observation.from_array(image, timestamp=timestamp)

    assert native.predict(at("2018-12-31T23:59:59Z")).flux_scale_factor == 1
    assert legacy.predict(at("2018-12-31T23:59:59Z")).flux_scale_factor == .7
    assert legacy.predict(at("2019-01-01")).flux_scale_factor == 1


@pytest.mark.parametrize("bad", ["true", "false", 1])
def test_invalid_legacy_flag_rejected_before_loading(bad):
    with pytest.raises(ValueError, match="legacy_correction_enabled must be a boolean"):
        helio_foxes.FOXESModel("missing.ckpt", legacy_correction_enabled=bad)


def test_timestamp_stems_and_invalid_dates():
    calibration = LegacyCalibration(True)
    assert calibration.factor("2016-11-18T23_59_59.npy") == .7
    assert calibration.factor("2016-11-19T00_00_00") == 1
    assert calibration.cutoff == datetime(2016, 11, 19, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="ISO timestamp"):
        calibration.factor("not_a_timestamp")
    with pytest.raises(ValueError, match="ISO timestamp"):
        LegacyCalibration(True, "not_a_date")


def test_batch_exports_scale_maps_and_keep_attention(model, tmp_path):
    times = ["2015-01-01T00:00:00", "2016-11-19T00:00:00", "2023-01-01T00:00:00"]
    dataset = TensorDataset(torch.ones(3, 16, 16, 7), torch.full((3,), 1e-6))
    options = {"num_workers": 0, "pin_memory": False, "use_amp": False}
    outputs = []
    for legacy in (False, True):
        folder = tmp_path / str(legacy)
        outputs.append(list(evaluate_model_on_dataset(
            model, dataset, batch_size=2, times=times,
            config_data={**options, "legacy_model": legacy,
                         "flux_path": str(folder / "flux"),
                         "patch_uncertainty_path": str(folder / "uncertainty")},
            save_weights=True, input_size=16, patch_size=8,
            save_flux=True, save_patch_uncertainty=True,
        )))
    for index, (native, corrected) in enumerate(zip(*outputs)):
        scale = .7 if index == 0 else 1
        np.testing.assert_allclose(corrected[0], native[0] * scale)
        np.testing.assert_array_equal(corrected[1], native[1])
        np.testing.assert_allclose(corrected[2], native[2])
        np.testing.assert_allclose(corrected[3], native[3] * scale)
        assert corrected[3].sum() == pytest.approx(float(corrected[0].item()))
        np.testing.assert_allclose(
            np.load(tmp_path / "True" / "flux" / f"{times[index]}.npy"), corrected[3],
        )
        assert len(corrected) == 5
    assert not (tmp_path / "True" / "uncertainty").exists()
    with pytest.raises(ValueError, match="one timestamp per"):
        next(evaluate_model_on_dataset(model, dataset, times=None,
                                       config_data={**options, "legacy_model": True}))


def test_cli_csv_corrects_only_deterministic_predictions(model, monkeypatch, tmp_path):
    from forecasting import inference
    dataset = TensorDataset(torch.ones(2, 16, 16, 7), torch.full((2,), 1e-6))
    dataset.samples = ["2015-01-01T00:00:00", "2023-01-01T00:00:00"]
    monkeypatch.setattr(inference, "AIAGOESDataset", lambda **kwargs: dataset)
    monkeypatch.setattr(inference, "load_model_from_config", lambda config: model)
    output = tmp_path / "predictions.csv"
    config = {
        "data": {"aia_dir": "unused", "sxr_dir": "unused"},
        "model_params": {"input_size": 16, "patch_size": 8, "batch_size": 2,
                         "no_weights": True},
        "wavelengths": [94, 131, 171, 193, 211, 304, 335],
        "legacy_model": True, "num_workers": 0, "pin_memory": False,
        "output_path": str(output),
    }
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["inference", "--config", str(config_path)])
    inference.main()
    frame = pd.read_csv(output)
    np.testing.assert_allclose(frame.flux_scale_factor, [.7, 1])
    np.testing.assert_allclose(frame.groundtruth, [1e-6, 1e-6])
    assert frame.legacy_model.all()
    assert "variance_normalized" not in frame
    assert "groundtruth_normalized" not in frame


def test_checkpoint_loader_records_opt_in_without_modifying_checkpoint(model, tmp_path):
    from forecasting.inference import load_model_from_config
    path = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__}, path)
    before = path.read_bytes()
    loaded = load_model_from_config({"data": {"checkpoint_path": str(path)},
                                     "device": "cpu", "legacy_model": True})
    assert loaded.legacy_model is True
    assert loaded.legacy_calibration.factor("2015-01-01") == .7
    assert path.read_bytes() == before


@pytest.mark.parametrize('checkpoint_format', ['lightning', 'serialized'])
def test_gaussian_checkpoints_reject_legacy_correction(tmp_path, checkpoint_format):
    architecture = dict(embed_dim=8, hidden_dim=16, num_channels=7, num_heads=2,
                        num_layers=1, patch_size=1, num_patches=4, dropout=0.0,
                        mask_mode="local", local_window=1)
    gaussian = CoupledGaussianViTLocal(
        model_kwargs=architecture, sxr_norm=np.array([-6, .7], dtype=np.float32),
    ).eval()
    if checkpoint_format == 'lightning':
        checkpoint = tmp_path / 'gaussian.ckpt'
        torch.save({"state_dict": gaussian.state_dict(),
                    "hyper_parameters": dict(gaussian.hparams),
                    "pytorch-lightning_version": pl.__version__}, checkpoint)
    else:
        checkpoint = tmp_path / 'gaussian.pt'
        torch.save({'model': gaussian}, checkpoint)
    with pytest.raises(ValueError, match="only supported for deterministic"):
        helio_foxes.FOXESModel(checkpoint, device="cpu", legacy_correction_enabled=True)
    dataset = TensorDataset(torch.ones(1, 2, 2, 7), torch.tensor([1e-6]))
    with pytest.raises(ValueError, match="only supported for deterministic"):
        next(evaluate_model_on_dataset(
            gaussian, dataset, times=["2015-01-01"],
            config_data={"legacy_model": True, "num_workers": 0},
        ))


@pytest.mark.parametrize("checkpoint,legacy_model,expected", [
    ("uncertainty", None, False),
    ("inverted-attention-mask", None, True),
    ("localized-attention-mask", None, True),
    ("no-attention-mask", None, True),
    ("no-attention-mask", False, False),
    ("uncertainty", True, True),
])
def test_legacy_correction_defaults_by_checkpoint_name(model, monkeypatch, checkpoint, legacy_model, expected):
    monkeypatch.setattr("forecasting.inference.load_model_from_config", lambda config: model)
    monkeypatch.setattr("helio_foxes.api._resolve_checkpoint", lambda checkpoint: "model.ckpt")
    options = {} if legacy_model is None else {"legacy_correction_enabled": legacy_model}
    assert helio_foxes.FOXESModel(checkpoint, device="cpu", **options).legacy_correction_enabled is expected


def test_checkpoint_files_default_to_no_legacy_correction(model, monkeypatch, tmp_path):
    native, _ = predictors(model, monkeypatch, tmp_path)
    assert native.legacy_correction_enabled is False
