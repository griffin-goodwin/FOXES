from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from training.train import (
    build_checkpoint_callback,
    get_four_class_macro_weights,
    get_resume_fit_kwargs,
    load_training_config,
    resolve_resume_checkpoint,
)


def test_power_inverse_frequency_weights_are_normalized(tmp_path):
    fluxes = [5e-7, 2e-6, 3e-6, 2e-5, 3e-5, 4e-5, 2e-4]
    for index, flux in enumerate(fluxes):
        np.save(tmp_path / f"{index}.npy", np.array(flux))
    data_module = SimpleNamespace(train_ds=SimpleNamespace(
        samples=[str(index) for index in range(len(fluxes))],
        sxr_dir=tmp_path,
    ))

    counts, weights = get_four_class_macro_weights(data_module, exponent=2 / 3)

    assert counts == {'quiet': 1, 'c_class': 2, 'm_class': 3, 'x_class': 1}
    assert sum(counts[key] * weights[key] for key in counts) / 7 == pytest.approx(1)
    assert weights['x_class'] / weights['m_class'] == pytest.approx(3 ** (2 / 3))


@pytest.mark.parametrize('exponent', [None, 0, 1.1, float('nan')])
def test_power_inverse_frequency_rejects_invalid_exponents(exponent):
    with pytest.raises(ValueError, match='exponent must be in'):
        get_four_class_macro_weights(None, exponent=exponent)


def test_resume_checkpoint_uses_config_path(tmp_path):
    checkpoint = tmp_path / "epoch=09.ckpt"
    checkpoint.touch()

    resolved = resolve_resume_checkpoint({"resume_from": str(checkpoint)})

    assert resolved == str(checkpoint.resolve())


def test_cli_resume_checkpoint_overrides_config(tmp_path):
    checkpoint = tmp_path / "override.ckpt"
    checkpoint.touch()

    resolved = resolve_resume_checkpoint(
        {"resume_from": "/missing/config.ckpt"}, str(checkpoint)
    )

    assert resolved == str(checkpoint.resolve())


def test_missing_resume_checkpoint_fails_before_training():
    with pytest.raises(FileNotFoundError, match="Resume checkpoint"):
        resolve_resume_checkpoint({"resume_from": "/missing/model.ckpt"})


def test_full_state_resume_disables_weights_only_loading():
    assert get_resume_fit_kwargs("/trusted/model.ckpt") == {
        "ckpt_path": "/trusted/model.ckpt",
        "weights_only": False,
    }
    assert get_resume_fit_kwargs(None) == {}


def test_checkpoint_callback_can_keep_all_and_save_last(tmp_path):
    config = {
        "data": {"checkpoints_dir": str(tmp_path)},
        "wandb": {"run_name": "recovery"},
    }
    checkpoint_config = {
        "monitor": "val_class/X/rmse_dex",
        "mode": "min",
        "save_top_k": -1,
        "save_last": True,
    }

    callback = build_checkpoint_callback(config, checkpoint_config)

    assert callback.dirpath == str(tmp_path)
    assert callback.monitor == "val_class/X/rmse_dex"
    assert callback.save_top_k == -1
    assert callback.save_last is True


def test_recovery_config_resumes_epoch_nine_and_keeps_every_epoch():
    config_path = (
        Path(__file__).parents[1]
        / "training/configs/recover_weighted_huber015_from_epoch09.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["epochs"] == 25
    assert config["checkpoint"]["save_top_k"] == -1
    assert config["checkpoint"]["save_last"] is True
    assert "epoch=09-step=006080.ckpt" in config["checkpoint"]["resume_from"]
    assert config["data"]["checkpoints_dir"].endswith(
        "-RECOVERY-FROM-EPOCH09"
    )


def test_patch_mean_experiment_configs_differ_only_in_parameterization_and_metadata():
    config_dir = (
        Path(__file__).parents[1]
        / "experiments"
        / "patch_mean_comparison"
    )
    original = load_training_config(config_dir / "original.yaml")
    multiplier = load_training_config(config_dir / "multiplier.yaml")
    positive = load_training_config(config_dir / "positive.yaml")

    assert original["mean_parameterization"] == "original"
    assert multiplier["mean_parameterization"] == "multiplier"
    assert positive["mean_parameterization"] == "positive"
    assert original["uncertainty"] == multiplier["uncertainty"]
    assert original["uncertainty"] == positive["uncertainty"]
    assert original["uncertainty"]["enabled"] is False

    for config in (original, multiplier, positive):
        config.pop("mean_parameterization")
        config["data"].pop("checkpoints_dir")
        config.pop("wandb")
    assert original == multiplier
    assert original == positive


def test_positive_contrast_followup_changes_only_prior_and_output_metadata():
    config_dir = (
        Path(__file__).parents[1]
        / "experiments"
        / "patch_mean_comparison"
    )
    positive = load_training_config(config_dir / "positive.yaml")
    contrast = load_training_config(config_dir / "positive_contrast.yaml")

    assert contrast["mean_parameterization"] == "positive"
    assert contrast["uncertainty"]["enabled"] is False
    assert contrast["contrast_prior"] == {
        "enabled": True,
        "weight": 0.001,
        "wavelengths": [94, 131],
        "minimum_contrast_sum": 0.000001,
    }
    assert contrast["callbacks"]["attention_enabled"] is True
    assert contrast["callbacks"]["spatial_uncertainty_enabled"] is False

    contrast.pop("contrast_prior")
    contrast["callbacks"] = positive["callbacks"]
    contrast["data"].pop("checkpoints_dir")
    positive["data"].pop("checkpoints_dir")
    contrast.pop("wandb")
    positive.pop("wandb")
    assert contrast == positive


def test_positive_contrast_10x_changes_only_weight_and_output_metadata():
    config_dir = (
        Path(__file__).parents[1]
        / "experiments"
        / "patch_mean_comparison"
    )
    baseline = load_training_config(config_dir / "positive_contrast.yaml")
    stronger = load_training_config(
        config_dir / "positive_contrast_10x.yaml"
    )

    assert baseline["contrast_prior"]["weight"] == 0.001
    assert stronger["contrast_prior"]["weight"] == 0.01
    assert stronger["mean_parameterization"] == "positive"
    assert stronger["uncertainty"]["enabled"] is False
    assert stronger["callbacks"]["attention_enabled"] is True

    for config in (baseline, stronger):
        config["contrast_prior"].pop("weight")
        config["data"].pop("checkpoints_dir")
        config.pop("wandb")
    assert baseline == stronger


def test_uncertainty_schedule_configs_differ_only_in_start_epoch_and_metadata():
    config_dir = (
        Path(__file__).parents[1]
        / "experiments"
        / "uncertainty_schedule"
    )
    start_zero = load_training_config(config_dir / "start_epoch_0.yaml")
    start_eight = load_training_config(config_dir / "start_epoch_8.yaml")

    assert start_zero["mean_parameterization"] == "original"
    assert start_eight["mean_parameterization"] == "original"
    assert start_zero["uncertainty"]["enabled"] is True
    assert start_eight["uncertainty"]["enabled"] is True
    assert start_zero["uncertainty"]["start_epoch"] == 0
    assert start_eight["uncertainty"]["start_epoch"] == 8
    assert start_zero["checkpoint"]["monitor"] == "val_class/macro_nll"
    assert start_eight["checkpoint"]["monitor"] == "val_class/macro_nll"
    for config in (start_zero, start_eight):
        assert config["callbacks"]["attention_enabled"] is True
        assert config["callbacks"]["attention_num_samples"] == 5
        assert config["callbacks"]["spatial_uncertainty_enabled"] is False

    for config in (start_zero, start_eight):
        config["uncertainty"].pop("start_epoch")
        config["data"].pop("checkpoints_dir")
        config.pop("wandb")
    assert start_zero == start_eight


def test_positive_uncertainty_schedule_keeps_positive_mean_and_attention():
    config_dir = (
        Path(__file__).parents[1]
        / "experiments"
        / "uncertainty_schedule"
    )
    start_zero = load_training_config(
        config_dir / "positive_start_epoch_0.yaml"
    )
    start_eight = load_training_config(
        config_dir / "positive_start_epoch_8.yaml"
    )

    for config, expected_start in ((start_zero, 0), (start_eight, 8)):
        assert config["mean_parameterization"] == "positive"
        assert config["uncertainty"]["enabled"] is True
        assert config["uncertainty"]["start_epoch"] == expected_start
        assert config["checkpoint"]["monitor"] == "val_class/macro_nll"
        assert config["callbacks"]["attention_enabled"] is True
        assert config["callbacks"]["attention_num_samples"] == 5
        assert config["callbacks"]["spatial_uncertainty_enabled"] is False

    for config in (start_zero, start_eight):
        config["uncertainty"].pop("start_epoch")
        config["data"].pop("checkpoints_dir")
        config.pop("wandb")
    assert start_zero == start_eight
