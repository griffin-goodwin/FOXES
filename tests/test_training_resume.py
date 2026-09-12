from pathlib import Path

import pytest
import yaml

from training.train import (
    build_checkpoint_callback,
    get_resume_fit_kwargs,
    resolve_resume_checkpoint,
)


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
