"""Context-conditioned uncertainty heads and head-only fine-tuning."""

from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from forecasting.dataset import AIAGOESDataModule
from forecasting.inference import _auto_checkpoint_model_class
from forecasting.model import CoupledGaussianViTLocal, CoupledVisionTransformerLocal
from training.train import (
    build_checkpoint_callback, configure_class_weights, initialize_from_checkpoint,
    load_training_config, resolve_finetune_config,
)


ARCHITECTURE = dict(
    embed_dim=8, hidden_dim=16, num_channels=1, num_heads=2,
    num_layers=1, patch_size=1, num_patches=4, dropout=0.5,
    mask_mode='local', local_window=1, global_patch_adjustment=True,
)
SXR_NORM = torch.tensor([-6.0, 1.0])
IMAGES = torch.randn(4, 2, 2, 1, generator=torch.Generator().manual_seed(0))


def build(uncertainty_input='logit', **architecture):
    return CoupledGaussianViTLocal(
        model_kwargs={**ARCHITECTURE, **architecture}, sxr_norm=SXR_NORM,
        uncertainty_kwargs={
            'uncertainty_hidden_dim': 8, 'uncertainty_input': uncertainty_input,
        },
    )


def save_checkpoint(model, path):
    torch.save({
        'pytorch-lightning_version': pl.__version__,
        'hyper_parameters': dict(model.hparams),
        'state_dict': model.state_dict(),
    }, path)
    return path


def mean_state(model):
    prefixes = tuple(
        f'model.{name}.'
        for name in CoupledVisionTransformerLocal.UNCERTAINTY_MODULE_NAMES
    )
    return {
        key: value.detach().clone() for key, value in model.state_dict().items()
        if not key.startswith(prefixes)
    }


class CheckFrozenTraining(pl.Callback):
    batches = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        head_modules = {
            id(module) for head in pl_module.model.uncertainty_modules()
            for module in head.modules()
        }
        for name, module in pl_module.model.named_modules():
            assert module.training == (id(module) in head_modules), name
        self.batches += 1

    def on_after_backward(self, trainer, pl_module):
        head_parameters = {
            id(p) for head in pl_module.model.uncertainty_modules()
            for p in head.parameters()
        }
        for name, parameter in pl_module.named_parameters():
            if id(parameter) in head_parameters:
                assert parameter.grad is not None, name
            else:
                assert not parameter.requires_grad and parameter.grad is None, name


@pytest.mark.parametrize('global_patch_adjustment', [False, True])
def test_context_head_starts_at_initial_relative_std(global_patch_adjustment):
    torch.manual_seed(0)
    model = build('context', global_patch_adjustment=global_patch_adjustment).eval()
    context_dim = 8 * (3 if global_patch_adjustment else 1)
    assert model.model.patch_uncertainty_head[0].in_features == context_dim + 1
    with torch.no_grad():
        if global_patch_adjustment:
            model.model.global_patch_head[-1].bias.fill_(0.5)
        _, _, patches, patch_variance = model(IMAGES)
    torch.testing.assert_close(patch_variance, patches.square())


def test_logit_head_keeps_released_state_dict_layout():
    state = build('logit').state_dict()
    assert not any('patch_uncertainty_context_norm' in key for key in state)
    assert build('logit').model.patch_uncertainty_head[0].in_features == 1


def test_bundled_config_defaults_to_released_head_without_finetuning():
    config = load_training_config(
        Path(__file__).resolve().parents[1] / 'training/train_coupled_gaussian.yaml'
    )
    assert config['uncertainty']['uncertainty_input'] == 'logit'
    assert resolve_finetune_config(config, 'coupled_uncertainty') == (
        None, False, False, None,
    )


@pytest.mark.parametrize('global_patch_adjustment', [False, True])
def test_train_only_uncertainty_head_from_checkpoint(tmp_path, global_patch_adjustment):
    torch.manual_seed(0)
    source = build('logit', global_patch_adjustment=global_patch_adjustment)
    if global_patch_adjustment:
        with torch.no_grad():
            source.model.global_patch_head[-1].bias.fill_(0.5)
    path = save_checkpoint(source, tmp_path / 'source.ckpt')

    model = build('context', global_patch_adjustment=global_patch_adjustment)
    loaded_head = initialize_from_checkpoint(
        model, path, {**ARCHITECTURE, 'global_patch_adjustment': global_patch_adjustment},
    )
    assert not loaded_head
    model.freeze_all_but_uncertainty_head()
    before = mean_state(model)
    initial_head_bias = model.model.patch_uncertainty_head[-1].bias.detach().clone()

    check = CheckFrozenTraining()
    loader = DataLoader(TensorDataset(IMAGES, torch.full((4,), 2.0)), batch_size=2)
    trainer = pl.Trainer(
        accelerator='cpu', devices=1, max_epochs=2,
        limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0,
        logger=pl.loggers.CSVLogger(tmp_path, name='logs'), log_every_n_steps=1,
        callbacks=[check], enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=tmp_path,
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert check.batches == 2
    torch.testing.assert_close(
        trainer.callback_metrics['val/total_loss'],
        trainer.callback_metrics['val/uncertainty_nll'],
    )

    after = mean_state(model)
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    assert not torch.equal(
        model.model.patch_uncertainty_head[-1].bias, initial_head_bias,
    )
    model.eval()
    source.eval()
    with torch.no_grad():
        flux, _, patches, _ = model(IMAGES)
        source_flux, _, source_patches, _ = source(IMAGES)
    torch.testing.assert_close(flux, source_flux, rtol=0, atol=0)
    torch.testing.assert_close(patches, source_patches, rtol=0, atol=0)

    tuned = tmp_path / 'tuned.ckpt'
    trainer.save_checkpoint(tuned)
    checkpoint = torch.load(tuned, map_location='cpu', weights_only=False)
    assert _auto_checkpoint_model_class(checkpoint) is CoupledGaussianViTLocal
    restored = CoupledGaussianViTLocal.load_from_checkpoint(
        tuned, map_location='cpu', weights_only=False, strict=True,
    ).eval()
    assert restored.model.uncertainty_input == 'context'
    with torch.no_grad():
        for actual, expected in zip(restored(IMAGES), model(IMAGES)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('reset_head', [False, True])
@pytest.mark.parametrize('uncertainty_input', ['logit', 'context'])
@pytest.mark.parametrize('global_patch_adjustment', [False, True])
def test_matching_head_can_load_or_start_fresh(
    tmp_path, reset_head, uncertainty_input, global_patch_adjustment,
):
    torch.manual_seed(0)
    architecture = {**ARCHITECTURE, 'global_patch_adjustment': global_patch_adjustment}
    source = build(uncertainty_input, global_patch_adjustment=global_patch_adjustment)
    with torch.no_grad():
        for module in source.model.uncertainty_modules():
            for parameter in module.parameters():
                parameter.fill_(0.25)
    path = save_checkpoint(source, tmp_path / 'source.ckpt')
    model = build(uncertainty_input, global_patch_adjustment=global_patch_adjustment)
    fresh_state = {key: value.clone() for key, value in model.state_dict().items()}
    loaded = initialize_from_checkpoint(
        model, path, architecture, reset_uncertainty_head=reset_head,
    )
    assert loaded is not reset_head
    expected_mean = mean_state(source)
    expected_head = fresh_state if reset_head else source.state_dict()
    for key, actual in model.state_dict().items():
        expected = expected_mean[key] if key in expected_mean else expected_head[key]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_finetune_rejects_mismatched_architecture_and_normalization(tmp_path):
    path = save_checkpoint(build('logit'), tmp_path / 'source.ckpt')
    with pytest.raises(ValueError, match='mask_mode'):
        initialize_from_checkpoint(
            build('logit', mask_mode='inverted'), path,
            {**ARCHITECTURE, 'mask_mode': 'inverted'},
        )
    other_norm = CoupledGaussianViTLocal(
        model_kwargs=ARCHITECTURE, sxr_norm=torch.tensor([-5.0, 1.0]),
        uncertainty_kwargs={'uncertainty_hidden_dim': 8},
    )
    with pytest.raises(ValueError, match='sxr_norm_path'):
        initialize_from_checkpoint(other_norm, path, ARCHITECTURE)
    # Learning rate and dropout are training choices, so they may differ.
    initialize_from_checkpoint(
        build('logit', dropout=0.0, learning_rate=1e-3), path,
        {**ARCHITECTURE, 'dropout': 0.0, 'learning_rate': 1e-3},
    )


def test_finetune_config_validation():
    assert resolve_finetune_config({
        'finetune': {
            'checkpoint_path': 'x.ckpt', 'train_only_uncertainty_head': True,
            'reset_uncertainty_head': True,
        },
    }, 'coupled_uncertainty') == ('x.ckpt', True, True, None)
    assert resolve_finetune_config({
        'finetune': {
            'checkpoint_path': 'x.ckpt', 'train_only_uncertainty_head': True,
            'val_fit_fraction': 0.8,
        },
    }, 'coupled_uncertainty') == ('x.ckpt', True, False, 0.8)
    with pytest.raises(ValueError, match='early_stopping needs validation'):
        resolve_finetune_config({
            'finetune': {
                'checkpoint_path': 'x.ckpt', 'train_only_uncertainty_head': True,
                'val_fit_fraction': 1,
            },
            'early_stopping': {'enabled': True},
        }, 'coupled_uncertainty')
    with pytest.raises(ValueError, match='requires finetune.train_only'):
        resolve_finetune_config({
            'finetune': {'checkpoint_path': 'x.ckpt', 'val_fit_fraction': 0.8},
        }, 'coupled_uncertainty')
    for fraction in (0, 1.5, True, '0.8'):
        with pytest.raises(ValueError, match='at most 1'):
            resolve_finetune_config({
                'finetune': {
                    'checkpoint_path': 'x.ckpt',
                    'train_only_uncertainty_head': True,
                    'val_fit_fraction': fraction,
                },
            }, 'coupled_uncertainty')
    with pytest.raises(ValueError, match='requires finetune.checkpoint_path'):
        resolve_finetune_config(
            {'finetune': {'train_only_uncertainty_head': True}}, 'coupled_uncertainty',
        )
    with pytest.raises(ValueError, match='coupled_uncertainty'):
        resolve_finetune_config(
            {'finetune': {'checkpoint_path': 'x.ckpt'}}, 'deterministic',
        )
    with pytest.raises(ValueError, match='Unsupported finetune'):
        resolve_finetune_config({'finetune': {'path': 'x.ckpt'}}, 'coupled_uncertainty')
    with pytest.raises(ValueError, match='reset_uncertainty_head requires'):
        resolve_finetune_config(
            {'finetune': {'reset_uncertainty_head': True}}, 'coupled_uncertainty',
        )
    with pytest.raises(ValueError, match='reset_uncertainty_head must be a boolean'):
        resolve_finetune_config({
            'finetune': {'checkpoint_path': 'x.ckpt', 'reset_uncertainty_head': 'true'},
        }, 'coupled_uncertainty')
    with pytest.raises(ValueError, match='start_epoch'):
        CoupledGaussianViTLocal(
            model_kwargs=ARCHITECTURE, sxr_norm=SXR_NORM,
            uncertainty_kwargs={'start_epoch': 1},
        ).freeze_all_but_uncertainty_head()
    with pytest.raises(ValueError, match='uncertainty.enabled'):
        CoupledGaussianViTLocal(
            model_kwargs=ARCHITECTURE, sxr_norm=SXR_NORM,
            uncertainty_kwargs={'enabled': False},
        ).freeze_all_but_uncertainty_head()
    with pytest.raises(ValueError, match='uncertainty_input'):
        build('embedding')


def write_split(root, split, stamps, flux=1e-7):
    for kind in ('aia', 'sxr'):
        (root / kind / split).mkdir(parents=True)
    for stamp in stamps:
        np.save(root / 'aia' / split / f'{stamp}.npy', np.ones((7, 2, 2), np.float32))
        np.save(root / 'sxr' / split / f'{stamp}.npy', np.array(flux))


def test_val_fit_fraction_splits_val_in_time_without_reading_train(tmp_path):
    stamps = [f'2021-01-01T00_{minute:02d}_00' for minute in range(10)]
    # Write the val files out of order; the split must still follow time.
    write_split(tmp_path, 'val', stamps[::-1])
    write_split(tmp_path, 'test', stamps[:1])
    dirs = {
        f'{kind}_{split}_dir': str(tmp_path / kind / split)
        for kind in ('aia', 'sxr') for split in ('train', 'val', 'test')
    }
    data = AIAGOESDataModule(**dirs, sxr_norm=SXR_NORM.numpy(), val_fit_fraction=0.8)
    data.setup()
    assert data.train_ds.samples == stamps[:8]
    assert data.val_ds.samples == stamps[8:]
    val_dir = (tmp_path / 'sxr' / 'val').resolve()
    assert data.train_ds.sxr_dir == data.val_ds.sxr_dir == val_dir
    # Lightning calls setup again inside fit.
    data.setup()
    assert data.val_ds.samples == stamps[8:]

    # A frozen mean loss does not need every flare class in the fit split.
    config = {
        'mean_loss': {'class_weighting': 'inverse_frequency'}, 'uncertainty': {},
    }
    with pytest.raises(ValueError, match='empty bins'):
        configure_class_weights(dict(config), data, 'coupled_uncertainty')
    _, uncertainty = configure_class_weights(
        dict(config), data, 'coupled_uncertainty', mean_loss_frozen=True,
    )
    assert set(uncertainty['class_weights']) == {
        'quiet', 'c_class', 'm_class', 'x_class',
    }

    with pytest.raises(ValueError, match='both parts must be non-empty'):
        AIAGOESDataModule(
            **dirs, sxr_norm=SXR_NORM.numpy(), val_fit_fraction=0.01,
        ).setup()


def test_val_fit_fraction_one_trains_on_all_val_without_validation(tmp_path):
    stamps = [f'2021-01-01T00_{minute:02d}_00' for minute in range(4)]
    write_split(tmp_path, 'val', stamps)
    write_split(tmp_path, 'test', stamps[:1])
    dirs = {
        f'{kind}_{split}_dir': str(tmp_path / kind / split)
        for kind in ('aia', 'sxr') for split in ('train', 'val', 'test')
    }
    data = AIAGOESDataModule(**dirs, sxr_norm=SXR_NORM.numpy(), val_fit_fraction=1)
    data.setup()
    assert data.train_ds.samples == stamps
    assert data.val_ds.samples == []

    config = {
        'model_type': 'coupled_uncertainty', 'wandb': {'run_name': 'head'},
        'data': {'checkpoints_dir': str(tmp_path / 'checkpoints')},
    }
    checkpoint_cfg = {
        'include_monitor_in_filename': True, 'save_top_k': 2, 'save_last': True,
    }
    checkpoint = build_checkpoint_callback(config, checkpoint_cfg, validate=False)
    assert checkpoint.monitor is None
    torch.manual_seed(0)
    model = build('logit')
    loader = DataLoader(TensorDataset(IMAGES, torch.full((4,), 2.0)), batch_size=2)
    trainer = pl.Trainer(
        accelerator='cpu', devices=1, max_epochs=3, limit_train_batches=1,
        limit_val_batches=0, logger=False, callbacks=[checkpoint],
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(
        model, train_dataloaders=loader,
        val_dataloaders=DataLoader(TensorDataset(IMAGES[:0], torch.zeros(0))),
    )
    saved = sorted(path.name for path in (tmp_path / 'checkpoints').iterdir())
    assert saved == [
        'head-epoch=00-step=000001.ckpt', 'head-epoch=01-step=000002.ckpt',
        'head-epoch=02-step=000003.ckpt', 'last.ckpt',
    ]
