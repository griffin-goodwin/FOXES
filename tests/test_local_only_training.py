"""Retraining supports configurable patch masking and an optional global branch."""

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from forecasting.inference import _auto_checkpoint_model_class
from forecasting.model import CoupledGaussianViTLocal, CoupledVisionTransformerLocal
from training.train import load_training_config


ARCHITECTURE = dict(
    embed_dim=8, hidden_dim=16, num_channels=1, num_heads=2,
    num_layers=1, patch_size=1, num_patches=4, dropout=0.0,
    mask_mode='local', local_window=1,
)
SXR_NORM = torch.tensor([-6.0, 1.0])


class CheckTrainingGradients(pl.Callback):
    backward_steps = 0

    def on_after_backward(self, trainer, pl_module):
        missing = [
            name for name, parameter in pl_module.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert not missing, f'Trainable parameters without gradients: {missing}'
        assert all(torch.isfinite(p.grad).all() for p in pl_module.parameters())
        self.backward_steps += 1


@pytest.mark.parametrize('uncertainty_enabled', [False, True])
@pytest.mark.parametrize('return_attention', [False, True])
@pytest.mark.parametrize('mask_mode', ['local', 'inverted', 'none'])
def test_local_only_outputs_and_architecture(
    uncertainty_enabled, return_attention, mask_mode,
):
    network = CoupledVisionTransformerLocal(
        **dict(ARCHITECTURE, mask_mode=mask_mode), global_patch_adjustment=False,
        uncertainty_enabled=uncertainty_enabled,
    )
    assert not any(name.startswith('global_') for name, _ in network.named_modules())
    assert not any(name.startswith('global_') for name in network.state_dict())
    assert not hasattr(network, 'global_query')
    expected_mask = None
    if mask_mode != 'none':
        expected_mask = torch.eye(ARCHITECTURE['num_patches'], dtype=torch.bool)
        if mask_mode == 'local':
            expected_mask = ~expected_mask
    for block in network.transformer_blocks:
        assert block.mask_mode == mask_mode
        if expected_mask is None:
            assert block.attention_mask is None
        else:
            torch.testing.assert_close(block.attention_mask, expected_mask)
    images = torch.randn(2, 2, 2, 1)

    for training in (True, False):
        network.train(training)
        embeddings, _ = network._encode_patch_embeddings(images, return_attention)
        expected = network._patch_flux_from_logits(
            network.mlp_head(embeddings).squeeze(-1), SXR_NORM,
        )
        outputs = network(images, SXR_NORM, return_attention=return_attention)
        patches = outputs[-2] if uncertainty_enabled else outputs[-1]
        assert torch.isfinite(outputs[0]).all()
        torch.testing.assert_close(patches, expected, rtol=0, atol=0)
        torch.testing.assert_close(outputs[0], patches.sum(dim=1, keepdim=True))
        assert torch.count_nonzero(network.last_log10_global_patch_adjustment) == 0
        if uncertainty_enabled:
            assert torch.isfinite(outputs[2]).all()
            assert (outputs[-1] > 0).all()
        if return_attention:
            attention = outputs[3] if uncertainty_enabled else outputs[2]
            assert len(attention) == ARCHITECTURE['num_layers']
            for weights in attention:
                assert torch.isfinite(weights).all()
                if expected_mask is not None:
                    assert torch.count_nonzero(weights[..., expected_mask]) == 0


@pytest.mark.parametrize('uncertainty_enabled', [False, True])
@pytest.mark.parametrize('global_patch_adjustment', [False, True])
@pytest.mark.parametrize('mask_mode', ['local', 'inverted', 'none'])
def test_config_training_and_checkpoint(
    tmp_path, uncertainty_enabled, global_patch_adjustment, mask_mode,
):
    torch.manual_seed(0)
    config_path = tmp_path / 'local_only.yaml'
    config_path.write_text(yaml.safe_dump({
        'base_config': str((
            Path(__file__).resolve().parents[1]
            / 'training/train_coupled_gaussian.yaml'
        )),
        'vit_architecture': {
            **ARCHITECTURE, 'global_patch_adjustment': global_patch_adjustment,
            'mask_mode': mask_mode,
        },
        'uncertainty': {'enabled': uncertainty_enabled, 'uncertainty_hidden_dim': 8},
    }))
    config = load_training_config(config_path)
    model = CoupledGaussianViTLocal(
        model_kwargs=config['vit_architecture'], sxr_norm=SXR_NORM,
        uncertainty_kwargs=config['uncertainty'],
    )
    initial_local_weight = model.model.mlp_head[-1].weight.detach().clone()
    if uncertainty_enabled:
        initial_uncertainty_bias = (
            model.model.patch_uncertainty_head[-1].bias.detach().clone()
        )
    images = torch.randn(4, 2, 2, 1)
    loader = DataLoader(TensorDataset(images, torch.full((4,), 2.0)), batch_size=2)
    gradient_check = CheckTrainingGradients()
    trainer = pl.Trainer(
        accelerator='cpu', devices=1, max_epochs=1,
        limit_train_batches=2, limit_val_batches=1, num_sanity_val_steps=0,
        logger=pl.loggers.CSVLogger(tmp_path, name='logs'), log_every_n_steps=1,
        callbacks=[gradient_check],
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=tmp_path,
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.global_step == 2
    assert gradient_check.backward_steps == 2
    assert not torch.equal(model.model.mlp_head[-1].weight, initial_local_weight)
    if uncertainty_enabled:
        assert not torch.equal(
            model.model.patch_uncertainty_head[-1].bias, initial_uncertainty_bias,
        )
    assert any(
        name.startswith('model.global_') for name in model.state_dict()
    ) == global_patch_adjustment
    if not global_patch_adjustment:
        assert trainer.callback_metrics['val/global_patch_adjustment_dex'] == 0

    path = tmp_path / 'local_only.ckpt'
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert (
        checkpoint['hyper_parameters']['model_kwargs']['global_patch_adjustment']
        is global_patch_adjustment
    )
    assert checkpoint['hyper_parameters']['model_kwargs']['mask_mode'] == mask_mode
    assert _auto_checkpoint_model_class(checkpoint) is CoupledGaussianViTLocal
    restored = CoupledGaussianViTLocal.load_from_checkpoint(
        path, map_location='cpu', weights_only=False, strict=True,
    )
    assert restored.model.global_patch_adjustment is global_patch_adjustment
    assert all(
        block.mask_mode == mask_mode for block in restored.model.transformer_blocks
    )
    assert set(restored.state_dict()) == set(model.state_dict())
    model.eval()
    restored.eval()
    with torch.no_grad():
        for actual, expected in zip(restored(images), model(images)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_global_head_remains_enabled_by_default():
    torch.manual_seed(0)
    default = CoupledVisionTransformerLocal(**ARCHITECTURE)
    torch.manual_seed(0)
    explicit = CoupledVisionTransformerLocal(
        **ARCHITECTURE, global_patch_adjustment=True,
    )
    assert default.global_patch_adjustment
    assert 'global_query' in default.state_dict()
    with torch.no_grad():
        default.global_patch_head[-1].bias.fill_(0.5)
        explicit.global_patch_head[-1].bias.fill_(0.5)
        images = torch.randn(2, 2, 2, 1)
        for actual, expected in zip(
            default(images, SXR_NORM), explicit(images, SXR_NORM),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert default.last_log10_global_patch_adjustment.abs().sum() > 0


@pytest.mark.parametrize('value', [None, 0, 1, 'false'])
def test_global_patch_adjustment_requires_boolean(value):
    with pytest.raises(ValueError, match='global_patch_adjustment must be a boolean'):
        CoupledVisionTransformerLocal(**ARCHITECTURE, global_patch_adjustment=value)
