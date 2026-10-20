"""Checks for five coupled patch quantiles trained with pinball loss."""

import numpy as np
import pytest
import torch

from forecasting.model import (
    CoupledPinballViTLocal, mean_class_signed_error_penalty, unnormalize_sxr,
)


ARCHITECTURE = dict(
    embed_dim=8, hidden_dim=16, num_channels=1, num_heads=1,
    num_layers=1, patch_size=1, num_patches=4, dropout=0.0,
    mask_mode='local', local_window=3, global_patch_adjustment=True,
)
SXR_NORM = np.array([-5.0, 1.0], dtype=np.float32)


def test_pinball_trains_five_local_and_global_quantile_heads():
    model = CoupledPinballViTLocal(ARCHITECTURE, SXR_NORM)
    assert model.model.mlp_head[-1].out_features == 5
    assert model.model.global_patch_head[-1].out_features == 5
    logged = {}
    model.log = lambda name, value, **kwargs: logged.update({name: value})
    images = torch.randn(3, 2, 2, 1)
    targets = torch.tensor([0.2, 0.4, 0.6])

    result = model.validation_step((images, targets), 0)
    raw_quantiles, quantile_predictions, patch_quantiles = model.forward(images)
    assert raw_quantiles.shape == (3, 5)
    assert patch_quantiles.shape == (3, 4, 5)
    torch.testing.assert_close(raw_quantiles, patch_quantiles.sum(dim=1))
    errors = targets[:, None] - quantile_predictions
    expected = torch.maximum(
        model.quantile_levels * errors,
        (model.quantile_levels - 1) * errors,
    ).mean()
    torch.testing.assert_close(logged['val/total_loss'], expected)
    torch.testing.assert_close(logged['val/pinball_loss'], expected)
    torch.testing.assert_close(result['loss'], expected.detach())

    logged['val/total_loss'].backward()
    assert model.model.mlp_head[-1].weight.grad.abs().sum() > 0
    assert model.model.global_patch_head[-1].weight.grad.abs().sum() > 0
    assert model.model.patch_uncertainty_head is None
    assert model.predict_quantiles(images).shape == (3, 5)


def test_quantile_heads_preserve_different_spatial_allocations():
    model = CoupledPinballViTLocal(ARCHITECTURE, SXR_NORM)
    relative_flux = torch.tensor([
        [1., 2., 3., 4., 5.],
        [5., 4., 3., 2., 1.],
        [1., 2., 3., 4., 5.],
        [1., 2., 3., 4., 5.],
    ])

    class FixedPatchHead(torch.nn.Module):
        def forward(self, embeddings):
            return relative_flux.log10().unsqueeze(0).expand(embeddings.shape[0], -1, -1)

    model.model.mlp_head = FixedPatchHead()
    raw, _, patches, local = model.model(
        torch.zeros(1, 2, 2, 1), model.sxr_norm, return_local=True,
    )
    expected = model.model._patch_flux_from_logits(
        relative_flux.log10().unsqueeze(0), model.sxr_norm,
    )
    torch.testing.assert_close(patches, expected)
    torch.testing.assert_close(local, expected)
    torch.testing.assert_close(raw, expected.sum(dim=1))
    # This patch contributes less to higher heads, while global totals increase.
    assert (patches[0, 1, 1:] < patches[0, 1, :-1]).all()
    assert (raw[:, 1:] > raw[:, :-1]).all()


def test_global_quantile_correction_changes_patch_flux():
    model = CoupledPinballViTLocal(ARCHITECTURE, SXR_NORM)
    images = torch.randn(2, 2, 2, 1)
    _, _, patches, local_patches = model.model(
        images, model.sxr_norm, return_local=True,
    )
    torch.testing.assert_close(patches, local_patches)
    with torch.no_grad():
        model.model.global_patch_head[-1].bias.fill_(0.5)
    _, _, corrected, local = model.model(
        images, model.sxr_norm, return_local=True,
    )
    assert not torch.allclose(corrected, local)


def test_optional_bias_penalty_adds_to_pinball_loss():
    model = CoupledPinballViTLocal(
        ARCHITECTURE, SXR_NORM,
        uncertainty_kwargs={'bias_penalty_weight': 0.7},
    )
    logged = {}
    model.log = lambda name, value, **kwargs: logged.update({name: value})
    images = torch.randn(3, 2, 2, 1)
    targets = torch.tensor([-1.0, 0.0, 1.0])

    result = model.validation_step((images, targets), 0)
    expected_bias = mean_class_signed_error_penalty(
        result['mean_norm'], targets,
        unnormalize_sxr(targets, model.sxr_norm), 0.7,
    )
    torch.testing.assert_close(logged['val/bias_penalty'], expected_bias)
    torch.testing.assert_close(
        result['loss'], logged['val/pinball_loss'] + expected_bias,
    )


@pytest.mark.parametrize('quantiles', [
    (), (0.1, 0.2, 0.5), (0.1, 0.2, 0.5, 0.5, 0.9),
    (0.1, 0.8, 0.5, 0.9, 0.95), (0.1, 0.2, 0.3, 0.8, 0.9),
    (0, 0.2, 0.5, 0.8, 0.9),
])
def test_invalid_quantiles_rejected(quantiles):
    with pytest.raises(ValueError, match='quantiles'):
        CoupledPinballViTLocal(ARCHITECTURE, SXR_NORM, quantiles=quantiles)


def test_pinball_checkpoint_restores_quantiles(tmp_path):
    model = CoupledPinballViTLocal(
        ARCHITECTURE, SXR_NORM, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    )
    import pytorch_lightning as pl
    path = tmp_path / 'pinball.ckpt'
    torch.save({
        'pytorch-lightning_version': pl.__version__,
        'hyper_parameters': dict(model.hparams),
        'state_dict': model.state_dict(),
    }, path)
    restored = CoupledPinballViTLocal.load_from_checkpoint(
        path, map_location='cpu', weights_only=False,
    )
    assert restored.quantiles == (0.1, 0.25, 0.5, 0.75, 0.9)
    assert restored.model.state_dict().keys() == model.model.state_dict().keys()
