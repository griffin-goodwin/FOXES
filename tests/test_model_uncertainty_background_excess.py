import numpy as np
import pytest
import torch
import torch.nn.functional as F

from forecasting.model_uncertainty_background_excess import (
    BackgroundExcessGaussianNLLViTLocal,
)


def make_model(uncertainty_overrides=None, model_overrides=None):
    uncertainty_kwargs = {
        'patch_flux_scale_multiplier': 1.0,
        'max_abs_log10_patch_multiplier': 8.0,
        'relative_std_floor': 0.0025,
        'relative_std_max': 20.0,
        'patch_scale_floor_fraction': 1.0,
        'background_flux_max': 5e-6,
        'background_initial_cap_fraction': 0.5,
        'excess_initial_fraction': 0.2,
        'solar_disk_radius_fraction': 0.48,
        'background_scale_floor_fraction': 0.05,
        'background_uncertainty_initial_raw': -3.0,
    }
    uncertainty_kwargs.update(uncertainty_overrides or {})
    model_kwargs = {
        'embed_dim': 8,
        'hidden_dim': 16,
        'num_channels': 7,
        'num_heads': 1,
        'num_layers': 2,
        'patch_size': 1,
        'num_patches': 4,
        'dropout': 0.0,
        'mask_mode': 'local',
        'local_window': 3,
    }
    model_kwargs.update(model_overrides or {})
    return BackgroundExcessGaussianNLLViTLocal(
        model_kwargs=model_kwargs,
        sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
        uncertainty_kwargs=uncertainty_kwargs,
    )


def test_background_excess_mean_and_variance_are_exactly_additive():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)

    assert model.patch_uncertainty_semantics == 'variance_attribution'

    raw, _, variance_norm, patch_flux, patch_variance = model.model(
        images, model.sxr_norm
    )
    components = model.model.latest_mean_components()

    assert torch.all(components['background_flux_raw'] > 0)
    assert torch.all(components['background_flux_raw'] < 5e-6)
    assert torch.all(components['excess_patch_flux_raw'] >= 0)
    torch.testing.assert_close(
        components['background_patch_flux_raw'].sum(dim=1),
        components['background_flux_raw'],
    )
    torch.testing.assert_close(
        patch_flux,
        components['background_patch_flux_raw']
        + components['excess_patch_flux_raw'],
    )
    torch.testing.assert_close(raw.squeeze(-1), patch_flux.sum(dim=1))

    derivative = 1.0 / (
        torch.log(torch.tensor(10.0))
        * model.sxr_norm[1]
        * (raw.squeeze(-1) + 1e-8)
    )
    torch.testing.assert_close(
        variance_norm,
        patch_variance.sum(dim=1) * derivative.square(),
    )


def test_opt_in_components_are_explicit_and_preserve_public_tuple():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)

    assert len(model(images)) == 4
    assert len(model(images, return_attention=True)) == 5
    output = model(images, return_components=True)

    assert len(output) == 5
    global_flux, _, accounting_flux, variance_attribution, components = output
    assert set(components) == {
        'background_flux_raw',
        'background_cap_fraction',
        'excess_patch_flux_raw',
        'background_allocation_raw',
        'total_accounting_patch_flux_raw',
        'background_variance_raw',
        'excess_variance_contribution_raw',
        'total_variance_attribution_raw',
        'global_variance_raw',
    }
    assert all(value.shape[0] == images.shape[0]
               for value in components.values())
    torch.testing.assert_close(
        components['background_allocation_raw'].sum(dim=-1),
        components['background_flux_raw'],
    )
    torch.testing.assert_close(
        components['total_accounting_patch_flux_raw'],
        components['background_allocation_raw']
        + components['excess_patch_flux_raw'],
    )
    torch.testing.assert_close(
        accounting_flux,
        components['total_accounting_patch_flux_raw'],
    )
    torch.testing.assert_close(
        global_flux.squeeze(-1), accounting_flux.sum(dim=-1)
    )
    torch.testing.assert_close(
        components['global_variance_raw'],
        components['background_variance_raw']
        + components['excess_variance_contribution_raw'].sum(dim=-1),
    )
    torch.testing.assert_close(
        variance_attribution,
        components['total_variance_attribution_raw'],
    )
    torch.testing.assert_close(
        components['total_variance_attribution_raw'].sum(dim=-1),
        components['global_variance_raw'],
    )


def test_background_query_is_not_added_to_patch_attention_outputs():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)

    outputs = model.model(images, model.sxr_norm, return_attention=True)
    attention = outputs[3]

    assert len(attention) == 2
    assert all(weights.shape[-2:] == (4, 4) for weights in attention)


def test_off_limb_patches_can_predict_local_excess_and_uncertainty():
    model = make_model(model_overrides={'num_patches': 16})
    images = torch.randn(2, 4, 4, 7)

    _, _, _, _, patch_variance = model.model(images, model.sxr_norm)
    components = model.model.latest_mean_components()
    off_disk = ~model.model.solar_disk_patch_mask.bool().squeeze(0)

    assert torch.any(off_disk)
    assert torch.all(
        components['background_patch_flux_raw'][:, off_disk] == 0
    )
    assert torch.all(components['excess_patch_flux_raw'][:, off_disk] > 0)
    assert torch.all(patch_variance[:, off_disk] > 0)


def test_whole_image_background_query_reads_off_mask_tokens():
    torch.manual_seed(123)
    model = make_model(model_overrides={'num_patches': 16})
    embeddings = torch.randn(2, 16, 8, requires_grad=True)
    off_disk = ~model.model.solar_disk_patch_mask.bool().squeeze(0)
    with torch.no_grad():
        model.model.background_head[-1].weight.copy_(
            torch.arange(8, dtype=torch.float32).reshape(1, 8) / 8
        )

    background, _, _ = model.model._background_from_embeddings(embeddings)
    gradient = torch.autograd.grad(background.sum(), embeddings)[0]

    assert torch.any(off_disk)
    assert gradient[:, off_disk].abs().sum() > 0


def test_variance_only_gradients_do_not_train_either_mean_branch():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)
    _, mean_norm, variance_norm, _, _ = model.model(images, model.sxr_norm)

    target = mean_norm.detach()
    loss = F.gaussian_nll_loss(
        mean_norm.detach(), target, variance_norm, full=True
    )
    loss.backward()

    assert model.model.patch_uncertainty_head[-1].bias.grad is not None
    assert model.model.background_uncertainty_head[-1].bias.grad is not None
    assert model.model.mlp_head[-1].weight.grad is None
    assert model.model.background_head[-1].weight.grad is None
    assert model.model.background_attention.in_proj_weight.grad is None
    assert model.model.input_layer.weight.grad is None


def test_parameterization_version_rejects_standard_gaussian_checkpoint():
    model = make_model()
    checkpoint = {
        'state_dict': {
            'model.mean_parameterization_version': torch.tensor(2),
        },
    }

    with pytest.raises(RuntimeError, match='does not match'):
        model.on_load_checkpoint(checkpoint)
