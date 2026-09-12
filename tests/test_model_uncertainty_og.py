import numpy as np
import pytest
import torch
import torch.nn.functional as F

from forecasting.model import SXR_LOG_OFFSET
from forecasting.model_uncertainty_og import OGGaussianNLLViTLocal


def make_model():
    return OGGaussianNLLViTLocal(
        model_kwargs={
            'embed_dim': 8,
            'hidden_dim': 16,
            'num_channels': 7,
            'num_heads': 1,
            'num_layers': 1,
            'patch_size': 1,
            'num_patches': 4,
            'dropout': 0.0,
            'mask_mode': 'none',
        },
        sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
        uncertainty_kwargs={
            'relative_std_floor': 0.0025,
            'relative_std_max': 20.0,
            'patch_scale_floor_fraction': 1.0,
        },
    )


def test_og_patch_mean_uses_normalized_sxr_inverse_and_zero_clamp():
    model = make_model()
    logits = torch.tensor([[-3.0, -2.0, 0.0, 1.0]])

    actual = model.model._patch_flux_from_logits(logits, model.sxr_norm)
    expected = torch.clamp(
        10 ** (logits * model.sxr_norm[1] + model.sxr_norm[0])
        - SXR_LOG_OFFSET,
        min=0.0,
        max=1.0,
    )

    torch.testing.assert_close(actual, expected)
    assert actual[0, 0] == 0
    assert actual[0, 2] > 0


def test_og_global_prediction_still_sums_patch_flux_and_predicts_variance():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)

    raw, _, variance, patches, patch_variance = model.model(
        images, model.sxr_norm
    )

    torch.testing.assert_close(raw.squeeze(-1), patches.sum(dim=1))
    assert torch.all(variance > 0)
    assert torch.all(patch_variance > 0)


def test_og_mean_head_uses_original_nonzero_linear_initialization():
    model = make_model()

    assert torch.count_nonzero(model.model.mlp_head[-1].weight) > 0


def test_og_variance_path_remains_detached_from_mean_and_backbone():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)
    _, mean, variance, _, _ = model.model(images, model.sxr_norm)

    variance_loss = F.gaussian_nll_loss(
        mean.detach(), mean.detach(), variance, full=True
    )
    variance_loss.backward()

    uncertainty_output = model.model.patch_uncertainty_head[-1]
    assert uncertainty_output.bias.grad is not None
    assert uncertainty_output.bias.grad.abs().sum() > 0
    assert model.model.mlp_head[-1].weight.grad is None
    assert model.model.input_layer.weight.grad is None


def test_mean_parameterization_version_prevents_wrong_model_loading():
    model = make_model()
    checkpoint = {
        'state_dict': {
            'model.mean_parameterization_version': torch.tensor(2),
        },
    }

    with pytest.raises(RuntimeError, match='does not match'):
        model.on_load_checkpoint(checkpoint)
