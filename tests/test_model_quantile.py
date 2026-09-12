import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from forecasting.model_quantile import QuantileViTLocal


def make_quantile_model(quantile_overrides=None):
    quantile_kwargs = {
        'loss_weighting': 'unweighted',
        'spatial_sparsity': {
            'weight': 0.0,
            'top_fraction': 0.25,
            'gate_center_flux': 5e-6,
            'gate_width_dex': 0.25,
        },
    }
    quantile_kwargs.update(quantile_overrides or {})
    return QuantileViTLocal(
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
        quantile_kwargs=quantile_kwargs,
    )


def test_patch_quantiles_are_ordered_and_sum_to_global_quantiles():
    model = make_quantile_model()
    images = torch.randn(3, 2, 2, 7)

    global_raw, quantiles_norm, patch_raw = model.predict_quantiles(images)
    prediction, public_global, patch_median, public_patch = model(images)

    assert global_raw.shape == (3, 5)
    assert quantiles_norm.shape == (3, 5)
    assert patch_raw.shape == (3, 4, 5)
    assert torch.all(torch.diff(patch_raw, dim=-1) > 0)
    assert torch.all(torch.diff(global_raw, dim=-1) > 0)
    torch.testing.assert_close(global_raw, patch_raw.sum(dim=1))
    torch.testing.assert_close(public_global, global_raw)
    torch.testing.assert_close(public_patch, patch_raw)
    torch.testing.assert_close(patch_median, patch_raw[:, :, 2])
    torch.testing.assert_close(prediction[:, 0], global_raw[:, 2])


def test_pinball_loss_averages_all_requested_quantiles():
    model = make_quantile_model()
    target = torch.tensor([0.0, 0.0])
    quantiles = torch.tensor([
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-1.0, -0.5, 0.0, 0.5, 1.0],
    ])

    losses = model._pinball_per_sample(quantiles, target)

    expected = torch.tensor([
        (0.05 + 0.16 + 0.0 + 0.16 + 0.05) / 5,
        (0.025 + 0.08 + 0.0 + 0.08 + 0.025) / 5,
    ])
    torch.testing.assert_close(losses, expected)


def test_five_class_pinball_uses_static_macro_weights():
    model = make_quantile_model({
        'loss_weighting': 'five_class',
        'five_class_weights': {
            'below_b': 0.5,
            'b_class': 1.0,
            'c_class': 2.0,
            'm_class': 4.0,
            'x_class': 8.0,
        },
    })
    target_raw = torch.tensor([9e-8, 1e-7, 1e-6, 1e-5, 1e-4])

    torch.testing.assert_close(
        model._loss_weights(target_raw),
        torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0]),
    )


def test_pinball_gradients_reach_median_and_spatial_width_heads():
    model = make_quantile_model()
    images = torch.randn(2, 2, 2, 7)
    _, quantiles_norm, _ = model.predict_quantiles(images)
    target = torch.tensor([-0.4, 0.8])

    model._pinball_per_sample(quantiles_norm, target).mean().backward()

    median_head = model.model.mlp_head[-1]
    width_head = model.model.patch_quantile_width_head[-1]
    assert torch.isfinite(median_head.weight.grad).all()
    assert median_head.weight.grad.abs().sum() > 0
    assert torch.isfinite(width_head.weight.grad).all()
    assert width_head.weight.grad.abs().sum() > 0


def test_quantile_model_uses_regular_cosine_annealing():
    model = make_quantile_model()
    configured = model.configure_optimizers()

    assert isinstance(
        configured['lr_scheduler']['scheduler'], CosineAnnealingLR
    )
    assert configured['lr_scheduler']['interval'] == 'epoch'
