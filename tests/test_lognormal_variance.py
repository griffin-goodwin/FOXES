"""The full-disk log-normal carries exactly the summed patch variance."""

import math

import pytest
import torch

from forecasting.model import (
    SXR_LOG_OFFSET, CoupledGaussianViTLocal, lognormal_log_variance,
)


SXR_NORM = torch.tensor([-6.0, 0.7])


@pytest.mark.parametrize('ratio', [1e-12, 1e-4, 0.02, 0.2, 1.0, 100.0, 1e6])
def test_log_variance_gives_lognormal_with_requested_raw_variance(ratio):
    median = torch.tensor([1e-9, 3e-6, 2e-4], dtype=torch.float64)
    variance = ratio * median.square()
    log_variance = lognormal_log_variance(median, variance)
    raw_variance = median.square() * torch.exp(log_variance) * torch.expm1(log_variance)
    torch.testing.assert_close(raw_variance, variance, rtol=1e-9, atol=0)


def test_sampled_lognormal_has_requested_raw_variance():
    median, variance = torch.tensor(2e-6), torch.tensor(0.2 * 2e-6 ** 2)
    s = torch.sqrt(lognormal_log_variance(median, variance))
    generator = torch.Generator().manual_seed(0)
    samples = median * torch.exp(
        s * torch.randn(4_000_000, generator=generator, dtype=torch.float64)
    )
    assert samples.var().item() == pytest.approx(variance.item(), rel=0.02)


def test_small_ratios_reduce_to_delta_method_in_float32():
    median = torch.tensor([1e-6, 1e-3], dtype=torch.float32)
    variance = 1e-10 * median.square()
    torch.testing.assert_close(
        lognormal_log_variance(median, variance),
        torch.full_like(median, 1e-10), rtol=1e-5, atol=0,
    )


def test_model_variance_and_interval_match_summed_patch_variance():
    torch.manual_seed(0)
    model = CoupledGaussianViTLocal(
        model_kwargs=dict(
            embed_dim=8, hidden_dim=16, num_channels=1, num_heads=2,
            num_layers=1, patch_size=1, num_patches=4, dropout=0.0,
            mask_mode='local', local_window=1,
        ),
        sxr_norm=SXR_NORM,
        uncertainty_kwargs={'initial_relative_std': 0.8},
    ).eval()
    with torch.no_grad():
        model.model.global_patch_head[-1].bias.fill_(0.5)
        images = torch.randn(3, 2, 2, 1)
        flux, variance_norm, _, patch_variance = model(images)
        _, lower, upper = model.predict_interval(images, z=1.0)

    median = flux.squeeze(-1).double() + SXR_LOG_OFFSET
    s = variance_norm.double().sqrt() * math.log(10.0) * SXR_NORM[1].item()
    u = torch.exp(s.square())
    torch.testing.assert_close(
        median.square() * u * (u - 1), patch_variance.double().sum(dim=1),
        rtol=1e-4, atol=0,
    )
    torch.testing.assert_close(
        lower.double() + SXR_LOG_OFFSET, median * torch.exp(-s), rtol=1e-4, atol=0,
    )
    torch.testing.assert_close(
        upper.double() + SXR_LOG_OFFSET, median * torch.exp(s), rtol=1e-4, atol=0,
    )
