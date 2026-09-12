import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.sxr_normalization import compute_sxr_norm
from forecasting.dataset import SXRLogNormTransform
from forecasting.model import normalize_sxr, unnormalize_sxr
from forecasting.model_uncertainty import GaussianNLLViTLocal


def make_model(uncertainty_overrides=None):
    uncertainty_kwargs = {
        "patch_flux_scale_multiplier": 1.0,
        "max_abs_log10_patch_multiplier": 8.0,
        "relative_std_floor": 0.0025,
        "relative_std_max": 20.0,
        "patch_scale_floor_fraction": 1.0,
    }
    uncertainty_kwargs.update(uncertainty_overrides or {})
    return GaussianNLLViTLocal(
        model_kwargs={
            "embed_dim": 8,
            "hidden_dim": 16,
            "num_channels": 7,
            "num_heads": 1,
            "num_layers": 1,
            "patch_size": 1,
            "num_patches": 4,
            "dropout": 0.0,
            "mask_mode": "none",
        },
        sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
        uncertainty_kwargs=uncertainty_kwargs,
    )


def test_sxr_normalization_uses_epsilon_log10_and_exact_inverse():
    sxr_norm = torch.tensor([-7.0, 1.0])
    raw = torch.tensor([9e-8, 9.9e-7])
    normalized = normalize_sxr(raw, sxr_norm)

    torch.testing.assert_close(normalized, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(unnormalize_sxr(normalized, sxr_norm), raw)
    transform = SXRLogNormTransform(mean=-7.0, std=1.0)
    np.testing.assert_allclose(transform(9.9e-7), 1.0)


def test_sxr_statistics_loader_and_model_share_epsilon_log10_contract(tmp_path):
    fluxes = np.array([1e-8, 1e-7, 1e-6, 1e-4], dtype=np.float64)
    for index, flux in enumerate(fluxes):
        np.save(tmp_path / f"sample_{index}.npy", flux)

    mean, std = compute_sxr_norm(tmp_path)
    expected = np.log10(fluxes + 1e-8)
    np.testing.assert_allclose([mean, std], [expected.mean(), expected.std()])


def test_global_prediction_is_sum_of_patch_flux():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)

    raw, _, variance_norm, patch_flux, patch_variance = model.model(
        images, model.sxr_norm
    )

    torch.testing.assert_close(raw.squeeze(-1), patch_flux.sum(dim=1))
    assert torch.all(patch_flux > 0)
    assert torch.all(patch_variance > 0)
    assert torch.all(variance_norm > 0)


def test_model_has_patch_uncertainty_head_and_no_global_head():
    model = make_model()
    parameter_names = [name for name, _ in model.named_parameters()]

    assert any("patch_uncertainty_head" in name for name in parameter_names)
    assert not any("global_uncertainty_head" in name for name in parameter_names)


def test_global_variance_is_sum_of_patch_variances_after_log_conversion():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)

    raw, _, variance_norm, _, patch_variance = model.model(
        images, model.sxr_norm
    )
    derivative = 1.0 / (
        torch.log(torch.tensor(10.0))
        * model.sxr_norm[1]
        * (raw.squeeze(-1) + 1e-8)
    )
    expected = patch_variance.sum(dim=1) * derivative.square()

    torch.testing.assert_close(variance_norm, expected)


def test_gaussian_nll_trains_global_mean_and_patch_uncertainty_head():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)
    _, mean_norm, variance_norm, _, _ = model.model(images, model.sxr_norm)
    targets = torch.tensor([0.2, -0.3])

    loss = F.gaussian_nll_loss(
        mean_norm, targets, variance_norm, full=True
    )
    loss.backward()

    assert model.model.mlp_head[-1].bias.grad is not None
    assert model.model.mlp_head[-1].bias.grad.abs().sum() > 0
    patch_output = model.model.patch_uncertainty_head[-1]
    assert patch_output.bias.grad is not None
    assert patch_output.bias.grad.abs().sum() > 0


def test_variance_branch_does_not_train_mean_head_or_backbone():
    """Variance-only gradients must remain inside the uncertainty head."""
    model = make_model()
    images = torch.randn(2, 2, 2, 7)
    _, mean_norm, variance_norm, _, _ = model.model(images, model.sxr_norm)

    # Detaching the mean isolates the variance term of Gaussian NLL. The
    # uncertainty head should still learn, but no variance gradient may reach
    # the patch-mean head or the shared transformer backbone.
    target = mean_norm.detach()
    variance_only_loss = F.gaussian_nll_loss(
        mean_norm.detach(), target, variance_norm, full=True
    )
    variance_only_loss.backward()

    uncertainty_output = model.model.patch_uncertainty_head[-1]
    assert uncertainty_output.bias.grad is not None
    assert uncertainty_output.bias.grad.abs().sum() > 0
    assert model.model.mlp_head[-1].weight.grad is None
    assert model.model.mlp_head[-1].bias.grad is None
    assert model.model.input_layer.weight.grad is None
    assert model.model.pos_embedding_2d.grad is None


def test_uncertainty_nll_has_no_mean_gradient():
    model = make_model()
    mean = torch.tensor([0.4, -0.2], requires_grad=True)
    target = torch.tensor([0.1, 0.3])
    variance = torch.tensor([0.25, 0.5], requires_grad=True)

    uncertainty_nll = model._uncertainty_nll_loss(mean, target, variance)
    uncertainty_nll.backward()

    error = mean.detach() - target
    expected_variance_gradient = (
        0.5
        * (
            1.0 / variance.detach()
            - error.square() / variance.detach().square()
        )
        / mean.numel()
    )
    assert mean.grad is None
    torch.testing.assert_close(variance.grad, expected_variance_gradient)


def test_huber_and_uncertainty_nll_train_separate_outputs():
    model = make_model()
    mean = torch.tensor([0.4, -0.2], requires_grad=True)
    target = torch.tensor([0.1, 0.3])
    variance = torch.tensor([0.25, 0.5], requires_grad=True)

    variance_only = model._uncertainty_nll_loss(mean, target, variance)
    huber = F.huber_loss(mean, target, delta=0.2)
    (huber + variance_only).backward()

    torch.testing.assert_close(mean.grad, torch.tensor([0.1, -0.1]))
    assert variance.grad is not None
    assert variance.grad.abs().sum() > 0


def test_huber_mean_uses_training_derived_four_class_macro_weights():
    class_weights = {
        "quiet": 1.0,
        "c_class": 2.0,
        "m_class": 3.0,
        "x_class": 4.0,
    }
    model = make_model({
        "huber_delta": 0.3,
        "class_weighting": "inverse_frequency",
        "class_weights": class_weights,
    })
    mean = torch.tensor([0.1, 0.2, 0.4, 0.5], requires_grad=True)
    target = torch.zeros(4)
    target_raw = torch.tensor([5e-7, 2e-6, 2e-5, 2e-4])

    loss = model._weighted_huber_loss(mean, target, target_raw)
    expected = (
        F.huber_loss(mean, target, delta=0.3, reduction="none")
        * torch.tensor([1.0, 2.0, 3.0, 4.0])
    ).mean()

    torch.testing.assert_close(loss, expected)


def test_four_class_macro_mean_weighting_requires_computed_weights():
    try:
        make_model({
            "class_weighting": "inverse_frequency",
        })
    except ValueError as error:
        assert "training-derived" in str(error)
    else:
        raise AssertionError("macro-weighted Huber must require class weights")


def test_canonical_training_step_trains_mean_and_uncertainty_heads():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)
    targets = torch.tensor([0.2, -0.3])
    model.log = lambda *args, **kwargs: None

    loss = model._calculate_loss((images, targets), "test")
    loss.backward()

    assert model.model.mlp_head[-1].bias.grad is not None
    assert model.model.mlp_head[-1].bias.grad.abs().sum() > 0
    assert model.model.patch_uncertainty_head[-1].weight.grad is not None
    assert model.model.patch_uncertainty_head[-1].bias.grad is not None


def test_log10_patch_parameterization_reaches_large_flux_with_small_logits():
    model = make_model()
    sxr_norm = model.sxr_norm
    zero_logits = torch.zeros(2, 4)
    x_scale_logits = torch.full((2, 4), 2.61)

    initial_flux = model.model._patch_flux_from_logits(
        zero_logits, sxr_norm
    ).sum(dim=1)
    x_scale_flux = model.model._patch_flux_from_logits(
        x_scale_logits, sxr_norm
    ).sum(dim=1)

    torch.testing.assert_close(
        x_scale_flux / initial_flux,
        torch.full_like(initial_flux, 10 ** 2.61),
        rtol=1e-5,
        atol=0,
    )


def test_patch_uncertainty_bounds_are_validated():
    try:
        make_model({"relative_std_floor": 0.0})
    except ValueError as error:
        assert "must be positive" in str(error)
    else:
        raise AssertionError("zero relative std floor must fail")


def test_gaussian_model_uses_cosine_annealing_without_restarts():
    model = make_model()
    model.scheduler_kwargs = {"T_max": 12, "eta_min": 1e-6}

    optimizer_config = model.configure_optimizers()
    scheduler = optimizer_config["lr_scheduler"]["scheduler"]

    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.T_max == 12
    assert scheduler.eta_min == 1e-6
