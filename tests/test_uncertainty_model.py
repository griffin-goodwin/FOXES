import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.sxr_normalization import compute_sxr_norm
from forecasting.dataset import SXRLogNormTransform
from forecasting.model import normalize_sxr, unnormalize_sxr
from forecasting.uncertainty_model import GaussianNLLViTLocal


def make_model(uncertainty_overrides=None, mean_parameterization="multiplier",
               contrast_prior_overrides=None, num_patches=4,
               model_kwargs_overrides=None):
    uncertainty_kwargs = {
        "patch_flux_scale_multiplier": 1.0,
        "max_abs_log10_patch_multiplier": 8.0,
        "initial_relative_std": 1.0,
        "min_log10_relative_std": -4.0,
        "max_log10_relative_std": 4.0,
        "uncertainty_hidden_dim": 8,
    }
    uncertainty_kwargs.update(uncertainty_overrides or {})
    model_kwargs = {
        "embed_dim": 8,
        "hidden_dim": 16,
        "num_channels": 7,
        "num_heads": 1,
        "num_layers": 1,
        "patch_size": 1,
        "num_patches": num_patches,
        "dropout": 0.0,
        "mask_mode": "none",
    }
    model_kwargs.update(model_kwargs_overrides or {})
    return GaussianNLLViTLocal(
        model_kwargs=model_kwargs,
        sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
        uncertainty_kwargs=uncertainty_kwargs,
        mean_parameterization=mean_parameterization,
        contrast_prior_kwargs=contrast_prior_overrides,
    )


def test_global_aia_patch_adjustment_starts_from_local_map_and_stays_bounded():
    model = make_model(
        mean_parameterization="positive",
        num_patches=16,
        model_kwargs_overrides={
            "mask_mode": "local",
            "local_window": 3,
            "global_patch_adjustment": True,
            "max_abs_log10_global_adjustment": 0.3,
        },
    )
    model.eval()
    images = torch.randn(2, 4, 4, 7)

    with torch.no_grad():
        embeddings, _ = model.model._encode_patch_embeddings(images, False)
        logits = model.model.mlp_head(embeddings).squeeze(-1)
        local_flux = model.model._patch_flux_from_logits(
            logits, model.sxr_norm
        )
        initial_raw, _, _, initial_flux, _ = model.model(
            images, model.sxr_norm
        )

    torch.testing.assert_close(initial_flux, local_flux)
    torch.testing.assert_close(initial_raw.squeeze(-1), initial_flux.sum(dim=1))
    torch.testing.assert_close(
        model.model.last_log10_global_patch_adjustment,
        torch.zeros_like(initial_flux),
    )

    with torch.no_grad():
        model.model.global_patch_head[-1].weight[0, 0] = 1.0
        adjusted_raw, _, _, adjusted_flux, adjusted_variance = model.model(
            images, model.sxr_norm
        )

    correction = torch.log10(adjusted_flux / local_flux)
    assert correction.abs().max() <= 0.3 + 1e-6
    assert correction.std() > 1e-6
    torch.testing.assert_close(
        adjusted_raw.squeeze(-1), adjusted_flux.sum(dim=1)
    )
    assert torch.all(adjusted_variance >= 0)


def test_global_aia_patch_adjustment_requires_local_attention():
    try:
        make_model(
            mean_parameterization="positive",
            model_kwargs_overrides={
                "mask_mode": "inverted",
                "global_patch_adjustment": True,
            },
        )
    except ValueError as error:
        assert "requires local patch attention" in str(error)
    else:
        raise AssertionError("global adjustment accepted inverted attention")


def test_random_global_head_initialization_is_nonzero_and_backpropagates():
    torch.manual_seed(7)
    model = make_model(
        mean_parameterization="positive",
        num_patches=16,
        model_kwargs_overrides={
            "mask_mode": "local",
            "local_window": 3,
            "global_patch_adjustment": True,
            "global_patch_head_init_std": 0.01,
        },
    )
    model.eval()
    final_layer = model.model.global_patch_head[-1]
    assert final_layer.weight.std() > 0
    torch.testing.assert_close(final_layer.bias, torch.zeros_like(final_layer.bias))

    images = torch.randn(1, 4, 4, 7)
    raw, _, _, patch_flux, _ = model.model(images, model.sxr_norm)
    adjustment = model.model.last_log10_global_patch_adjustment
    assert adjustment.std() > 0
    assert adjustment.abs().max() <= 0.3
    raw.sum().backward()
    assert model.model.global_raw_patch_projection.weight.grad.abs().sum() > 0
    torch.testing.assert_close(raw.squeeze(-1), patch_flux.sum(dim=1))


def test_global_aia_summary_reaches_a_distant_patch_adjustment():
    model = make_model(
        mean_parameterization="positive",
        num_patches=16,
        model_kwargs_overrides={
            "mask_mode": "local",
            "local_window": 3,
            "global_patch_adjustment": True,
        },
    )
    model.eval()
    with torch.no_grad():
        model.model.global_patch_head[-1].weight[0, 0] = 1.0

    images = torch.randn(1, 4, 4, 7, requires_grad=True)
    local_embeddings, _ = model.model._encode_patch_embeddings(
        images.detach(), False
    )
    _, corrected_flux = model.model._patch_mean_from_embeddings(
        local_embeddings.detach(), model.sxr_norm, images=images
    )
    image_gradient = torch.autograd.grad(corrected_flux[0, 0], images)[0]

    # On a 4x4 grid with local 3x3 attention, the opposite corner is reachable
    # here only through the summary's attention to the original AIA patches.
    assert image_gradient[0, 3, 3].abs().sum() > 0


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


def test_patch_head_initializes_to_configured_relative_uncertainty():
    model = make_model({"initial_relative_std": 0.75})
    images = torch.randn(2, 2, 2, 7)

    _, _, _, patch_flux, patch_variance = model.model(images, model.sxr_norm)
    relative_std = torch.sqrt(patch_variance) / patch_flux

    torch.testing.assert_close(
        relative_std, torch.full_like(relative_std, 0.75)
    )
    np.testing.assert_allclose(
        model.model.initial_log10_relative_std, np.log10(0.75)
    )
    assert model.uncertainty_parameterization == (
        "logit_conditioned_relative_log10_patch_std"
    )


def test_initial_relative_uncertainty_does_not_depend_on_patch_count():
    four_patches = make_model(num_patches=4)
    sixteen_patches = make_model(num_patches=16)

    np.testing.assert_allclose(
        four_patches.model.initial_log10_relative_std, 0.0
    )
    np.testing.assert_allclose(
        sixteen_patches.model.initial_log10_relative_std, 0.0
    )


def test_relative_patch_uncertainty_scales_with_predicted_patch_flux():
    model = make_model()
    images = torch.randn(2, 2, 2, 7)

    _, _, _, patch_flux_before, patch_variance_before = model.model(
        images, model.sxr_norm
    )
    with torch.no_grad():
        model.model.mlp_head[-1].bias.add_(1.0)
    _, _, _, patch_flux_after, patch_variance_after = model.model(
        images, model.sxr_norm
    )

    assert not torch.allclose(patch_flux_before, patch_flux_after)
    assert not torch.allclose(
        patch_variance_before, patch_variance_after, rtol=1e-5, atol=0.0
    )
    relative_before = (
        torch.sqrt(patch_variance_before)
        / patch_flux_before
    )
    relative_after = (
        torch.sqrt(patch_variance_after)
        / patch_flux_after
    )
    torch.testing.assert_close(relative_before, relative_after)


def test_uncertainty_head_receives_detached_patch_logits_directly():
    model = make_model()
    patch_logits = torch.tensor([[-2.0, -1.0, 0.0, 1.0]], requires_grad=True)
    patch_flux = torch.tensor([[1e-9, 2e-8, 3e-7, 4e-6]])
    captured_inputs = []
    hook = model.model.patch_uncertainty_head.register_forward_pre_hook(
        lambda _module, inputs: captured_inputs.append(inputs[0].detach())
    )

    model.model._patch_variance_from_logits(patch_logits, patch_flux)
    hook.remove()

    torch.testing.assert_close(
        captured_inputs[0], patch_logits.detach().unsqueeze(-1)
    )
    assert captured_inputs[0].shape == (1, 4, 1)
    assert model.model.patch_uncertainty_head[0].in_features == 1
    assert captured_inputs[0].requires_grad is False


def test_identical_patch_logit_has_identical_relative_uncertainty():
    model = make_model()
    with torch.no_grad():
        model.model.patch_uncertainty_head[-1].weight.fill_(0.25)
    patch_flux = torch.tensor([
        [1e-8, 2e-8, 3e-8, 4e-8],
        [1e-8, 2e-7, 3e-7, 4e-7],
    ])
    patch_logits = torch.tensor([
        [-1.0, -0.5, 0.0, 0.5],
        [-1.0, 1.0, 1.5, 2.0],
    ])
    patch_variance = model.model._patch_variance_from_logits(
        patch_logits, patch_flux
    )
    relative_std = torch.sqrt(patch_variance) / patch_flux

    # Patch zero has the same logit in two globally different maps. Its
    # predicted relative uncertainty must therefore be identical.
    torch.testing.assert_close(relative_std[0, 0], relative_std[1, 0])


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


def test_positive_mean_is_strictly_positive_and_uses_normalized_log_scale():
    model = make_model(
        {"enabled": False}, mean_parameterization="positive"
    )
    sxr_norm = torch.tensor([-6.0, 0.5])
    logits = torch.tensor([
        [0.0, 1.0, -1.0, -100.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    patch_flux = model.model._patch_flux_from_logits(logits, sxr_norm)
    typical_global_flux = 10 ** sxr_norm[0] - 1e-8
    expected = (
        typical_global_flux / logits.shape[1]
        * torch.pow(
            torch.tensor(10.0),
            torch.clamp(logits * sxr_norm[1], min=-8.0, max=8.0),
        )
    )

    assert torch.all(patch_flux > 0)
    torch.testing.assert_close(patch_flux, expected)
    torch.testing.assert_close(
        patch_flux[1].sum(), typical_global_flux
    )


def test_positive_contrast_prior_prefers_flux_on_hot_bright_patches():
    model = make_model(
        {"enabled": False},
        mean_parameterization="positive",
        contrast_prior_overrides={
            "enabled": True,
            "weight": 0.001,
            "channel_indices": [0, 1],
        },
    )
    images = torch.zeros(1, 2, 2, 7)
    images[0, 0, 0, 0] = 4.0
    images[0, 0, 0, 1] = 4.0
    aligned_flux = torch.tensor([[4.0, 1.0, 1.0, 1.0]])
    reversed_flux = torch.tensor([[1.0, 4.0, 4.0, 4.0]])

    aligned = model._contrast_alignment_loss(images, aligned_flux)
    reversed_loss = model._contrast_alignment_loss(images, reversed_flux)

    assert aligned < reversed_loss
    assert 0 <= aligned <= 1
    assert 0 <= reversed_loss <= 1


def test_positive_contrast_model_returns_patch_flux_map_with_attention():
    model = make_model(
        {"enabled": False},
        mean_parameterization="positive",
        contrast_prior_overrides={
            "enabled": True,
            "weight": 0.001,
            "channel_indices": [0, 1],
        },
    )
    images = torch.randn(2, 2, 2, 7)

    global_flux, attention, patch_flux = model(
        images, return_attention=True
    )

    assert patch_flux.shape == (2, 4)
    assert len(attention) == 1
    torch.testing.assert_close(
        global_flux.squeeze(-1), patch_flux.sum(dim=1)
    )


def test_contrast_prior_is_restricted_to_positive_mean():
    with np.testing.assert_raises_regex(
        ValueError, "restricted to the positive mean"
    ):
        make_model(
            {"enabled": False},
            mean_parameterization="original",
            contrast_prior_overrides={
                "enabled": True,
                "weight": 0.001,
                "channel_indices": [0, 1],
            },
        )


def test_original_patch_parameterization_restores_normalized_log_semantics():
    model = make_model(mean_parameterization="original")
    logits = torch.tensor([[0.0, 1.0, 7.0, -10.0]], dtype=torch.float16)

    patch_flux = model.model._patch_flux_from_logits(logits, model.sxr_norm)
    expected = torch.clamp(
        torch.pow(torch.tensor(10.0), logits.float() - 6.0) - 1e-8,
        min=0.0,
        max=1.0,
    )

    assert patch_flux.dtype == torch.float32
    torch.testing.assert_close(patch_flux, expected)
    assert patch_flux[0, 2] == 1.0


def test_original_unclamped_flux_has_no_floor_ceiling_or_patch_offset():
    model = make_model(mean_parameterization="original_unclamped")
    logits = torch.tensor([[0.0, 9.0, -4.0, -10.0]], requires_grad=True)

    flux = model.model._patch_flux_from_logits(logits, model.sxr_norm)

    # Include flux above the old upper bound and below the old offset/floor.
    torch.testing.assert_close(
        flux, torch.tensor([[1e-6, 1e3, 1e-10, 1e-16]]),
        rtol=1e-6, atol=0,
    )
    flux.sum().backward()
    torch.testing.assert_close(
        logits.grad, flux.detach() * np.log(10.0), rtol=1e-6, atol=0,
    )
    half_flux = model.model._patch_flux_from_logits(
        logits.detach().half(), model.sxr_norm,
    )
    assert half_flux.dtype == torch.float32
    torch.testing.assert_close(half_flux, flux.detach())


def test_original_unclamped_local_flux_retains_detached_uncertainty():
    model = make_model(
        mean_parameterization="original_unclamped", num_patches=16,
        model_kwargs_overrides={"mask_mode": "local", "local_window": 3},
    )
    model.eval()
    images = torch.randn(2, 4, 4, 7, requires_grad=True)
    raw, mean, variance, flux, patch_variance = model.predict_distribution(images)

    torch.testing.assert_close(raw.squeeze(-1), flux.sum(dim=1))
    torch.testing.assert_close(mean, normalize_sxr(raw.squeeze(-1), model.sxr_norm))
    torch.testing.assert_close(patch_variance, flux.detach().square())
    assert model.model.global_patch_adjustment is False
    assert model.model.patch_global_context is None
    assert torch.isfinite(variance).all()
    assert (flux > 0).all()
    local_gradient = torch.autograd.grad(flux[0, 0], images, retain_graph=True)[0]
    assert torch.count_nonzero(local_gradient[0, 3, 3]) == 0
    assert local_gradient[0, 0, 0].abs().sum() > 0

    model._uncertainty_nll_loss(mean, mean.detach() + 0.2, variance).backward()
    assert model.model.patch_uncertainty_head[-1].bias.grad.abs().sum() > 0
    assert model.model.mlp_head[-1].weight.grad is None
    assert model.model.input_layer.weight.grad is None

    model.zero_grad(set_to_none=True)
    model.log = lambda *args, **kwargs: None
    loss = model._calculate_loss((images.detach(), torch.tensor([0.2, -0.3])), "test")
    loss.backward()
    assert torch.isfinite(loss)
    assert model.model.mlp_head[-1].weight.grad.abs().sum() > 0
    assert model.model.patch_uncertainty_head[-1].bias.grad.abs().sum() > 0
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.grad is not None
    )


def test_original_unclamped_uncertainty_checkpoint_roundtrip(tmp_path):
    import pytorch_lightning as pl

    model = make_model(
        mean_parameterization="original_unclamped",
        model_kwargs_overrides={"mask_mode": "local", "local_window": 3},
    ).eval()
    checkpoint = tmp_path / "unclamped.ckpt"
    torch.save({
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": pl.__version__,
    }, checkpoint)
    restored = GaussianNLLViTLocal.load_from_checkpoint(
        checkpoint, weights_only=False,
    ).eval()
    images = torch.randn(2, 2, 2, 7)

    assert restored.mean_parameterization == "original_unclamped"
    assert restored.uncertainty_enabled
    for expected, actual in zip(model(images), restored(images)):
        torch.testing.assert_close(actual, expected)


def test_mean_head_initialization_depends_on_parameterization():
    torch.manual_seed(7)
    original = make_model(mean_parameterization="original")
    torch.manual_seed(7)
    multiplier = make_model(mean_parameterization="multiplier")
    torch.manual_seed(7)
    positive = make_model(mean_parameterization="positive")

    assert original.mean_head_initialization == "pytorch_linear_default"
    assert multiplier.mean_head_initialization == "zeros"
    assert positive.mean_head_initialization == "pytorch_linear_default"
    assert torch.count_nonzero(original.model.mlp_head[-1].weight) > 0
    assert torch.count_nonzero(multiplier.model.mlp_head[-1].weight) == 0
    assert torch.count_nonzero(multiplier.model.mlp_head[-1].bias) == 0
    torch.testing.assert_close(
        positive.model.mlp_head[-1].weight,
        original.model.mlp_head[-1].weight,
    )
    torch.testing.assert_close(
        positive.model.mlp_head[-1].bias,
        original.model.mlp_head[-1].bias,
    )


def test_mean_only_mode_omits_uncertainty_head_and_trains_mean():
    model = make_model({"enabled": False}, mean_parameterization="original")
    images = torch.randn(2, 2, 2, 7)
    targets = torch.tensor([0.2, -0.3])
    model.log = lambda *args, **kwargs: None

    prediction, patch_flux = model(images)
    loss = model._calculate_loss((images, targets), "test")
    loss.backward()

    assert model.predicts_uncertainty is False
    assert model.model.patch_uncertainty_head is None
    torch.testing.assert_close(prediction.squeeze(-1), patch_flux.sum(dim=1))
    assert model.model.mlp_head[-1].weight.grad is not None
    assert all("uncertainty_head" not in name for name, _ in model.named_parameters())


def test_delayed_uncertainty_head_has_no_gradient_before_start_epoch():
    model = make_model({"start_epoch": 2})
    images = torch.randn(2, 2, 2, 7)
    targets = torch.tensor([0.2, -0.3])
    model.log = lambda *args, **kwargs: None

    assert model._uncertainty_active_for_epoch(0) is False
    assert model._uncertainty_active_for_epoch(1) is False
    assert model._uncertainty_active_for_epoch(2) is True
    loss = model._calculate_loss((images, targets), "test")
    loss.backward()

    assert model.model.mlp_head[-1].weight.grad is not None
    assert model.model.patch_uncertainty_head[-1].weight.grad is None


def test_uncertainty_start_epoch_rejects_invalid_values():
    for value in (-1, 1.5, True):
        try:
            make_model({"start_epoch": value})
        except ValueError as error:
            assert "nonnegative integer or null" in str(error)
        else:
            raise AssertionError(f"invalid start_epoch {value!r} must fail")


def test_patch_uncertainty_bounds_are_validated():
    try:
        make_model({
            "min_log10_relative_std": -4.0,
            "max_log10_relative_std": -5.0,
        })
    except ValueError as error:
        assert "must exceed" in str(error)
    else:
        raise AssertionError("reversed log-standard-deviation bounds must fail")


def test_legacy_patch_uncertainty_settings_are_rejected():
    try:
        make_model({"relative_std_floor": 0.0025})
    except ValueError as error:
        assert "legacy patch uncertainty" in str(error)
    else:
        raise AssertionError("obsolete flux-relative scaling must fail")


def test_incompatible_uncertainty_checkpoint_is_rejected():
    model = make_model()
    checkpoint = {
        "hyper_parameters": {"mean_parameterization": "multiplier"},
        "state_dict": {
            "model.patch_uncertainty_head.1.weight": torch.zeros(1, 8),
        },
    }

    try:
        model.on_load_checkpoint(checkpoint)
    except RuntimeError as error:
        assert "logit-conditioned relative" in str(error)
    else:
        raise AssertionError("old uncertainty semantics must not load silently")


def test_gaussian_model_uses_cosine_annealing_without_restarts():
    model = make_model()
    model.scheduler_kwargs = {"T_max": 12, "eta_min": 1e-6}

    optimizer_config = model.configure_optimizers()
    scheduler = optimizer_config["lr_scheduler"]["scheduler"]

    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.T_max == 12
    assert scheduler.eta_min == 1e-6
