import numpy as np
import pytest
import torch

from forecasting.model import (
    CoupledGaussianViTLocal, ViTLocal, mean_class_signed_error_penalty,
    normalize_sxr, unnormalize_sxr,
)


MODEL_KWARGS = dict(
    embed_dim=8, hidden_dim=16, num_channels=1, num_heads=1,
    num_layers=1, patch_size=1, num_patches=4, dropout=0.0,
    mask_mode='none',
)
COUPLED_KWARGS = dict(
    MODEL_KWARGS, mask_mode='local', local_window=3,
    global_patch_adjustment=True,
)


def test_class_bias_penalty_preserves_opposing_class_biases_and_has_gradient():
    predictions = torch.tensor([1.0, -1.0], requires_grad=True)
    target = torch.zeros(2)
    flux = torch.tensor([1e-7, 1e-4])
    loss = mean_class_signed_error_penalty(predictions, target, flux, 2.0)
    torch.testing.assert_close(loss, torch.tensor(2.0))
    loss.backward()
    torch.testing.assert_close(predictions.grad, torch.tensor([2.0, -2.0]))

    # Errors within the same class still cancel in its signed mean.
    torch.testing.assert_close(
        mean_class_signed_error_penalty(predictions, target,
                                        torch.tensor([1e-7, 1e-7]), 2.0),
        torch.tensor(0.0),
    )

    # Two quiet samples and one X sample still give each class one share.
    torch.testing.assert_close(
        mean_class_signed_error_penalty(
            torch.tensor([1.0, 1.0, -2.0]), torch.zeros(3),
            torch.tensor([1e-7, 1e-7, 2e-4]), 1.0,
        ),
        torch.tensor(2.5),
    )


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf'), True])
def test_invalid_bias_penalty_weight_rejected(weight):
    with pytest.raises(ValueError, match='bias_penalty_weight'):
        ViTLocal(MODEL_KWARGS, np.array([-5.0, 1.0]), bias_penalty_weight=weight)
    with pytest.raises(ValueError, match='bias_penalty_weight'):
        CoupledGaussianViTLocal(
            COUPLED_KWARGS, np.array([-5.0, 1.0]),
            uncertainty_kwargs={'enabled': False, 'bias_penalty_weight': weight},
        )


def test_coupled_total_loss_adds_penalty_without_changing_mean_loss():
    torch.manual_seed(1)
    model = CoupledGaussianViTLocal(
        COUPLED_KWARGS, np.array([-5.0, 1.0]),
        uncertainty_kwargs={'enabled': False, 'bias_penalty_weight': 0.7},
    )
    logged = {}
    model.log = lambda name, value, **kwargs: logged.update({name: value})
    images = torch.randn(3, 2, 2, 1)
    targets = torch.tensor([0.2, 0.4, 0.6])

    result = model.validation_step((images, targets), 0)
    expected_bias = (result['mean_norm'] - targets).mean()
    torch.testing.assert_close(logged['val/bias_dex'], expected_bias)
    torch.testing.assert_close(logged['val/bias_penalty'], 0.7 * expected_bias.square())
    torch.testing.assert_close(
        result['loss'], logged['val/mean_loss'] + logged['val/bias_penalty'],
    )


def test_deterministic_total_loss_adds_penalty():
    torch.manual_seed(2)
    sxr_norm = np.array([-5.0, 1.0])
    model = ViTLocal(MODEL_KWARGS, sxr_norm, bias_penalty_weight=0.4)
    logged = {}
    model.log = lambda name, value, **kwargs: logged.update({name: value})
    images = torch.randn(3, 2, 2, 1)
    targets = torch.tensor([0.2, 0.4, 0.6])
    predictions_raw, _ = model.model(images, sxr_norm)
    predictions_norm = normalize_sxr(predictions_raw.squeeze(), sxr_norm)
    base_loss, _ = model.adaptive_loss.calculate_loss(
        predictions_norm, targets, unnormalize_sxr(targets, sxr_norm),
        update_tracking=False,
    )

    loss = model._calculate_loss((images, targets), mode='val')
    expected_bias = (predictions_norm - targets).mean()
    torch.testing.assert_close(logged['val/bias_dex'], expected_bias)
    torch.testing.assert_close(logged['val/bias_penalty'], 0.4 * expected_bias.square())
    torch.testing.assert_close(loss, base_loss + logged['val/bias_penalty'])


def test_coupled_bias_penalty_balances_classes_independent_of_huber_weights():
    model = CoupledGaussianViTLocal(
        COUPLED_KWARGS, np.array([-5.0, 1.0]),
        uncertainty_kwargs={
            'enabled': False, 'bias_penalty_weight': 0.5,
            'class_weighting': 'inverse_frequency',
            'class_weights': {
                'quiet': 0.5, 'c_class': 1.0,
                'm_class': 2.0, 'x_class': 4.0,
            },
        },
    )
    logged = {}
    model.log = lambda name, value, **kwargs: logged.update({name: value})
    result = model.validation_step(
        (torch.randn(3, 2, 2, 1), torch.tensor([-2.0, 0.0, 2.0])), 0,
    )
    residual = result['mean_norm'] - result['target_norm']
    torch.testing.assert_close(logged['val/bias_penalty'],
                               0.5 * residual.square().mean())
