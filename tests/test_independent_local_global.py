import numpy as np
import pytest
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, TensorDataset

from forecasting.model import normalize_sxr, unnormalize_sxr
from forecasting.uncertainty_model import GaussianNLLViTLocal
from training.train import load_training_config


GLOBAL_PREFIXES = (
    'global_raw_patch_projection.', 'global_query', 'global_attention.',
    'global_patch_head.', 'patch_global_context.',
)


def make_model(branch='queries', uncertainty=True, independent=True,
               start_epoch=0, contrast=False):
    torch.manual_seed(71)
    architecture = {
        'embed_dim': 8, 'hidden_dim': 16, 'num_channels': 7,
        'num_heads': 1, 'num_layers': 1, 'patch_size': 1,
        'num_patches': 16, 'dropout': 0.0, 'mask_mode': 'local',
        'local_window': 3, 'learning_rate': 1e-3,
        'independent_local_global': independent,
        'global_patch_adjustment': branch in {'summary', 'both'},
        'global_patch_head_init_std': 0.01,
    }
    if branch in {'queries', 'both'}:
        architecture['patch_global_context'] = {
            'enabled': True, 'grid_size': 2, 'attention_dim': 8,
            'num_heads': 2, 'max_abs_adjustment': 1.5,
        }
    model = GaussianNLLViTLocal(
        model_kwargs=architecture,
        sxr_norm=np.array([-6.0, 0.7], dtype=np.float32),
        mean_parameterization='positive',
        uncertainty_kwargs={
            'enabled': uncertainty, 'start_epoch': start_epoch,
            'huber_delta': 0.15, 'class_weighting': 'inverse_frequency',
            'class_weights': dict(quiet=1.0, c_class=2.0, m_class=3.0, x_class=4.0),
        },
        contrast_prior_kwargs={
            'enabled': contrast, 'weight': 0.01, 'channel_indices': [0, 1],
        },
    )
    # Exercise every route through the correction, beyond its zero-init step.
    with torch.no_grad():
        if model.model.patch_global_context is not None:
            torch.nn.init.normal_(model.model.patch_global_context.output[-1].weight,
                                  std=0.05)
        if model.model.global_patch_adjustment:
            model.model.global_patch_head[-1].bias.fill_(0.1)
    return model


def parameter_group(name):
    if name.startswith(GLOBAL_PREFIXES):
        return 'global'
    if name.startswith('patch_uncertainty_head.'):
        return 'uncertainty'
    return 'local'


@pytest.mark.parametrize('branch', ['summary', 'queries', 'both'])
def test_each_mean_loss_updates_only_its_own_parameters(branch):
    model = make_model(branch)
    images = torch.randn(2, 4, 4, 7)
    outputs = model.model(images, model.sxr_norm, return_local=True)
    _, global_mean, _, _, _, local_flux = outputs
    local_mean = normalize_sxr(local_flux.sum(dim=1), model.sxr_norm)
    targets = global_mean.detach() + 0.4
    target_raw = unnormalize_sxr(targets, model.sxr_norm)
    global_loss = model._weighted_huber_loss(global_mean, targets, target_raw)
    local_loss = model._weighted_huber_loss(local_mean, targets, target_raw)
    parameters = dict(model.model.named_parameters())

    global_loss.backward(retain_graph=True)
    assert parameters['input_layer.weight'].grad is None
    assert parameters['pos_embedding_2d'].grad is None
    assert parameters['mlp_head.1.weight'].grad is None
    for name, parameter in parameters.items():
        if parameter_group(name) != 'global':
            assert parameter.grad is None, name
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in parameters.items() if parameter_group(n) == 'global')
    global_gradients = {
        n: p.grad.clone() for n, p in parameters.items() if p.grad is not None
    }

    model.zero_grad(set_to_none=True)
    local_loss.backward(retain_graph=True)
    for name, parameter in parameters.items():
        if parameter_group(name) != 'local':
            assert parameter.grad is None, name
    for name in ('input_layer.weight', 'pos_embedding_2d', 'mlp_head.1.weight'):
        assert parameters[name].grad.abs().sum() > 0, name
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in parameters.items() if n.startswith('transformer_blocks.'))
    local_gradients = {
        n: p.grad.clone() for n, p in parameters.items() if p.grad is not None
    }

    model.zero_grad(set_to_none=True)
    (local_loss + global_loss).backward()
    for name, expected in {**local_gradients, **global_gradients}.items():
        torch.testing.assert_close(parameters[name].grad, expected)


@pytest.mark.parametrize('uncertainty,start_epoch', [(False, 0), (True, 0), (True, 5)])
def test_training_objective_is_two_equal_weighted_losses_plus_active_nll(
    uncertainty, start_epoch,
):
    model = make_model(uncertainty=uncertainty, start_epoch=start_epoch)
    images = torch.randn(2, 4, 4, 7)
    target_raw = torch.tensor([5e-7, 2e-4])
    targets = normalize_sxr(target_raw, model.sxr_norm)
    logs = {}
    model.log = lambda name, value, **kwargs: logs.update({name: value})

    outputs = model.model(images, model.sxr_norm, return_local=True)
    global_mean, local_flux = outputs[1], outputs[-1]
    local_mean = normalize_sxr(local_flux.sum(dim=1), model.sxr_norm)
    weights = torch.tensor([1.0, 4.0])
    local_loss = (F.huber_loss(local_mean, targets, delta=0.15, reduction='none')
                  * weights).mean()
    global_loss = (F.huber_loss(global_mean, targets, delta=0.15, reduction='none')
                   * weights).mean()
    expected = local_loss + global_loss
    if uncertainty and start_epoch == 0:
        expected = expected + model._uncertainty_nll_loss(global_mean, targets, outputs[2])
    expected.backward()
    expected_gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters() if parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    actual = model._calculate_loss((images, targets), 'test')
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logs['test/local_mean_loss'], local_loss)
    torch.testing.assert_close(logs['test/global_mean_loss'], global_loss)
    torch.testing.assert_close(logs['test/mean_loss'], local_loss + global_loss)
    torch.testing.assert_close(logs['test/local_mse'], F.mse_loss(local_mean, targets))
    torch.testing.assert_close(logs['test/mse'], F.mse_loss(global_mean, targets))
    actual.backward()
    for name, parameter in model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(parameter.grad, expected_gradients[name])
        else:
            assert parameter.grad is None, name


@pytest.mark.parametrize('branch', ['summary', 'queries', 'both'])
def test_uncertainty_loss_cannot_update_either_mean_branch(branch):
    model = make_model(branch)
    images = torch.randn(2, 4, 4, 7)
    _, mean, variance, _, _ = model.model(images, model.sxr_norm)
    model._uncertainty_nll_loss(mean, mean.detach() + 0.2, variance).backward()
    for name, parameter in model.model.named_parameters():
        if parameter_group(name) != 'uncertainty':
            assert parameter.grad is None, name
    assert model.model.patch_uncertainty_head[-1].bias.grad.abs().sum() > 0


def test_contrast_prior_trains_local_map_in_independent_mode():
    model = make_model(contrast=True, uncertainty=False)
    images = torch.randn(2, 4, 4, 7)
    logs = {}
    model.log = lambda name, value, **kwargs: logs.update({name: value})
    loss = model._calculate_loss((images, torch.tensor([0.1, 0.2])), 'test')
    torch.testing.assert_close(
        loss, logs['test/mean_loss'] + logs['test/weighted_contrast_loss'],
    )
    logs['test/contrast_loss'].backward()
    assert model.model.mlp_head[-1].weight.grad.abs().sum() > 0
    for name, parameter in model.model.named_parameters():
        if parameter_group(name) != 'local':
            assert parameter.grad is None, name


@pytest.mark.parametrize('branch', ['summary', 'queries', 'both'])
@pytest.mark.parametrize('uncertainty', [False, True])
@pytest.mark.parametrize('attention', [False, True])
def test_detaching_changes_gradients_without_changing_inference(branch, uncertainty, attention):
    independent = make_model(branch, uncertainty=uncertainty).eval()
    joint = make_model(branch, uncertainty=uncertainty, independent=False).eval()
    joint.load_state_dict(independent.state_dict(), strict=True)
    images = torch.randn(2, 4, 4, 7)
    with torch.no_grad():
        actual = independent(images, return_attention=attention)
        expected = joint(images, return_attention=attention)
    assert len(actual) == len(expected)
    for before, after in zip(actual, expected):
        if isinstance(before, list):
            for a, b in zip(before, after):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_independent_mode_requires_global_correction_and_boolean_setting():
    with pytest.raises(ValueError, match='requires a global patch correction'):
        make_model(branch='none')
    with pytest.raises(ValueError, match='must be a boolean'):
        make_model(independent='true')


def test_independent_config_changes_only_objective_and_output_metadata():
    baseline = load_training_config('training/configs/finetune_patch_global_full_unfrozen.yaml')
    independent = load_training_config('training/configs/finetune_patch_global_independent.yaml')
    assert independent['vit_architecture'].pop('independent_local_global') is True
    assert independent['checkpoint']['monitor'] == 'val/mse'
    assert independent['data']['checkpoints_dir'] != baseline['data']['checkpoints_dir']
    for config in (baseline, independent):
        config['data'].pop('checkpoints_dir')
        config.pop('wandb')
    assert independent == baseline


def test_lightning_training_and_checkpoint_reload_preserve_independence(tmp_path):
    model = make_model()
    images = torch.randn(4, 4, 4, 7)
    targets = torch.tensor([0.2, -0.3, 0.5, 0.8])
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)
    local_before = model.model.mlp_head[-1].weight.detach().clone()
    global_before = model.model.patch_global_context.output[-1].weight.detach().clone()
    trainer = Trainer(
        accelerator='cpu', devices=1, fast_dev_run=2, logger=False,
        plugins=[LightningEnvironment()],
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=str(tmp_path),
    )
    trainer.fit(model, loader, loader)
    assert not torch.equal(model.model.mlp_head[-1].weight, local_before)
    assert not torch.equal(model.model.patch_global_context.output[-1].weight, global_before)
    assert 'val/local_mse' in trainer.callback_metrics
    assert 'val/global_mean_loss' in trainer.callback_metrics

    checkpoint = tmp_path / 'independent.ckpt'
    trainer.save_checkpoint(checkpoint)
    restored = GaussianNLLViTLocal.load_from_checkpoint(checkpoint, weights_only=False)
    assert restored.model.independent_local_global is True
    model.eval()
    restored.eval()
    with torch.no_grad():
        for before, after in zip(model(images), restored(images)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    # Reloaded checkpoints retain the gradient boundary as well as predictions.
    restored.model(images, restored.sxr_norm)[1].sum().backward()
    assert restored.model.input_layer.weight.grad is None
    assert restored.model.pos_embedding_2d.grad is None
    assert restored.model.mlp_head[-1].weight.grad is None
    assert restored.model.patch_global_context.output[-1].weight.grad.abs().sum() > 0
