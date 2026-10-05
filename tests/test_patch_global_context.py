import numpy as np
import torch

from forecasting.model import normalize_sxr
from forecasting.uncertainty_model import GaussianNLLViTLocal
from training.train import initialize_model_weights, load_training_config


def make_model(with_queries, lr=5e-6):
    model_kwargs = {
        'embed_dim': 8, 'hidden_dim': 16, 'num_channels': 7,
        'num_heads': 1, 'num_layers': 1, 'patch_size': 1,
        'num_patches': 16, 'dropout': 0.1, 'mask_mode': 'local',
        'local_window': 3, 'global_patch_adjustment': True,
        'learning_rate': lr,
    }
    if with_queries:
        model_kwargs['global_patch_adjustment'] = False
        model_kwargs['patch_global_context'] = {
            'enabled': True, 'grid_size': 2,
            'attention_dim': 8, 'num_heads': 2,
            'max_abs_adjustment': 1.5, 'dropout': 0.0,
        }
    return GaussianNLLViTLocal(
        model_kwargs=model_kwargs,
        sxr_norm=np.array([-5.475177, 0.66483974], dtype=np.float32),
        mean_parameterization='positive',
        uncertainty_kwargs={'enabled': True, 'start_epoch': 100},
    )


def test_old_global_path_is_skipped_and_new_branch_can_change_prediction(tmp_path):
    torch.manual_seed(31)
    source = make_model(False).eval()
    with torch.no_grad():
        source.model.global_patch_head[-1].bias.fill_(-0.8)
    images = torch.randn(2, 4, 4, 7)
    checkpoint = tmp_path / 'source.ckpt'
    torch.save({
        'state_dict': source.state_dict(),
        'hyper_parameters': dict(source.hparams),
        'epoch': 10,
    }, checkpoint)

    target = make_model(True).eval()
    loaded = initialize_model_weights(target, checkpoint)
    assert loaded['source_epoch'] == 10
    assert loaded['new_parameter_keys']
    assert all(name.startswith('model.patch_global_context.')
               for name in loaded['new_parameter_keys'])
    assert loaded['skipped_parameter_keys']
    assert all(name.startswith((
        'model.global_raw_patch_projection.', 'model.global_query',
        'model.global_attention.', 'model.global_patch_head.',
    )) for name in loaded['skipped_parameter_keys'])
    assert not target.model.global_patch_adjustment
    assert not hasattr(target.model, 'global_attention')
    with torch.no_grad():
        old = source.model(images, source.sxr_norm)
        new = target.model(images, target.sxr_norm)
        embeddings, _ = source.model._encode_patch_embeddings(images, False)
        local_logits = source.model.mlp_head(embeddings).squeeze(-1)
        local_flux = source.model._patch_flux_from_logits(
            local_logits, source.sxr_norm
        )
        local_mean = normalize_sxr(local_flux.sum(dim=1), source.sxr_norm)
    torch.testing.assert_close(new[3], local_flux, rtol=0, atol=0)
    torch.testing.assert_close(new[1], local_mean, rtol=0, atol=0)
    assert (old[1] - new[1]).abs().max() > 0.1

    target.freeze_except(['model.patch_global_context'])
    target.train()
    assert not target.model.training
    assert target.model.patch_global_context.training
    assert not target.model.dropout.training
    optimizer = target.configure_optimizers()['optimizer']
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert optimized == {
        id(p) for p in target.model.patch_global_context.parameters()
    }

    target.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = target.model(images, target.sxr_norm)[1]
    (prediction - (prediction.detach() + 0.3)).square().mean().backward()
    assert target.model.patch_global_context.output[-1].weight.grad.abs().sum() > 0
    assert source.model.mlp_head[-1].weight.grad is None
    assert target.model.mlp_head[-1].weight.grad is None
    optimizer.step()
    target.eval()
    with torch.no_grad():
        updated = target.model(images, target.sxr_norm)[1]
    assert (updated - new[1]).abs().max() > 0


def test_each_patch_has_its_own_global_query():
    model = make_model(True).eval()
    branch = model.model.patch_global_context
    assert branch.attention.num_heads == 2
    with torch.no_grad():
        branch.output[-1].weight.fill_(0.1)
        images = torch.randn(1, 4, 4, 7)
        model.model(images, model.sxr_norm)
        first = model.model.last_log10_patch_global_adjustment.clone()
        changed = images.clone()
        changed[:, 3, 3, :] += 20.0  # Distant from top-left in local 3x3.
        model.model(changed, model.sxr_norm)
        second = model.model.last_log10_patch_global_adjustment
    assert first.shape == (1, 16)
    assert (first[:, 0] - second[:, 0]).abs().max() > 1e-7
    assert first.std() > 0


def test_run_config_preserves_source_objective_and_new_branch_only():
    config = load_training_config(
        'training/configs/finetune_sqrt_e10_patch_global_queries.yaml'
    )
    assert config['uncertainty']['class_weighting'] == 'sqrt_inverse_frequency'
    assert config['uncertainty']['huber_delta'] == 0.15
    assert config['uncertainty']['start_epoch'] > config['epochs']
    assert config['finetune']['trainable_modules'] == ['model.patch_global_context']
    assert config['vit_architecture']['global_patch_adjustment'] is False
    assert config['vit_architecture']['patch_global_context']['max_abs_adjustment'] == 1.5
    assert config['vit_architecture']['learning_rate'] > 0
    assert 'epoch=10-' in config['checkpoint']['initialize_from']
    assert config['checkpoint']['monitor'] == 'val/mse'


def test_full_unfreeze_config_trains_all_parameters_at_one_lr():
    config = load_training_config(
        'training/configs/finetune_patch_global_full_unfrozen.yaml'
    )
    assert config['finetune']['trainable_modules'] is None
    assert config['uncertainty']['start_epoch'] == 0
    assert config['vit_architecture']['global_patch_adjustment'] is False
    assert 'REPLACE_WITH_CHOSEN_CHECKPOINT' in config['checkpoint']['initialize_from']
    assert config['checkpoint']['resume_from'] is None
    assert config['vit_architecture']['learning_rate'] == 5e-6
    assert config['optimizer']['scheduler']['T_max'] == config['epochs']

    model = make_model(
        True,
        lr=config['vit_architecture']['learning_rate'],
    )
    assert all(parameter.requires_grad for parameter in model.parameters())
    optimizer = model.configure_optimizers()['optimizer']
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 5e-6
    all_parameter_ids = {id(parameter) for parameter in model.parameters()}
    grouped_ids = {
        id(parameter)
        for group in optimizer.param_groups for parameter in group['params']
    }
    assert grouped_ids == all_parameter_ids

    image = torch.randn(2, 4, 4, 7)
    _, mean, variance, _, _ = model.model(image, model.sxr_norm)
    (mean.square().mean() + variance.mean()).backward()
    assert model.model.input_layer.weight.grad.abs().sum() > 0
    assert model.model.mlp_head[-1].weight.grad.abs().sum() > 0
    assert model.model.patch_global_context.output[-1].weight.grad.abs().sum() > 0
    assert model.model.patch_uncertainty_head[-1].weight.grad.abs().sum() > 0
