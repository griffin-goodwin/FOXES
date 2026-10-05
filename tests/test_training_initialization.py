import numpy as np
import pytest
import torch

from forecasting.uncertainty_model import GaussianNLLViTLocal
from training.train import (
    initialize_model_weights,
    resolve_initialization_checkpoint,
)


def make_model(weighting='none', lr=1e-5):
    return GaussianNLLViTLocal(
        model_kwargs={
            'embed_dim': 8, 'hidden_dim': 16, 'num_channels': 7,
            'num_heads': 1, 'num_layers': 1, 'patch_size': 1,
            'num_patches': 16, 'dropout': 0.0, 'mask_mode': 'local',
            'local_window': 3, 'global_patch_adjustment': True,
            'learning_rate': lr,
        },
        sxr_norm=np.array([-5.475177, 0.66483974], dtype=np.float32),
        mean_parameterization='positive',
        scheduler_kwargs={'T_max': 5, 'eta_min': 1e-6},
        uncertainty_kwargs={
            'class_weighting': weighting, 'huber_delta': 0.15,
            'class_weights': dict(quiet=2.0, c_class=1.0, m_class=3.0, x_class=4.0),
        },
    )


def test_initialization_restores_predictions_but_not_objective_or_optimizer(tmp_path):
    source = make_model('inverse_frequency', lr=5e-5)
    source_optimizer = source.configure_optimizers()['optimizer']
    images = torch.randn(3, 4, 4, 7)
    source.model(images, source.sxr_norm)[1].square().mean().backward()
    source_optimizer.step()
    path = tmp_path / 'source.ckpt'
    torch.save({
        'state_dict': source.state_dict(), 'epoch': 14, 'global_step': 300,
        'hyper_parameters': dict(source.hparams),
        'optimizer_states': [source_optimizer.state_dict()],
        'lr_schedulers': [{'last_epoch': 15, 'T_max': 25}],
    }, path)

    target = make_model()
    metadata = initialize_model_weights(target, path)
    assert metadata == {'source_epoch': 14, 'source_global_step': 300}
    source.eval()
    target.eval()
    with torch.no_grad():
        for before, after in zip(source.model(images, source.sxr_norm),
                                 target.model(images, target.sxr_norm)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert target.class_weighting == 'none'
    assert torch.equal(target.class_weight_values, torch.ones(4))
    assert target.current_epoch == 0
    optimizers = target.configure_optimizers()
    assert not optimizers['optimizer'].state
    assert optimizers['optimizer'].param_groups[0]['lr'] == 1e-5
    scheduler = optimizers['lr_scheduler']['scheduler']
    assert scheduler.T_max == 5
    assert scheduler.last_epoch == 0


@pytest.mark.parametrize('field', ['sxr_norm', 'attention_mask', 'mean_parameterization'])
def test_initialization_rejects_changed_mean_interpretation(tmp_path, field):
    source = make_model()
    state = source.state_dict()
    hparams = dict(source.hparams)
    if field == 'sxr_norm':
        state['sxr_norm'] = state['sxr_norm'] + 1
    elif field == 'attention_mask':
        key = next(k for k in state if k.endswith('.attention_mask'))
        state[key] = ~state[key]
    else:
        hparams[field] = 'original'
    path = tmp_path / 'source.ckpt'
    torch.save({'state_dict': state, 'hyper_parameters': hparams}, path)
    with pytest.raises(ValueError, match=field):
        initialize_model_weights(make_model(), path)


def test_initialization_cannot_be_combined_with_full_resume(tmp_path):
    path = tmp_path / 'source.ckpt'
    path.touch()
    config = {'initialize_from': str(path)}
    assert resolve_initialization_checkpoint(config) == str(path.resolve())
    with pytest.raises(ValueError, match='cannot be combined'):
        resolve_initialization_checkpoint(config, resume_checkpoint=str(path))
    assert resolve_initialization_checkpoint({}) is None
    with pytest.raises(FileNotFoundError, match='Initialization checkpoint'):
        resolve_initialization_checkpoint({'initialize_from': str(tmp_path / 'missing')})
