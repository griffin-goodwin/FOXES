from copy import deepcopy
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import LightningEnvironment
from forecasting.model import normalize_sxr
from forecasting.uncertainty_model import GaussianNLLViTLocal
from training.configuration import resolve_model_settings
from training.train import load_training_config


def make_model(global_enabled=True, uncertainty=True, **global_arch):
    torch.manual_seed(42)
    return GaussianNLLViTLocal(
        model_kwargs=dict(embed_dim=8, hidden_dim=16, num_channels=7, num_heads=1,
                          num_layers=1, patch_size=1, num_patches=16, dropout=0.,
                          patch_global_context=dict(enabled=global_enabled, grid_size=2,
                                                    attention_dim=8, num_heads=2, **global_arch)),
        sxr_norm=np.array([-5.4, 0.7], dtype=np.float32),
        uncertainty_kwargs=dict(enabled=uncertainty, class_weighting='none', huber_delta=0.3),
    )


@pytest.mark.parametrize('global_enabled', [False, True])
@pytest.mark.parametrize('uncertainty', [False, True])
def test_loss_accounting_and_outputs(global_enabled, uncertainty):
    m = make_model(global_enabled, uncertainty)
    logged = {}
    m.log = lambda key, value, **kw: logged.update({key:value})
    images = torch.randn(3,4,4,7)
    target = torch.tensor([0.5, -0.3, 0.8])
    loss = m.training_step((images,target),0)
    torch.testing.assert_close(logged['train/mean_loss'], logged['train/local_huber']+logged['train/global_huber'])
    torch.testing.assert_close(loss, logged['train/mean_loss']+logged['train/nll'])
    if not global_enabled:
        assert logged['train/global_huber'] == 0
        torch.testing.assert_close(logged['train/mse'],logged['train/local_mse'])
    if not uncertainty:
        assert logged['train/nll'] == 0
    assert torch.isfinite(loss)
    result = m.validation_step((images,target),0)
    assert result['prediction_raw'].shape == (3,)
    assert ('variance_norm' in result) == uncertainty
    distribution = m.predict_distribution(images)
    if uncertainty:
        _, _, _, patches, _ = distribution
    else:
        _, _, patches = distribution
    torch.testing.assert_close(patches.sum(dim=1),distribution[0].squeeze(-1))


def test_three_objectives_have_disjoint_gradients():
    m=make_model()
    out=m.model(torch.randn(2,4,4,7),m.sxr_norm,return_local=True)
    raw, mean, variance, patches, patch_var, local = out
    target=mean.detach()+0.5
    local_mean=normalize_sxr(local.sum(dim=1),m.sxr_norm)
    losses=dict(local=torch.nn.functional.huber_loss(local_mean,target),
                global_head=torch.nn.functional.huber_loss(mean,target),
                uncertainty=m._uncertainty_nll_loss(mean,target,variance))
    def group(name):
        if name.startswith('patch_global_context.'):return 'global_head'
        if name.startswith('patch_uncertainty_head.'):return 'uncertainty'
        return 'local'
    for owner,loss in losses.items():
        m.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        assert any(p.grad is not None and p.grad.abs().sum()>0
                   for n,p in m.model.named_parameters() if group(n)==owner)
        for n,p in m.model.named_parameters():
            if group(n)!=owner:assert p.grad is None,n


def test_global_architecture_and_default_switches():
    m=make_model(num_layers=3,hidden_dim=12,mlp_depth=2)
    g=m.model.patch_global_context
    assert len(g.extra_attention)==2
    assert g.output[1].out_features==12
    assert len([layer for layer in g.output if isinstance(layer,torch.nn.Linear)])==3
    assert m.model(torch.randn(2,4,4,7),m.sxr_norm)[0].shape==(2,1)
    default=GaussianNLLViTLocal(model_kwargs=dict(embed_dim=8,hidden_dim=16,num_heads=1,
             num_layers=1,num_channels=7,patch_size=1,num_patches=16,dropout=0.),sxr_norm=[-5.4,0.7])
    assert default.uncertainty_enabled and default.model.patch_global_context_enabled
    assert default.model.mask_mode=='local'


def test_flat_baseline_formula():
    m=make_model(False,False)
    logits=torch.zeros(2,16)
    baseline=(10**m.sxr_norm[0]-1e-8)/16
    torch.testing.assert_close(m.model._patch_flux_from_logits(logits,m.sxr_norm),torch.full_like(logits,baseline))
    logits[:,0]=1
    flux=m.model._patch_flux_from_logits(logits,m.sxr_norm)
    torch.testing.assert_close(flux[:,0],10*flux[:,1])


def test_config_sections_and_disable_flags():
    c=load_training_config('training/train_config.yaml')
    assert c['vit_architecture']['patch_global_context']['enabled']
    assert c['uncertainty']['huber_delta']==c['mean_loss']['huber_delta']
    raw=dict(model_type='uncertainty',global_head=dict(enabled=False),uncertainty=dict(enabled=False),
             mean_loss=dict(huber_delta=1.),local_head=dict(max_abs_log10_multiplier=7.))
    result=resolve_model_settings(raw)
    assert not result['vit_architecture']['patch_global_context']['enabled']
    assert result['uncertainty']['huber_delta']==1.
    assert result['uncertainty']['max_abs_log10_patch_multiplier']==7.
    assert 'vit_architecture' not in raw


def test_lightning_epoch_rmse_and_checkpoint_roundtrip(tmp_path):
    m=make_model()
    data=DataLoader(TensorDataset(torch.randn(5,4,4,7),torch.tensor([-2.,0.,0.1,0.2,3.])),batch_size=2)
    trainer=Trainer(accelerator='cpu',devices=1,max_epochs=1,logger=False,
                    plugins=[LightningEnvironment()],
                    enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
                    num_sanity_val_steps=0,default_root_dir=tmp_path)
    trainer.fit(m,data,data)
    metrics=trainer.callback_metrics
    torch.testing.assert_close(metrics['val/local_rmse'].square(),metrics['val/local_mse'],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(metrics['val/rmse'].square(),metrics['val/mse'],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(metrics['val/total_loss'],metrics['val/mean_loss']+metrics['val/nll'])
    path=tmp_path/'model.ckpt'
    trainer.save_checkpoint(path)
    loaded=GaussianNLLViTLocal.load_from_checkpoint(path,weights_only=False)
    m.eval();loaded.eval()
    x=torch.randn(2,4,4,7)
    with torch.no_grad():
        for a,b in zip(m(x),loaded(x)):torch.testing.assert_close(a,b)
