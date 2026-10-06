"""Global output initialization must preserve independent gradient paths."""
import numpy as np
import pytest
import torch

from forecasting.model import normalize_sxr
from forecasting.uncertainty_model import GaussianNLLViTLocal, PatchGlobalContext


def test_default_random_linear_initialization_and_bounded_output():
    torch.manual_seed(17)
    head = PatchGlobalContext(8, grid_size=2, attention_dim=8, num_heads=2,
                              max_abs_adjustment=1.5)
    assert head.output[-1].weight.count_nonzero() > 0
    assert head.output[-1].bias.count_nonzero() > 0
    limit = head.output[-1].in_features ** -0.5
    assert head.output[-1].weight.abs().max() <= limit
    assert head.output[-1].bias.abs().max() <= limit
    with torch.no_grad():
        head.output[-1].bias.fill_(100.)
    result = head(torch.randn(2, 16, 8), torch.randn(2, 16, 8))
    assert result.abs().max() <= 1.5


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_rejects_invalid_correction_range(value):
    with pytest.raises(ValueError, match='max_abs_adjustment'):
        PatchGlobalContext(8, attention_dim=8, num_heads=2, max_abs_adjustment=value)


def test_first_update_reaches_global_attention_but_not_local_model():
    torch.manual_seed(71)
    model = GaussianNLLViTLocal(
        model_kwargs=dict(embed_dim=8, hidden_dim=16, num_channels=7, num_heads=1,
                          num_layers=1, patch_size=1, num_patches=16, dropout=0.,
                          mask_mode='local', local_window=3,
                          patch_global_context=dict(enabled=True, grid_size=2,
                                                    attention_dim=8, num_heads=2,
                                                    max_abs_adjustment=1.5)),
        sxr_norm=np.array([-5.475177, 0.66483974], dtype=np.float32),
        uncertainty_kwargs=dict(enabled=True, class_weighting='none', huber_delta=0.15),
    )
    outputs = model.model(torch.randn(2, 4, 4, 7), model.sxr_norm, return_local=True)
    global_mean, local_flux = outputs[1], outputs[-1]
    target = global_mean.detach() + 0.4
    global_loss = torch.nn.functional.huber_loss(global_mean, target, delta=0.15)
    global_loss.backward(retain_graph=True)
    parameters = dict(model.model.named_parameters())
    for name, parameter in parameters.items():
        if not name.startswith('patch_global_context.'):
            assert parameter.grad is None, name
    for name in ('query_projection.weight', 'memory_projection.weight',
                 'attention.in_proj_weight', 'output.3.weight'):
        grad = parameters['patch_global_context.' + name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    before = model.model.patch_global_context.query_projection.weight.detach().clone()
    optimizer = model.configure_optimizers()['optimizer']
    # Check the reverse gradient path before modifying parameters in the graph.
    model.zero_grad(set_to_none=True)
    local_mean = normalize_sxr(local_flux.sum(dim=1), model.sxr_norm)
    torch.nn.functional.huber_loss(local_mean, target, delta=0.15).backward()
    assert model.model.input_layer.weight.grad.abs().sum() > 0
    assert all(p.grad is None for n,p in parameters.items() if n.startswith('patch_global_context.'))
    # A fresh global-only forward must actually update the attention parameters.
    model.zero_grad(set_to_none=True)
    global_mean = model.model(torch.randn(2,4,4,7), model.sxr_norm)[1]
    torch.nn.functional.huber_loss(global_mean, global_mean.detach()+0.4, delta=0.15).backward()
    optimizer.step()
    assert not torch.equal(before, model.model.patch_global_context.query_projection.weight)
