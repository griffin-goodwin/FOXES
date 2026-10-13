from unittest.mock import Mock

import pytest
import torch

from training import strategies
from training.train import resolve_devices


@pytest.mark.parametrize('devices', ['all', [0, 1]])
def test_multiple_gpus_use_training_stream_strategy(monkeypatch, devices):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    accelerator, selected, strategy = resolve_devices(devices)
    assert accelerator == 'gpu'
    assert selected == (-1 if devices == 'all' else devices)
    assert isinstance(strategy, strategies.TrainingStreamDDPStrategy)


@pytest.mark.parametrize('devices', ['all', [0], 0, -1])
def test_single_device_keeps_auto_strategy(monkeypatch, devices):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    assert resolve_devices(devices)[2] == 'auto'


def test_no_cuda_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_devices('all') == ('cpu', 1, 'auto')


def test_ddp_setup_preserves_current_stream_and_options(monkeypatch):
    strategy = strategies.TrainingStreamDDPStrategy(find_unused_parameters=True)
    monkeypatch.setattr(strategy, 'determine_ddp_device_ids', lambda: [0])
    new_stream = Mock(side_effect=AssertionError('DDP must use the training stream'))
    monkeypatch.setattr(torch.cuda, 'Stream', new_stream)
    switch_stream = Mock(side_effect=AssertionError('DDP must not switch streams'))
    monkeypatch.setattr(torch.cuda, 'stream', switch_stream)
    ddp = Mock()
    monkeypatch.setattr(strategies, 'DistributedDataParallel', ddp)
    model = torch.nn.Linear(2, 1)

    assert strategy._setup_model(model) is ddp.return_value
    ddp.assert_called_once_with(
        module=model, device_ids=[0], find_unused_parameters=True,
    )
    new_stream.assert_not_called()
    switch_stream.assert_not_called()
