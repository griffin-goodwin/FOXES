"""Independent inference shards preserve samples, maps, and CSV provenance."""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

from forecasting import inference, run_parallel_inference as parallel
from forecasting.model import CoupledGaussianViTLocal, ViTLocal


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_sampling_then_sharding_preserves_every_selected_sample_once():
    dataset = TensorDataset(torch.arange(20))
    dataset.samples = [f'timestamp_{i:02d}' for i in range(20)]
    sampled, selected = inference.select_inference_samples(dataset, {'count': 11, 'seed': 5})
    pairs = []
    for index in range(3):
        shard, times = inference.shard_inference_samples(sampled, selected, index, 3)
        assert times == selected[index::3]
        pairs.extend((stamp, shard[i][0].item()) for i, stamp in enumerate(times))
    assert len({stamp for stamp, _ in pairs}) == 11
    assert sorted(pairs) == [(stamp, sampled[i][0].item()) for i, stamp in enumerate(selected)]


@pytest.mark.parametrize('index,count', [(-1, 2), (2, 2), (0, 0), (1, 2)])
def test_invalid_or_empty_shard_is_rejected(index, count):
    dataset = TensorDataset(torch.arange(1))
    with pytest.raises(ValueError):
        inference.shard_inference_samples(dataset, ['a'], index, count)


def test_sharding_rejects_misaligned_timestamps():
    with pytest.raises(ValueError, match='one timestamp per'):
        inference.shard_inference_samples(TensorDataset(torch.arange(2)), ['a'], 0, 2)


def test_merge_sorts_rows_and_preserves_all_numeric_text(tmp_path):
    fields = ['timestamp', 'predictions', 'groundtruth', 'variance_normalized',
              'lower_95', 'upper_95', 'legacy_model', 'flux_scale_factor']
    rows = [dict(zip(fields, [str(i), '1.234567890123456789e-08', '', '0.00100000000000001',
                             '9.999999999e-10', '2.000000000000000001e-08', 'True', '0.7']))
            for i in range(4)]
    paths = [tmp_path / 'a.csv', tmp_path / 'b.csv']
    write_csv(paths[0], rows[::2])
    write_csv(paths[1], rows[1::2])
    output = tmp_path / 'merged' / 'predictions.csv'
    assert parallel.merge_shard_csvs(paths, output) == 4
    assert read_csv(output) == rows
    assert not output.with_name('predictions.csv.partial').exists()


@pytest.mark.parametrize('problem', ['duplicate', 'columns', 'timestamp', 'empty'])
def test_invalid_merge_preserves_existing_output(tmp_path, problem):
    paths = [tmp_path / 'a.csv', tmp_path / 'b.csv']
    row = {'timestamp': 'a', 'predictions': '1e-6'}
    write_csv(paths[0], [row])
    if problem == 'duplicate':
        write_csv(paths[1], [row])
    elif problem == 'columns':
        write_csv(paths[1], [{'timestamp': 'b', 'different': '1'}])
    elif problem == 'timestamp':
        write_csv(paths[1], [{'different': '1'}])
    else:
        for path in paths:
            write_csv(path, [], fields=list(row))
    output = tmp_path / 'merged.csv'
    output.write_text('previous result')
    with pytest.raises(ValueError):
        parallel.merge_shard_csvs(paths, output)
    assert output.read_text() == 'previous result'


@pytest.mark.parametrize('value', ['', '0', '0,0', '0,00', '0,-1', '0,', 'a,1'])
def test_invalid_gpu_lists_are_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parallel.parse_gpu_ids(value)


def test_runner_resolves_paths_and_assigns_one_cuda_device_per_worker(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    config_dir = project / 'configs'
    config_dir.mkdir(parents=True)
    config_path = config_dir / 'inference.yaml'
    config_path.write_text(yaml.safe_dump({
        'path_root': '..', 'output_path': 'outputs/predictions.csv', 'device': 'cpu',
    }))
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    calls = []

    def launch(command, **kwargs):
        index = int(command[command.index('--shard-index') + 1])
        output = Path(command[command.index('--output-path') + 1])
        assert command[command.index('--config') + 1] == str(config_path)
        assert command[command.index('--num-shards') + 1] == '2'
        assert command[command.index('--device') + 1] == 'cuda'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ['2', '5'][index]
        assert Path(command[1]).is_absolute()
        assert output.parent == project / 'outputs'
        write_csv(output, [
            {'timestamp': str(position), 'predictions': '1.00000000000001e-6'}
            for position in range(index, 4, 2)
        ])
        calls.append((command, kwargs))
        return SimpleNamespace(wait=lambda: 0, poll=lambda: 0)

    monkeypatch.setattr(parallel.subprocess, 'Popen', launch)
    monkeypatch.setattr('sys.argv', ['parallel', '--config', str(config_path), '--gpus', '2,5'])
    parallel.main()
    assert len(calls) == 2
    assert all(kwargs['stdout'].closed for _, kwargs in calls)
    assert [row['timestamp'] for row in read_csv(project / 'outputs/predictions.csv')] == list('0123')
    assert yaml.safe_load(config_path.read_text())['device'] == 'cpu'


def test_failed_worker_does_not_replace_merged_output(tmp_path, monkeypatch):
    output = tmp_path / 'predictions.csv'
    output.write_text('previous result')
    config = tmp_path / 'inference.yaml'
    config.write_text(yaml.safe_dump({'output_path': str(output)}))
    logs = []

    def launch(command, **kwargs):
        index = int(command[command.index('--shard-index') + 1])
        logs.append(kwargs['stdout'])
        return SimpleNamespace(wait=lambda: index, poll=lambda: index)

    monkeypatch.setattr(parallel.subprocess, 'Popen', launch)
    monkeypatch.setattr('sys.argv', ['parallel', '--config', str(config), '--gpus', '0,1'])
    with pytest.raises(RuntimeError, match='Inference shards failed'):
        parallel.main()
    assert output.read_text() == 'previous result'
    assert all(log.closed for log in logs)


def test_spawn_failure_terminates_started_workers(tmp_path, monkeypatch):
    config = tmp_path / 'inference.yaml'
    config.write_text(yaml.safe_dump({'output_path': str(tmp_path / 'predictions.csv')}))
    state = {'launched': 0, 'terminated': False, 'waited': False}
    logs = []

    def terminate():
        state['terminated'] = True

    def wait():
        state['waited'] = True
        return -15

    def launch(command, **kwargs):
        logs.append(kwargs['stdout'])
        state['launched'] += 1
        if state['launched'] == 2:
            raise OSError('could not spawn')
        return SimpleNamespace(poll=lambda: None, terminate=terminate, wait=wait)

    monkeypatch.setattr(parallel.subprocess, 'Popen', launch)
    monkeypatch.setattr('sys.argv', ['parallel', '--config', str(config), '--gpus', '0,1'])
    with pytest.raises(OSError, match='could not spawn'):
        parallel.main()
    assert state['terminated'] and state['waited']
    assert all(log.closed for log in logs)


@pytest.mark.parametrize('args', [
    ['--num-shards', '0'], ['--shard-index', '1'],
    ['--num-shards', '2'], ['--num-shards', '2', '--shard-index', '2'],
])
def test_invalid_shard_cli_fails_before_loading_model(monkeypatch, args):
    monkeypatch.setattr('sys.argv', ['inference', '--config', 'missing.yaml', *args])
    with pytest.raises(SystemExit) as error:
        inference.main()
    assert error.value.code == 2


@pytest.mark.parametrize('kind', ['deterministic', 'legacy', 'gaussian', 'mean_only'])
@pytest.mark.parametrize('prediction_only', [False, True])
def test_cpu_sharded_cli_matches_single_run_csv_and_maps(
    tmp_path, monkeypatch, kind, prediction_only,
):
    torch.manual_seed(0)
    architecture = dict(
        embed_dim=8, hidden_dim=16, num_channels=7, num_heads=2,
        num_layers=1, patch_size=1, num_patches=4, dropout=0.0,
        mask_mode='local', local_window=1,
    )
    norm = np.array([-6.0, 1.0], dtype=np.float32)
    if kind in ('gaussian', 'mean_only'):
        model = CoupledGaussianViTLocal(
            architecture, norm, uncertainty_kwargs={'enabled': kind == 'gaussian'},
        ).eval()
    else:
        model = ViTLocal(architecture, norm).eval()
    dataset = TensorDataset(torch.randn(9, 2, 2, 7), torch.full((9,), 1e-6))
    start = datetime(2016, 11, 18, 23, 56)
    dataset.samples = [(start + timedelta(minutes=i)).isoformat() for i in range(9)]
    monkeypatch.setattr(inference, 'AIAGOESDataset', lambda **kwargs: dataset)

    def load_model(config):
        assert config['device'] == 'cpu'
        return model

    monkeypatch.setattr(inference, 'load_model_from_config', load_model)
    config_path = tmp_path / 'inference.yaml'
    config = {
        'data': {'aia_dir': 'unused', 'sxr_dir': 'unused'},
        'model_params': {'input_size': 2, 'patch_size': 1, 'batch_size': 1,
                         'no_weights': True, 'no_flux': False, 'no_patch_uncertainty': False},
        'device': 'cuda', 'wavelengths': [94, 131, 171, 193, 211, 304, 335],
        'prediction_only': prediction_only, 'legacy_model': kind == 'legacy',
        'sampling': {'count': 7, 'seed': 11}, 'num_workers': 0, 'pin_memory': False,
    }
    for mode in ('single', 'sharded'):
        config.update({
            'output_path': str(tmp_path / f'{mode}.csv'),
            'flux_path': str(tmp_path / mode / 'flux'),
            'patch_uncertainty_path': str(tmp_path / mode / 'uncertainty'),
        })
        config_path.write_text(yaml.safe_dump(config))
        if mode == 'single':
            monkeypatch.setattr('sys.argv', [
                'inference', '--config', str(config_path), '--device', 'cpu',
            ])
            inference.main()
        else:
            paths = [tmp_path / f'shard-{i}.csv' for i in range(3)]
            for index, path in enumerate(paths):
                monkeypatch.setattr('sys.argv', [
                    'inference', '--config', str(config_path), '--device', 'cpu',
                    '--num-shards', '3', '--shard-index', str(index),
                    '--output-path', str(path),
                ])
                inference.main()
            assert parallel.merge_shard_csvs(paths, tmp_path / 'sharded.csv') == 7
    assert read_csv(tmp_path / 'sharded.csv') == read_csv(tmp_path / 'single.csv')
    single_maps = {path.relative_to(tmp_path / 'single')
                   for path in (tmp_path / 'single').rglob('*.npy')}
    sharded_maps = {path.relative_to(tmp_path / 'sharded')
                    for path in (tmp_path / 'sharded').rglob('*.npy')}
    assert single_maps == sharded_maps
    assert len([path for path in single_maps if path.parts[0] == 'flux']) == 7
    assert len([path for path in single_maps if path.parts[0] == 'uncertainty']) == (
        7 if kind == 'gaussian' else 0
    )
    for path in single_maps:
        np.testing.assert_array_equal(
            np.load(tmp_path / 'single' / path), np.load(tmp_path / 'sharded' / path),
        )
