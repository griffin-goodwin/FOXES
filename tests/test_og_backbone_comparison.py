"""Protect the subset experiment against confounded or misdirected runs."""
import shutil

import pytest
import yaml

from experiments.og_backbone_comparison import run_comparison as runner
from experiments.og_backbone_comparison.run_comparison import HERE, ARMS, validate_configs


def copy_configs(tmp_path):
    for name in ARMS:
        shutil.copyfile(HERE / f'{name}.yaml', tmp_path / f'{name}.yaml')
    return tmp_path


def change(path, section, key, value):
    c = yaml.safe_load(path.read_text())
    c[section][key] = value
    path.write_text(yaml.safe_dump(c))


def test_standalone_configs_are_matched():
    configs = validate_configs()
    assert len(configs) == 6
    assert configs['original_legacy']['model_type'] == 'deterministic'
    assert len({(c['batch_size'], c['accumulate_grad_batches']) for c in configs.values()}) == 1


def test_accepts_user_edited_matched_batch_settings(tmp_path):
    directory = copy_configs(tmp_path)
    for name in ARMS:
        path = directory/f'{name}.yaml'
        c = yaml.safe_load(path.read_text())
        c.update(batch_size=8, accumulate_grad_batches=3)
        path.write_text(yaml.safe_dump(c))
    assert all(c['batch_size'] == 8 for c in validate_configs(directory).values())


def test_rejects_mismatched_batch_settings(tmp_path):
    directory = copy_configs(tmp_path)
    path = directory/'local_multiplier.yaml'
    c = yaml.safe_load(path.read_text())
    c['batch_size'] += 1
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='documented experimental factors'):
        validate_configs(directory)


def test_rejects_full_data_training_path(tmp_path):
    directory = copy_configs(tmp_path)
    change(directory/'local_multiplier.yaml', 'data', 'aia_dir', '/data/FOXES_OG/AIA_processed')
    with pytest.raises(ValueError, match='screening subset'):
        validate_configs(directory)


def test_rejects_optimizer_confound(tmp_path):
    directory = copy_configs(tmp_path)
    change(directory/'local_global_multiplier.yaml', 'optimizer', 'weight_decay', 0.1)
    with pytest.raises(ValueError, match='documented experimental factors'):
        validate_configs(directory)


def test_rejects_wrong_mae_monitor(tmp_path):
    directory = copy_configs(tmp_path)
    change(directory/'original_legacy.yaml', 'checkpoint', 'monitor', 'val/mae')
    with pytest.raises(ValueError, match='validation MAE'):
        validate_configs(directory)


def test_parallel_gpu_assignment():
    assert runner.assign_gpus(list(ARMS), parallel=True) == list(range(6))
    assert runner.assign_gpus(list(ARMS), gpu_ids='0,1', parallel=True) == [0, 1, 0, 1, 0, 1]
    assert runner.assign_gpus(list(ARMS), gpu_ids='0,1,2,3,4,5,6,7', parallel=True) == list(range(6))
    assert runner.assign_gpus(['original_legacy', 'local_multiplier'], gpu=3) == [3, 3]


@pytest.mark.parametrize('ids', ['0,0,1,2,3,4', '', '0,-1,2,3,4,5', '0,x'])
def test_rejects_invalid_parallel_gpus(ids):
    with pytest.raises(ValueError):
        runner.assign_gpus(list(ARMS), gpu_ids=ids, parallel=True)


def test_parallel_starts_all_before_waiting_and_reports_failure(monkeypatch, tmp_path):
    events = []
    environments = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.arm = command[0]
            self.code = None
            environments.append(kwargs['env'])
            events.append(('start', self.arm))

        def wait(self, timeout=None):
            events.append(('wait', self.arm))
            self.code = 1 if self.arm == 'failed' else 0
            return self.code

        def poll(self):
            if self.code is None:
                return self.wait()
            return self.code

    monkeypatch.setattr(runner.subprocess, 'Popen', FakeProcess)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    with pytest.raises(RuntimeError, match='failed .*exit 1'):
        runner.run_jobs([('failed', 2, ['failed']), ('ok', 5, ['ok'])], tmp_path)
    assert events == [('start', 'failed'), ('start', 'ok'), ('wait', 'failed'), ('wait', 'ok')]
    assert [env['CUDA_VISIBLE_DEVICES'] for env in environments] == ['2', '5']
    assert runner.os.environ['CUDA_VISIBLE_DEVICES'] == '7'
    assert (tmp_path/'failed.log').exists() and (tmp_path/'ok.log').exists()


def test_gpu_queues_advance_independently_without_overlap(monkeypatch, tmp_path):
    live = set()
    starts = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.name = command[0]
            self.gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
            assert self.gpu not in live
            live.add(self.gpu)
            starts.append(self.name)
            self.remaining = 4 if self.name == 'slow' else 1
            self.code = None

        def poll(self):
            if self.code is None:
                self.remaining -= 1
                if self.remaining == 0:
                    self.code = 0
                    live.remove(self.gpu)
            return self.code

    monkeypatch.setattr(runner.subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    jobs = [('slow', 0, ['slow']), ('fast', 1, ['fast']),
            ('next0', 0, ['next0']), ('next1', 1, ['next1'])]
    runner.run_jobs(jobs, tmp_path)
    assert starts == ['slow', 'fast', 'next1', 'next0']
    assert not live
