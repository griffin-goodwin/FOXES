#!/usr/bin/env python3
"""Validate standalone subset configs, optionally smoke-check or train them."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUBSET = Path('/data/FOXES_screening/subset-og-quick')
# Ordered so each successive comparison has an explicit control.
ARMS = {
    #'original_legacy': ('deterministic', 'original', 'inverted', 9, False),
    #'original_matched': ('uncertainty', 'original', 'inverted', 9, False),
    #'local_original': ('uncertainty', 'original', 'local', 3, False),
    #'local_multiplier': ('uncertainty', 'multiplier', 'local', 3, False),
    'local_global_multiplier': ('uncertainty', 'multiplier', 'local', 3, True),
    'local_unclamped': ('uncertainty', 'original_unclamped', 'local', 3, False),
}
GLOBAL_KEYS = ('global_patch_adjustment', 'max_abs_log10_global_adjustment',
               'global_patch_head_init_std', 'independent_local_global', 'patch_global_context')


def comparison_payload(config):
    c = deepcopy(config)
    c.pop('model_type')
    c.pop('mean_parameterization')
    c.pop('wandb')
    c['data'].pop('checkpoints_dir')
    c['checkpoint'].pop('monitor')
    for key in ('mask_mode', 'local_window', *GLOBAL_KEYS):
        c['vit_architecture'].pop(key, None)
    return c


def validate_configs(directory=HERE):
    configs = {}
    for name, (kind, mean, mask, window, correction) in ARMS.items():
        path = directory / f'{name}.yaml'
        c = yaml.safe_load(path.read_text())
        if 'base_config' in c:
            raise ValueError(f'{name}: configs must be standalone')
        a = c['vit_architecture']
        actual = (c['model_type'], c['mean_parameterization'], a['mask_mode'],
                  a['local_window'], a.get('global_patch_adjustment', False))
        if actual != (kind, mean, mask, window, correction):
            raise ValueError(f'{name}: unexpected arm settings {actual}')
        if kind == 'deterministic' and any(k in a for k in GLOBAL_KEYS):
            raise ValueError('Legacy model.py does not accept correction kwargs')
        if kind == 'uncertainty' and (
            a['max_abs_log10_global_adjustment'] != 0.3
            or a['global_patch_head_init_std'] != 0.0
            or a['independent_local_global']
            or a['patch_global_context'] != {'enabled': False}
        ):
            raise ValueError(f'{name}: unexpected global correction settings')
        expected_arch = dict(embed_dim=256, hidden_dim=1024, num_layers=8,
                             num_heads=8, patch_size=8, num_patches=4096,
                             num_channels=7, num_classes=1, dropout=0.1)
        if any(a.get(k) != v for k, v in expected_arch.items()):
            raise ValueError(f'{name}: backbone differs from the published dimensions')
        expected_data = dict(aia_dir=SUBSET/'AIA_processed', sxr_dir=SUBSET/'SXR_processed',
                             sxr_norm_path=SUBSET/'SXR_processed'/'normalized_sxr.npy')
        if any(Path(c['data'][k]) != v for k, v in expected_data.items()):
            raise ValueError(f'{name}: must use the requested screening subset')
        if c['data']['aia_norm_path'] is not None:
            raise ValueError(f'{name}: preprocessed subset must not be renormalized')
        if c['gpu_ids'] != [0]:
            raise ValueError(f'{name}: use one visible GPU per arm')
        for key in ('batch_size', 'accumulate_grad_batches'):
            if type(c.get(key)) is not int or c[key] < 1:
                raise ValueError(f'{name}: {key} must be a positive integer')
        if c['uncertainty']['enabled'] or c['early_stopping']['enabled']:
            raise ValueError(f'{name}: uncertainty and early stopping must be disabled')
        if c['epochs'] != c['optimizer']['scheduler']['T_max']:
            raise ValueError(f'{name}: cosine schedule must match the epoch budget')
        monitor = 'val_mae_loss' if kind == 'deterministic' else 'val/mae'
        if c['checkpoint']['monitor'] != monitor or c['checkpoint']['mode'] != 'min':
            raise ValueError(f'{name}: checkpoint selection must minimize validation MAE')
        if c['checkpoint']['initialize_from'] or c['checkpoint']['resume_from']:
            raise ValueError(f'{name}: comparison must start from fresh weights')
        configs[name] = c
    reference = comparison_payload(configs['original_legacy'])
    if any(comparison_payload(c) != reference for c in configs.values()):
        raise ValueError('Configs differ outside the documented experimental factors')
    outputs = [c['data']['checkpoints_dir'] for c in configs.values()]
    if len(set(outputs)) != len(outputs):
        raise ValueError('Checkpoint directories must be distinct')
    return configs


def check_data():
    # Check actual paired filenames; do not touch the held-out test values.
    for split in ('train', 'val', 'test'):
        aia = {p.stem for p in (SUBSET/'AIA_processed'/split).glob('*.npy')}
        sxr = {p.stem for p in (SUBSET/'SXR_processed'/split).glob('*.npy')}
        if not aia or aia != sxr:
            raise ValueError(f'{split}: missing/unpaired subset samples ({len(aia)} AIA, {len(sxr)} SXR)')
        print(f'{split}: {len(aia):,} paired subset samples', flush=True)
    if not (SUBSET/'SXR_processed'/'normalized_sxr.npy').is_file():
        raise FileNotFoundError('Subset SXR normalization is missing')


def smoke_check(configs):
    """Check real model constructors and finite gradients on smaller spatial grids."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from forecasting.model import ViTLocal, normalize_sxr
    from forecasting.uncertainty_model import GaussianNLLViTLocal
    torch.set_num_threads(2)
    norm = np.load(SUBSET/'SXR_processed'/'normalized_sxr.npy')
    for name, c in configs.items():
        torch.manual_seed(c['seed'])
        architecture = deepcopy(c['vit_architecture'])
        # 12x12 grid leaves unmasked distant patches for inverted 9x9 attention.
        # All backbone widths, depth, patch size, and head settings stay unchanged.
        architecture['num_patches'] = 144
        kwargs = dict(model_kwargs=architecture, sxr_norm=norm,
                      weight_decay=c['optimizer']['weight_decay'],
                      scheduler_kwargs=c['optimizer']['scheduler'])
        if c['model_type'] == 'deterministic':
            model = ViTLocal(**kwargs, base_weights=c['loss']['base_weights'],
                             loss_kwargs={k:c['loss'][k] for k in
                                          ('huber_delta', 'window_size', 'adaptive_multipliers')})
        else:
            u = deepcopy(c['uncertainty'])
            # Synthetic weights only for this finite-gradient check; train.py
            # computes the actual weights from the subset training targets.
            u['class_weights'] = dict(quiet=1., c_class=1., m_class=1., x_class=1.)
            model = GaussianNLLViTLocal(**kwargs, uncertainty_kwargs=u,
                                       mean_parameterization=c['mean_parameterization'])
        images = torch.rand(2, 96, 96, 7) * 2 - 1
        outputs = model.model(images, model.sxr_norm)
        raw = outputs[0] if c['model_type'] == 'deterministic' else outputs[1]
        loss = F.huber_loss(normalize_sxr(raw, model.sxr_norm).reshape(-1),
                            torch.zeros(2), delta=0.3)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f'{name}: nonfinite loss/gradient')
        optimizer = model.configure_optimizers()['optimizer']
        optimizer.step()
        print(f'{name}: constructor, forward, backward, optimizer OK (reduced spatial grid)', flush=True)
        del optimizer, model, outputs, loss, gradients


def assign_gpus(arms, gpu=0, gpu_ids=None, parallel=False):
    if len(set(arms)) != len(arms):
        raise ValueError('Each arm may be selected only once')
    if gpu < 0:
        raise ValueError('--gpu must be nonnegative')
    if gpu_ids is not None:
        try:
            ids = [int(value.strip()) for value in gpu_ids.split(',')]
        except ValueError as error:
            raise ValueError('--gpu-ids must be comma-separated GPU indices') from error
        if any(value < 0 for value in ids) or len(set(ids)) != len(ids):
            raise ValueError('--gpu-ids must contain distinct nonnegative indices')
        if not parallel:
            raise ValueError('--gpu-ids requires --parallel; use --gpu for sequential runs')
    else:
        ids = list(range(len(arms))) if parallel else [gpu]
    return [ids[index % len(ids)] for index in range(len(arms))] if parallel else [gpu] * len(arms)


def run_jobs(jobs, log_dir):
    """Run one child per GPU, advancing each GPU's queue independently."""
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    pending = list(jobs)
    active = {}
    try:
        failures = []
        while pending or active:
            for job in pending[:]:
                arm, gpu, command = job
                if gpu in active:
                    continue
                log_path = log_dir / f'{arm}.log'
                handle = log_path.open('x')
                handles.append(handle)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
                process = subprocess.Popen(command, cwd=ROOT, env=env,
                                           stdout=handle, stderr=subprocess.STDOUT)
                processes.append((arm, process, log_path))
                active[gpu] = (arm, process, log_path, handle)
                pending.remove(job)
                print(f'Started {arm} on GPU {gpu}; log: {log_path}', flush=True)
            for gpu, (arm, process, log_path, handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[gpu]
                print(f'{arm}: exit {code}; log: {log_path}', flush=True)
                if code:
                    failures.append(f'{arm} (exit {code}, {log_path})')
            if active:
                time.sleep(0.2)
        if failures:
            raise RuntimeError('Training failed: ' + '; '.join(failures))
    finally:
        # Clean up only children launched by this invocation on interruption/error.
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', nargs='+', choices=list(ARMS), default=list(ARMS))
    parser.add_argument('--gpu', type=int, default=0, help='Physical GPU for sequential runs')
    parser.add_argument('--parallel', action='store_true', help='Run one arm at a time per GPU, queueing remaining arms')
    parser.add_argument('--gpu-ids', help='Comma-separated GPUs for parallel runs (default: 0 through number of arms minus one)')
    parser.add_argument('--log-dir', type=Path, help='Directory for separate console logs (default: timestamped logs/ directory)')
    parser.add_argument('--python', default=sys.executable, help='Training environment Python')
    parser.add_argument('--smoke', action='store_true', help='CPU finite-gradient check of all six arms')
    parser.add_argument('--run', action='store_true', help='Launch selected arms')
    args = parser.parse_args()
    configs = validate_configs()
    check_data()
    try:
        gpu_ids = assign_gpus(args.arm, args.gpu, args.gpu_ids, args.parallel)
    except ValueError as error:
        parser.error(str(error))
    if args.smoke:
        smoke_check(configs)
    jobs = [(arm, gpu, [args.python, str(ROOT/'training'/'train.py'), '--config', str(HERE/f'{arm}.yaml')])
            for arm, gpu in zip(args.arm, gpu_ids)]
    for arm, gpu, command in jobs:
        print(f'[{arm}] CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}', flush=True)
    if not args.run:
        print('Validated. Add --run to train; no training launched.')
        return
    for arm in args.arm:
        directory = Path(configs[arm]['data']['checkpoints_dir'])
        if directory.exists() and any(directory.glob('*.ckpt')):
            raise FileExistsError(f'{directory} contains checkpoints; choose new output paths before rerunning')
    visible = list(dict.fromkeys(gpu_ids))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, visible)))
    subprocess.run([args.python, '-c',
                    'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; '
                    f'assert torch.cuda.device_count() == {len(visible)}, "Not all requested GPUs are available"'],
                   env=env, cwd=ROOT, check=True)
    log_dir = args.log_dir or HERE/'logs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.parallel:
        run_jobs(jobs, log_dir)
    else:
        for job in jobs:
            run_jobs([job], log_dir)


if __name__ == '__main__':
    main()
