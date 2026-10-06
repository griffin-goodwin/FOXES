#!/usr/bin/env python3
"""Validate and launch the subset independent-local/global parameter sweep."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from training.configuration import resolve_model_settings
from experiments.og_backbone_comparison.run_comparison import (
    SUBSET, assign_gpus, check_data, run_jobs,
)

ARMS = {f'range{r:.1f}_huber{d:.2f}'.replace('.', 'p'): (r, d)
        for r in (1.5,) for d in (0.15, 0.30, 1.0)}


def comparison_payload(config):
    c = deepcopy(config)
    c['uncertainty'].pop('huber_delta')
    c.pop('mean_loss', None)
    c['data'].pop('checkpoints_dir')
    c.pop('wandb')
    return c


def validate_configs(directory=HERE):
    configs = {}
    for name, (limit, delta) in ARMS.items():
        raw_config = yaml.safe_load((directory/f'{name}.yaml').read_text())
        c = resolve_model_settings(raw_config)
        if 'base_config' in c:
            raise ValueError('Sweep configs must be standalone')
        a = c['vit_architecture']
        if c['model_type'] != 'uncertainty':
            raise ValueError(f'{name}: use the independent flat-field multiplier model')
        p = a['patch_global_context']
        if type(p['enabled']) is not bool or p['max_abs_adjustment'] != limit or c['uncertainty']['huber_delta'] != delta:
            raise ValueError(f'{name}: incorrect sweep factors')
        if type(c['uncertainty']['enabled']) is not bool:
            raise ValueError(f'{name}: use a boolean uncertainty toggle')
        for key, value in dict(embed_dim=512, hidden_dim=2048, num_layers=12,
                               num_heads=8, patch_size=8, num_patches=4096,
                               mask_mode='local', local_window=3).items():
            if a[key] != value:
                raise ValueError(f'{name}: {key} differs from OG-PatchGlobal-Independent')
        for key, value in dict(grid_size=32, attention_dim=128, num_heads=4, dropout=0.0).items():
            if p[key] != value:
                raise ValueError(f'{name}: global {key} differs from the reference')
        expected = dict(aia_dir=SUBSET/'AIA_processed', sxr_dir=SUBSET/'SXR_processed',
                        sxr_norm_path=SUBSET/'SXR_processed'/'normalized_sxr.npy')
        if any(Path(c['data'][key]) != value for key, value in expected.items()):
            raise ValueError(f'{name}: must use the screening subset')
        if c['data']['aia_norm_path'] is not None or c['gpu_ids'] != [0]:
            raise ValueError(f'{name}: preserve normalized input and one GPU per run')
        for key in ('batch_size', 'accumulate_grad_batches', 'epochs'):
            if type(c[key]) is not int or c[key] < 1:
                raise ValueError(f'{name}: {key} must be a positive integer')
        if c['epochs'] != c['optimizer']['scheduler']['T_max'] or c['early_stopping']['enabled']:
            raise ValueError(f'{name}: match the full cosine schedule without early stopping')
        if c['checkpoint']['monitor'] != 'val/mae' or c['checkpoint']['mode'] != 'min':
            raise ValueError(f'{name}: select corrected validation MAE')
        if c['checkpoint']['initialize_from'] is not None or c['checkpoint']['resume_from'] is not None:
            raise ValueError(f'{name}: start from scratch without any checkpoint')
        configs[name] = c
    reference = comparison_payload(next(iter(configs.values())))
    if any(comparison_payload(c) != reference for c in configs.values()):
        raise ValueError('Arms differ outside Huber delta and output metadata')
    for section, key in [('data', 'checkpoints_dir'), ('wandb', 'run_name')]:
        if len({c[section][key] for c in configs.values()}) != len(configs):
            raise ValueError('Every arm needs distinct output names')
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', nargs='+', choices=list(ARMS), default=list(ARMS))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--gpu-ids', help='GPU queues, e.g. 0,1 for two concurrent runs')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    configs = validate_configs()
    check_data()
    try:
        ids = assign_gpus(args.arm, args.gpu, args.gpu_ids, args.parallel)
    except ValueError as error:
        parser.error(str(error))
    jobs = [(name, gpu, [args.python, str(ROOT/'training'/'train.py'), '--config', str(HERE/f'{name}.yaml')])
            for name, gpu in zip(args.arm, ids)]
    for name, gpu, command in jobs:
        c = configs[name]
        source = c['checkpoint']['initialize_from']
        if source is not None and not Path(source).is_file():
            raise FileNotFoundError(f'{name}: initialization checkpoint is missing: {source}')
        print(f'[{name}] CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}', flush=True)
    if not args.run:
        print('Validated configs and subset. No training launched.')
        return
    for name in args.arm:
        out = Path(configs[name]['data']['checkpoints_dir'])
        if out.exists() and any(out.glob('*.ckpt')):
            raise FileExistsError(f'{out} already contains checkpoints; choose new output paths')
    visible = list(dict.fromkeys(ids))
    subprocess.run([args.python, '-c', 'import torch; '
                    'assert torch.cuda.is_available(), "CUDA is unavailable"; '
                    f'assert torch.cuda.device_count() == {len(visible)}, "Requested GPUs unavailable"'],
                   cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, visible))), check=True)
    logs = args.log_dir or HERE/'logs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.parallel:
        run_jobs(jobs, logs)
    else:
        for job in jobs:
            run_jobs([job], logs)


if __name__ == '__main__':
    main()
