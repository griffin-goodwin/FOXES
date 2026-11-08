"""Run independent single-GPU inference shards and merge their CSVs.

Ported from the pinball branch, with config-root path resolution.

Example:
    python forecasting/run_parallel_inference.py \
        --config forecasting/inference_config.yaml --gpus 0,1,2,3
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.config import load_config


def merge_shard_csvs(paths, output_path):
    """Merge completed shards in timestamp order without changing numeric text."""
    rows = []
    fieldnames = None
    for path in paths:
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or 'timestamp' not in reader.fieldnames:
                raise ValueError(f'Missing timestamp column in {path}')
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f'CSV columns differ in {path}')
            rows.extend(reader)
    rows.sort(key=lambda row: row['timestamp'])
    timestamps = [row['timestamp'] for row in rows]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError('Duplicate timestamps across inference shards')
    if not rows:
        raise ValueError('Inference shards contain no predictions')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + '.partial')
    with temporary_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_path)
    return len(rows)


def parse_gpu_ids(value):
    """Parse distinct physical CUDA device IDs for at least two workers."""
    values = [part.strip() for part in value.split(',')]
    if not values or any(not part.isascii() or not part.isdigit() for part in values):
        raise argparse.ArgumentTypeError('--gpus must contain nonnegative CUDA device IDs')
    gpus = [str(int(part)) for part in values]
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError('--gpus must contain distinct CUDA device IDs')
    if len(gpus) < 2:
        raise argparse.ArgumentTypeError('Use forecasting/inference.py directly for one GPU')
    return gpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpus', required=True, type=parse_gpu_ids,
                        help='Comma-separated physical CUDA device IDs, e.g. 0,1,2,3')
    args = parser.parse_args()
    config_path = Path(os.path.expandvars(args.config)).expanduser().resolve()
    config = load_config(config_path)
    gpus = args.gpus
    output_path = Path(config['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_paths = [
        output_path.with_name(f'{output_path.stem}.shard-{index}-of-{len(gpus)}{output_path.suffix}')
        for index in range(len(gpus))
    ]
    processes = []
    logs = []
    try:
        for index, (gpu, shard_path) in enumerate(zip(gpus, shard_paths)):
            log_path = shard_path.with_suffix('.log')
            log = log_path.open('w')
            logs.append(log)
            environment = os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES'] = gpu
            command = [
                sys.executable, str(Path(__file__).with_name('inference.py')),
                '--config', str(config_path),
                '--device', 'cuda',
                '--num-shards', str(len(gpus)),
                '--shard-index', str(index),
                '--output-path', str(shard_path),
            ]
            print(f'GPU {gpu}: shard {index + 1}/{len(gpus)} -> {shard_path}; log: {log_path}', flush=True)
            processes.append(subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT,
            ))
        results = [process.wait() for process in processes]
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    finally:
        for log in logs:
            log.close()
    failures = [(index, code) for index, code in enumerate(results) if code != 0]
    if failures:
        raise RuntimeError(f'Inference shards failed: {failures}; inspect shard logs')
    count = merge_shard_csvs(shard_paths, output_path)
    print(f'Merged {count} predictions in timestamp order to {output_path}')


if __name__ == '__main__':
    main()
