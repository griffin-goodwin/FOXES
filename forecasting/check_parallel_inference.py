"""Compare single-GPU and DataParallel pinball outputs on identical images.

Run on a server with at least two CUDA GPUs:

    python forecasting/check_parallel_inference.py --config forecasting/inference_config.yaml

Use --timestamps TIMESTAMP ... to check specific rows from an inference CSV.
"""

import argparse
import copy
import hashlib
import os
import sys
from pathlib import Path

import numpy  # Initialize NumPy's native libraries before PyTorch on macOS.
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.dataset import AIAGOESDataset, AIANormTransform
from forecasting.inference import load_model_from_config, select_inference_samples


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--count', type=int, default=None,
                        help='Number of selected images; default is one per GPU')
    parser.add_argument('--timestamps', nargs='+', help='Exact AIA filename stems')
    parser.add_argument('--single-device', action='store_true',
                        help='Check one image on the only visible GPU')
    args = parser.parse_args()

    device_count = torch.cuda.device_count()
    if device_count < (1 if args.single_device else 2):
        raise SystemExit('No suitable visible CUDA GPUs for this check.')
    print(f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "all")}')
    for index in range(device_count):
        print(f'cuda:{index}: {torch.cuda.get_device_name(index)}')

    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(config.get('matmul_precision', 'high'))
    checkpoint_path = config['data']['checkpoint_path']
    print(f'checkpoint: {checkpoint_path}')
    print(f'checkpoint SHA256: {checkpoint_hash(checkpoint_path)}', flush=True)

    single_config = copy.deepcopy(config)
    single_config['device'] = 'cuda'
    single_config['multi_gpu'] = False
    single_config['use_amp'] = False
    model = load_model_from_config(single_config).eval()
    if getattr(model, 'uncertainty_kind', None) != 'quantile':
        raise SystemExit('Checkpoint is not a pinball quantile model.')

    aia_norm_path = config['data'].get('aia_norm_path')
    aia_transform = (
        AIANormTransform.from_file(aia_norm_path, config['wavelengths'])
        if aia_norm_path else None
    )
    prediction_only = str(config.get('prediction_only', False)).lower() in ('true', '1')
    dataset = AIAGOESDataset(
        aia_dir=config['data']['aia_dir'],
        sxr_dir=None if prediction_only else config['data']['sxr_dir'],
        wavelengths=config['wavelengths'],
        aia_transform=aia_transform,
        only_prediction=prediction_only,
    )
    dataset, times = select_inference_samples(dataset, config.get('sampling'))
    if args.single_device:
        if args.timestamps and len(args.timestamps) != 1:
            raise SystemExit('--single-device accepts exactly one timestamp')
        timestamp = args.timestamps[0] if args.timestamps else times[0]
        if timestamp not in times:
            raise SystemExit(f'Timestamp is not in the selected inference set: {timestamp}')
        image, _ = dataset[times.index(timestamp)]
        image = image.unsqueeze(0).to('cuda:0')
        with torch.inference_mode():
            raw, normalized, patches, local = model.model(
                image, model.sxr_norm, return_local=True,
            )
        median = model.quantiles.index(0.5)
        print(f'{timestamp}: raw={raw[0, median].item():.9g}, '
              f'normalized={normalized[0, median].item():.9g}, '
              f'patch_sum={patches[0, :, median].sum().item():.9g}, '
              f'patch_min={patches[0, :, median].min().item():.9g}, '
              f'local_sum={local[0, :, median].sum().item():.9g} W/m^2')
        return
    if args.timestamps:
        positions = {timestamp: index for index, timestamp in enumerate(times)}
        missing = [timestamp for timestamp in args.timestamps if timestamp not in positions]
        if missing:
            raise SystemExit(f'Timestamps are not in the selected inference set: {missing}')
        selected = args.timestamps
        indices = [positions[timestamp] for timestamp in selected]
    else:
        count = device_count if args.count is None else args.count
        if count < device_count or count > len(times):
            raise SystemExit(f'--count must be between {device_count} and {len(times)}')
        selected = times[:count]
        indices = list(range(len(selected)))
    if len(indices) < device_count:
        raise SystemExit(f'Provide at least {device_count} timestamps to exercise every GPU.')

    images = torch.stack([dataset[index][0] for index in indices]).to('cuda:0')
    parallel = torch.nn.DataParallel(model, device_ids=list(range(device_count))).eval()
    median = model.quantiles.index(0.5)
    single_raw, local_medians = [], []
    with torch.inference_mode():
        for image in images.split(1):
            raw, _, patches, local = model.model(
                image, model.sxr_norm, return_local=True,
            )
            single_raw.append(raw)
            local_medians.append(local[:, :, median].sum(dim=1))
        single_raw = torch.cat(single_raw)
        local_medians = torch.cat(local_medians)
        parallel_raw, parallel_norm, parallel_patches = parallel(images)
    with torch.no_grad():
        no_grad_raw, _, no_grad_patches = parallel(images)

    # Running each image directly on its assigned device separates a replica
    # computation error from an output-gather error in DataParallel.
    direct_device_medians = []
    del parallel
    if len(images) == device_count:
        with torch.inference_mode():
            for index in range(device_count):
                device = torch.device(f'cuda:{index}')
                model.to(device)
                raw, _, _ = model(images[index:index + 1].to(device))
                direct_device_medians.append(raw[0, median].item())
            model.to('cuda:0')

    difference = (single_raw - parallel_raw).abs()
    relative = difference / single_raw.abs().clamp_min(1e-12)
    map_error = (parallel_raw - parallel_patches.sum(dim=1)).abs().max()
    print(f'CUDA GPUs: {device_count}; images: {len(images)}')
    print(f'max absolute difference, all heads: {difference.max().item():.6g} W/m^2')
    print(f'max relative difference, all heads: {relative.max().item():.6g}')
    print(f'max global prediction vs patch-sum difference: {map_error.item():.6g} W/m^2')
    print('max single vs DataParallel under no_grad difference: '
          f'{(single_raw - no_grad_raw).abs().max().item():.6g} W/m^2')
    for index, (timestamp, local, single, multi) in enumerate(zip(
        selected, local_medians, single_raw[:, median], parallel_raw[:, median]
    )):
        patch_median = parallel_patches[index, :, median]
        direct = (f'{direct_device_medians[index]:.9g}'
                  if index < len(direct_device_medians) else 'not_run')
        print(
            f'{timestamp}: local={local.item():.9g}, '
            f'single_global={single.item():.9g}, '
            f'direct_cuda{index}={direct}, '
            f'parallel_global={multi.item():.9g}, '
            f'parallel_no_grad={no_grad_raw[index, median].item():.9g}, '
            f'parallel_norm={parallel_norm[index, median].item():.9g}, '
            f'parallel_patch_sum={patch_median.sum().item():.9g}, '
            f'parallel_no_grad_patch_sum={no_grad_patches[index, :, median].sum().item():.9g}, '
            f'parallel_patch_min={patch_median.min().item():.9g} W/m^2'
        )


if __name__ == '__main__':
    main()
