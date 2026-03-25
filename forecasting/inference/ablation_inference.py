"""
Ablation Inference Script — Gaussian Noise Channel Masking
==========================================================

Runs inference with the pretrained model while applying Gaussian noise to
specific AIA wavelength channels, one condition at a time. This lets you
measure how much each channel (or combination of channels) contributes to
forecast skill.

Each ablation condition produces its own output CSV so results can be
compared directly against the clean baseline.
"""

import argparse
import re
import sys
import gc
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.data_loaders.SDOAIA_dataloader import AIA_GOESDataset, NoisyAIA_GOESDataset
import forecasting.models as models
# Reuse inference helpers from the main inference script
from forecasting.inference.inference import load_model_from_config, evaluate_model_on_dataset


def build_dataset(config_data, noisy_wavelengths, noise_std):
    """
    Build an AIA_GOESDataset (or NoisyAIA_GOESDataset if wavelengths are given).

    Parameters
    ----------
    config_data : dict
    noisy_wavelengths : list of int
        Empty list → clean baseline (plain AIA_GOESDataset).
    noise_std : float or dict

    Returns
    -------
    dataset : torch.utils.data.Dataset
    """
    aia_dir = config_data['data']['aia_dir']
    sxr_dir = config_data['data']['sxr_dir']
    wavelengths = config_data['wavelengths']
    prediction_only = config_data.get('prediction_only', 'false').lower() == 'true'

    common_kwargs = dict(
        aia_dir=aia_dir,
        sxr_dir=sxr_dir,
        wavelengths=wavelengths,
        only_prediction=prediction_only,
    )

    if noisy_wavelengths:
        return NoisyAIA_GOESDataset(
            **common_kwargs,
            noisy_wavelengths=noisy_wavelengths,
            noise_std=noise_std,
        )
    else:
        return AIA_GOESDataset(**common_kwargs)


def run_condition(model, dataset, label, config_data, model_params, output_dir):
    """
    Run inference for a single ablation condition and save results to CSV.

    Parameters
    ----------
    model : torch.nn.Module
    dataset : torch.utils.data.Dataset
    label : str
        Human-readable name for this condition (used as filename stem).
    config_data : dict
    model_params : dict
    output_dir : Path
    """
    times = dataset.samples
    total_samples = len(times)
    batch_size = model_params.get('batch_size', 10)
    input_size = model_params.get('input_size', 512)
    patch_size = model_params.get('patch_size', 16)
    save_weights = not model_params.get('no_weights', True)
    save_flux = not model_params.get('no_flux', True)
    prediction_only = config_data.get('prediction_only', 'false').lower() == 'true'

    print(f"\n{'='*60}")
    print(f"  Condition: {label}  ({total_samples} samples)")
    print(f"{'='*60}")

    timestamps, predictions, ground_truths = [], [], []

    pbar = tqdm(
        evaluate_model_on_dataset(
            model, dataset, batch_size, times, config_data,
            save_weights, input_size, patch_size, save_flux,
        ),
        total=total_samples,
        desc=label,
        unit="sample",
        ncols=100,
    )

    for prediction, sxr, _weight, _flux, idx in pbar:
        predictions.append(float(prediction.item() if hasattr(prediction, 'item') else prediction))
        ground_truths.append(0.0 if prediction_only else
                             float(sxr.item() if hasattr(sxr, 'item') else sxr))
        timestamps.append(str(times[idx]))

    output_path = output_dir / f"{label}.csv"
    pd.DataFrame({
        'timestamp': timestamps,
        'predictions': predictions,
        'groundtruth': ground_truths,
    }).to_csv(output_path, index=False)

    print(f"  Saved → {output_path}")

    # Free dataset memory between conditions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_config_variables(config_dict):
    """Recursively resolve ${variable} references within config."""
    variables = {k: v for k, v in config_dict.items()
                 if isinstance(v, str) and not v.startswith('${')}

    def substitute(obj):
        if isinstance(obj, dict):
            return {k: substitute(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute(item) for item in obj]
        elif isinstance(obj, str):
            for match in re.finditer(r'\$\{([^}]+)\}', obj):
                var = match.group(1)
                if var in variables:
                    obj = obj.replace(f'${{{var}}}', variables[var])
            return obj
        return obj

    return substitute(config_dict)


def main():
    parser = argparse.ArgumentParser(description="Ablation inference with Gaussian noise masking.")
    parser.add_argument('-config', type=str, required=True,
                        help='Path to the ablation inference YAML config.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.SafeLoader)
    config_data = resolve_config_variables(config_data)
    sys.modules['models'] = models

    model_params = config_data.get('model_params', {})
    ablation_cfg = config_data.get('ablation', {})
    noise_std = ablation_cfg.get('noise_std', 1.0)
    conditions = ablation_cfg.get('conditions', [])

    if not conditions:
        raise ValueError("No ablation conditions defined in config under 'ablation.conditions'.")

    output_dir = Path(config_data['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model...")
    model = load_model_from_config(config_data)

    for condition in conditions:
        label = condition.get('label', 'unnamed')
        noisy_wavelengths = condition.get('wavelengths', [])

        dataset = build_dataset(config_data, noisy_wavelengths, noise_std)
        run_condition(model, dataset, label, config_data, model_params, output_dir)

    print(f"\nAll conditions complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
