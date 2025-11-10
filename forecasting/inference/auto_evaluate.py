#!/usr/bin/env python3
"""
Automated Evaluation Script for Solar Flare Models

This script automates the generation of inference and evaluation configurations,
and runs the complete end-to-end evaluation pipeline for trained solar flare models.

It supports both directory-based checkpoint discovery and direct checkpoint paths,
automatically detecting the model type and setting up inference/evaluation YAMLs.

Usage
-----
Example commands:
    python auto_evaluate.py -checkpoint_dir /path/to/checkpoint/dir -model_name my_model
    python auto_evaluate.py -checkpoint_path /path/to/checkpoint.pth -model_name my_model
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime
import glob

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def find_checkpoint_files(checkpoint_dir):
    """
    Find all checkpoint files (.pth, .ckpt, .pt) within a directory.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Path to the directory containing model checkpoint files.

    Returns
    -------
    list of str
        Sorted list of checkpoint file paths discovered within the directory.
    """
    patterns = ['*.pth', '*.ckpt', '*.pt']
    checkpoints = []
    
    for pattern in patterns:
        checkpoints.extend(glob.glob(str(Path(checkpoint_dir) / pattern)))
        checkpoints.extend(glob.glob(str(Path(checkpoint_dir) / '**' / pattern), recursive=True))
    
    return sorted(checkpoints)


def detect_model_type(checkpoint_path):
    """
    Infer the model type from a checkpoint filename.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.

    Returns
    -------
    str
        Model type inferred from filename (e.g., 'vitlocal', 'vitpatch', 'fusion', etc.).
    """
    filename = Path(checkpoint_path).name.lower()
    
    return 'vitlocal'


def check_sxr_data_availability(base_data_dir):
    """
    Check if SXR data is available in the specified directory.

    Parameters
    ----------
    base_data_dir : str
        Base directory containing the SXR data.

    Returns
    -------
    bool
        True if SXR data is available, False otherwise.
    """
    sxr_dir = Path(base_data_dir) / "SXR"
    sxr_norm_path = Path(base_data_dir) / "SXR" / "normalized_sxr.npy"
    
    # Check if SXR directory exists and has files
    if not sxr_dir.exists():
        print(f"SXR directory not found: {sxr_dir}")
        return False
    
    # Check if normalized SXR file exists
    if not sxr_norm_path.exists():
        print(f"Normalized SXR file not found: {sxr_norm_path}")
        return False
    
    # Check if there are any .npy files in the SXR directory
    sxr_files = list(sxr_dir.glob("*.npy"))
    if not sxr_files:
        print(f"No SXR data files found in: {sxr_dir}")
        return False
    
    print(f"Found {len(sxr_files)} SXR data files in {sxr_dir}")
    return True


def create_inference_config(checkpoint_path, model_name, base_data_dir="/mnt/data/NO-OVERLAP", prediction_only=False):
    """
    Dynamically create an inference configuration dictionary for a given checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.
    model_name : str
        Name for the model (used for output folder and file naming).
    base_data_dir : str, optional
        Root directory of dataset and normalization files.
    prediction_only : bool, optional
        If True, run in prediction-only mode (no SXR ground truth required).
    Returns
    -------
    tuple(dict, str)
        - Inference configuration dictionary.
        - Path to the output directory where results will be saved.
    """
    # Detect model type
    model_type = detect_model_type(checkpoint_path)
    
    # Create output directory
    output_dir = f"/mnt/data/batch_results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/weights", exist_ok=True)
    
    # Create flux directory for patch-based models
    if model_type == 'vitlocal':
        os.makedirs(f"{output_dir}/flux", exist_ok=True)
    
    # Generate config
    config = {
        'SolO': 'false',
        'Stereo': 'false',
        'prediction_only': 'true' if prediction_only else 'false',
        'base_data_dir': base_data_dir,
        'data': {
            'aia_dir': f"{base_data_dir}/AIA/",
            'checkpoint_path': checkpoint_path,
            'sxr_dir': f"{base_data_dir}/SXR/" if not prediction_only else "",
            'sxr_norm_path': f"{base_data_dir}/SXR/normalized_sxr.npy" if not prediction_only else ""
        },
        'model': model_type,
        'wavelengths': [94, 131, 171, 193, 211, 304, 335],
        'mc': {
            'active': 'false',
            'runs': 5
        },
        'model_params': {
            'batch_size': 16,  # Match training batch size. If you get OOM errors, reduce this.
                              # Note: Inference with attention weights uses more memory than training
            'input_size': 512,
            'no_weights': True,  # Set to False to save attention weights (uses more memory)
            'no_flux': True,  # Set to False to save flux contributions (uses more memory)
            'patch_size': 8
        },
        'vit_custom': {
            'embed_dim': 256,
            'hidden_dim': 1024,
            'num_channels': 7,
            'num_classes': 1,
            'patch_size': 8,
            'num_patches': 4096,
            'num_heads': 8,
            'num_layers': 8,
            'dropout': 0.1
        },
        'output_path': f"{output_dir}/{model_name}_predictions.csv",
        'weight_path': f"{output_dir}/weights"
    }
    
    # Add flux_path for patch-based models
    if model_type in ['vitpatch', 'vitlocal']:
        config['flux_path'] = f"{output_dir}/flux/"
    
    
    return config, output_dir


def create_evaluation_config(model_name, output_dir, base_data_dir="/mnt/data/NO-OVERLAP", prediction_only=False):
    """
    Create evaluation configuration for computing metrics and visualizations.

    Parameters
    ----------
    model_name : str
        Name of the model under evaluation.
    output_dir : str
        Path to output directory containing prediction results.
    base_data_dir : str, optional
        Base dataset directory containing AIA and SXR test data.
    prediction_only : bool, optional
        If True, create config for prediction-only mode (no ground truth evaluation).

    Returns
    -------
    dict
        Evaluation configuration dictionary with metrics, time range, and plotting settings.
    """
    config = {
        'base_data_dir': base_data_dir,
        'output_base_dir': f"{base_data_dir}/solar_flare_comparison_results",
        'prediction_only': prediction_only,
        'data': {
            'aia_dir': f"{base_data_dir}/AIA/test/",
            'weight_path': f"{output_dir}/weights"
        },
        'model_predictions': {
            'main_model_csv': f"{output_dir}/{model_name}_predictions.csv",
            'baseline_csv': ''
        },
        'evaluation': {
            'output_dir': output_dir,
            'sxr_cutoff': 1e-9 if not prediction_only else None
        },
        'time_range': {
            'start_time': '2023-08-08T20:00:00',
            'end_time': '2023-08-08T23:59:00',
            'interval_minutes': 1
        },
        'plotting': {
            'figure_size': [12, 8],
            'dpi': 300,
            'colormap': 'sdoaia171'
        },
        'metrics': {
            'include_rmse': True,
            'include_mae': True,
            'include_r2': True,
            'include_correlation': True
        }
    }
    return config


def run_inference(inference_config_path):
    """
    Execute model inference using the generated YAML configuration.

    Parameters
    ----------
    inference_config_path : str
        Path to the inference configuration YAML file.

    Returns
    -------
    bool
        True if inference completes successfully, False if an error occurs.
    """
    print(f"Running inference with config: {inference_config_path}")
    
    cmd = [
        sys.executable, 
        str(PROJECT_ROOT / "forecasting/inference/inference.py"),
        "-config", inference_config_path
    ]
    
    # Use Popen with real-time output streaming to show progress bar
    # Both stdout and stderr go to terminal so tqdm progress bar (which writes to stderr) is visible
    process = subprocess.Popen(
        cmd,
        stdout=None,  # Let stdout go directly to terminal
        stderr=subprocess.STDOUT,  # Merge stderr into stdout so progress bar is visible
        text=True,
        bufsize=1  # Line buffered for real-time output
    )
    
    # Wait for process to complete
    process.wait()
    
    if process.returncode != 0:
        print(f"Error: Inference process exited with code {process.returncode}")
        return False
    
    print("Inference completed successfully!")
    return True


def run_evaluation(evaluation_config_path):
    """
    Execute evaluation of inference outputs using the generated YAML configuration.

    Parameters
    ----------
    evaluation_config_path : str
        Path to the evaluation configuration YAML file.

    Returns
    -------
    bool
        True if evaluation completes successfully, False otherwise.
    """
    print(f"Running evaluation with config: {evaluation_config_path}")
    
    cmd = [
        sys.executable, 
        str(PROJECT_ROOT / "forecasting/inference/evaluation.py"),
        "-config", evaluation_config_path
    ]
    
    # Use Popen with real-time output streaming
    # Both stdout and stderr go to terminal for real-time output
    process = subprocess.Popen(
        cmd,
        stdout=None,  # Let stdout go directly to terminal
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1  # Line buffered for real-time output
    )
    
    # Wait for process to complete
    process.wait()
    
    if process.returncode != 0:
        print(f"Error: Evaluation process exited with code {process.returncode}")
        return False
    
    print("Evaluation completed successfully!")
    return True


def main():
    """
    Main function for automating inference and evaluation.

    Steps:
      1. Parse command-line arguments.
      2. Locate checkpoint file or directory.
      3. Generate inference and evaluation YAML configs.
      4. Optionally run inference and/or evaluation scripts.
      5. Output results and metrics to specified directory.
    """
    parser = argparse.ArgumentParser(description='Automated evaluation for solar flare models')
    parser.add_argument('-checkpoint_dir', type=str, help='Directory containing checkpoint files')
    parser.add_argument('-checkpoint_path', type=str, help='Specific checkpoint file path')
    parser.add_argument('-model_name', type=str, required=True, help='Name for the model (used for output naming)')
    parser.add_argument('-base_data_dir', type=str, default='/mnt/data/', help='Base data directory')
    parser.add_argument('-skip_inference', action='store_true', help='Skip inference and only run evaluation')
    parser.add_argument('-skip_evaluation', action='store_true', help='Skip evaluation and only run inference')
    parser.add_argument('-prediction_only', action='store_true', help='Force prediction-only mode (no SXR ground truth)')
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)
    elif args.checkpoint_dir:
        checkpoints = find_checkpoint_files(args.checkpoint_dir)
        if not checkpoints:
            print(f"Error: No checkpoint files found in {args.checkpoint_dir}")
            sys.exit(1)
        elif len(checkpoints) > 1:
            print(f"Found multiple checkpoints: {checkpoints}")
            print("Using the first one. Use -checkpoint_path to specify a specific file.")
        checkpoint_path = checkpoints[0]
    else:
        print("Error: Must specify either -checkpoint_dir or -checkpoint_path")
        sys.exit(1)
    
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Model name: {args.model_name}")
    
    # Check SXR data availability and determine if we should use prediction-only mode
    prediction_only_mode = args.prediction_only
    
    if not prediction_only_mode:
        print("Checking SXR data availability...")
        sxr_available = check_sxr_data_availability(args.base_data_dir)
        if not sxr_available:
            print("⚠️  SXR data not available. Switching to prediction-only mode.")
            prediction_only_mode = True
        else:
            print("✅ SXR data found. Running with ground truth evaluation.")
    else:
        print("🔮 Running in prediction-only mode (as requested).")
    
    # Create configs
    inference_config, output_dir = create_inference_config(checkpoint_path, args.model_name, args.base_data_dir, prediction_only_mode)
    evaluation_config = create_evaluation_config(args.model_name, output_dir, args.base_data_dir, prediction_only_mode)
    
    # Save configs
    inference_config_path = f"/tmp/inference_config_{args.model_name}.yaml"
    evaluation_config_path = f"/tmp/evaluation_config_{args.model_name}.yaml"
    
    with open(inference_config_path, 'w') as f:
        yaml.dump(inference_config, f, default_flow_style=False)
    
    with open(evaluation_config_path, 'w') as f:
        yaml.dump(evaluation_config, f, default_flow_style=False)
    
    print(f"Configs saved to:")
    print(f"  Inference: {inference_config_path}")
    print(f"  Evaluation: {evaluation_config_path}")
    print(f"  Output directory: {output_dir}")
    
    # Run inference
    if not args.skip_inference:
        if not run_inference(inference_config_path):
            print("Inference failed. Stopping.")
            sys.exit(1)
    else:
        print("Skipping inference...")
    
    # Run evaluation
    if not args.skip_evaluation:
        if prediction_only_mode:
            print("Skipping evaluation (prediction-only mode - no ground truth available)")
        else:
            if not run_evaluation(evaluation_config_path):
                print("Evaluation failed. Stopping.")
                sys.exit(1)
    else:
        print("Skipping evaluation...")
    
    print(f"\n✅ Complete! Results saved to: {output_dir}")
    if prediction_only_mode:
        print(f"🔮 Prediction-only mode: No ground truth evaluation performed")
        print(f"📊 Check the prediction results in: {output_dir}")
    else:
        print(f"📊 Check the plots and metrics in: {output_dir}")


if __name__ == '__main__':
    main()
