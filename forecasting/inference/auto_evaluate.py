#!/usr/bin/env python3
"""
Automated Evaluation Script for Solar Flare Models

This script automatically generates inference and evaluation configs
and runs the complete evaluation pipeline based on a directory input.

Usage:
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
    """Find checkpoint files in directory"""
    patterns = ['*.pth', '*.ckpt', '*.pt']
    checkpoints = []
    
    for pattern in patterns:
        checkpoints.extend(glob.glob(str(Path(checkpoint_dir) / pattern)))
        checkpoints.extend(glob.glob(str(Path(checkpoint_dir) / '**' / pattern), recursive=True))
    
    return sorted(checkpoints)

def detect_model_type(checkpoint_path):
    """Detect model type from checkpoint filename or content"""
    filename = Path(checkpoint_path).name.lower()
    
    return 'vitlocal'

def create_inference_config(checkpoint_path, model_name, base_data_dir="/mnt/data/NO-OVERLAP"):
    """Create inference config for checkpoint"""
    
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
        'base_data_dir': base_data_dir,
        'data': {
            'aia_dir': f"{base_data_dir}/AIA/",
            'checkpoint_path': checkpoint_path,
            'sxr_dir': f"{base_data_dir}/SXR/",
            'sxr_norm_path': f"{base_data_dir}/SXR/normalized_sxr.npy"
        },
        'model': model_type,
        'wavelengths': [94, 131, 171, 193, 211, 304],
        'mc': {
            'active': 'false',
            'runs': 5
        },
        'model_params': {
            'batch_size': 16,
            'input_size': 512,
            'no_weights': False,
            'patch_size': 16
        },
        'vit_custom': {
            'embed_dim': 512,
            'hidden_dim': 512,
            'num_channels': 6,
            'num_classes': 1,
            'patch_size': 16,
            'num_patches': 1024,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1
        },
        'megsai': {
            'cnn_model': 'updated',
            'cnn_dp': 0.2,
            'weight_decay': 1e-5,
            'cosine_restart_T0': 50,
            'cosine_restart_Tmult': 2,
            'cosine_eta_min': 1e-7
        },
        'output_path': f"{output_dir}/{model_name}_predictions.csv",
        'weight_path': f"{output_dir}/weights"
    }
    
    # Add flux_path for patch-based models
    if model_type in ['vitpatch', 'vitlocal']:
        config['flux_path'] = f"{output_dir}/flux/"
    
    # Add model-specific configs
    if model_type == 'fusion':
        config['fusion'] = {
            'scalar_branch': 'hybrid',
            'lr': 0.0001,
            'lambda_vit_to_target': 0.3,
            'lambda_scalar_to_target': 0.1,
            'learnable_gate': True,
            'gate_init_bias': 5.0,
            'scalar_kwargs': {
                'd_input': 6,
                'd_output': 1,
                'cnn_model': 'updated',
                'cnn_dp': 0.75
            }
        }
    
    return config, output_dir

def create_evaluation_config(model_name, output_dir, base_data_dir="/mnt/data/NO-OVERLAP"):
    """Create evaluation config"""
    
    config = {
        'base_data_dir': base_data_dir,
        'output_base_dir': f"{base_data_dir}/solar_flare_comparison_results",
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
            'sxr_cutoff': 1e-9
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
    """Run inference with the generated config"""
    print(f"Running inference with config: {inference_config_path}")
    
    cmd = [
        sys.executable, 
        str(PROJECT_ROOT / "forecasting/inference/inference.py"),
        "-config", inference_config_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running inference: {result.stderr}")
        return False
    
    print("Inference completed successfully!")
    return True

def run_evaluation(evaluation_config_path):
    """Run evaluation with the generated config"""
    print(f"Running evaluation with config: {evaluation_config_path}")
    
    cmd = [
        sys.executable, 
        str(PROJECT_ROOT / "forecasting/inference/evaluation.py"),
        "-config", evaluation_config_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}")
        return False
    
    print("Evaluation completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Automated evaluation for solar flare models')
    parser.add_argument('-checkpoint_dir', type=str, help='Directory containing checkpoint files')
    parser.add_argument('-checkpoint_path', type=str, help='Specific checkpoint file path')
    parser.add_argument('-model_name', type=str, required=True, help='Name for the model (used for output naming)')
    parser.add_argument('-base_data_dir', type=str, default='/mnt/data/NO-OVERLAP', help='Base data directory')
    parser.add_argument('-skip_inference', action='store_true', help='Skip inference and only run evaluation')
    parser.add_argument('-skip_evaluation', action='store_true', help='Skip evaluation and only run inference')
    
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
    
    # Create configs
    inference_config, output_dir = create_inference_config(checkpoint_path, args.model_name, args.base_data_dir)
    evaluation_config = create_evaluation_config(args.model_name, output_dir, args.base_data_dir)
    
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
        if not run_evaluation(evaluation_config_path):
            print("Evaluation failed. Stopping.")
            sys.exit(1)
    else:
        print("Skipping evaluation...")
    
    print(f"\n✅ Complete! Results saved to: {output_dir}")
    print(f"📊 Check the plots and metrics in: {output_dir}")

if __name__ == '__main__':
    main()
