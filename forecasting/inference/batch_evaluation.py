#!/usr/bin/env python3
"""
Batch Evaluation Script for Solar Flare Models

This script automates running inference and evaluation across multiple model checkpoints.
It dynamically creates inference and evaluation configuration files for each model,
executes inference, computes evaluation metrics, and logs the results.

Usage
-----
    python batch_evaluation.py -checkpoints checkpoint_list.yaml -base_config base_config.yaml -base_eval_config eval_config.yaml

Each model in the checkpoint list is processed sequentially, and results are saved in
organized subdirectories within a user-defined output directory.
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime
import shutil
import glob

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.inference.evaluation import SolarFlareEvaluator


def load_checkpoint_list(checkpoint_file):
    """
    Load a list of model checkpoints from a YAML file.

    Parameters
    ----------
    checkpoint_file : str
        Path to YAML file containing a list of checkpoints under the key `checkpoints`.

    Returns
    -------
    list of dict
        List of dictionaries, each containing:
        - `name`: model name (string)
        - `checkpoint_path`: absolute or relative checkpoint path (string)
    """
    with open(checkpoint_file, 'r') as f:
        data = yaml.safe_load(f)
    return data['checkpoints']


def create_inference_config(base_config, checkpoint_path, output_path, weight_path, model_name):
    """
    Create an inference configuration file for a specific model checkpoint.

    Parameters
    ----------
    base_config : str
        Path to a base inference configuration template YAML file.
    checkpoint_path : str
        Path to the model checkpoint to be evaluated.
    output_path : str
        Path where prediction CSVs will be saved.
    weight_path : str
        Directory where model attention or patch weights will be stored.
    model_name : str
        Name of the model (used to name the temporary config file).

    Returns
    -------
    str
        Path to the newly created temporary inference configuration YAML.
    """
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths
    config['data']['checkpoint_path'] = checkpoint_path
    config['output_path'] = output_path
    config['weight_path'] = weight_path
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(weight_path, exist_ok=True)
    
    # Save config
    config_path = f"temp_inference_config_{model_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def create_evaluation_config(base_eval_config, model_csv_path, output_dir, model_name, weight_path):
    """
    Create an evaluation configuration file for a specific model.

    Parameters
    ----------
    base_eval_config : str
        Path to a base evaluation configuration template YAML.
    model_csv_path : str
        Path to the model predictions CSV file.
    output_dir : str
        Directory where evaluation results will be saved.
    model_name : str
        Model identifier used for naming temporary config files.
    weight_path : str
        Directory containing model attention weights.

    Returns
    -------
    str
        Path to the generated temporary evaluation configuration YAML file.
    """
    with open(base_eval_config, 'r') as f:
        config = yaml.safe_load(f)
    
    from forecasting.inference.evaluation import resolve_config_variables
    config = resolve_config_variables(config)
    
    # Update paths
    config['model_predictions']['main_model_csv'] = model_csv_path
    config['evaluation']['output_dir'] = output_dir
    config['data']['weight_path'] = weight_path
    
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = f"temp_evaluation_config_{model_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def run_inference(inference_config_path, model_name, input_size=512, patch_size=8, batch_size=16, no_weights=False):
    """
    Run model inference using `inference_on_patch.py`.

    Parameters
    ----------
    inference_config_path : str
        Path to the inference configuration YAML.
    model_name : str
        Model name for logging.
    input_size : int, optional
        Input image size (default: 512).
    patch_size : int, optional
        Patch size for ViT models (default: 8).
    batch_size : int, optional
        Batch size used during inference (default: 16).
    no_weights : bool, optional
        If True, disables saving attention weights for faster inference.

    Returns
    -------
    bool
        True if inference completes successfully, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Running inference for model: {model_name}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 
        "inference_on_patch.py",
        "-config", inference_config_path,
        "-input_size", str(input_size),
        "-patch_size", str(patch_size),
        "--batch_size", str(batch_size)
    ]
    
    if no_weights:
        cmd.append("--no_weights")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True, check=True)
        print("Inference completed successfully!")
        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Inference failed for {model_name}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def run_evaluation(evaluation_config_path, model_name):
    """
    Run evaluation using the `SolarFlareEvaluator` class.

    Parameters
    ----------
    evaluation_config_path : str
        Path to the evaluation configuration YAML.
    model_name : str
        Name of the model being evaluated.

    Returns
    -------
    bool
        True if evaluation completes successfully, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Running evaluation for model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load evaluation config
        with open(evaluation_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Generate timestamps if specified
        timestamps = None
        if 'time_range' in config:
            from forecasting.inference.evaluation import generate_timestamps
            timestamps = generate_timestamps(
                config['time_range']['start_time'],
                config['time_range']['end_time'],
                config['time_range']['interval_minutes']
            )
            print(f"Generated {len(timestamps)} timestamps for movie frames")
        
        # Initialize evaluator
        evaluator = SolarFlareEvaluator(
            csv_path=config['model_predictions']['main_model_csv'],
            aia_dir=config['data']['aia_dir'],
            weight_path=config['data']['weight_path'],
            baseline_csv_path=config['model_predictions']['baseline_csv'],
            output_dir=config['evaluation']['output_dir'],
            sxr_cutoff=config['evaluation']['sxr_cutoff']
        )
        
        evaluator.run_full_evaluation(timestamps=timestamps)
        print(f"Evaluation completed successfully for {model_name}!")
        return True
        
    except Exception as e:
        print(f"Evaluation failed for {model_name}: {str(e)}")
        return False


def cleanup_temp_files(model_name):
    """
    Remove temporary inference and evaluation YAML configuration files.

    Parameters
    ----------
    model_name : str
        Name of the model used to identify the temp config files.
    """
    temp_files = [
        f"temp_inference_config_{model_name}.yaml",
        f"temp_evaluation_config_{model_name}.yaml"
    ]
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up {temp_file}")


def main():
    """
    Main entry point for batch evaluation of solar flare models.

    Workflow:
    ----------
    1. Loads a list of checkpoints from a YAML file.
    2. Creates temporary inference and evaluation configs for each model.
    3. Runs inference (unless skipped).
    4. Runs evaluation to compute metrics and generate plots.
    5. Records and summarizes results for all models in a YAML report.

    Command-line Arguments
    ----------------------
    -checkpoints : YAML file listing checkpoints to evaluate.
    -base_config : Base inference config template.
    -base_eval_config : Base evaluation config template.
    -output_base_dir : Directory to store all model outputs.
    -input_size : Input image size.
    -patch_size : Patch size for transformer-based models.
    -batch_size : Batch size for inference.
    --no_weights : Skip saving attention weights.
    --skip_inference : Skip inference step and only evaluate results.
    """
    parser = argparse.ArgumentParser(description='Batch evaluation for solar flare models')
    parser.add_argument('-checkpoints', type=str, required=True, 
                       help='YAML file containing list of checkpoints to evaluate')
    parser.add_argument('-base_config', type=str, required=True,
                       help='Base inference config template')
    parser.add_argument('-base_eval_config', type=str, required=True,
                       help='Base evaluation config template')
    parser.add_argument('-output_base_dir', type=str, default='./batch_evaluation_results',
                       help='Base directory for all outputs')
    parser.add_argument('-input_size', type=int, default=512,
                       help='Input size for models')
    parser.add_argument('-patch_size', type=int, default=16,
                       help='Patch size for models')
    parser.add_argument('-batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--no_weights', action='store_true',
                       help='Skip saving attention weights to speed up')
    parser.add_argument('--skip_inference', action='store_true',
                       help='Skip inference and only run evaluation (useful for re-running evaluation)')
    
    args = parser.parse_args()
    
    # Load checkpoint list
    checkpoints = load_checkpoint_list(args.checkpoints)
    
    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    results = {'successful': [], 'failed': []}
    
    print(f"Starting batch evaluation for {len(checkpoints)} checkpoints")
    print(f"Output base directory: {args.output_base_dir}")
    print(f"Input size: {args.input_size}, Patch size: {args.patch_size}, Batch size: {args.batch_size}")
    
    for i, checkpoint_info in enumerate(checkpoints, 1):
        model_name = checkpoint_info['name']
        checkpoint_path = checkpoint_info['checkpoint_path']
        
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(checkpoints)}: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*80}")
        
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            results['failed'].append({
                'name': model_name,
                'checkpoint': checkpoint_path,
                'error': 'Checkpoint not found'
            })
            continue
        
        model_output_dir = os.path.join(args.output_base_dir, model_name)
        model_csv_path = os.path.join(model_output_dir, f"{model_name}_predictions.csv")
        weight_path = os.path.join(model_output_dir, "weights")
        
        try:
            inference_config_path = create_inference_config(
                args.base_config, checkpoint_path, model_csv_path, weight_path, model_name
            )
            
            evaluation_config_path = create_evaluation_config(
                args.base_eval_config, model_csv_path, model_output_dir, model_name, weight_path
            )
            
            inference_success = True
            if not args.skip_inference:
                inference_success = run_inference(
                    inference_config_path, model_name, args.input_size,
                    args.patch_size, args.batch_size, args.no_weights
                )
            
            evaluation_success = False
            if inference_success or args.skip_inference:
                evaluation_success = run_evaluation(evaluation_config_path, model_name)
            
            if inference_success and evaluation_success:
                results['successful'].append({
                    'name': model_name,
                    'checkpoint': checkpoint_path,
                    'output_dir': model_output_dir,
                    'csv_path': model_csv_path
                })
                print(f"✅ SUCCESS: {model_name}")
            else:
                results['failed'].append({
                    'name': model_name,
                    'checkpoint': checkpoint_path,
                    'error': 'Inference or evaluation failed'
                })
                print(f"❌ FAILED: {model_name}")
            
        except Exception as e:
            print(f"ERROR processing {model_name}: {str(e)}")
            results['failed'].append({
                'name': model_name,
                'checkpoint': checkpoint_path,
                'error': str(e)
            })
        
        finally:
            cleanup_temp_files(model_name)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['successful']:
        print(f"\n✅ SUCCESSFUL MODELS:")
        for result in results['successful']:
            print(f"  - {result['name']}: {result['output_dir']}")
    
    if results['failed']:
        print(f"\n❌ FAILED MODELS:")
        for result in results['failed']:
            print(f"  - {result['name']}: {result['error']}")
    
    summary_path = os.path.join(args.output_base_dir, "batch_evaluation_summary.yaml")
    with open(summary_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\nResults summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
