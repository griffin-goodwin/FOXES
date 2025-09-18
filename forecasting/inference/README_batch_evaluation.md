# Batch Evaluation Script

This script automates the process of running inference and evaluation for multiple model checkpoints.

## Files Created

- `batch_evaluation.py` - Main batch evaluation script
- `checkpoint_list_example.yaml` - Example checkpoint list
- `inference_template.yaml` - Base inference configuration template
- `README_batch_evaluation.md` - This documentation

## Usage

### 1. Prepare Checkpoint List

Create a YAML file listing all checkpoints you want to evaluate:

```yaml
checkpoints:
  - name: "vit-patch-epoch-10"
    checkpoint_path: "/path/to/checkpoint1.ckpt"
    
  - name: "vit-patch-epoch-20"
    checkpoint_path: "/path/to/checkpoint2.ckpt"
    
  - name: "best-model"
    checkpoint_path: "/path/to/best_checkpoint.ckpt"
```

### 2. Run Batch Evaluation

```bash
python batch_evaluation.py \
  -checkpoints checkpoint_list.yaml \
  -base_config inference_template.yaml \
  -base_eval_config evaluation_config.yaml \
  -output_base_dir ./batch_results \
  -input_size 512 \
  -patch_size 16 \
  -batch_size 16
```

### 3. Command Line Options

- `-checkpoints`: YAML file with list of checkpoints to evaluate
- `-base_config`: Base inference configuration template
- `-base_eval_config`: Base evaluation configuration template
- `-output_base_dir`: Base directory for all outputs (default: ./batch_evaluation_results)
- `-input_size`: Input size for models (default: 512)
- `-patch_size`: Patch size for models (default: 8)
- `-batch_size`: Batch size for inference (default: 16)
- `--no_weights`: Skip saving attention weights to speed up
- `--skip_inference`: Skip inference and only run evaluation

## Output Structure

```
batch_evaluation_results/
├── model1/
│   ├── model1_predictions.csv
│   ├── weights/
│   │   ├── attention_weights_0.txt
│   │   └── ...
│   ├── metrics/
│   │   └── performance_comparison.csv
│   ├── plots/
│   │   └── regression_comparison.png
│   └── ...
├── model2/
│   └── ...
└── batch_evaluation_summary.yaml
```

## Features

- **Automated Pipeline**: Runs inference and evaluation for each checkpoint
- **Error Handling**: Continues processing even if some models fail
- **Progress Tracking**: Shows progress and detailed logs
- **Results Summary**: Generates summary of successful/failed evaluations
- **Flexible Configuration**: Uses templates for easy customization
- **Cleanup**: Automatically cleans up temporary files

## Example Workflow

1. **Train multiple models** with different configurations
2. **Create checkpoint list** with all trained models
3. **Run batch evaluation** to get comprehensive results
4. **Compare results** across all models using the generated metrics

## Troubleshooting

- **Checkpoint not found**: Verify checkpoint paths in your list
- **Inference fails**: Check model configuration matches training config
- **Evaluation fails**: Verify data paths and evaluation config
- **Memory issues**: Reduce batch_size or use --no_weights flag

## Advanced Usage

### Skip Inference (Re-run Evaluation Only)

```bash
python batch_evaluation.py \
  -checkpoints checkpoint_list.yaml \
  -base_eval_config evaluation_config.yaml \
  --skip_inference
```

### Speed Up with No Weights

```bash
python batch_evaluation.py \
  -checkpoints checkpoint_list.yaml \
  -base_config inference_template.yaml \
  -base_eval_config evaluation_config.yaml \
  --no_weights
```

This script makes it easy to systematically evaluate multiple model checkpoints and compare their performance!
