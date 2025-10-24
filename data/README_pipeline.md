# Data Processing Pipeline

This directory contains a comprehensive data processing pipeline for solar flare data analysis.

## Pipeline Scripts

### Main Orchestrator
- **`process_data_pipeline.py`** - Main orchestrator script that runs all processing steps in sequence

### Individual Processing Steps
1. **`euv_data_cleaning.py`** - Removes bad AIA files based on timestamp validation
2. **`iti_data_processing.py`** - Processes good AIA data using ITI methods  
3. **`align_data.py`** - Concatenates GOES data and checks for missing data

### Configuration
- **`pipeline_config.py`** - Configuration system for all directory paths and settings
- **`pipeline_config_template.yaml`** - YAML template for creating custom configurations
- **`pipeline_config_template.py`** - Python template (backward compatibility)

## Usage

### Run the Complete Pipeline
```bash
# Run all steps (skip completed ones) with default configuration
python process_data_pipeline.py

# Force rerun all steps
python process_data_pipeline.py --force

# Use custom configuration file (YAML or Python)
python process_data_pipeline.py --config my_config.yaml

# Display current configuration
python process_data_pipeline.py --show-config

# Validate configuration paths
python process_data_pipeline.py --validate

# Create YAML configuration template
python process_data_pipeline.py --create-template
```

### Run Individual Steps
```bash
# EUV data cleaning
python euv_data_cleaning.py

# ITI data processing  
python iti_data_processing.py

# Data alignment
python align_data.py
```

## Pipeline Features

- **Modular Configuration**: Easy to change directory paths and settings
- **Automatic Step Skipping**: Steps that are already completed are automatically skipped
- **Comprehensive Logging**: All operations are logged to both console and file (`data_processing_pipeline.log`)
- **Error Handling**: Pipeline stops on first error with detailed error reporting
- **Progress Tracking**: Real-time progress updates and timing information
- **Configuration Validation**: Check if all required paths exist before running
- **Template Generation**: Create custom configuration templates easily

## Configuration

### Default Directories

The pipeline uses the following default directories (configurable):

**Input Directories:**
- `/mnt/data/AUGUST/SDO-AIA-timespan/` - Raw AIA data
- `/mnt/data/NEW-FLARE/combined/` - GOES CSV files
- `/mnt/data/NEW-FLARE/AIA_processed/` - Processed AIA files

**Output Directories:**
- `/mnt/data/AUGUST/SDO-AIA_bad/` - Bad AIA files (from EUV cleaning)
- `/mnt/data/AUGUST/AIA_ITI/` - Processed AIA data (from ITI processing)
- `/mnt/data/NEW-FLARE/GOES-SXR-A/` - GOES SXR-A data (from alignment)
- `/mnt/data/NEW-FLARE/GOES-SXR-B/` - GOES SXR-B data (from alignment)
- `/mnt/data/NEW-FLARE/AIA_ITI_MISSING/` - AIA files with missing GOES data

### Custom Configuration

To use custom directories:

1. **Create a YAML configuration file:**
   ```bash
   python process_data_pipeline.py --create-template
   # Edit pipeline_config_template.yaml with your paths
   ```

2. **Use your custom configuration:**
   ```bash
   python process_data_pipeline.py --config my_config.yaml
   ```

3. **Validate your configuration:**
   ```bash
   python process_data_pipeline.py --config my_config.yaml --validate
   ```

**YAML Configuration Example:**
```yaml
# Data Processing Pipeline Configuration
base_data_dir: /your/data/path

euv:
  input_folder: /your/data/AIA-raw
  bad_files_dir: /your/data/AIA-bad
  wavelengths: [94, 131, 171, 193, 211, 304]

iti:
  input_folder: /your/data/AIA-raw
  output_folder: /your/data/AIA-processed
  wavelengths: [94, 131, 171, 193, 211, 304]

alignment:
  goes_data_dir: /your/data/GOES
  aia_processed_dir: /your/data/AIA-processed
  output_sxr_a_dir: /your/data/GOES-SXR-A
  output_sxr_b_dir: /your/data/GOES-SXR-B
  aia_missing_dir: /your/data/AIA-missing

processing:
  max_processes: null  # null = use all CPU cores
  batch_size_multiplier: 4
  min_batch_size: 1
```

## Requirements

- Python 3.6+
- Required packages: pandas, numpy, astropy, tqdm, multiprocessing, pyyaml
- Sufficient disk space for data processing
- Access to the specified data directories

## Logging

The pipeline creates detailed logs in `data_processing_pipeline.log` including:
- Start/end times for each step
- Success/failure status
- Error messages and stack traces
- Processing statistics
