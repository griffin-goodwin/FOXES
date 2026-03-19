"""
PipelineConfig: loads and validates the data processing pipeline configuration.

Used by process_data_pipeline.py. Config is read from a YAML file and passed
to each sub-script as a JSON string via the PIPELINE_CONFIG environment variable.
"""

import json
from pathlib import Path

import yaml


TEMPLATE_PATH = Path(__file__).parent / "pipeline_config.yaml"

# Paths that must exist before the pipeline runs
REQUIRED_INPUT_PATHS = [
    ("euv", "input_folder"),
    ("iti", "input_folder"),
]

# Paths that should be created before the pipeline runs
OUTPUT_PATHS = [
    ("euv",       "bad_files_dir"),
    ("iti",       "output_folder"),
    ("alignment", "output_sxr_dir"),
    ("alignment", "aia_missing_dir"),
]


class PipelineConfig:
    def __init__(self, config_path: str = None):
        if config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._defaults()

    # ------------------------------------------------------------------

    def get_path(self, section: str, key: str) -> str:
        """Return config[section][key], or config[section] if key == section."""
        section_data = self.config.get(section, {})
        if isinstance(section_data, dict):
            return section_data.get(key, "")
        return section_data  # scalar value (e.g. base_data_dir)

    def to_json(self) -> str:
        """Serialize config to JSON string for passing via environment variable."""
        return json.dumps(self.config)

    # ------------------------------------------------------------------

    def validate_paths(self) -> tuple[bool, list[str]]:
        """Check that all required input paths exist. Returns (valid, missing)."""
        missing = []
        for section, key in REQUIRED_INPUT_PATHS:
            p = self.get_path(section, key)
            if p and not Path(p).exists():
                missing.append(f"{section}.{key}: {p}")
        return (len(missing) == 0, missing)

    def create_directories(self):
        """Create all output directories."""
        for section, key in OUTPUT_PATHS:
            p = self.get_path(section, key)
            if p:
                Path(p).mkdir(parents=True, exist_ok=True)

    def print_config(self):
        print(yaml.dump(self.config, default_flow_style=False))

    def save_config_template(self, path: str = None):
        dest = Path(path) if path else TEMPLATE_PATH
        with open(dest, "w") as f:
            yaml.dump(self._defaults(), f, default_flow_style=False)
        print(f"Template saved to {dest}")

    # ------------------------------------------------------------------

    @staticmethod
    def _defaults() -> dict:
        return {
            "base_data_dir": "/Volumes/T9/Data_FOXES",
            "euv": {
                "input_folder":  "/Volumes/T9/Data_FOXES/AIA_raw",
                "bad_files_dir": "/Volumes/T9/Data_FOXES/AIA_bad",
                "wavelengths":   [94, 131, 171, 193, 211, 304, 335],
            },
            "iti": {
                "input_folder":  "/Volumes/T9/Data_FOXES/AIA_raw",
                "output_folder": "/Volumes/T9/Data_FOXES/AIA_processed",
                "wavelengths":   [94, 131, 171, 193, 211, 304, 335],
            },
            "alignment": {
                "goes_data_dir":    "/Volumes/T9/Data_FOXES/SXR_raw/combined",
                "aia_processed_dir": "/Volumes/T9/Data_FOXES/AIA_processed",
                "output_sxr_dir":   "/Volumes/T9/Data_FOXES/SXR_processed",
                "aia_missing_dir":  "/Volumes/T9/Data_FOXES/AIA_missing",
            },
            "processing": {
                "max_processes":        None,
                "batch_size_multiplier": 4,
                "min_batch_size":        1,
            },
        }
