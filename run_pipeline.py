#!/usr/bin/env python3
"""
FOXES End-to-End Pipeline Orchestrator

Runs any combination of pipeline steps in order:

  0. hf_download    - Download processed+split data from HuggingFace (replaces steps 1-5)
  1. download_aia   - Download SDO/AIA EUV images from JSOC (download/download_sdo.py)
  2. download_sxr   - Download GOES SXR flux data (download/sxr_downloader.py)
  3. combine_sxr    - Combine raw GOES .nc files into per-satellite CSVs (data/sxr_data_processing.py)
  4. preprocess     - EUV cleaning, ITI processing, data alignment (data/process_data_pipeline.py)
  5. split          - Split AIA + SXR into train/val/test (data/split_data.py)
  6. normalize      - Compute SXR normalization stats on train split (data/sxr_normalization.py)
  7. train          - Train the ViTLocal forecasting model (forecasting/training/train.py)
  8. inference      - Run batch inference on val/test data (forecasting/inference/inference.py)
  9. flare_analysis - Detect, track, and match flares (forecasting/inference/flare_analysis.py)

Usage:
  python run_pipeline.py --list
  python run_pipeline.py --config pipeline_config.yaml --steps all
  python run_pipeline.py --config pipeline_config.yaml --steps train,inference,flare_analysis
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base, modifying base in-place."""
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def write_merged_config(base_path: str, overrides: dict, out_name: str) -> Path:
    """
    Load base_path YAML, apply overrides, write merged result to ROOT/.{out_name}.yaml.
    Returns the path of the merged file.
    """
    with open(base_path) as f:
        base = yaml.safe_load(f) or {}
    deep_merge(base, overrides)
    out = ROOT / f".merged_{out_name}.yaml"
    with open(out, "w") as f:
        yaml.dump(base, f, default_flow_style=False)
    log.info(f"  Merged config written to {out}")
    return out


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEP_ORDER = [
    "hf_download",
    "download_aia",
    "download_sxr",
    "combine_sxr",
    "preprocess",
    "split",
    "normalize",
    "train",
    "inference",
    "evaluate",
    "flare_analysis",
]

STEP_INFO = {
    "hf_download": {
        "description": "Download processed+split AIA/SXR data from HuggingFace Hub (replaces download→preprocess→split)",
        "script": ROOT / "download" / "hugging_face_data_download.py",
    },
    "download_aia": {
        "description": "Download SDO/AIA EUV images from JSOC",
        "script": ROOT / "download" / "download_sdo.py",
    },
    "download_sxr": {
        "description": "Download GOES SXR flux data via SXRDownloader",
        "script": None,  # invoked inline via python -c
    },
    "combine_sxr": {
        "description": "Combine raw GOES .nc files into per-satellite CSVs for alignment",
        "script": ROOT / "data" / "sxr_data_processing.py",
    },
    "preprocess": {
        "description": "EUV cleaning, ITI processing, and AIA/SXR data alignment",
        "script": ROOT / "data" / "process_data_pipeline.py",
    },
    "normalize": {
        "description": "Compute SXR log-normalization statistics (mean/std)",
        "script": ROOT / "data" / "sxr_normalization.py",
    },
    "split": {
        "description": "Split AIA and SXR data into train/val/test by date range",
        "script": ROOT / "data" / "split_data.py",
    },
    "train": {
        "description": "Train the ViTLocal solar flare forecasting model",
        "script": ROOT / "forecasting" / "training" / "train.py",
    },
    "inference": {
        "description": "Run batch inference and save predictions CSV",
        "script": ROOT / "forecasting" / "inference" / "inference.py",
    },
    "evaluate": {
        "description": "Compute metrics and generate evaluation plots from predictions CSV",
        "script": ROOT / "forecasting" / "inference" / "evaluation.py",
    },
    "flare_analysis": {
        "description": "Detect, track, and match flares; generate plots/movies",
        "script": ROOT / "forecasting" / "inference" / "flare_analysis.py",
    },
}


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def build_commands(step: str, cfg: dict, force: bool) -> list[list[str]] | None:
    """
    Return a list of subprocess commands for a given step, or None if required config is missing.
    Most steps return a single command; 'split' returns two (AIA then SXR).
    """

    def require(keys: list[str], section: str = None) -> bool:
        src = cfg.get(section, {}) if section else cfg
        missing = [k for k in keys if not src.get(k)]
        if missing:
            prefix = f"{section}." if section else ""
            log.error(f"pipeline_config.yaml missing required keys: {[prefix + k for k in missing]}")
            return False
        return True

    if step == "hf_download":
        hf = cfg.get("hf_download", {})
        config_path = hf.get("config", "download/hf_download_config.yaml")
        return [[sys.executable, str(STEP_INFO[step]["script"]), "--config", config_path]]

    if step == "download_aia":
        if not require(["download_dir", "email"], "aia") or not require(["start_date"]):
            return None
        aia = cfg["aia"]
        cmd = [sys.executable, str(STEP_INFO[step]["script"]),
               "--download_dir", aia["download_dir"],
               "--email",        aia["email"],
               "--start_date",   cfg["start_date"]]
        if cfg.get("end_date"):
            cmd += ["--end_date", cfg["end_date"]]
        if aia.get("cadence"):
            cmd += ["--cadence", str(aia["cadence"])]
        return [cmd]

    if step == "download_sxr":
        if not require(["save_dir"], "sxr") or not require(["start_date"]):
            return None
        start = cfg["start_date"]
        end = cfg.get("end_date", start)
        save_dir = cfg["sxr"]["save_dir"]
        inline = (
            f"import sys; sys.path.insert(0, r'{ROOT}'); "
            f"from download.sxr_downloader import SXRDownloader; "
            f"d = SXRDownloader(save_dir=r'{save_dir}'); "
            f"d.download_and_save_goes_data(start='{start}', end='{end}')"
        )
        return [[sys.executable, "-c", inline]]

    if step == "combine_sxr":
        if not require(["save_dir"], "sxr"):
            return None
        raw_dir = cfg["sxr"]["save_dir"]
        combined_dir = str(Path(raw_dir) / "combined")
        return [[sys.executable, str(STEP_INFO[step]["script"]),
                 "--data_dir", raw_dir,
                 "--output_dir", combined_dir]]

    script = STEP_INFO[step]["script"]
    base = [sys.executable, str(script)]

    if step == "preprocess":
        pre = cfg.get("preprocess", {})
        cmd = base[:]
        if pre.get("config"):
            cmd += ["--config", pre["config"]]
        if force:
            cmd += ["--force"]
        return [cmd]

    if step == "normalize":
        if not require(["sxr_dir", "output_path"], "normalize"):
            return None
        n = cfg["normalize"]
        return [base + ["--sxr_dir", n["sxr_dir"], "--output_path", n["output_path"]]]

    if step == "split":
        if not require(["aia_input_dir", "sxr_input_dir"], "split"):
            return None
        s = cfg["split"]
        date_args = []
        for key in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end"):
            if s.get(key):
                date_args += [f"--{key}", s[key]]
        # Each data type splits into its own input directory (creates train/val/test subdirs there)
        aia_cmd = base + ["--input_folder", s["aia_input_dir"], "--output_dir", s["aia_input_dir"],
                          "--data_type", "aia"] + date_args
        sxr_cmd = base + ["--input_folder", s["sxr_input_dir"], "--output_dir", s["sxr_input_dir"],
                          "--data_type", "sxr"] + date_args
        return [aia_cmd, sxr_cmd]

    if step == "train":
        if not require(["config"], "train"):
            return None
        t = cfg["train"]
        config_path = t["config"]
        if t.get("overrides"):
            config_path = str(write_merged_config(config_path, t["overrides"], "train_config"))
        return [base + ["-config", config_path]]

    if step == "inference":
        if not require(["config"], "inference"):
            return None
        inf = cfg["inference"]
        config_path = inf["config"]
        if inf.get("overrides"):
            config_path = str(write_merged_config(config_path, inf["overrides"], "inference_config"))
        return [base + ["-config", config_path]]

    if step == "evaluate":
        if not require(["config"], "evaluate"):
            return None
        ev = cfg["evaluate"]
        config_path = ev["config"]
        if ev.get("overrides"):
            config_path = str(write_merged_config(config_path, ev["overrides"], "evaluate_config"))
        return [base + ["-config", config_path]]

    if step == "flare_analysis":
        if not require(["config"], "inference"):
            return None
        inf = cfg["inference"]
        config_path = inf["config"]
        if inf.get("overrides"):
            config_path = str(write_merged_config(config_path, inf["overrides"], "inference_config"))
        return [base + ["--config", config_path]]

    return [base]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_step(step: str, cmds: list[list[str]]) -> bool:
    info = STEP_INFO[step]
    total_start = time.time()

    for i, cmd in enumerate(cmds):
        label = f"{step.upper()}" + (f" ({i + 1}/{len(cmds)})" if len(cmds) > 1 else "")
        log.info("")
        log.info("=" * 70)
        log.info(f"  STEP: {label}")
        log.info(f"  {info['description']}")
        log.info(f"  {' '.join(str(c) for c in cmd)}")
        log.info("=" * 70)

        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            log.error(f"  FAILED  {label} exited with code {result.returncode}")
            return False

    elapsed = time.time() - total_start
    log.info(f"  DONE  {step} completed in {elapsed:.1f}s")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_steps():
    print("\nAvailable pipeline steps (in order):\n")
    for i, step in enumerate(STEP_ORDER, 1):
        print(f"  {i}. {step:<16} {STEP_INFO[step]['description']}")
    print()
    print("Use --steps all to run every step, or comma-separate specific steps.")
    print("Example: --steps train,inference,flare_analysis\n")


def main():
    parser = argparse.ArgumentParser(
        description="FOXES End-to-End Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline_config.yaml")
    parser.add_argument("--steps",  type=str, default=None,
                        help=f"Comma-separated steps to run, or 'all'. Available: {', '.join(STEP_ORDER)}")
    parser.add_argument("--list",  action="store_true", help="List all available steps and exit")
    parser.add_argument("--force", action="store_true", help="Force re-run (forwarded to preprocess step)")

    args = parser.parse_args()

    if args.list:
        list_steps()
        return

    if not args.steps:
        parser.print_help()
        return

    if not args.config:
        log.error("--config is required. Point it at your pipeline_config.yaml.")
        sys.exit(1)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve step list
    if args.steps.strip().lower() == "all":
        selected = list(STEP_ORDER)
    else:
        selected = [s.strip() for s in args.steps.split(",")]
        unknown = [s for s in selected if s not in STEP_INFO]
        if unknown:
            log.error(f"Unknown steps: {', '.join(unknown)}")
            list_steps()
            sys.exit(1)
        selected = [s for s in STEP_ORDER if s in selected]  # preserve order

    log.info(f"Config: {args.config}")
    log.info(f"Running {len(selected)} step(s): {' -> '.join(selected)}")

    passed, failed = [], []

    for step in selected:
        cmds = build_commands(step, cfg, args.force)
        if cmds is None:
            failed.append(step)
            break

        if run_step(step, cmds):
            passed.append(step)
        else:
            failed.append(step)
            log.error(f"Pipeline stopped at '{step}'.")
            break

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 70)
    for s in passed:
        log.info(f"  PASSED   {s}")
    for s in failed:
        log.error(f"  FAILED   {s}")
    for s in [s for s in selected if s not in passed and s not in failed]:
        log.info(f"  SKIPPED  {s}")
    log.info("=" * 70)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
