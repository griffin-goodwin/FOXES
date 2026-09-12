from argparse import Namespace
from pathlib import Path

import pytest

from experiments.parameter_screening.create_subset import QUICK_COUNTS
from experiments.parameter_screening.run_experiment import build_config
from experiments.parameter_screening.run_sweep import (
    build_commands,
    group_commands_by_gpu,
    parse_gpu_ids,
    parse_local_window_sizes,
    run_command_queue,
    snapshot_base_config,
    stage_values,
    subset_preflight_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sweep_args(**overrides):
    values = {
        "stage": "patch-gaussian",
        "sparsity_weight": 0.1,
        "top_fraction": 0.02,
        "learning_rate": 1e-4,
        "base_config": PROJECT_ROOT / "training" / "train_config.yaml",
        "scheduler_t_max": 10,
        "subset_root": Path("/tmp/subset"),
        "dataset_root": None,
        "aia_pre_normalized": False,
        "runs_root": Path("/tmp/runs"),
        "max_steps": 1000,
        "seed": 42,
        "run": True,
        "parallel": 3,
        "gpu_ids": (0, 1),
        "local_window_sizes": (3, 5, 9),
        "patch_batch_size": 4,
        "patch_accumulate_grad_batches": 16,
    }
    values.update(overrides)
    return Namespace(**values)


def test_patch_gaussian_commands_pin_one_gpu_per_process_round_robin():
    commands = build_commands(sweep_args())
    assigned = [command[command.index("--gpu-id") + 1] for command in commands]

    assert assigned == ["0", "1", "0"]
    assert all("--run" in command for command in commands)
    assert parse_gpu_ids("0, 2") == (0, 2)


def test_patch_gaussian_stage_changes_one_regularizer_at_a_time():
    args = sweep_args(stage="patch-gaussian", learning_rate=5e-5)
    runs = stage_values(args)
    commands = build_commands(args)

    assert [(run.weight_decay, run.dropout) for run in runs] == [
        (1e-4, 0.1),
        (1e-3, 0.1),
        (1e-4, 0.2),
    ]
    assert [run.name for run in runs] == [
        "screen-patch-base-s0.1-top0.02-lr5e-05-steps1000-tmax10-do0.1-wd0.0001",
        "screen-patch-strong-wd-s0.1-top0.02-lr5e-05-steps1000-tmax10-do0.1-wd0.001",
        "screen-patch-dropout-s0.1-top0.02-lr5e-05-steps1000-tmax10-do0.2-wd0.0001",
    ]
    assert all("--global-sigma-init-dex" not in command for command in commands)
    assert [command[command.index("--dropout") + 1] for command in commands] == [
        "0.1", "0.1", "0.2",
    ]

    queues = group_commands_by_gpu(commands)
    assert len(queues) == 2
    assert [
        command[command.index("--gpu-id") + 1]
        for command in queues[0]
    ] == ["0", "0"]
    assert [
        command[command.index("--gpu-id") + 1]
        for command in queues[1]
    ] == ["1"]


def test_quantile_stage_remains_available():
    args = sweep_args(stage="quantile")
    runs = stage_values(args)
    commands = build_commands(args)

    assert len(runs) == 1
    assert runs[0].objective == "quantile"
    assert commands[0][commands[0].index("--objective") + 1] == "quantile"
    assert commands[0][commands[0].index("--sparsity-weight") + 1] == "0.1"


def test_detached_beta_stage_compares_only_beta_zero_and_one():
    args = sweep_args(stage="beta-detached", parallel=2)
    runs = stage_values(args)
    commands = build_commands(args)

    assert [run.beta_nll_beta for run in runs] == [0.0, 1.0]
    assert [command[command.index("--beta-nll-beta") + 1]
            for command in commands] == ["0.0", "1.0"]
    assert [command[command.index("--gpu-id") + 1]
            for command in commands] == ["0", "1"]


def test_mean_loss_stage_isolates_huber_with_no_sparsity():
    args = sweep_args(stage="mean-loss", sparsity_weight=0.7, parallel=2)
    runs = stage_values(args)
    commands = build_commands(args)

    assert [run.mean_loss for run in runs] == ["beta_nll", "huber"]
    assert [run.beta_nll_beta for run in runs] == [1.0, 1.0]
    assert [run.sparsity_weight for run in runs] == [0.0, 0.0]
    assert [command[command.index("--mean-loss") + 1]
            for command in commands] == ["beta_nll", "huber"]


def test_mean_gradient_controls_stage_has_mean_only_and_no_clip_runs():
    args = sweep_args(
        stage="mean-gradient-controls", sparsity_weight=0.7, parallel=2,
    )
    runs = stage_values(args)
    commands = build_commands(args)

    assert len(runs) == 4
    assert runs[0].name.startswith("screen-meanonly-beta0-local-w3-")
    assert runs[0].mean_only is True
    assert runs[0].mean_loss == "mse"
    assert runs[0].mask_mode == "local"
    assert runs[0].local_window == 3
    assert all(run.beta_nll_beta == 0.0 for run in runs)
    assert [run.disable_gradient_clipping for run in runs] == [
        False, True, True, True,
    ]
    assert [(run.mask_mode, run.local_window) for run in runs[1:]] == [
        ("local", 1), ("local", 3), ("none", None),
    ]
    assert all(run.embed_dim == 512 for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert "--mean-only" in commands[0]
    assert "--disable-gradient-clipping" not in commands[0]
    assert all(
        "--disable-gradient-clipping" in command
        for command in commands[1:]
    )


def test_contextual_local_depth_stage_is_matched_and_uses_small_subset():
    args = sweep_args(
        stage="contextual-local-depth", sparsity_weight=0.7, parallel=2,
        subset_root=Path("/data/FOXES_screening/subset-og-quick"),
        aia_pre_normalized=True,
    )
    runs = stage_values(args)
    commands = build_commands(args)

    assert len(runs) == 5
    assert [run.num_layers for run in runs] == [1, 2, 4, 8, 2]
    assert [run.gaussian_model_type for run in runs] == [
        None, None, None, None, "gaussian_nll_background_excess",
    ]
    assert all(run.beta_nll_beta == 0.0 for run in runs)
    assert all(run.mean_loss == "mse" for run in runs)
    assert all(run.disable_gradient_clipping for run in runs)
    assert all(run.mask_mode == "local" for run in runs)
    assert all(run.local_window == 3 for run in runs)
    assert all(run.embed_dim == 512 for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert all(
        command[command.index("--subset-root") + 1]
        == "/data/FOXES_screening/subset-og-quick"
        for command in commands
    )
    assert (
        commands[-1][commands[-1].index("--gaussian-model-type") + 1]
        == "gaussian_nll_background_excess"
    )


def test_inverted_mask_window_stage_is_matched_and_uses_symmetric_windows():
    args = sweep_args(
        stage="inverted-mask-window", sparsity_weight=0.7, parallel=2,
        subset_root=Path("/data/FOXES_screening/subset-og-quick"),
        aia_pre_normalized=True,
    )
    runs = stage_values(args)
    commands = build_commands(args)

    assert len(runs) == 4
    assert [run.local_window for run in runs] == [3, 5, 9, 17]
    assert "requestedw16-effectivew17" in runs[-1].name
    assert all(run.mask_mode == "inverted" for run in runs)
    assert all(run.embed_dim == 512 for run in runs)
    assert all(run.num_layers == 8 for run in runs)
    assert all(run.patch_size == 8 for run in runs)
    assert all(run.beta_nll_beta == 0.0 for run in runs)
    assert all(run.mean_loss == "mse" for run in runs)
    assert all(run.disable_gradient_clipping for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert all(
        run.checkpoint_monitor == "val_class/X/rmse_dex"
        for run in runs
    )
    assert [
        command[command.index("--local-window") + 1]
        for command in commands
    ] == ["3", "5", "9", "17"]
    assert all(
        command[command.index("--checkpoint-monitor") + 1]
        == "val_class/X/rmse_dex"
        for command in commands
    )


def test_local_weighted_huber_stage_matches_requested_experiment():
    args = sweep_args(
        stage="local-weighted-huber", sparsity_weight=0.7, parallel=1,
        subset_root=Path("/data/FOXES_screening/subset-og-quick"),
        aia_pre_normalized=True,
    )
    runs = stage_values(args)
    commands = build_commands(args)

    assert len(runs) == 1
    run = runs[0]
    assert run.embed_dim == 512
    assert run.num_layers == 8
    assert run.patch_size == 8
    assert run.mask_mode == "local"
    assert run.local_window == 3
    assert run.mean_loss == "huber"
    assert run.mean_huber_delta == 0.3
    assert run.mean_loss_weighting == "four_class_macro"
    assert run.beta_nll_beta == 0.0
    assert run.sparsity_weight == 0.0
    assert run.disable_gradient_clipping is True
    assert run.checkpoint_monitor == "val_class/X/rmse_dex"
    assert commands[0][commands[0].index("--mean-huber-delta") + 1] == "0.3"
    assert (
        commands[0][commands[0].index("--mean-loss-weighting") + 1]
        == "four_class_macro"
    )


def test_weighted_huber_followup_is_one_factor_at_a_time():
    args = sweep_args(
        stage="weighted-huber-followup", sparsity_weight=0.7, parallel=2,
        subset_root=Path("/data/FOXES_screening/subset-og-quick"),
        aia_pre_normalized=True,
    )
    runs = stage_values(args)
    commands = build_commands(args)

    assert [
        (
            run.num_layers, run.mean_huber_delta,
            run.mean_loss_weighting, run.local_window,
        )
        for run in runs
    ] == [
        (8, 0.15, "four_class_macro", 3),
        (8, 1.0, "four_class_macro", 3),
        (8, 0.15, "sqrt_four_class_macro", 3),
        (8, 1.0, "sqrt_four_class_macro", 3),
        (8, 0.3, "four_class_macro", 5),
        (8, 0.3, "sqrt_four_class_macro", 3),
        (4, 0.3, "four_class_macro", 3),
        (8, 0.3, "four_class_macro", 9),
        (4, 0.3, "sqrt_four_class_macro", 3),
        (2, 0.3, "four_class_macro", 3),
        (2, 0.3, "sqrt_four_class_macro", 3),
    ]
    assert all(run.embed_dim == 512 for run in runs)
    assert all(run.patch_size == 8 for run in runs)
    assert all(run.mask_mode == "local" for run in runs)
    assert [run.local_window for run in runs if run.num_layers == 8] == [
        3, 3, 3, 3, 5, 3, 9,
    ]
    assert all(run.mean_loss == "huber" for run in runs)
    assert all(run.beta_nll_beta == 0.0 for run in runs)
    assert all(run.disable_gradient_clipping for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert all(
        run.checkpoint_monitor == "val_class/X/rmse_dex" for run in runs
    )
    assert [command[command.index("--gpu-id") + 1] for command in commands] == [
        "0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0",
    ]


def test_canonical_weighted_huber_compares_inverse_and_sqrt_only():
    args = sweep_args(
        stage="canonical-weighted-huber", sparsity_weight=0.7, parallel=2,
        subset_root=Path("/data/FOXES_screening/subset-og-quick"),
        aia_pre_normalized=True,
    )
    runs = stage_values(args)
    commands = build_commands(args)

    assert len(runs) == 2
    assert [run.class_weighting for run in runs] == [
        "inverse_frequency", "sqrt_inverse_frequency",
    ]
    assert all(run.mean_huber_delta == 0.15 for run in runs)
    assert all(run.gaussian_model_type is None for run in runs)
    assert all(run.embed_dim == 512 for run in runs)
    assert all(run.num_layers == 8 for run in runs)
    assert all(run.patch_size == 8 for run in runs)
    assert all(run.mask_mode == "local" for run in runs)
    assert all(run.local_window == 3 for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert all(run.disable_gradient_clipping for run in runs)
    assert [
        command[command.index("--class-weighting") + 1]
        for command in commands
    ] == ["inverse_frequency", "sqrt_inverse_frequency"]


def test_attention_capacity_stage_matches_requested_four_experiments():
    args = sweep_args(
        stage="attention-capacity-weighted-huber",
        sparsity_weight=0.7,
        parallel=2,
        subset_root=Path("/data/FOXES_screening/subset-og-quick"),
        aia_pre_normalized=True,
    )
    runs = stage_values(args)
    commands = build_commands(args)

    assert [
        (run.num_layers, run.num_heads, run.hidden_dim)
        for run in runs
    ] == [
        (8, 8, 2048),
        (12, 8, 2048),
        (12, 16, 2048),
        (12, 8, 4096),
    ]
    assert all(run.embed_dim == 512 for run in runs)
    assert all(run.patch_size == 8 for run in runs)
    assert all(run.mask_mode == "local" for run in runs)
    assert all(run.local_window == 3 for run in runs)
    assert all(run.mean_huber_delta == 0.15 for run in runs)
    assert all(run.class_weighting == "inverse_frequency" for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert all(run.disable_gradient_clipping for run in runs)
    assert all(run.batch_size == 8 for run in runs)
    assert all(run.accumulate_grad_batches == 8 for run in runs)
    assert all(
        run.batch_size * run.accumulate_grad_batches == 64
        for run in runs
    )
    assert all(
        run.checkpoint_monitor == "val_class/macro_mse_dex2"
        for run in runs
    )
    assert [
        command[command.index("--num-heads") + 1]
        for command in commands
    ] == ["8", "8", "16", "8"]
    assert [
        command[command.index("--hidden-dim") + 1]
        for command in commands
    ] == ["2048", "2048", "2048", "4096"]
    assert all(
        command[command.index("--patch-size") + 1] == "8"
        for command in commands
    )


def test_background_excess_depth_completion_runs_only_missing_depths():
    args = sweep_args(
        stage="background-excess-depth-completion",
        sparsity_weight=0.7,
        parallel=2,
        subset_root=Path("/data/FOXES_screening/subset-og-quick"),
        aia_pre_normalized=True,
    )
    runs = stage_values(args)
    commands = build_commands(args)

    assert [run.num_layers for run in runs] == [4, 8, 1]
    assert all(
        run.gaussian_model_type == "gaussian_nll_background_excess"
        for run in runs
    )
    assert all(run.beta_nll_beta == 0.0 for run in runs)
    assert all(run.mean_loss == "mse" for run in runs)
    assert all(run.disable_gradient_clipping for run in runs)
    assert all(run.mask_mode == "local" for run in runs)
    assert all(run.local_window == 3 for run in runs)
    assert all(run.embed_dim == 512 for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert [
        command[command.index("--gpu-id") + 1] for command in commands
    ] == ["0", "1", "0"]
    assert all(
        command[command.index("--subset-root") + 1]
        == "/data/FOXES_screening/subset-og-quick"
        for command in commands
    )


def test_background_excess_depth_stage_is_self_contained():
    runs = stage_values(sweep_args(stage="background-excess-depth"))

    assert [run.num_layers for run in runs] == [1, 2, 4, 8]
    assert all(
        run.gaussian_model_type == "gaussian_nll_background_excess"
        for run in runs
    )


def test_local_window_beta_stage_isolates_window_and_beta():
    args = sweep_args(
        stage="local-window-beta", sparsity_weight=0.7, parallel=2,
    )
    runs = stage_values(args)
    commands = build_commands(args)

    assert parse_local_window_sizes("3,5,9") == (3, 5, 9)
    assert [(run.local_window, run.beta_nll_beta) for run in runs] == [
        (3, 0.0), (3, 1.0),
        (5, 0.0), (5, 1.0),
        (9, 0.0), (9, 1.0),
    ]
    assert all(run.mask_mode == "local" for run in runs)
    assert all(run.mean_loss == "beta_nll" for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert [command[command.index("--gpu-id") + 1]
            for command in commands] == ["0", "1", "0", "1", "0", "1"]
    assert [command[command.index("--local-window") + 1]
            for command in commands] == ["3", "3", "5", "5", "9", "9"]


def test_og_patch_attention_beta_stage_has_requested_eight_runs():
    args = sweep_args(
        stage="og-patch-attention-beta", sparsity_weight=0.7, parallel=2,
    )
    runs = stage_values(args)
    commands = build_commands(args)

    assert [
        (run.mask_mode, run.local_window, run.beta_nll_beta)
        for run in runs
    ] == [
        ("none", None, 0.0), ("none", None, 1.0),
        ("local", 1, 0.0), ("local", 1, 1.0),
        ("local", 3, 0.0), ("local", 3, 1.0),
        ("local", 5, 0.0), ("local", 5, 1.0),
    ]
    assert all(run.gaussian_model_type == "gaussian_nll_og" for run in runs)
    assert all(run.embed_dim == 512 for run in runs)
    assert all(run.num_layers == 8 for run in runs)
    assert all(run.patch_size == 8 for run in runs)
    assert all(run.mean_loss == "beta_nll" for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert [command[command.index("--gpu-id") + 1]
            for command in commands] == ["0", "1"] * 4
    assert all(
        command[command.index("--gaussian-model-type") + 1]
        == "gaussian_nll_og"
        for command in commands
    )


def test_patch_size_stage_uses_beta_zero_and_two_local_scales():
    args = sweep_args(stage="patch-size-beta0", parallel=2)
    runs = stage_values(args)
    commands = build_commands(args)

    assert [(run.patch_size, run.local_window) for run in runs] == [
        (4, 5), (4, 3),
    ]
    assert all(run.beta_nll_beta == 0.0 for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert all(run.batch_size == 4 for run in runs)
    assert all(run.accumulate_grad_batches == 16 for run in runs)
    assert [command[command.index("--patch-size") + 1]
            for command in commands] == ["4", "4"]
    assert [command[command.index("--batch-size") + 1]
            for command in commands] == ["4", "4"]


def test_architecture_capacity_stage_changes_one_axis_per_run():
    args = sweep_args(stage="architecture-capacity-beta0", parallel=2)
    runs = stage_values(args)
    commands = build_commands(args)

    assert [
        (run.embed_dim, run.num_layers, run.hidden_dim)
        for run in runs
    ] == [
        (512, None, None),
        (None, 16, None),
        (64, None, None),
        (128, None, None),
        (None, 2, None),
        (None, 4, None),
        (None, 12, None),
        (None, None, 4096),
        (None, None, 256),
        (None, None, 512),
        (None, None, 2048),
    ]
    assert all(run.patch_size == 8 for run in runs)
    assert all(run.local_window == 3 for run in runs)
    assert all(run.beta_nll_beta == 0.0 for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert [command[command.index("--gpu-id") + 1]
            for command in commands] == [
                "0", "1", "0", "1", "0", "1",
                "0", "1", "0", "1", "0",
            ]


def test_architecture_combo_stage_preserves_effective_batch_size():
    args = sweep_args(stage="architecture-combos-beta0", parallel=2)
    runs = stage_values(args)
    commands = build_commands(args)

    assert [
        (run.embed_dim, run.num_layers, run.batch_size,
         run.accumulate_grad_batches)
        for run in runs
    ] == [
        (1024, 16, 8, 8),
        (768, 24, 8, 8),
        (512, 16, 32, 2),
        (768, 16, 16, 4),
        (512, 24, 16, 4),
    ]
    assert all(
        run.batch_size * run.accumulate_grad_batches == 64
        for run in runs
    )
    assert all(run.hidden_dim is None for run in runs)
    assert all(run.patch_size == 8 for run in runs)
    assert all(run.local_window == 3 for run in runs)
    assert all(run.beta_nll_beta == 0.0 for run in runs)
    assert all(run.sparsity_weight == 0.0 for run in runs)
    assert [command[command.index("--gpu-id") + 1]
            for command in commands] == ["0", "1", "0", "1", "0"]


def test_gpu_queue_continues_after_failed_run(monkeypatch):
    attempted = []

    def fake_run(command, *, check):
        attempted.append(command)
        if command == ["first"]:
            raise __import__("subprocess").CalledProcessError(7, command)

    monkeypatch.setattr("subprocess.run", fake_run)

    failures = run_command_queue([["first"], ["second"]])

    assert attempted == [["first"], ["second"]]
    assert len(failures) == 1
def test_base_config_snapshot_is_content_addressed(tmp_path):
    source = tmp_path / "base.yaml"
    source.write_text("seed: 42\n")

    first = snapshot_base_config(source, tmp_path / "runs")
    second = snapshot_base_config(source, tmp_path / "runs")
    source.write_text("seed: 7\n")
    third = snapshot_base_config(source, tmp_path / "runs")

    assert first == second
    assert first != third
    assert first.read_text() == "seed: 42\n"


def test_subset_preflight_reports_missing_and_existing_alternative(tmp_path):
    existing = tmp_path / "subset"
    (existing / "AIA_processed" / "train").mkdir(parents=True)
    (existing / "SXR_processed" / "train").mkdir(parents=True)
    (existing / "subset_summary.json").write_text("{}\n")

    requested = tmp_path / "subset-quick"
    error = subset_preflight_error(requested)

    assert "Create the quick subset first" in error
    assert str(existing) in error
    assert subset_preflight_error(existing) is None


def test_quick_subset_keeps_full_evaluation_splits():
    assert QUICK_COUNTS == {
        "train": {"quiet": 10_000, "c": 3_200, "m": 1_000, "x": 50},
        "val": {"quiet": None, "c": None, "m": None, "x": None},
        "test": {"quiet": None, "c": None, "m": None, "x": None},
    }


def experiment_args(tmp_path, **overrides):
    subset_root = tmp_path / "subset"
    for kind in ("AIA_processed", "SXR_processed"):
        for split in ("train", "val", "test"):
            (subset_root / kind / split).mkdir(parents=True)
    (subset_root / "subset_summary.json").write_text("{}\n")
    values = {
        "name": "patch-gaussian",
        "objective": "gaussian",
        "gaussian_model_type": "gaussian_nll",
        "base_config": PROJECT_ROOT / "training" / "train_config.yaml",
        "subset_root": subset_root,
        "dataset_root": None,
        "aia_pre_normalized": False,
        "runs_root": tmp_path / "runs",
        "sparsity_weight": 0.0,
        # Keep this fixture independent of edits to the full-training config.
        "beta_nll_beta": 0.0,
        "mean_loss": None,
        "class_weighting": "none",
        "mean_loss_weighting": None,
        "mean_huber_delta": 1.0,
        "top_fraction": 0.02,
        "learning_rate": 1e-4,
        "scheduler_t_max": 10,
        "weight_decay": None,
        "dropout": None,
        "mask_mode": None,
        "local_window": None,
        "patch_size": None,
        "batch_size": None,
        "embed_dim": None,
        "hidden_dim": None,
        "num_heads": None,
        "num_layers": None,
        "checkpoint_monitor": None,
        "accumulate_grad_batches": None,
        "max_steps": 1000,
        "seed": 42,
        "gpu_id": 0,
        "wandb_project": "test-screening",
    }
    values.update(overrides)
    return Namespace(**values)


def test_gaussian_experiment_contains_patch_uncertainty_settings(tmp_path):
    config, _ = build_config(experiment_args(tmp_path))

    assert config["model_type"] == "gaussian_nll"
    assert "quantile" not in config
    assert config["uncertainty"] == {
        "class_weighting": "none",
        "huber_delta": 1.0,
        "patch_flux_scale_multiplier": 1.0,
        "max_abs_log10_patch_multiplier": 8.0,
        "relative_std_floor": 0.0025,
        "relative_std_max": 20.0,
        "patch_scale_floor_fraction": 1.0,
    }
    assert config["checkpoint"]["monitor"] == "val/mse"
    assert config["screening"]["uncertainty_source"] == "summed_patch_variance"
    assert "beta_nll_beta" not in config["screening"]
    assert config["callbacks"]["spatial_gaussian_enabled"] is True
    assert config["callbacks"]["spatial_gaussian_num_samples"] == 5
    assert config["callbacks"]["attention_enabled"] is False


def test_canonical_gaussian_rejects_removed_mean_only_mse_path(tmp_path):
    with pytest.raises(ValueError, match="weighted Huber only"):
        build_config(experiment_args(
            tmp_path,
            mean_loss="mse",
            mean_only=True,
        ))


def test_weighted_huber_experiment_defers_weights_to_training_split(tmp_path):
    config, _ = build_config(experiment_args(
        tmp_path,
        mean_loss="huber",
        mean_huber_delta=0.3,
        class_weighting="inverse_frequency",
    ))

    assert config["uncertainty"]["huber_delta"] == 0.3
    assert "mean_loss" not in config["uncertainty"]
    assert config["uncertainty"]["class_weighting"] == "inverse_frequency"
    assert "class_weights" not in config["uncertainty"]
    assert config["screening"]["class_weighting"] == "inverse_frequency"


def test_background_excess_experiment_has_guarded_component_defaults(tmp_path):
    config, _ = build_config(experiment_args(
        tmp_path,
        gaussian_model_type="gaussian_nll_background_excess",
        beta_nll_beta=0.0,
        mean_loss="mse",
        disable_gradient_clipping=True,
        mask_mode="local",
        local_window=3,
        num_layers=2,
    ))

    assert config["model_type"] == "gaussian_nll_background_excess"
    assert config["uncertainty"]["train_uncertainty"] is True
    assert config["uncertainty"]["background_flux_max"] == 5e-6
    assert config["uncertainty"]["background_initial_cap_fraction"] == 0.5
    assert config["uncertainty"]["excess_initial_fraction"] == 0.2
    assert config["uncertainty"]["solar_disk_radius_fraction"] == 0.48
    assert config["optimizer"]["gradient_clip_val"] is None
    assert "background-excess-patch-mean" in config["wandb"]["tags"]


def test_og_gaussian_experiment_selects_original_patch_forecast(tmp_path):
    config, _ = build_config(experiment_args(
        tmp_path, gaussian_model_type="gaussian_nll_og",
    ))

    assert config["model_type"] == "gaussian_nll_og"
    assert config["screening"]["model_type"] == "gaussian_nll_og"
    assert "patch_flux_scale_multiplier" not in config["uncertainty"]
    assert "max_abs_log10_patch_multiplier" not in config["uncertainty"]
    assert "og-patch-mean" in config["wandb"]["tags"]
    assert "beta-0" in config["wandb"]["tags"]
    assert "beta-1" not in config["wandb"]["tags"]
    assert "subset-data" in config["wandb"]["tags"]
    assert "full-data" not in config["wandb"]["tags"]


def test_gaussian_experiment_overrides_regularization(tmp_path):
    config, _ = build_config(experiment_args(
        tmp_path, weight_decay=1e-3, dropout=0.2,
    ))

    assert config["optimizer"]["weight_decay"] == 1e-3
    assert config["vit_architecture"]["dropout"] == 0.2
    assert config["screening"]["weight_decay"] == 1e-3
    assert config["screening"]["dropout"] == 0.2


def test_gaussian_experiment_overrides_local_attention_window(tmp_path):
    config, _ = build_config(experiment_args(
        tmp_path, mask_mode="local", local_window=5,
    ))

    assert config["vit_architecture"]["mask_mode"] == "local"
    assert config["vit_architecture"]["local_window"] == 5
    assert config["screening"]["mask_mode"] == "local"
    assert config["screening"]["local_window"] == 5


def test_experiment_can_select_checkpoints_by_x_class_rmse(tmp_path):
    config, _ = build_config(experiment_args(
        tmp_path,
        checkpoint_monitor="val_class/X/rmse_dex",
    ))

    assert config["checkpoint"]["monitor"] == "val_class/X/rmse_dex"
    assert config["screening"]["checkpoint_monitor"] == (
        "val_class/X/rmse_dex"
    )
    assert config["callbacks"]["per_class_metrics_enabled"] is True


def test_experiment_derives_num_patches_from_patch_size(tmp_path):
    config, _ = build_config(experiment_args(
        tmp_path, patch_size=4, batch_size=4,
    ))

    assert config["vit_architecture"]["patch_size"] == 4
    assert config["vit_architecture"]["num_patches"] == 16_384
    assert config["batch_size"] == 4
    assert config["screening"]["patch_size"] == 4
    assert config["screening"]["num_patches"] == 16_384


def test_experiment_overrides_transformer_capacity(tmp_path):
    config, _ = build_config(experiment_args(
        tmp_path, embed_dim=128, hidden_dim=512, num_heads=16, num_layers=4,
    ))

    assert config["vit_architecture"]["embed_dim"] == 128
    assert config["vit_architecture"]["hidden_dim"] == 512
    assert config["vit_architecture"]["num_heads"] == 16
    assert config["vit_architecture"]["num_layers"] == 4
    assert config["screening"]["embed_dim"] == 128
    assert config["screening"]["hidden_dim"] == 512
    assert config["screening"]["num_heads"] == 16
    assert config["screening"]["num_layers"] == 4


def test_canonical_gaussian_drops_legacy_beta_zero_setting(tmp_path):
    config, _ = build_config(experiment_args(
        tmp_path, beta_nll_beta=0.0,
    ))

    assert "beta_nll_beta" not in config["uncertainty"]
    assert "beta_nll_beta" not in config["screening"]


def test_quantile_experiment_enables_spatial_maps(tmp_path):
    config, _ = build_config(experiment_args(
        tmp_path, name="quantile", objective="quantile",
        sparsity_weight=0.1,
    ))

    assert config["model_type"] == "quantile"
    assert "uncertainty" not in config
    assert config["quantile"]["spatial_sparsity"]["weight"] == 0.1
    assert config["callbacks"]["spatial_quantile_enabled"] is True
