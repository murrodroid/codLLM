import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import codllm.wandb_utils as wandb_utils
from codllm.config import Config
from codllm.experiments import (
    LsfProfile,
    SpecError,
    build_trial_environment,
    build_wandb_agent_spec,
    load_experiment_spec,
    load_trial_parameters,
    prepare_lsf_submission,
    validate_trial_parameters,
)


def _write_bayesian_base(tmp_path: Path) -> Path:
    """Write a minimal single-run base config for wrapper tests."""
    path = tmp_path / "bayesian_base.toml"
    path.write_text(
        """
name = "bayesian-base"
command = "train"

[env]
CODLLM_OUTPUT_DIR = "checkpoints/publication/bayesian"
CODLLM_SEED = 777
CODLLM_DATA_SEED = 777
CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS = 32
""",
        encoding="utf-8",
    )
    return path


def test_load_trial_parameters_requires_json_object(tmp_path: Path) -> None:
    """The W&B parameter file must contain one JSON object."""
    valid_path = tmp_path / "valid.json"
    valid_path.write_text('{"balance_floor": 300}', encoding="utf-8")
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("[1, 2]", encoding="utf-8")

    assert load_trial_parameters(valid_path) == {"balance_floor": 300}
    with pytest.raises(SpecError, match="JSON object"):
        load_trial_parameters(invalid_path)


def test_validate_trial_parameters_rejects_unknown_and_invalid_values() -> None:
    """Only declared, well-formed Bayesian dimensions may reach the environment."""
    with pytest.raises(SpecError, match="Unsupported"):
        validate_trial_parameters({"shell_command": "unsafe"})
    with pytest.raises(SpecError, match="multicod_synthetic_ratio"):
        validate_trial_parameters({"multicod_synthetic_ratio": 1.1})
    with pytest.raises(SpecError, match="scheduler"):
        validate_trial_parameters({"scheduler": "linear"})


def test_build_trial_environment_maps_all_parameters_and_scopes_output(
    tmp_path: Path,
) -> None:
    """Sampled values should override the base through an isolated trial root."""
    base_path = _write_bayesian_base(tmp_path)
    parameters = {
        "balance_floor": 450.0,
        "multicod_synthetic_ratio": 0.45,
        "pretrain_epochs": 16,
        "pretrain_synthetic_ratio": 0.3,
        "pretrain_target_per_label": 12,
        "learning_rate": 2.5e-5,
        "scheduler": "constant",
        "warmup_ratio": 0.15,
        "base_perturbation_rate": 0.25,
    }

    env = build_trial_environment(
        base_path,
        parameters,
        environ={
            "WANDB_RUN_ID": "abc123",
            "RUN_STORAGE_DIR": "/work3/s234805/codllm",
        },
    )

    assert env["CODLLM_SEED"] == "777"
    assert env["CODLLM_DATA_SEED"] == "777"
    assert env["CODLLM_BALANCE_FLOOR"] == "450"
    assert env["CODLLM_MULTICOD_SYNTHETIC_RATIO"] == "0.45"
    assert env["CODLLM_PRETRAIN_ENABLED"] == "1"
    assert env["CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS"] == "16"
    assert env["CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_RATIO"] == "0.3"
    assert env["CODLLM_PRETRAIN_UPSAMPLE_TARGET_PER_LABEL"] == "12"
    assert env["CODLLM_LR"] == "2.5e-05"
    assert env["CODLLM_LR_SCHEDULER_TYPE"] == "constant"
    assert env["CODLLM_WARMUP_RATIO"] == "0.15"
    assert env["CODLLM_BASE_PERTURBATION_RATE"] == "0.25"
    assert env["CODLLM_OUTPUT_DIR"] == (
        "/work3/s234805/codllm/checkpoints/publication/bayesian/trial-abc123"
    )
    assert env["CODLLM_AUTO_RESUME"] == "0"
    assert env["CODLLM_WANDB_RUN_NAME"] == "publication-bayes-abc123"
    assert json.loads(env["CODLLM_EXPERIMENT_SWEEP_VALUES"]) == parameters


def test_zero_pretraining_budget_disables_without_exporting_zero_epochs(
    tmp_path: Path,
) -> None:
    """The valid no-pretraining form should retain a positive ignored epoch value."""
    base_path = _write_bayesian_base(tmp_path)

    env = build_trial_environment(
        base_path,
        {"pretrain_epochs": 0},
        environ={"WANDB_RUN_ID": "off123"},
    )

    assert env["CODLLM_PRETRAIN_ENABLED"] == "0"
    assert env["CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS"] == "32"


def test_wandb_agent_spec_and_lsf_script_run_one_nonresumable_trial(
    tmp_path: Path,
) -> None:
    """Every Bayesian array element should claim exactly one W&B trial."""
    spec = build_wandb_agent_spec(
        "murromanden_data/codllm/abc12345",
        agents=4,
    )
    profile = LsfProfile(
        name="test",
        queue="gpu",
        wall_time="24:00",
        cores=2,
        memory="2GB",
        sync_env=False,
    )
    submission = prepare_lsf_submission(
        spec,
        profile,
        project_dir=tmp_path,
        output_root="jobs/generated",
    )
    script = submission.script_path.read_text(encoding="utf-8")
    env_file = (submission.env_dir / "run-1.env").read_text(encoding="utf-8")

    assert spec.command == "wandb-agent"
    assert len(spec.expanded_runs()) == 4
    assert '#BSUB -J "codllm-publication_bayesian_agents[1-4]"' in script
    assert "wandb agent --forward-signals --count 1" in script
    assert "wandb-agent) run_wandb_agent" in script
    assert "export STORAGE_FOLDER RUN_STORAGE_DIR" in script
    assert "export CODLLM_AUTO_RESUME=0" in env_file
    assert "export WANDB_SWEEP_ID=murromanden_data/codllm/abc12345" in env_file


def test_checked_in_publication_specs_pin_seed_grids_and_isolated_outputs() -> None:
    """Publication curve specs should preserve the declared discovery protocol."""
    expected = {
        "runs/publication/balance_floor_curve.toml": (
            "CODLLM_BALANCE_FLOOR",
            ["0", "100", "200", "300", "450", "600", "900"],
        ),
        "runs/publication/multicod_synthetic_curve.toml": (
            "CODLLM_MULTICOD_SYNTHETIC_RATIO",
            ["0", "0.15", "0.3", "0.45", "0.6", "0.8", "1"],
        ),
    }
    for path, (env_name, values) in expected.items():
        spec = load_experiment_spec(path)
        runs = spec.expanded_runs()
        assert [run.env[env_name] for run in runs] == values
        assert {run.env["CODLLM_SEED"] for run in runs} == {"777"}
        assert {run.env["CODLLM_DATA_SEED"] for run in runs} == {"777"}
        assert {run.env["CODLLM_FINAL_TEST_EVAL_ENABLED"] for run in runs} == {"0"}
        assert {run.env["CODLLM_WANDB_MODE"] for run in runs} == {"online"}
        assert {run.env["CODLLM_WANDB_LOG_MODEL"] for run in runs} == {"false"}
        assert len({run.env["CODLLM_OUTPUT_DIR"] for run in runs}) == len(runs)

    pretrain = load_experiment_spec(
        "runs/publication/pretraining_dose_curve.toml"
    ).expanded_runs()
    assert [run.env["CODLLM_PRETRAIN_ENABLED"] for run in pretrain] == [
        "0",
        "1",
        "1",
        "1",
        "1",
        "1",
    ]
    assert [run.env["CODLLM_PRETRAIN_NUM_TRAIN_EPOCHS"] for run in pretrain] == [
        "1",
        "4",
        "8",
        "16",
        "32",
        "48",
    ]
    assert len({run.env["CODLLM_OUTPUT_DIR"] for run in pretrain}) == len(pretrain)


def test_completed_sweep_trial_logs_one_optimization_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Bayesian controller should receive the best validation macro F1."""
    fake_run = SimpleNamespace(summary={})
    logged: list[dict[str, float]] = []
    fake_wandb = SimpleNamespace(run=fake_run, log=logged.append)
    monkeypatch.setenv("WANDB_SWEEP_ID", "abc12345")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(wandb_utils, "_import_wandb", lambda: fake_wandb)
    trainer = SimpleNamespace(state=SimpleNamespace(best_metric=0.734))

    wandb_utils.log_optimization_result(
        Config(save_strategy_best_metric="macro_f1"), trainer
    )

    assert logged == [{"optimization/val_macro_f1": 0.734}]
    assert fake_run.summary["optimization/val_macro_f1"] == 0.734
