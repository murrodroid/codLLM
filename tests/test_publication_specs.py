"""Offline publication specification, launch-guide, and LSF syntax contracts."""

import re
import shlex
import subprocess
from pathlib import Path

import pytest

from codllm.config import config_from_env
from codllm.experiments.lsf import load_lsf_profile, prepare_lsf_submission
from codllm.experiments.specs import load_experiment_spec


@pytest.mark.parametrize(
    ("name", "count"),
    [
        ("smoke", 1),
        ("interaction_confirmation", 8),
        ("screening_controls", 2),
        ("source_transfer_pilot", 4),
        ("source_transfer_validation", 6),
        ("row_split_confirmation", 1),
        ("source_ablation_controls", 4),
        ("baselines_grouped", 3),
        ("baselines_row", 3),
        ("baselines_copenhagen", 3),
        ("baselines_belgium", 3),
        ("baselines_final", 3),
        ("reduced_data_calibration", 4),
        ("reduced_split_sensitivity", 8),
        ("reduced_training_seed_sensitivity", 4),
        ("metadata_confirmation", 3),
        ("perturbation_confirmation", 3),
        ("model_scale_confirmation", 2),
        ("model_scale_xl", 1),
        ("scale_source_confirmation", 2),
        ("frozen_test_evaluation", 1),
        ("external_evaluation", 1),
        ("final_model", 1),
    ],
)
def test_publication_launch_cells(
    name: str, count: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every documented stage expands to isolated, correctly configured launch cells."""
    path = Path("runs/publication") / f"{name}.toml"
    spec = load_experiment_spec(path)
    runs = spec.expanded_runs()
    assert len(runs) == count
    assert len({run.env["CODLLM_OUTPUT_DIR"] for run in runs}) == count
    for run in runs:
        with monkeypatch.context() as context:
            for key, value in run.env.items():
                context.setenv(key, value)
            cfg = config_from_env()
            assert cfg.publication_eval_enabled and cfg.prediction_export_enabled
            assert "historic_strings_en_2024" in cfg.train_excluded_source_ids
            assert cfg.hold_out_evaluate_per is None
            assert cfg.wandb.mode == "online"
            assert cfg.data_seed != 777 or cfg.dataset_sample_seed == 777
            if "scale" in name:
                assert (
                    cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
                    == 192
                )
            if name not in {"final_model", "baselines_final"}:
                assert not cfg.final_test_eval_enabled
    guide = Path("docs/publication/publication_runs.md").read_text()
    assert f"hpc.build --config {path}" in guide
    assert f"hpc.submit --config {path}" in guide
    profile = load_lsf_profile("h100")
    submission = prepare_lsf_submission(
        spec, profile, project_dir=Path.cwd(), output_root=tmp_path
    )
    scripts = list(tmp_path.rglob("*.lsf"))
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
    assert submission.bsub_command()


def test_launch_guide_shell_blocks_parse() -> None:
    """Copyable bash commands contain no invisible separators or malformed quoting."""
    guide = Path("docs/publication/publication_runs.md").read_text()
    assert "\u2028" not in guide and "\u2029" not in guide
    blocks = re.findall(r"```shell\n(.*?)```", guide, re.DOTALL)
    for block in blocks:
        subprocess.run(
            ["bash", "-n"], input=block, text=True, check=True, capture_output=True
        )
        for line in block.splitlines():
            if "--config" in line:
                tokens = shlex.split(line)
                assert Path(tokens[tokens.index("--config") + 1]).is_file()
