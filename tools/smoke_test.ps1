#requires -Version 5.1
<#
Smoke test for branch 63: exercises the new code paths end-to-end on a tiny
slice of the dataset without W&B, then verifies the expected artifacts exist
and have the expected structure.

Run from the repo root:

    .\tools\smoke_test.ps1

A second invocation should auto-resume into the same run-* dir.
#>

$ErrorActionPreference = "Stop"

# Minimal training config: 1 epoch, 1% of data, flan-t5-small, no W&B.
$env:CODLLM_NUM_TRAIN_EPOCHS = "1"
$env:CODLLM_DATASET_SIZE = "0.01"
$env:CODLLM_HF_MODEL = "google/flan-t5-small"
$env:CODLLM_PER_DEVICE_TRAIN_BATCH_SIZE = "8"
$env:CODLLM_PER_DEVICE_EVAL_BATCH_SIZE = "8"
$env:CODLLM_GRADIENT_ACCUMULATION_STEPS = "1"
$env:CODLLM_MAX_LABEL_COUNT = "3"

# Skip the heavy pretraining + balance machinery for the smoke run.
$env:CODLLM_PRETRAIN_ENABLED = "0"
$env:CODLLM_BALANCE_STRATEGY = "none"
$env:CODLLM_BASE_PERTURBATION_RATE = "0"

# Cheap eval/save cadence so the run finishes in a few minutes.
$env:CODLLM_EVAL_STRATEGY = "epoch"
$env:CODLLM_SAVE_STRATEGY = "epoch"
$env:CODLLM_SAVE_STRATEGY_BEST_METRIC = "sample_f1"

# Exercise the new branches.
$env:CODLLM_AUTO_RESUME = "1"
$env:CODLLM_PER_SIZE_OUTPUT_DIR = "1"
$env:CODLLM_UNCERTAINTY_EVAL_ENABLED = "1"
$env:CODLLM_EARLY_STOPPING_PATIENCE = "0"

# Disable W&B for the smoke - we just want to confirm local artifacts.
$env:WANDB_MODE = "disabled"

# Point the run dir at a smoke-only base so we don't pollute runs/.
$env:CODLLM_OUTPUT_DIR = "runs_smoke"

Write-Output "=== Smoke test: launching minimal training ==="
python -m codllm.training
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Output ""
Write-Output "=== Verifying smoke-test artifacts ==="
python tools\verify_smoke.py
exit $LASTEXITCODE
