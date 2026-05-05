> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* The project uses `uv` for management of virtual environments. This means:
  * To install packages, use `uv add <package-name>`.
  * To run Python scripts, use `uv run <script-name>.py`.
  * To run other commands related to Python, prefix them with `uv run `, e.g., `uv run <command>`.
  * To run training locally, use `uv run python -m codllm.training`.
  * To run inference locally, use `uv run python -m codllm.inference`.
* The project uses `pytest` for testing. To run tests, use `uv run pytest tests/`.
* The project uses `ruff` for linting and formatting:
    * To format code, use `uv run ruff format .`.
    * To lint code, use `uv run ruff check . --fix`.
* The project uses `invoke` for task management. To see available tasks, use `uv run invoke --list` or refer to the
    `tasks.py` file.
  * To sync dependencies, use `uv run invoke sync`.
  * To list experiment specs, use `uv run invoke experiments.list`.
  * To inspect expanded experiment runs, use `uv run invoke experiments.plan --config <path>`.
  * To list configured LSF profiles, use `uv run invoke hpc.profiles`.
  * To inspect uv/cache paths before HPC work, source `hpc/env.sh` and run `bash hpc/storage-check.sh`.
    If invoke is already installed, `uv run --no-sync invoke hpc.storage` provides the same check.
  * To prebuild processed-data and prepared-split caches for an experiment spec on HPC, use
    `uv run invoke hpc.build --config <path> --profile <profile>`. Use the same profile intended for `hpc.submit`.
    By default this builds all expanded runs; pass `--sweep-index <n>` to build one run.
  * To generate and submit an LSF job, use
    `uv run invoke hpc.submit --config <path> --profile <profile> --user <lucas|elias>`.
  * To generate an LSF job without submitting it, add `--dry-run`.
* The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use
    `uv run pre-commit run --all-files`. For more information, refer to the `.pre-commit-config.yaml` file.

# Application overview

codLLM trains and evaluates transformer models for mapping historical free-text causes of death to ICD10h labels.
The production runtime is the Python package under `src/codllm`, with explicit entrypoints for training
(`python -m codllm.training`) and inference (`python -m codllm.inference`). Runtime behavior is configured through the
`Config` dataclass in `src/codllm/settings/schema.py` and environment overrides parsed by
`src/codllm/settings/env.py`. The core pipeline loads raw historical datasets, harmonizes input fields, preprocesses and
balances data, optionally pretrains on the ICD10h masterlist, fine-tunes Hugging Face models, records metadata, and
writes run-scoped outputs.

Experiment orchestration is handled separately from model code. Human-editable experiment specs live under
`runs/**/*.toml`, LSF resource profiles live in `hpc/lsf_profiles.toml`, and `tasks.py` exposes the
supported workflow through `uv run invoke ...`. Generated LSF scripts and per-run env files are written under
`jobs/generated/` and are intentionally ignored by git. Prefer adding or editing TOML specs and LSF profiles over adding
new handwritten shell scripts in `jobs/`.
Processed raw-data caches live under `Config.data_processed_dir`; prepared split caches live beside the processed file
under `<processed-stem>.splits/<cache-key>/` and include split-time transformations such as multi-COD synthesis,
balancing, hold-out sampling, and masterlist injection. Keep cache-key metadata in sync with any option that changes
prepared split content.
Generated TOML sweep runs export `WANDB_SWEEP_ID=codllm-<experiment-name-slug>` and
`WANDB_RUN_GROUP=<experiment-name>` unless those values are explicitly set in `[env]`.
Training runs log compact W&B data visualizations under `data/*`, and evaluation error tables under
`<scope>/errors/*`, where scopes include `val`, `test`, `holdout/val`, `holdout/test`, and pretraining scopes.
Error tables aggregate ICD10h labels to the chapter-block prefix, i.e. the first three characters of each code.
Prepared split cache metadata includes training balance diagnostics used by these visualizations; keep those diagnostics
cache-safe and summary-only rather than adding visualization-only columns to training dataframes.
H100 profiles request 17 CPU cores so H100 runtime specs can use 16 DataLoader workers plus the main process.

On HPC systems, source `hpc/env.sh` before any `uv` command. This puts `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`,
`UV_PYTHON_INSTALL_DIR`, Hugging Face caches, torch caches, and W&B caches under the configured storage unit instead of
personal user space. After syncing once, prefer `uv run --no-sync invoke ...` for plan/submit commands to avoid
unexpected dependency downloads.

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints
* Do not add inline comments unless absolutely necessary.
* Keep implementations config-first and consistency-first:
  * User-editable defaults must be defined in `Config` (inside the class), not as
    module-level constants.
  * Reuse existing config fields and shared helpers (for example `_default_device`,
    `resolved_max_target_length`, and existing path/metadata resolvers) instead of
    duplicating logic.
  * Do not hardcode defaults in multiple places when the same value already exists in
    `Config`; wire code to `Config` so behavior updates dynamically when config changes.
  * When adding runtime options, add them to `Config` and `config_from_env`, and ensure
    all relevant call sites and tests use the config-driven value.
  * Label harmonization is standard processed-data behavior controlled by `Config.label_harmonization_enabled`,
    `Config.label_harmonization_masterlist_path`, `Config.label_harmonization_masterlist_sheet_name`,
    `Config.label_harmonization_transfer_sheet_name`, and the matching `CODLLM_LABEL_HARMONIZATION_*` env vars. Do not
    use `pretrain_*` settings for processed-data harmonization or classifier label vocabulary.
  * Pretraining warmup is controlled independently by `Config.pretrain_warmup_ratio` and
    `CODLLM_PRETRAIN_WARMUP_RATIO`; do not reuse fine-tuning `warmup_ratio` for pretraining.
  * Synthetic multi-COD rows for masterlist pretraining are controlled independently by
    `Config.pretrain_multicod_synthetic_ratio`, `Config.pretrain_multicod_synthetic_text_separator`,
    `CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_RATIO`, and
    `CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_TEXT_SEPARATOR`; do not reuse fine-tuning
    `multicod_synthetic_ratio` for pretraining.
  * Processed input field prefixes are owned by `Config.input_field_prefixes`; do not hardcode
    `cod: `, `age: `, or `sex: ` when building or parsing processed text.
  * Floor-upsample copy perturbation count is controlled by `Config.balance_perturbation_mean`,
    `Config.balance_perturbation_variance`, `CODLLM_BALANCE_PERTURBATION_MEAN`, and
    `CODLLM_BALANCE_PERTURBATION_VARIANCE`. Whole-training-set base perturbation is controlled separately by
    `Config.base_perturbation_rate`, `Config.base_perturbations`, `Config.base_perturbation_mean`,
    `Config.base_perturbation_variance`, `CODLLM_BASE_PERTURBATION_RATE`, `CODLLM_BASE_PERTURBATIONS`,
    `CODLLM_BASE_PERTURBATION_MEAN`, and `CODLLM_BASE_PERTURBATION_VARIANCE`. These values scale by the length of the
    processed `cod` text segment; do not reintroduce a fixed perturbations-per-sample control for training rows.
  * Multi-COD dataset behavior is part of split preparation. Use the existing `multicod_*` config fields for
    label-order shuffling and training-only synthetic single-COD merges, and keep cross-source synthetic merging opt-in
    rather than the default.
  * Dataset leave-one-source-out evaluation is controlled by `Config.hold_out_dataset` and
    `CODLLM_HOLD_OUT_DATASET`. Hold-out matching uses processed `source_id` values, removes the entire matching source
    from train/val/test splitting, keeps normal val/test splits on the remaining sources, and evaluates the held-out
    rows after training with `holdout_*` metrics. During-training sampled hold-out evaluation is controlled separately
    by `Config.hold_out_evaluate_per`, `Config.hold_out_evaluate_ratio`, `CODLLM_HOLD_OUT_EVALUATE_PER`, and
    `CODLLM_HOLD_OUT_EVALUATE_RATIO`; the final post-training hold-out evaluation must always use the full held-out
    source.
  * `CODLLM_SAVE_STRATEGY_BEST_METRIC` supports single-label metrics plus multi-COD metrics such as `exact_match`,
    `sample_f1`, `sample_jaccard`, `micro_jaccard`, `hamming_loss`, and `hamming_score`.
  * W&B run-page config volume is controlled by `Config.wandb.run_config_mode` and
    `CODLLM_WANDB_RUN_CONFIG_MODE=minimal|standard|full`; scalar metric volume is controlled by
    `Config.wandb.metric_mode` and `CODLLM_WANDB_METRIC_MODE=core|standard|all`. Keep full reproducibility payloads in
    W&B artifacts instead of flattening every metadata field onto the run page by default.
* Ensure new or updated tests are compatible with GitHub Actions (CPU-only Linux runners
  by default) and do not depend on local-only resources or hardware.

# Documentation

* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Ensure new or updated tests are compatible with GitHub Actions (CPU-only Linux runners by default) and do not
  depend on local-only resources or hardware.
* Update this `AGENTS.md` file whenever a change affects how future agents should work in this repo. This includes:
  * adding, removing, or changing project commands, dependencies, task names, test commands, lint commands, or HPC
    submission workflows;
  * changing configuration ownership rules, environment variable behavior, experiment spec formats, generated file
    locations, or required local/HPC setup;
  * introducing new conventions for code style, documentation, tests, data handling, generated artifacts, or secrets;
  * adding a tool or workflow that future agents should prefer over an older path.
* Do not update `AGENTS.md` for ordinary feature code, bug fixes, or tests when the existing commands and conventions
  remain accurate.
