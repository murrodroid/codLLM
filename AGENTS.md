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
  * To inspect processed-data and prepared-split caches, use
    `uv run invoke maintenance.data-cache`. To clear those caches, use
    `uv run invoke maintenance.clear-data-cache --yes`; without `--yes` it only prints a dry run. Use `--locks` only
    when no jobs are building or reading dataset caches.
  * To inspect broad storage usage, use `uv run invoke maintenance.status`. This reports shared filesystem capacity for
    the workspace, profile/home, and configured HPC storage roots, DTU quota lines when `getquota_zhome.sh`,
    `getquota_work1.sh`, or `getquota_work3.sh` are available, plus storage usage by code, raw data, processed data,
    model outputs, generated jobs/logs, runtime caches, managed HPC runtime roots, uncategorized run-storage usage, the
    largest direct children of the run-storage root, large siblings under the configured HPC storage folder, and quota
    gaps when DTU reports more quota usage than is visible below the configured storage folder. Treat `filesystem_*`
    values as shared capacity only; capacity failures on DTU HPC usually correspond to the separate `quota:` line.
  * Maintenance data-cache, clear-data-cache, clear-cache, status, and hpc-env tasks should normally inspect the
    current environment after `hpc/env.sh` has been sourced on HPC. Use user shortcuts such as `--lucas` or
    `--user lucas` only when the HPC storage environment is not already set; Lucas's shortcut resolves to
    `/work3/s234805` and `/work3/s234805/codllm`.
  * To clear broad generated caches and outputs, use `uv run invoke maintenance.clear-cache --standard` for generated
    paths unused for 14+ days, or `uv run invoke maintenance.clear-cache --aggressive` for all maintenance-managed
    generated paths. Both modes dry-run unless `--yes` is passed. These policies still protect raw data, source code,
    tracked experiment specs, and generic profile caches. Use `--locks` only when no jobs are building or reading
    dataset caches; with `--aggressive --locks --yes`, prepared-split roots and their lock files are removed before the
    next build regenerates them. Aggressive cleanup may also remove managed HPC runtime roots under `$RUN_STORAGE_DIR`,
    including `cache/`, `.venv/`, and `python/`, because uv and the job bootstrap can regenerate them.
  * To check generated-output git hygiene before pushing, use `uv run invoke maintenance.git-hygiene`. To write a
    local git status/recent-commit snapshot, use `uv run invoke maintenance.git-snapshot`; snapshots are written under
    ignored `logs/git/`.
  * To inspect HPC cache/storage environment hygiene through the maintenance package, use
    `uv run --no-sync invoke maintenance.hpc-env`.
  * To prebuild processed-data and prepared-split caches for an experiment spec on HPC, use
    `uv run invoke hpc.build --config <path> --profile <profile> --user <lucas|elias>`. Use the same profile and user
    alias intended for `hpc.submit`. By default this builds all expanded runs; pass `--sweep-index <n>` to build one
    run.
  * To generate and submit an LSF job, use
    `uv run invoke hpc.submit --config <path> --profile <profile> --user <lucas|elias>`.
    Both `hpc.build` and `hpc.submit` also accept shortcut flags such as `--lucas`.
  * To generate an LSF job without submitting it, add `--dry-run`.
  * Publication response curves and Bayesian-search setup live under `runs/publication/`; read its `README.md` before
    submitting. Create a native W&B sweep with `uv run --no-sync invoke experiments.bayes-create`, then submit
    single-trial agent waves with `uv run --no-sync invoke hpc.bayes-submit --sweep-id <entity/project/sweep-id>`.
    Bayesian agent jobs deliberately run one trial per allocation and do not use duration-based auto-resume.
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
`jobs/generated/` and logs under `logs/` are intentionally ignored by git. Prefer adding or editing TOML specs and LSF
profiles over adding new handwritten shell scripts in `jobs/`. Do not ignore the whole `runs/` tree; only generated
run output directories such as `runs/run-*/` and `runs/*/run-*/` should be ignored so new TOML specs remain addable.
In TOML specs, relative `CODLLM_OUTPUT_DIR` values are resolved below the selected profile/user's `$RUN_STORAGE_DIR`;
prefer this for isolated resumable run roots instead of hardcoded DTU account paths.
Processed raw-data caches live under `Config.data_processed_dir`; prepared split caches live beside the processed file
under `<processed-stem>.splits/<cache-key>/` and include split-time transformations such as multi-COD synthesis,
balancing, hold-out sampling, configured train-source exclusions, and masterlist injection. Keep cache-key metadata in
sync with any option that changes prepared split content.
For reduced-data studies, `Config.dataset_sample_seed` and `CODLLM_DATASET_SAMPLE_SEED` control cohort subsampling
independently of `Config.data_seed` and `CODLLM_DATA_SEED`. When unset, the cohort-sampling seed inherits the resolved
data seed for backward compatibility. Pin the sample seed while varying the data seed to compare split/preparation
variability on one fixed sampled cohort.
Repository maintenance helpers live under `src/codllm/maintenance/` and are exposed via `invoke maintenance.*` tasks.
Keep dataset cache cleanup config-driven and dry-run by default; do not delete raw data as part of maintenance cache
clearing. Broad cache cleanup may delete processed-data caches, prepared split caches, local run/checkpoint directories,
generated LSF submissions, logs, Python/tool caches, managed HPC runtime roots under `$RUN_STORAGE_DIR`, and explicit
runtime caches such as Hugging Face, torch, W&B, and uv cache paths. It must not delete source code, checked-in TOML
specs, raw datasets, arbitrary files under `models/`, or generic profile cache roots that are not clearly owned by
codLLM.
Maintenance tasks that inspect or clear generated storage should use the currently sourced HPC storage environment, or
`--lucas`/`--user lucas` when inspecting Lucas's `/work3/s234805` storage from a shell where the env is not set. On DTU
HPC, status output labels shared filesystem capacity as `filesystem_*` and reports user quota separately when the DTU
quota scripts are installed; do not interpret shared filesystem totals as available user quota.
Generated TOML sweep runs export `CODLLM_EXPERIMENT_SWEEP_ID=codllm-<experiment-name-slug>` and
`WANDB_RUN_GROUP=<experiment-name>`. Do not auto-generate `WANDB_SWEEP_ID`; W&B treats it as a native sweep id and fails
unless that sweep exists. Only set `WANDB_SWEEP_ID` explicitly in `[env]` when attaching to a real W&B sweep.
New runs default to the W&B project `codllmdev/codllm` through `WandbConfig`; use the
`CODLLM_WANDB_ENTITY` and `CODLLM_WANDB_PROJECT` environment variables only for deliberate per-run overrides.
Resumable LSF training persists `wandb_run_id.txt` in the stable `CODLLM_RUN_STATE_DIR`, not a scheduler-slot-specific
checkpoint path; retain the checkpoint-local read fallback for older runs. Publication specs should set
`CODLLM_WANDB_MODE=online` explicitly. When `.resume_needed` is present, the Transformers W&B train-end hook must skip
the intermediate final-model artifact so checkpoint save, telemetry flush, and resubmission fit inside the wall-time
safety margin.
Native W&B samples do not automatically become codLLM settings. Use the allowlisted wrapper in
`src/codllm/experiments/bayesian.py`, keep agent trials single-slot, and isolate every trial's output root.
Use `[sweep]` for Cartesian environment-variable dimensions. Use `[[variants]]` for lockstep dimensions such as
model/profile pairs; each variant may set `base = "../profiles/..."` and optional `[variants.env]`, and variants cross
with `[sweep]` without crossing with one another.
Training runs log compact W&B data visualizations under `data/*`, and evaluation error tables under
`<scope>/errors/*`, where scopes include `val`, `test`, final full `holdout/full` plus legacy `holdout`, sampled
`holdout/sample` plus legacy `holdout/val`, and pretraining scopes. Keep final leave-one-source-out metrics visible
under both `holdout/full/*` and compatibility `holdout/*` so W&B runs visibly contain both `val/*` and hold-out metrics.
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
  * Label standardization is a tracked curation overlay, not a raw-data edit or a processed-cache patch. Use
    `data/curation/label_standardization.toml` for reviewed dataset-level rules and
    `data/curation/label_standardization_overrides.csv` for exact row-level exceptions. Runtime behavior is controlled
    by `Config.label_standardization_enabled`, `Config.label_standardization_rules_path`,
    `Config.label_standardization_overrides_path`, and the matching `CODLLM_LABEL_STANDARDIZATION_*` env vars. Keep
    these overlay files in processed-data cache metadata so curation edits trigger rebuilds.
  * Pretraining warmup is controlled independently by `Config.pretrain_warmup_ratio` and
    `CODLLM_PRETRAIN_WARMUP_RATIO`; do not reuse fine-tuning `warmup_ratio` for pretraining.
  * Pretraining early stopping and best-checkpoint reloading are independently controlled by
    `Config.pretrain_early_stopping_patience`, `Config.pretrain_load_best_model_at_end`,
    `CODLLM_PRETRAIN_EARLY_STOPPING_PATIENCE`, and `CODLLM_PRETRAIN_LOAD_BEST_MODEL_AT_END`. When unset, these inherit
    their fine-tuning counterparts.
  * Final test and test-based uncertainty evaluation can be suppressed during tuning with
    `Config.final_test_eval_enabled` and `CODLLM_FINAL_TEST_EVAL_ENABLED`; this must not change split construction.
  * Synthetic multi-COD rows for masterlist pretraining are controlled independently by
    `Config.pretrain_multicod_synthetic_ratio`, `Config.pretrain_multicod_synthetic_text_separators`,
    `CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_RATIO`, and `CODLLM_PRETRAIN_MULTICOD_SYNTHETIC_TEXT_SEPARATORS`; do not reuse
    fine-tuning `multicod_*` settings for pretraining. Separator env vars are JSON string arrays and preserve
    whitespace.
  * Processed input field prefixes are owned by `Config.input_field_prefixes`; do not hardcode
    `cod: `, `age: `, or `sex: ` when building or parsing processed text.
  * The default training input is COD text only. Set `Config.training_input` or `CODLLM_TRAINING_INPUT` explicitly for
    runs that should include age, sex, or other supported fields.
  * Floor-upsample copy perturbation count is controlled by `Config.balance_perturbation_mean`,
    `Config.balance_perturbation_variance`, `Config.balance_perturbation_loft`,
    `CODLLM_BALANCE_PERTURBATION_MEAN`, `CODLLM_BALANCE_PERTURBATION_VARIANCE`, and
    `CODLLM_BALANCE_PERTURBATION_LOFT`. Whole-training-set base perturbation is controlled separately by
    `Config.base_perturbation_rate`, `Config.base_perturbations`, `Config.base_perturbation_mean`,
    `Config.base_perturbation_variance`, `Config.base_perturbation_loft`, `CODLLM_BASE_PERTURBATION_RATE`,
    `CODLLM_BASE_PERTURBATIONS`, `CODLLM_BASE_PERTURBATION_MEAN`, `CODLLM_BASE_PERTURBATION_VARIANCE`, and
    `CODLLM_BASE_PERTURBATION_LOFT`. These values scale by the length of the processed `cod` text segment; loft caps
    the stochastic variance tail at mean plus loft standard deviations. Do not reintroduce a fixed
    perturbations-per-sample control for training rows.
  * Multi-COD dataset behavior is part of split preparation. Use the existing `multicod_*` config fields for
    label-order shuffling and training-only synthetic single-COD merges, and keep cross-source synthetic merging opt-in
    rather than the default. Use `Config.multicod_synthetic_text_separators` and
    `CODLLM_MULTICOD_SYNTHETIC_TEXT_SEPARATORS` for stochastic synthetic COD text separators; the env var is a JSON
    string array and preserves whitespace.
  * Dataset leave-one-source-out evaluation is controlled by `Config.hold_out_dataset` and
    `CODLLM_HOLD_OUT_DATASET`. Hold-out matching uses processed `source_id` values, removes the entire matching source
    from train/val/test splitting, keeps normal val/test splits on the remaining sources, and evaluates the held-out
    rows after training with `holdout_full_*` metrics that are logged as `holdout/full/*` plus legacy `holdout/*`
    aliases. `Config.train_excluded_source_ids` and `CODLLM_TRAIN_EXCLUDED_SOURCE_IDS` remove non-training reference
    sources from the remaining pool before sampling/splitting; hold-out sweeps should exclude
    `historic_strings_en_2024` unless deliberately measuring with that reference source in training. During-training
    sampled hold-out evaluation is controlled separately by `Config.hold_out_evaluate_per`,
    `Config.hold_out_evaluate_ratio`, `CODLLM_HOLD_OUT_EVALUATE_PER`, and `CODLLM_HOLD_OUT_EVALUATE_RATIO`, and logs
    as `holdout/sample/*` plus legacy `holdout/val/*`; the final post-training hold-out evaluation must always use the
    full held-out source.
  * `CODLLM_SAVE_STRATEGY_BEST_METRIC` supports single-label metrics plus multi-COD metrics such as `exact_match`,
    `sample_f1`, `sample_jaccard`, `micro_jaccard`, `hamming_loss`, and `hamming_score`; source-transfer metrics such
    as `source_transfer_label_accuracy`, `source_transfer_label_macro_f1`, and `source_transfer_label_recall` are also
    valid save metrics.
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
