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
  * Processed input field prefixes are owned by `Config.input_field_prefixes`; do not hardcode
    `cod: `, `age: `, or `sex: ` when building or parsing processed text.
  * Multi-COD dataset behavior is part of split preparation. Use the existing `multicod_*` config fields for
    label-order shuffling and training-only synthetic single-COD merges, and keep cross-source synthetic merging opt-in
    rather than the default.
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
