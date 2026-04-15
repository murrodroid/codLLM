> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* The project uses `uv` for management of virtual environments. This means:
  * To install packages, use `uv add <package-name>`.
  * To run Python scripts, use `uv run <script-name>.py`.
  * To run other commands related to Python, prefix them with `uv run `, e.g., `uv run <command>`.
  * To run training locally, use `uv run python -m codllm.training`.
* The project uses `pytest` for testing. To run tests, use `uv run pytest tests/`.
* The project uses `ruff` for linting and formatting:
    * To format code, use `uv run ruff format .`.
    * To lint code, use `uv run ruff check . --fix`.
* The project uses `invoke` for task management. To see available tasks, use `uv run invoke --list` or refer to the
    `tasks.py` file.
  * To sync dependencies and fetch the dataset, use `uv run invoke sync`.
* The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use
    `uv run pre-commit run --all-files`. For more information, refer to the `.pre-commit-config.yaml` file.

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
* Ensure new or updated tests are compatible with GitHub Actions (CPU-only Linux runners
  by default) and do not depend on local-only resources or hardware.

# Documentation

* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Ensure new or updated tests are compatible with GitHub Actions (CPU-only Linux runners by default) and do not
  depend on local-only resources or hardware.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.
