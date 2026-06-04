from __future__ import annotations

import itertools
import json
import re
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
SUPPORTED_COMMANDS = {"train", "inference"}


class SpecError(ValueError):
    """Raised when an experiment specification is invalid."""


@dataclass(frozen=True)
class ExperimentRun:
    """A concrete run produced from an experiment specification."""

    name: str
    command: str
    env: dict[str, str]
    force_reprocess: bool = False
    extra_args: tuple[str, ...] = ()
    sweep_index: int | None = None
    sweep_values: dict[str, str] = field(default_factory=dict)

    def env_with_runtime_metadata(self, spec: ExperimentSpec) -> dict[str, str]:
        """Return environment variables including orchestration metadata."""
        env = dict(self.env)
        env["CODLLM_JOB_COMMAND"] = self.command
        env["CODLLM_EXPERIMENT_NAME"] = spec.name
        env["CODLLM_EXPERIMENT_RUN_NAME"] = self.name
        env["CODLLM_EXPERIMENT_CONFIG"] = str(spec.path)
        env["CODLLM_FORCE_REPROCESS"] = "1" if self.force_reprocess else "0"
        if self.extra_args:
            env["TRAIN_EXTRA_ARGS"] = " ".join(self.extra_args)
        if self.sweep_index is not None:
            env["CODLLM_EXPERIMENT_SWEEP_INDEX"] = str(self.sweep_index)
            env["CODLLM_EXPERIMENT_SWEEP_ID"] = _experiment_sweep_id(spec)
            env["CODLLM_EXPERIMENT_SWEEP_VALUES"] = json.dumps(
                self.sweep_values,
                ensure_ascii=True,
                sort_keys=True,
            )
            env.setdefault("WANDB_RUN_GROUP", spec.name)
        env.setdefault("CODLLM_WANDB_RUN_NAME", self.name)
        return env


@dataclass(frozen=True)
class ExperimentVariant:
    """A lockstep experiment dimension with optional base-derived env values."""

    name: str
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    """Typed experiment specification loaded from TOML."""

    path: Path
    name: str
    command: str = "train"
    description: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    sweep: dict[str, tuple[str, ...]] = field(default_factory=dict)
    variants: tuple[ExperimentVariant, ...] = ()
    force_reprocess: bool = False
    extra_args: tuple[str, ...] = ()

    def expanded_runs(self) -> list[ExperimentRun]:
        """Expand the spec into one or more concrete runs."""
        if not self.sweep and not self.variants:
            return [
                ExperimentRun(
                    name=self.name,
                    command=self.command,
                    env=dict(self.env),
                    force_reprocess=self.force_reprocess,
                    extra_args=self.extra_args,
                )
            ]

        sweep_keys = list(self.sweep)
        runs: list[ExperimentRun] = []
        combinations = list(itertools.product(*(self.sweep[key] for key in sweep_keys)))
        if not combinations:
            combinations = [()]
        variants = self.variants or (ExperimentVariant(name="", env={}),)
        for index, (variant, values) in enumerate(
            itertools.product(variants, combinations),
            start=1,
        ):
            sweep_values = dict(zip(sweep_keys, values, strict=True))
            run_env = dict(self.env)
            run_env.update(variant.env)
            run_env.update(sweep_values)
            suffix_parts = []
            metadata_values = dict(sweep_values)
            if variant.name:
                suffix_parts.append(f"variant-{_slugify(variant.name)}")
                metadata_values = {"variant": variant.name, **metadata_values}
            suffix_parts.extend(
                f"{_env_key_slug(key)}-{_slugify(value)}"
                for key, value in sweep_values.items()
            )
            suffix = "__".join(suffix_parts)
            runs.append(
                ExperimentRun(
                    name=f"{self.name}__{suffix}" if suffix else self.name,
                    command=self.command,
                    env=run_env,
                    force_reprocess=self.force_reprocess,
                    extra_args=self.extra_args,
                    sweep_index=index,
                    sweep_values=metadata_values,
                )
            )
        return runs


def stringify_env_value(value: Any) -> str:
    """Convert a TOML value to a shell environment value."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ",".join(stringify_env_value(item) for item in value)
    raise SpecError(f"Unsupported environment value type: {type(value).__name__}.")


def format_env_file(run: ExperimentRun, spec: ExperimentSpec) -> str:
    """Return shell content for one generated run environment file."""
    lines = [
        "# Generated by codllm experiment orchestration.",
        "# Do not edit by hand; update the TOML experiment spec instead.",
    ]
    for key, value in sorted(run.env_with_runtime_metadata(spec).items()):
        if not ENV_NAME_PATTERN.match(key):
            raise SpecError(f"Invalid environment variable name: {key}.")
        lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def list_experiment_specs(root: Path | str = "runs") -> list[Path]:
    """Return sorted TOML experiment specs below a root path."""
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.rglob("*.toml") if path.is_file())


def load_experiment_spec(path: Path | str) -> ExperimentSpec:
    """Load an experiment specification and all inherited base specs."""
    spec_path = Path(path).resolve()
    raw = _load_raw_spec(spec_path, seen=())
    return _build_spec(spec_path, raw)


def _load_raw_spec(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    """Load and merge raw TOML mappings with base inheritance."""
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise SpecError(f"Circular experiment base inheritance detected: {chain}.")
    if not path.exists():
        raise SpecError(f"Experiment spec does not exist: {path}.")

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, dict):
        raise SpecError(f"Experiment spec must be a TOML table: {path}.")

    merged: dict[str, Any] = {}
    for base_path in _base_paths(path, raw.get("base")):
        merged = _merge_raw(merged, _load_raw_spec(base_path, seen=(*seen, path)))
    return _merge_raw(merged, raw)


def _base_paths(path: Path, base_value: object) -> list[Path]:
    """Resolve base spec paths from a raw TOML value."""
    if base_value is None:
        return []
    if isinstance(base_value, str):
        raw_paths = [base_value]
    elif isinstance(base_value, list) and all(
        isinstance(item, str) for item in base_value
    ):
        raw_paths = base_value
    else:
        raise SpecError("'base' must be a string path or a list of string paths.")
    return [
        (path.parent / raw_path).resolve()
        if not Path(raw_path).is_absolute()
        else Path(raw_path)
        for raw_path in raw_paths
    ]


def _merge_raw(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two raw TOML mappings with deep env and sweep tables."""
    result = dict(base)
    for key, value in override.items():
        if key == "base":
            continue
        if key in {"env", "sweep"}:
            merged_table = dict(_optional_mapping(result.get(key), key))
            merged_table.update(_optional_mapping(value, key))
            result[key] = merged_table
            continue
        if key == "variants":
            result[key] = value
            continue
        result[key] = value
    return result


def _build_spec(path: Path, raw: Mapping[str, Any]) -> ExperimentSpec:
    """Convert a merged raw mapping into an ExperimentSpec."""
    name = _optional_string(raw.get("name"), "name") or path.stem
    command = _optional_string(raw.get("command"), "command") or "train"
    if command not in SUPPORTED_COMMANDS:
        allowed = ", ".join(sorted(SUPPORTED_COMMANDS))
        raise SpecError(f"'command' must be one of: {allowed}.")

    description = _optional_string(raw.get("description"), "description")
    force_reprocess = _optional_bool(raw.get("force_reprocess"), "force_reprocess")
    extra_args = _optional_string_tuple(raw.get("extra_args"), "extra_args")
    raw_env = _optional_mapping(raw.get("env"), "env")
    env = _build_env(raw_env)
    sweep = _build_sweep(_optional_mapping(raw.get("sweep"), "sweep"))
    variants = _build_variants(raw.get("variants"), path, raw_env)
    return ExperimentSpec(
        path=path,
        name=name,
        command=command,
        description=description,
        env=env,
        sweep=sweep,
        variants=variants,
        force_reprocess=force_reprocess,
        extra_args=extra_args,
    )


def _build_env(raw_env: Mapping[str, Any]) -> dict[str, str]:
    """Validate and stringify an env table."""
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        if not ENV_NAME_PATTERN.match(key):
            raise SpecError(f"Invalid environment variable name in env table: {key}.")
        env[key] = stringify_env_value(value)
    return env


def _build_sweep(raw_sweep: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Validate and stringify a sweep table."""
    sweep: dict[str, tuple[str, ...]] = {}
    for key, value in raw_sweep.items():
        if not ENV_NAME_PATTERN.match(key):
            raise SpecError(f"Invalid environment variable name in sweep table: {key}.")
        if not isinstance(value, list) or len(value) == 0:
            raise SpecError(f"Sweep variable '{key}' must be a non-empty TOML array.")
        sweep[key] = tuple(stringify_env_value(item) for item in value)
    return sweep


def _build_variants(
    raw_variants: object,
    path: Path,
    common_env: Mapping[str, Any],
) -> tuple[ExperimentVariant, ...]:
    """Validate and resolve lockstep experiment variants."""
    if raw_variants is None:
        return ()
    if not isinstance(raw_variants, list) or not all(
        isinstance(item, Mapping) for item in raw_variants
    ):
        raise SpecError("'variants' must be a TOML array of tables.")

    variants: list[ExperimentVariant] = []
    for index, raw_variant in enumerate(raw_variants, start=1):
        variant_base_env: dict[str, Any] = {}
        base_paths = _base_paths(path, raw_variant.get("base"))
        for base_path in base_paths:
            base_raw = _load_raw_spec(base_path, seen=(path,))
            if _optional_mapping(base_raw.get("sweep"), "sweep"):
                raise SpecError("Variant base specs must not define [sweep].")
            if base_raw.get("variants") is not None:
                raise SpecError("Variant base specs must not define [[variants]].")
            variant_base_env.update(_optional_mapping(base_raw.get("env"), "env"))

        variant_env_raw = dict(variant_base_env)
        variant_env_raw.update(common_env)
        variant_env_raw.update(
            _optional_mapping(raw_variant.get("env"), "variants.env")
        )
        variant_name = _optional_string(raw_variant.get("name"), "variants.name")
        if variant_name is None:
            variant_name = _default_variant_name(base_paths, index)
        variants.append(
            ExperimentVariant(
                name=variant_name,
                env=_build_env(variant_env_raw),
            )
        )
    return tuple(variants)


def _default_variant_name(base_paths: list[Path], index: int) -> str:
    """Return a readable variant name when one is not specified."""
    if len(base_paths) == 1:
        return base_paths[0].stem
    return f"variant-{index}"


def _optional_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Return a mapping value or an empty mapping when omitted."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SpecError(f"'{name}' must be a TOML table.")
    return value


def _optional_string(value: object, name: str) -> str | None:
    """Return a string value or None when omitted."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise SpecError(f"'{name}' must be a string.")
    if value.strip() == "":
        raise SpecError(f"'{name}' must not be empty.")
    return value


def _optional_bool(value: object, name: str) -> bool:
    """Return a boolean value or False when omitted."""
    if value is None:
        return False
    if not isinstance(value, bool):
        raise SpecError(f"'{name}' must be a boolean.")
    return value


def _optional_string_tuple(value: object, name: str) -> tuple[str, ...]:
    """Return a tuple of strings or an empty tuple when omitted."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SpecError(f"'{name}' must be a TOML array of strings.")
    return tuple(value)


def _env_key_slug(value: str) -> str:
    """Convert an environment key to a compact suffix component."""
    lowered = value.lower()
    for prefix in ("codllm_", "inference_"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    return _slugify(lowered)


def _experiment_sweep_id(spec: ExperimentSpec) -> str:
    """Return the stable sweep id used for generated TOML sweep runs."""
    return f"codllm-{_slugify(spec.name)}"


def _slugify(value: str) -> str:
    """Convert arbitrary text to a stable lowercase slug."""
    slug = SLUG_PATTERN.sub("-", value.lower()).strip("-")
    return slug or "value"
