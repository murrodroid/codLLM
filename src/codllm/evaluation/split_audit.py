"""Score-blind audits of the exact original partitions used by publication screening."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from codllm.config import Config
from codllm.data import DataHandler
from codllm.evaluation.artifacts import write_json
from codllm.evaluation.provenance import (
    annotate_provenance,
    content_digest,
    provenance_signature,
)
from codllm.evaluation.reference import EvaluationReference, build_reference, cod_hash
from codllm.evaluation.scoring import crosslingual_codes
from codllm.evaluation.splitting import linked_groups, validate_group_integrity
from codllm.runtime.paths import resolve_source_path


def _codes(value: str, cfg: Config) -> tuple[str, ...]:
    """Canonicalize a complete target set independently of label-order shuffling."""
    return tuple(
        sorted(
            {
                part.strip()
                for part in str(value).split(cfg.label_separator)
                if part.strip()
            }
        )
    )


def _configuration(cfg: Config) -> dict[str, Any]:
    """Return reproducibility settings without credentials or non-JSON device objects."""
    values = asdict(cfg)
    values.pop("hf_token", None)
    values["device"] = str(cfg.device)
    values["device_map"] = str(cfg.device_map)
    return values


def validate_shared_audit_configs(configs: list[Config]) -> None:
    """Reject cells whose original partitions or language-resource support could differ."""
    if not configs:
        raise ValueError("At least one experiment cell is required.")
    signatures = []
    for cfg in configs:
        if not cfg.publication_eval_enabled:
            raise ValueError("Split audits require publication_eval_enabled=true.")
        if cfg.publication_audit_top_groups < 1:
            raise ValueError("publication_audit_top_groups must be positive.")
        if (
            cfg.balance_strategy not in {"none", "floor"}
            or cfg.masterlist_inject_enabled
        ):
            raise ValueError(
                "This original-partition audit supports none/floor balance without masterlist injection."
            )
        if (
            cfg.multicod_synthetic_ratio > 0
            and cfg.multicod_synthetic_source_scope != "within_source"
        ):
            raise ValueError(
                "Cross-source synthesis needs an actual augmented-exposure audit; not assumed equivalent."
            )
        if cfg.pretrain_enabled and cfg.pretrain_num_train_epochs <= 0:
            raise ValueError(
                "Enabled pretraining must have a positive dose for planned exposure accounting."
            )
        values = _configuration(cfg)
        for field in (
            "output_dir",
            "balance_floor",
            "multicod_synthetic_ratio",
            "pretrain_num_train_epochs",
        ):
            values.pop(field)
        values["wandb"].pop("run_name", None)
        signatures.append(content_digest(values))
    if len(set(signatures)) != 1:
        raise ValueError(
            "Selected cells differ beyond floor, synthesis ratio, or positive pretraining dose; audit separately."
        )


def file_identity(path: Path) -> dict[str, Any]:
    """Stream a content hash and reject files modified while they are being fingerprinted."""
    if not path.is_file():
        return {"path": str(path.resolve()), "exists": False}
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during audit hashing: {path}")
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def partition_fingerprint(frame: pd.DataFrame, cfg: Config) -> str:
    """Hash ordered original identities, inputs, provenance, and unordered targets without exporting them."""
    columns = [
        "row_uid",
        "source_id",
        "record_id",
        "cod_key",
        "language",
        "language_candidates",
        cfg.dataset_text_column,
        cfg.dataset_label_column,
    ]
    digest = hashlib.sha256()
    labels = {
        value: _codes(value, cfg) for value in frame[cfg.dataset_label_column].unique()
    }
    for row in frame[columns].fillna("").astype(str).itertuples(index=False, name=None):
        digest.update(
            json.dumps([*row[:-1], labels[row[-1]]], ensure_ascii=False).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _distribution(sizes: np.ndarray, top: int) -> dict[str, Any]:
    """Describe concentration without interpreting repeated cases as duplicate people."""
    sizes = np.sort(sizes)[::-1]
    total = int(sizes.sum())
    return {
        "groups": len(sizes),
        "rows": total,
        "largest_rows": int(sizes[0]) if len(sizes) else 0,
        "largest_row_share": float(sizes[0] / total) if total else None,
        "top_groups_row_share": float(sizes[:top].sum() / total) if total else None,
        "group_size_quantiles": dict(
            zip(
                ("p50", "p90", "p95", "p99"),
                np.quantile(sizes, [0.5, 0.9, 0.95, 0.99]).tolist(),
                strict=True,
            )
        )
        if len(sizes)
        else {},
    }


def _inventory(frame: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Summarize natural rows, label support, and connected-group concentration."""
    labels = frame[cfg.dataset_label_column].map(lambda value: _codes(value, cfg))
    long = (
        frame[["cod_key", "audit_group"]]
        .assign(code=labels)
        .explode("code")
        .dropna(subset=["code"])
    )
    supports = long.groupby("code", sort=True).agg(
        rows=("code", "size"),
        distinct_cods=("cod_key", "nunique"),
        connected_groups=("audit_group", "nunique"),
    )
    group_counts = frame["audit_group"].value_counts()
    top_groups = []
    for identifier in group_counts.head(cfg.publication_audit_top_groups).index:
        group = frame.loc[frame["audit_group"].eq(identifier)]
        top_groups.append(
            {
                "rows": len(group),
                "distinct_cods": group["cod_key"].nunique(),
                "source_rows": group["source_id"].value_counts().to_dict(),
            }
        )
    return {
        "rows": len(frame),
        "unique_row_ids": frame["row_uid"].nunique(),
        "distinct_cods": frame["cod_key"].nunique(),
        "natural_single_cod_rows": int(labels.map(len).eq(1).sum()),
        "natural_multicod_rows": int(labels.map(len).gt(1).sum()),
        "target_occurrences": int(labels.map(len).sum()),
        "distinct_codes": len(supports),
        "codes_with_fewer_than_10_rows": int(supports["rows"].lt(10).sum()),
        "codes_in_one_connected_group": int(supports["connected_groups"].eq(1).sum()),
        "normalized_cods_with_multiple_target_sets": int(
            frame.assign(targets=labels)
            .groupby("cod_key")["targets"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "languages": frame["language"].value_counts().to_dict(),
        "unknown_or_mixed_language_rows": int(
            frame["language"].isin(["und", "mul", ""]).sum()
        ),
        "connected_groups": _distribution(
            group_counts.to_numpy(), cfg.publication_audit_top_groups
        ),
        "largest_groups": top_groups,
        "per_code": supports.to_dict("index"),
    }


class _Support:
    """Accumulate eligible-target support without predictions or accuracy calculations."""

    def __init__(self) -> None:
        """Initialize distinct-support accumulators."""
        self.rows = self.single = self.multi = 0
        self.codes: Counter[str] = Counter()
        self.cods: set[str] = set()
        self.groups: set[int] = set()
        self.pairs: Counter[tuple[str, str]] = Counter()

    def add(
        self, eligible: set[str], target_size: int, cod: str, group: int, language: str
    ) -> None:
        """Count full qualifying rows and eligible code occurrences separately."""
        if not eligible:
            return
        self.rows += 1
        self.single += int(target_size == 1)
        self.multi += int(target_size > 1)
        self.codes.update(eligible)
        self.cods.add(cod)
        self.groups.add(group)
        self.pairs.update((code, language) for code in eligible)

    def payload(self) -> dict[str, Any]:
        """Return only aggregate support; an empty bucket remains explicitly unavailable."""
        return {
            "available": bool(self.rows),
            "rows": self.rows,
            "single_cod_rows": self.single,
            "multi_cod_rows": self.multi,
            "target_occurrences": sum(self.codes.values()),
            "distinct_codes": len(self.codes),
            "distinct_cods": len(self.cods),
            "connected_groups": len(self.groups),
            "code_language_pairs": len(self.pairs),
            "per_code_target_occurrences": dict(sorted(self.codes.items())),
            "per_code_language_target_occurrences": [
                {"code": code, "language": language, "targets": count}
                for (code, language), count in sorted(self.pairs.items())
            ],
        }


def _eligibility(
    frame: pd.DataFrame, reference: EvaluationReference, cfg: Config
) -> dict[str, Any]:
    """Report original-training-only eligibility overall and by evaluation source/language."""
    names = (
        "source_transfer",
        "crosslingual",
        "strict_crosslingual",
        "historically_unseen_label",
        "masterlist_only_label",
        "absent_from_all_adaptation",
        "seen_cod",
        "unseen_cod",
        "masterlist_seen_cod",
        "novel_known_code_combination",
    )
    scopes = [
        "overall",
        *(f"source/{v}" for v in sorted(frame["source_id"].unique())),
        *(f"language/{v}" for v in sorted(frame["language"].unique())),
    ]
    buckets = {scope: {name: _Support() for name in names} for scope in scopes}
    parse = {
        value: set(_codes(value, cfg))
        for value in frame[cfg.dataset_label_column].unique()
    }
    hashes = {value: cod_hash(value) for value in frame["cod_key"].unique()}
    classification: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    for source, language, cod, group, label in frame[
        ["source_id", "language", "cod_key", "audit_group", cfg.dataset_label_column]
    ].itertuples(index=False, name=None):
        target = parse[label]
        key = (source, language, label)
        if key not in classification:
            unseen = target - reference.label_sources.keys()
            classification[key] = {
                "source_transfer": {
                    c
                    for c in target
                    if reference.label_sources.get(c)
                    and source not in reference.label_sources[c]
                },
                "crosslingual": crosslingual_codes(target, language, reference),
                "strict_crosslingual": crosslingual_codes(
                    target, language, reference, strict_adaptation=True
                ),
                "historically_unseen_label": unseen,
                "masterlist_only_label": unseen & reference.masterlist_labels,
                "absent_from_all_adaptation": target
                - reference.adaptation_languages.keys(),
                "novel_known_code_combination": target
                if len(target) > 1
                and not unseen
                and ",".join(sorted(target)) not in reference.combinations
                else set(),
            }
        eligible = classification[key] | {
            "seen_cod": target if hashes[cod] in reference.historical_cods else set(),
            "unseen_cod": target
            if hashes[cod] not in reference.historical_cods
            else set(),
            "masterlist_seen_cod": target
            if hashes[cod] in reference.masterlist_cods
            else set(),
        }
        for scope in ("overall", f"source/{source}", f"language/{language}"):
            for name, codes in eligible.items():
                buckets[scope][name].add(codes, len(target), cod, group, language)
    return {
        scope: {name: value.payload() for name, value in values.items()}
        for scope, values in buckets.items()
    }


def _overlap_checks(
    parts: dict[str, pd.DataFrame], cfg: Config
) -> list[dict[str, Any]]:
    """Measure forbidden identity crossings separately from allowed held-out text overlap."""
    checks = []
    identities = {}
    for name, frame in parts.items():
        identifiers = set(frame["row_uid"])
        records = {
            (str(s), str(r))
            for s, r in frame[["source_id", "record_id"]].itertuples(
                index=False, name=None
            )
            if pd.notna(r) and str(r)
        }
        identities[name] = (identifiers, records, set(frame["cod_key"]) - {""})
        checks.append(
            {
                "check": f"{name}_unique_row_ids",
                "passed": len(identifiers) == len(frame),
            }
        )
        checks.append(
            {
                "check": f"{name}_no_excluded_source",
                "passed": not frame["source_id"]
                .isin(cfg.train_excluded_source_ids)
                .any(),
            }
        )
    for first, second in combinations(parts, 2):
        counts = [
            len(a & b)
            for a, b in zip(identities[first], identities[second], strict=True)
        ]
        cod_required = cfg.evaluation_protocol == "cod" and "holdout" not in {
            first,
            second,
        }
        checks.append(
            {
                "check": f"{first}__{second}",
                "shared_row_ids": counts[0],
                "shared_source_records": counts[1],
                "shared_cods": counts[2],
                "cod_disjoint_required": cod_required,
                "passed": counts[0] == counts[1] == 0
                and (not cod_required or counts[2] == 0),
            }
        )
    return checks


def _resource_identities(handler: DataHandler, cfg: Config) -> dict[str, Any]:
    """Fingerprint enabled raw sources, curation, masterlist, and the actual processed cache."""
    metadata = handler._build_processing_metadata()
    paths = {Path(entry["file"]["path"]) for entry in metadata["sources"]}
    paths.update(
        (
            handler.processed_path,
            handler.processed_metadata_path,
            Path(cfg.evaluation_language_metadata_path),
        )
    )
    if cfg.label_harmonization_enabled:
        paths.add(Path(metadata["label_harmonization"]["reference_file"]["path"]))
    if cfg.label_standardization_enabled:
        paths.update(
            Path(metadata["label_standardization"][name]["path"])
            for name in ("rules_file", "overrides_file")
        )
    if cfg.evaluation_language_overrides_path:
        paths.add(Path(cfg.evaluation_language_overrides_path))
    if cfg.pretrain_enabled:
        paths.add(resolve_source_path(cfg.pretrain_masterlist_path, cfg.data_raw_dir))
    return {
        "processing_metadata": metadata,
        "processing_digest": content_digest(metadata),
        "files": [file_identity(path) for path in sorted({p.resolve() for p in paths})],
        "languages": provenance_signature(cfg),
    }


def _check_existing_caches(
    configs: list[Config], fingerprints: dict[str, str]
) -> list[dict[str, Any]]:
    """Compare any current cached original partitions without loading augmented training data."""
    records = []
    visited: set[str] = set()
    for cfg in configs:
        handler = DataHandler(cfg)
        metadata = handler._build_prepared_splits_metadata()
        key = handler._prepared_splits_cache_key(metadata)
        if key in visited:
            continue
        visited.add(key)
        root = handler._prepared_splits_dir(metadata)
        record: dict[str, Any] = {
            "cache_key": key,
            "directory": str(root),
            "status": "not_built",
        }
        if (root / "metadata.json").exists():
            with FileLock(
                str(handler._prepared_splits_lock_path(metadata)),
                timeout=handler._processed_lock_timeout_seconds(),
            ):
                try:
                    payload = json.loads((root / "metadata.json").read_text())
                    if payload.get("metadata") != metadata:
                        record["status"] = "stale_will_rebuild"
                    else:
                        matches = {}
                        for name, digest in fingerprints.items():
                            path = root / f"{name}.parquet"
                            matches[name] = (
                                path.is_file()
                                and partition_fingerprint(pd.read_parquet(path), cfg)
                                == digest
                            )
                        record.update(
                            status="matches" if all(matches.values()) else "mismatch",
                            partitions=matches,
                        )
                except (OSError, ValueError, KeyError, AttributeError) as exc:
                    record.update(status="mismatch", error_type=type(exc).__name__)
        records.append(record)
    return records


def audit_splits(
    configs: list[Config],
    names: list[str],
    output: str,
    *,
    specification: str | None = None,
) -> dict[str, Any]:
    """Audit one shared original split and emit private JSON plus a concise Markdown review."""
    validate_shared_audit_configs(configs)
    if len(names) != len(configs):
        raise ValueError(
            "Every audited configuration needs its matching experiment cell name."
        )
    if Path(output).suffix != ".json":
        raise ValueError(
            "Use a .json output path; the review is written to its .md sibling."
        )
    cfg = configs[0]
    handler = DataHandler(cfg)
    print(
        "Split audit: loading the processed dataset once (no model or synthetic dataset builds).",
        flush=True,
    )
    processed = handler.ensure_processed()
    input_stat = handler._source_file_signature(handler.processed_path)
    annotated = annotate_provenance(processed, cfg)
    duplicates = annotated.loc[annotated["row_uid"].duplicated(keep=False)].copy()
    duplicates["audit_targets"] = duplicates[cfg.dataset_label_column].map(
        lambda value: _codes(value, cfg)
    )
    conflicting = int(
        duplicates.groupby("row_uid")[
            [
                "source_id",
                "record_id",
                "cod_key",
                "audit_targets",
                "language_candidates",
                cfg.dataset_text_column,
            ]
        ]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
        .sum()
    )
    print(
        "Split audit: constructing full-size original partitions with the training splitter.",
        flush=True,
    )
    splits, ledger = handler.prepare_original_splits(processed)
    del annotated, processed, duplicates
    parts = {
        "original_train": splits.original_train,
        "val": splits.val,
        "test": splits.test,
    }
    if splits.holdout is not None:
        parts["holdout"] = splits.holdout
    checks = _overlap_checks(parts, cfg)
    checks.append(
        {
            "check": "duplicate_uid_content_consistent",
            "passed": conflicting == 0,
            "conflicting_ids": conflicting,
        }
    )
    try:
        validate_group_integrity(splits, cfg)
        checks.append({"check": "training_integrity_validator", "passed": True})
    except ValueError as exc:
        checks.append(
            {
                "check": "training_integrity_validator",
                "passed": False,
                "reason": str(exc),
            }
        )
    if any(c.multicod_synthetic_ratio > 0 for c in configs):
        homogeneous = (
            splits.original_train.groupby("source_id")["language_candidates"]
            .nunique()
            .le(1)
            .all()
        )
        if not homogeneous:
            raise ValueError(
                "Mixed row-level languages within a synthesis source need actual augmented-exposure auditing."
            )
    masterlist = handler._load_pretraining_source() if cfg.pretrain_enabled else None
    if masterlist is not None and masterlist.empty:
        raise ValueError(
            "Enabled pretraining has an empty masterlist; resolve this before screening."
        )
    if (
        masterlist is not None
        and masterlist["language_candidates"].nunique() > 1
        and cfg.pretrain_multicod_synthetic_ratio > 0
    ):
        raise ValueError(
            "Mixed-language masterlist synthesis needs actual augmented-exposure auditing."
        )
    print(
        "Split audit: computing training-only code/language exposure and partition support.",
        flush=True,
    )
    reference = build_reference(
        splits.original_train, splits.original_train, masterlist, cfg
    )
    inventories, eligibility, fingerprints = {}, {}, {}
    warnings = []
    expected_sources = set(
        next(
            item["source_rows"] for item in ledger if item["stage"] == "sampled_cohort"
        )
    )
    for name, original in parts.items():
        frame = original.assign(
            audit_group=linked_groups(original, by_cod=cfg.evaluation_protocol == "cod")
        )
        print(f"Split audit: {name}, {len(frame):,} original rows.", flush=True)
        fingerprints[name] = partition_fingerprint(original, cfg)
        inventories[name] = _inventory(frame, cfg)
        inventories[name]["by_source"] = {
            source: _inventory(group, cfg)
            for source, group in frame.groupby("source_id")
        }
        if name != "holdout":
            for source in sorted(expected_sources - set(frame["source_id"])):
                warnings.append(f"{name}: source {source} has zero rows.")
        if inventories[name]["unknown_or_mixed_language_rows"]:
            warnings.append(
                f"{name}: unknown/mixed-language rows cannot be treated as verified cross-lingual targets."
            )
        if name != "original_train":
            eligibility[name] = _eligibility(frame, reference, cfg)
            for metric in ("crosslingual", "strict_crosslingual"):
                if not eligibility[name]["overall"][metric]["available"]:
                    warnings.append(
                        f"{name}: {metric} has zero eligible targets; unavailable for a tie-break."
                    )
    allocations = {}
    targets = dict(
        zip(
            ("original_train", "val", "test"),
            (cfg.train_size, cfg.val_size, cfg.test_size),
            strict=True,
        )
    )
    for source in sorted(expected_sources):
        counts = {
            name: int(frame["source_id"].eq(source).sum())
            for name, frame in parts.items()
            if name != "holdout"
        }
        total = sum(counts.values())
        allocations[source] = {
            "partition_rows": counts,
            "retained_rows": total,
            "actual_partition_fractions": {
                n: count / total if total else None for n, count in counts.items()
            },
            "nominal_partition_fractions": targets,
        }
    print(
        "Split audit: comparing existing original-partition caches and hashing data resources.",
        flush=True,
    )
    caches = _check_existing_caches(configs, fingerprints)
    checks.append(
        {
            "check": "existing_cached_partitions_match",
            "passed": all(c["status"] != "mismatch" for c in caches),
        }
    )
    identities = _resource_identities(handler, cfg)
    checks.append(
        {
            "check": "processed_file_unchanged",
            "passed": input_stat
            == handler._source_file_signature(handler.processed_path),
        }
    )
    checks.append(
        {
            "check": "all_data_resource_files_present",
            "passed": all(f["exists"] for f in identities["files"]),
        }
    )
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    code_files = {
        str(p.relative_to(root)): file_identity(p)["sha256"]
        for p in sorted((root / "src/codllm").rglob("*.py"))
    }
    code_files["tasks.py"] = file_identity(root / "tasks.py")["sha256"]
    result = {
        "version": 1,
        "kind": "publication_original_split_support",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "review_required"
        if all(c["passed"] for c in checks)
        else "integrity_failed",
        "scientific_approval": False,
        "model_scores_computed": False,
        "cells": [
            {
                "name": name,
                "configuration_digest": content_digest(_configuration(c)),
                "floor": c.balance_floor,
                "synthetic_ratio": c.multicod_synthetic_ratio,
                "pretraining_epochs": c.pretrain_num_train_epochs
                if c.pretrain_enabled
                else 0,
            }
            for name, c in zip(names, configs, strict=True)
        ],
        "shared_original_partition_count": 1,
        "configuration": _configuration(cfg),
        "specification": file_identity(Path(specification)) if specification else None,
        "data": identities,
        "code": {"git_revision": revision.stdout.strip(), "file_sha256": code_files},
        "preparation_ledger": ledger,
        "source_allocation": allocations,
        "partition_fingerprints": fingerprints,
        "partitions": inventories,
        "eligibility": eligibility,
        "masterlist": {
            "enabled": cfg.pretrain_enabled,
            "rows": len(masterlist) if masterlist is not None else 0,
            "codes": len(reference.masterlist_labels),
            "languages": masterlist["language"].value_counts().to_dict()
            if masterlist is not None
            else {},
        },
        "original_training_code_exposure": {
            code: {
                "historical_rows": reference.label_counts[code],
                "sources": sorted(reference.label_sources[code]),
                "historical_languages": sorted(reference.label_languages[code]),
                "planned_adaptation_languages": sorted(
                    reference.adaptation_languages.get(code, set())
                ),
            }
            for code in sorted(reference.label_sources)
        },
        "checks": checks,
        "review_notes": warnings,
        "expected_prepared_caches": caches,
        "limitations": [
            "No scientific pass/fail threshold is selected; coverage and sparse support require user review.",
            "This audits the selected original partition, not other seeds, fractions, or unconfigured source folds.",
            "Strict support uses planned full-code exposure, not measured trained-model knowledge or epoch completion.",
            "Checked source-homogeneous within-source synthesis and floor copies retain code/language support.",
            "No augmented datasets are generated: their lexical overlaps and constituent files are not checked.",
            "Planned PT support uses the original masterlist; checked homogeneous synthesis adds no language pairs.",
            "Connected COD/record groups are dependence units, not verified independent people or archives.",
            "Group counts are partition-local; source-level counts use those same partition group identities.",
            "Transfer rows contain eligible targets; target counts exclude ineligible co-occurring codes.",
            "Per-code/source/language summaries may expose small counts; outputs remain private until reviewed.",
        ],
    }
    write_json(Path(output), result)
    summary = Path(output).with_suffix(".md")
    summary.write_text(render_split_audit(result))
    summary.chmod(0o600)
    return result


def render_split_audit(report: dict[str, Any]) -> str:
    """Render a compact score-blind decision brief linked to the detailed JSON sections."""
    lines = [
        "# Publication split audit",
        "",
        f"Status: {report['status']}. No model scores or scientific approval.",
        "",
        f"Cells covered: {len(report['cells'])}; shared original partitions: 1.",
        "",
        "## Natural partition inventory",
        "",
        "| Split / source | Rows | Distinct CODs | Connected groups | Codes | Multi-COD rows | Largest-group share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, values in report["partitions"].items():
        for source, item in [("all", values), *values["by_source"].items()]:
            share = item["connected_groups"]["largest_row_share"]
            percent = f"{share:.2%}" if share is not None else "N/A"
            lines.append(
                f"| {split} / {source} | {item['rows']:,} | {item['distinct_cods']:,} | "
                f"{item['connected_groups']['groups']:,} | {item['distinct_codes']:,} | "
                f"{item['natural_multicod_rows']:,} | {percent} |"
            )
    lines += [
        "",
        "## Source allocation",
        "",
        "Fractions use each source's retained train/validation/test pool. Per-source stratification is not guaranteed.",
        "",
        "| Source | Train | Validation | Test |",
        "|---|---:|---:|---:|",
    ]
    for source, item in report["source_allocation"].items():
        fractions = item["actual_partition_fractions"]
        cells = [
            f"{fractions[name]:.2%}" if fractions[name] is not None else "N/A"
            for name in ("original_train", "val", "test")
        ]
        lines.append(f"| {source} | {' | '.join(cells)} |")
    lines += [
        "",
        "## Eligible targets — counts, not performance",
        "",
        "| Split | Bucket | Rows | Targets | Codes | CODs | Connected groups |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, values in report["eligibility"].items():
        for bucket, item in values["overall"].items():
            lines.append(
                f"| {split} | {bucket} | {item['rows']:,} | {item['target_occurrences']:,} | "
                f"{item['distinct_codes']:,} | {item['distinct_cods']:,} | {item['connected_groups']:,} |"
            )
    lines += [
        "",
        "## Transfer support by evaluation source",
        "",
        "Zero targets means unavailable, not zero performance. Detailed single/multi-COD support is in the JSON.",
        "",
        "| Split / source | Definition | Rows | Targets | Codes | CODs | Groups |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, scopes in report["eligibility"].items():
        for scope, buckets in scopes.items():
            if not scope.startswith("source/"):
                continue
            for bucket in ("source_transfer", "crosslingual", "strict_crosslingual"):
                item = buckets[bucket]
                lines.append(
                    f"| {split} / {scope.removeprefix('source/')} | {bucket} | {item['rows']:,} | "
                    f"{item['target_occurrences']:,} | {item['distinct_codes']:,} | "
                    f"{item['distinct_cods']:,} | {item['connected_groups']:,} |"
                )
    lines += ["", "## Integrity checks", ""]
    lines += [
        f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}" for c in report["checks"]
    ]
    lines += [
        "",
        "## Review notes",
        "",
        *[f"- {note}" for note in report["review_notes"]],
        "- Review source-allocation fractions and largest groups before choosing any split policy.",
        "- Review per-source/language eligibility and per-code support in the JSON before choosing support thresholds.",
        "- Do not equate a clean integrity check with representative coverage or model quality.",
        "",
        "## Scope and limits",
        "",
        *[f"- {note}" for note in report["limitations"]],
        "",
    ]
    return "\n".join(lines)
