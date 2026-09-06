"""Stable COD identities and reviewed language provenance."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from codllm.config import Config


def normalize_cod(value: Any) -> str:
    """Normalize full COD text without erasing punctuation or diacritics."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().split())


def cod_from_input(value: Any, cfg: Config) -> str:
    """Extract a COD segment using the configured input contract."""
    text = "" if value is None or pd.isna(value) else str(value)
    if cfg.training_input == ["cod"]:
        return text
    prefix = cfg.input_field_prefix("cod")
    for segment in text.split(cfg.text_field_separator):
        if segment.strip().startswith(prefix):
            return segment.strip()[len(prefix) :]
    return ""


def content_digest(value: Any) -> str:
    """Return a deterministic SHA256 digest of a JSON-serializable value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def language_metadata(cfg: Config) -> dict[str, Any]:
    """Load the reviewed language inventory; fail clearly on missing configuration."""
    if not cfg.evaluation_language_metadata_path:
        raise ValueError(
            "Publication evaluation requires evaluation_language_metadata_path."
        )
    path = Path(cfg.evaluation_language_metadata_path)
    with path.open("rb") as handle:
        metadata = tomllib.load(handle)
    if metadata.get("version") != 1:
        raise ValueError("Unsupported source-language metadata version.")
    for source, entry in metadata.get("sources", {}).items():
        languages = entry.get("languages", [])
        if not languages or not all(
            isinstance(code, str) and re.fullmatch(r"[a-z]{2,3}", code)
            for code in languages
        ):
            raise ValueError(f"Invalid reviewed language set for {source}.")
    return metadata


def provenance_signature(cfg: Config) -> dict[str, Any]:
    """Return content-based cache identity for reviewed language annotations."""
    paths = (
        cfg.evaluation_language_metadata_path,
        cfg.evaluation_language_overrides_path,
    )
    return {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in paths
        if path is not None
    }


def annotate_provenance(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach immutable raw COD identity and reviewed language candidates to rows."""
    result = frame.copy()
    if "cod_text" not in result:
        result["cod_text"] = result[cfg.dataset_text_column].map(
            lambda value: cod_from_input(value, cfg)
        )
    result["cod_key"] = result["cod_text"].map(normalize_cod)
    if "source_id" not in result or "record_id" not in result:
        raise ValueError(
            "Publication data requires source_id and record_id provenance."
        )
    if "row_uid" not in result:
        result["row_uid"] = [
            content_digest([str(source), str(record), key, str(label)])
            for source, record, key, label in result[
                ["source_id", "record_id", "cod_key", cfg.dataset_label_column]
            ].itertuples(index=False, name=None)
        ]
    inventory = language_metadata(cfg).get("sources", {})
    candidates = {
        source: ",".join(sorted(set(entry["languages"])))
        for source, entry in inventory.items()
    }
    result["language_candidates"] = result["source_id"].map(candidates).fillna("und")
    if cfg.evaluation_language_overrides_path:
        overrides = pd.read_csv(
            cfg.evaluation_language_overrides_path, dtype=str
        ).fillna("")
        required = {"source_id", "record_id", "language"}
        if not required.issubset(overrides.columns):
            raise ValueError(
                "Language overrides require source_id, record_id, language columns."
            )
        if overrides.duplicated(["source_id", "record_id"]).any():
            raise ValueError(
                "Language overrides must have unique source_id/record_id keys."
            )
        if not overrides["language"].str.fullmatch(r"[a-z]{2,3}").all():
            raise ValueError(
                "Language overrides must use ISO-style lower-case language codes."
            )
        lookup = overrides.set_index(["source_id", "record_id"])["language"].to_dict()
        result["language_candidates"] = [
            lookup.get((str(source), str(record)), language)
            for source, record, language in result[
                ["source_id", "record_id", "language_candidates"]
            ].itertuples(index=False, name=None)
        ]
    result["language"] = result["language_candidates"].map(
        lambda value: "mul" if "," in value else value
    )
    result["data_role"] = (
        result["source_id"]
        .map(
            {
                source: entry.get("role", "historical")
                for source, entry in inventory.items()
            }
        )
        .fillna("historical")
    )
    return result


def evaluation_records(frame: pd.DataFrame, cfg: Config) -> list[dict[str, str]]:
    """Return row-aligned metadata kept out of the model's tokenized features."""
    columns = [
        "row_uid",
        "source_id",
        "record_id",
        "cod_text",
        "cod_key",
        "language",
        "language_candidates",
    ]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Publication evaluation metadata missing: {sorted(missing)}")
    return frame[columns].fillna("").astype(str).to_dict("records")


def load_external_table(cfg: Config) -> pd.DataFrame:
    """Read explicitly curated external inputs without guessing mappings or language."""
    if not cfg.evaluation_data_path:
        raise ValueError(
            "Set CODLLM_EVALUATION_DATA_PATH to a canonical CSV or Parquet table."
        )
    path = Path(cfg.evaluation_data_path)
    if path.suffix not in {".csv", ".parquet"}:
        raise ValueError(
            "External evaluation accepts canonical .csv or .parquet tables only."
        )
    frame = (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, dtype=str, keep_default_na=False)
    )
    required = {
        "source_id",
        "record_id",
        cfg.dataset_text_column,
        cfg.dataset_label_column,
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"External table missing columns: {sorted(required - set(frame.columns))}"
        )
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(
            "External evaluation requires nonempty, non-null canonical records."
        )
    if (
        frame[list(required)]
        .astype(str)
        .apply(lambda column: column.str.strip().eq(""))
        .any()
        .any()
    ):
        raise ValueError(
            "External identifiers, model input, and reference labels must not be blank."
        )
    frame = annotate_provenance(frame, cfg)
    if frame["row_uid"].duplicated().any() or frame["cod_key"].eq("").any():
        raise ValueError(
            "External records must have unique row identities and nonempty COD text."
        )
    return frame
