import random
from collections import defaultdict
from typing import Any

import pandas as pd

from codllm.settings.schema import Config
from codllm.settings.types import MultiCodSyntheticSourceScope
from codllm.input.transform import _build_label, _coerce_row_codes

ANY_SOURCE_GROUP = "any_source"


def _split_text_parts(text: Any, field_separator: str) -> list[str]:
    """Split one processed text value into configured input-field segments."""
    rendered = "" if text is None else str(text)
    if field_separator == "":
        return [rendered]
    return rendered.split(field_separator)


def _cod_segment_value(text: Any, field_separator: str, cod_prefix: str) -> str | None:
    """Extract the processed cod segment value from one text field."""
    for part in _split_text_parts(text, field_separator):
        stripped = part.strip()
        if stripped.startswith(cod_prefix):
            return stripped[len(cod_prefix) :].strip()
    return None


def _merge_cod_texts(
    anchor_text: Any,
    source_texts: list[Any],
    cfg: Config,
    text_separator: str,
) -> str | None:
    """Merge the cod segments from source rows into the anchor row's text shape."""
    cod_values = [
        value
        for value in (
            _cod_segment_value(
                text, cfg.text_field_separator, cfg.input_field_prefix("cod")
            )
            for text in source_texts
        )
        if value
    ]
    if len(cod_values) < 2:
        return None

    parts = _split_text_parts(anchor_text, cfg.text_field_separator)
    cod_prefix = cfg.input_field_prefix("cod")
    for idx, part in enumerate(parts):
        if part.strip().startswith(cod_prefix):
            parts[idx] = f"{cod_prefix}{text_separator.join(cod_values)}"
            return cfg.text_field_separator.join(parts)
    return None


def _single_label_candidate_rows(
    dataframe: pd.DataFrame, cfg: Config
) -> list[dict[str, Any]]:
    """Return rows that can be used as single-COD synthetic merge inputs."""
    candidates: list[dict[str, Any]] = []
    cod_prefix = cfg.input_field_prefix("cod")
    for _, row in dataframe.iterrows():
        codes = _coerce_row_codes(row.get("y_codes"), cfg.label_separator)
        cod_text = _cod_segment_value(
            row.get(cfg.dataset_text_column), cfg.text_field_separator, cod_prefix
        )
        if len(codes) != 1 or cod_text is None:
            continue
        candidates.append(
            {
                "row": dict(row),
                "label": codes[0],
                "source_id": str(row.get("source_id", "unknown")),
            }
        )
    return candidates


def _group_candidate_rows(
    candidates: list[dict[str, Any]],
    source_scope: MultiCodSyntheticSourceScope,
) -> dict[str, list[dict[str, Any]]]:
    """Group synthetic merge candidates according to the configured source scope."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if source_scope == "any_source":
            group_key = ANY_SOURCE_GROUP
        else:
            group_key = candidate["source_id"]
        groups[group_key].append(candidate)
    return {
        group_key: rows
        for group_key, rows in groups.items()
        if len({row["label"] for row in rows}) >= 2
    }


def _sample_distinct_label_rows(
    rows: list[dict[str, Any]],
    max_label_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample candidate rows with distinct labels for one synthetic multi-COD row."""
    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_label[row["label"]].append(row)

    available_labels = list(rows_by_label)
    upper_label_count = min(max_label_count, len(available_labels))
    if upper_label_count < 2:
        return []

    label_count = rng.randint(2, upper_label_count)
    sampled_labels = rng.sample(available_labels, label_count)
    return [rng.choice(rows_by_label[label]) for label in sampled_labels]


def build_synthetic_multicod_rows(
    dataframe: pd.DataFrame,
    cfg: Config,
    *,
    synthetic_ratio: float | None = None,
    source_scope: MultiCodSyntheticSourceScope | None = None,
    text_separator: str | None = None,
    synthetic_source_prefix: str = "synthetic_multicod",
    seed_offset: int = 17,
) -> pd.DataFrame:
    """Build synthetic multi-COD rows by merging single-label rows."""
    resolved_ratio = (
        cfg.multicod_synthetic_ratio if synthetic_ratio is None else synthetic_ratio
    )
    resolved_source_scope = (
        cfg.multicod_synthetic_source_scope if source_scope is None else source_scope
    )
    resolved_text_separator = (
        cfg.multicod_synthetic_text_separator
        if text_separator is None
        else text_separator
    )

    if dataframe.empty or cfg.max_label_count < 2 or resolved_ratio <= 0:
        return dataframe.iloc[0:0].copy()
    if "cod" not in cfg.training_input:
        raise ValueError(
            "Synthetic multi-COD examples require 'cod' in training_input so cause strings can be merged."
        )
    if resolved_text_separator == "":
        raise ValueError("multicod_synthetic_text_separator must not be empty.")

    candidates = _single_label_candidate_rows(dataframe, cfg)
    groups = _group_candidate_rows(candidates, resolved_source_scope)
    if not groups:
        return dataframe.iloc[0:0].copy()

    synthetic_count = int(round(len(candidates) * resolved_ratio))
    if synthetic_count < 1:
        synthetic_count = 1

    rng = random.Random(cfg.resolved_data_seed() + seed_offset)
    group_keys = list(groups)
    group_weights = [len(groups[group_key]) for group_key in group_keys]
    synthetic_rows: list[dict[str, Any]] = []

    for idx in range(synthetic_count):
        group_key = rng.choices(group_keys, weights=group_weights, k=1)[0]
        sampled_rows = _sample_distinct_label_rows(
            rows=groups[group_key],
            max_label_count=cfg.max_label_count,
            rng=rng,
        )
        if len(sampled_rows) < 2:
            continue

        source_rows = [sample["row"] for sample in sampled_rows]
        anchor_row = dict(source_rows[0])
        merged_text = _merge_cod_texts(
            anchor_text=anchor_row.get(cfg.dataset_text_column),
            source_texts=[row.get(cfg.dataset_text_column) for row in source_rows],
            cfg=cfg,
            text_separator=resolved_text_separator,
        )
        if merged_text is None:
            continue

        labels = [sample["label"] for sample in sampled_rows]
        source_ids = sorted(
            {str(row.get("source_id", "unknown")) for row in source_rows}
        )
        source_paths = sorted({str(row.get("source_path", "")) for row in source_rows})
        synthetic_source_id = (
            f"{synthetic_source_prefix}:{group_key}"
            if resolved_source_scope == "within_source"
            else f"{synthetic_source_prefix}:any_source"
        )

        anchor_row["source_id"] = synthetic_source_id
        anchor_row["record_id"] = f"{synthetic_source_id}:{idx:06d}"
        anchor_row["source_path"] = resolved_text_separator.join(source_paths)
        anchor_row[cfg.dataset_text_column] = merged_text
        anchor_row["y_codes"] = labels
        anchor_row[cfg.dataset_label_column] = _build_label(
            labels, separator=cfg.label_separator
        )
        anchor_row["synthetic_source_ids"] = resolved_text_separator.join(source_ids)
        synthetic_rows.append(anchor_row)

    if not synthetic_rows:
        return dataframe.iloc[0:0].copy()
    return pd.DataFrame(synthetic_rows)


def shuffle_multicod_label_order(dataframe: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Randomize multi-label target order and rebuild target strings."""
    if dataframe.empty or cfg.max_label_count < 2 or not cfg.multicod_shuffle_labels:
        return dataframe.reset_index(drop=True)

    result = dataframe.copy()
    rng = random.Random(cfg.resolved_data_seed())
    shuffled_codes: list[list[str]] = []
    labels: list[str] = []
    for raw_codes in result["y_codes"].tolist():
        codes = _coerce_row_codes(raw_codes, cfg.label_separator)
        if len(codes) > 1:
            rng.shuffle(codes)
        shuffled_codes.append(codes)
        labels.append(_build_label(codes, separator=cfg.label_separator))

    result["y_codes"] = shuffled_codes
    result[cfg.dataset_label_column] = labels
    return result.reset_index(drop=True)


def prepare_multicod_training_split(
    dataframe: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Apply configured training-only multi-COD merging and label-order shuffling."""
    synthetic_rows = build_synthetic_multicod_rows(dataframe, cfg)
    if synthetic_rows.empty:
        combined = dataframe
    else:
        combined = pd.concat([dataframe, synthetic_rows], ignore_index=True)
    return shuffle_multicod_label_order(combined, cfg)
