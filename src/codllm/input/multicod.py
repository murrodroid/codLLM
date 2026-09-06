import random
import re
import json
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd

from codllm.settings.schema import Config
from codllm.settings.types import MultiCodSyntheticSourceScope
from codllm.input.transform import _build_label, _coerce_row_codes

ANY_SOURCE_GROUP = "any_source"


def _log_multicod_progress(message: str) -> None:
    """Print a timestamped multi-COD data-preparation progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Multi-COD: {message}", flush=True)


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


def _uses_bare_cod_input(cfg: Config) -> bool:
    """Return whether processed text consists only of the COD value."""
    return list(cfg.training_input) == ["cod"]


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
            str(text).strip()
            if _uses_bare_cod_input(cfg)
            else _cod_segment_value(
                text, cfg.text_field_separator, cfg.input_field_prefix("cod")
            )
            for text in source_texts
        )
        if value
    ]
    return _merge_cod_values(
        anchor_text=anchor_text,
        cod_values=cod_values,
        cfg=cfg,
        text_separator=text_separator,
    )


def _merge_cod_values(
    anchor_text: Any,
    cod_values: list[Any],
    cfg: Config,
    text_separator: str,
) -> str | None:
    """Merge pre-extracted COD values into the anchor row's text shape."""
    if len(cod_values) < 2:
        return None

    if _uses_bare_cod_input(cfg):
        return text_separator.join(str(value) for value in cod_values)

    parts = _split_text_parts(anchor_text, cfg.text_field_separator)
    cod_prefix = cfg.input_field_prefix("cod")
    for idx, part in enumerate(parts):
        if part.strip().startswith(cod_prefix):
            parts[idx] = f"{cod_prefix}{text_separator.join(cod_values)}"
            return cfg.text_field_separator.join(parts)
    return None


def _cod_segment_values(
    texts: pd.Series,
    field_separator: str,
    cod_prefix: str,
    bare_cod_input: bool = False,
) -> pd.Series:
    """Extract processed COD segment values from a text series."""
    if bare_cod_input:
        values = [str(text).strip() for text in texts.fillna("").tolist()]
    else:
        values = [
            _cod_segment_value(text, field_separator, cod_prefix)
            for text in texts.fillna("").tolist()
        ]
    return pd.Series(values, index=texts.index, dtype=object)


def _single_label_values(dataframe: pd.DataFrame, cfg: Config) -> pd.Series:
    """Return label values for single-label rows and nulls for multi-label rows."""
    labels = dataframe[cfg.dataset_label_column].fillna("").astype("string").str.strip()
    if cfg.label_separator:
        has_separator = labels.str.contains(
            re.escape(cfg.label_separator),
            regex=True,
            na=False,
        )
        return labels.where(labels.ne("") & ~has_separator, None).astype(object)

    label_lengths = dataframe["y_codes"].map(
        lambda raw_codes: len(_coerce_row_codes(raw_codes, cfg.label_separator))
    )
    return labels.where(labels.ne("") & label_lengths.eq(1), None).astype(object)


def _label_counts(dataframe: pd.DataFrame, cfg: Config) -> pd.Series:
    """Return per-row label cardinality from processed labels."""
    if "y_codes" in dataframe.columns:
        return dataframe["y_codes"].map(
            lambda raw_codes: len(_coerce_row_codes(raw_codes, cfg.label_separator))
        )

    labels = dataframe[cfg.dataset_label_column].fillna("").astype(str).str.strip()
    if cfg.label_separator:
        return labels.map(
            lambda value: len(
                [part for part in value.split(cfg.label_separator) if part.strip()]
            )
        )
    return labels.map(lambda value: 1 if value else 0)


def _multicod_label_count_distribution(
    dataframe: pd.DataFrame,
    cfg: Config,
) -> dict[int, int]:
    """Return the empirical multi-COD label-count distribution."""
    label_counts = _label_counts(dataframe, cfg)
    distribution = label_counts[
        label_counts.between(2, cfg.max_label_count, inclusive="both")
    ].value_counts()
    return {int(label_count): int(count) for label_count, count in distribution.items()}


def _resolve_text_separators(
    text_separators: list[str] | None,
) -> list[str]:
    """Return candidate text separators for synthetic COD merging."""
    resolved = [] if text_separators is None else list(text_separators)

    if not resolved or any(separator == "" for separator in resolved):
        raise ValueError("multicod_synthetic_text_separators must not be empty.")
    return resolved


def _candidate_source_values(
    dataframe: pd.DataFrame,
    column: str,
    fallback: str,
) -> pd.Series:
    """Return source metadata strings aligned to the candidate dataframe."""
    if column not in dataframe.columns:
        return pd.Series([fallback] * len(dataframe), index=dataframe.index)
    return dataframe[column].fillna(fallback).astype(str)


def _group_candidate_positions(
    candidates: pd.DataFrame,
    source_scope: MultiCodSyntheticSourceScope,
) -> dict[str, dict[str, list[int]]]:
    """Group candidate row positions by source scope and label."""
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for position, label, source_id in candidates[
        ["position", "label", "source_id"]
    ].itertuples(index=False, name=None):
        group_key = ANY_SOURCE_GROUP if source_scope == "any_source" else source_id
        groups[str(group_key)][str(label)].append(int(position))
    return {
        group_key: dict(positions_by_label)
        for group_key, positions_by_label in groups.items()
        if len(positions_by_label) >= 2
    }


def _sample_distinct_label_positions(
    positions_by_label: dict[str, list[int]],
    available_labels: list[str],
    max_label_count: int,
    label_count_distribution: dict[int, int],
    rng: random.Random,
) -> tuple[list[str], list[int]]:
    """Sample candidate row positions with distinct labels."""
    upper_label_count = min(max_label_count, len(available_labels))
    if upper_label_count < 2:
        return [], []

    eligible_counts = [
        label_count
        for label_count in sorted(label_count_distribution)
        if 2 <= label_count <= upper_label_count
    ]
    if eligible_counts:
        label_count = rng.choices(
            eligible_counts,
            weights=[label_count_distribution[count] for count in eligible_counts],
            k=1,
        )[0]
    else:
        label_count = rng.randint(2, upper_label_count)
    sampled_labels = rng.sample(available_labels, label_count)
    return sampled_labels, [
        rng.choice(positions_by_label[label]) for label in sampled_labels
    ]


def build_synthetic_multicod_rows(
    dataframe: pd.DataFrame,
    cfg: Config,
    *,
    synthetic_ratio: float | None = None,
    source_scope: MultiCodSyntheticSourceScope | None = None,
    text_separators: list[str] | None = None,
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
    resolved_text_separators = _resolve_text_separators(
        cfg.multicod_synthetic_text_separators
        if text_separators is None
        else text_separators
    )

    if dataframe.empty or cfg.max_label_count < 2 or resolved_ratio <= 0:
        return dataframe.iloc[0:0].copy()
    if "cod" not in cfg.training_input:
        raise ValueError(
            "Synthetic multi-COD examples require 'cod' in training_input so cause strings can be merged."
        )

    _log_multicod_progress(f"extracting candidates from {len(dataframe)} train rows.")
    label_values = _single_label_values(dataframe, cfg)
    cod_values = _cod_segment_values(
        dataframe[cfg.dataset_text_column],
        cfg.text_field_separator,
        cfg.input_field_prefix("cod"),
        bare_cod_input=_uses_bare_cod_input(cfg),
    )
    source_ids = _candidate_source_values(dataframe, "source_id", "unknown")
    source_paths = _candidate_source_values(dataframe, "source_path", "")
    positions = pd.Series(range(len(dataframe)), index=dataframe.index)
    candidate_mask = label_values.notna() & cod_values.notna()
    candidates = pd.DataFrame(
        {
            "position": positions,
            "label": label_values,
            "source_id": source_ids,
        }
    ).loc[candidate_mask]
    if candidates.empty:
        _log_multicod_progress("found no single-COD rows eligible for synthesis.")
        return dataframe.iloc[0:0].copy()

    _log_multicod_progress(f"grouping {len(candidates)} single-COD candidate rows.")
    groups = _group_candidate_positions(candidates, resolved_source_scope)
    if not groups:
        _log_multicod_progress("found no source groups with at least two labels.")
        return dataframe.iloc[0:0].copy()

    label_count_distribution = _multicod_label_count_distribution(dataframe, cfg)
    synthetic_count = int(round(len(candidates) * resolved_ratio))
    if synthetic_count < 1:
        synthetic_count = 1

    rng = random.Random(cfg.resolved_data_seed() + seed_offset)
    group_keys = list(groups)
    group_weights = [
        sum(
            len(positions_for_label)
            for positions_for_label in groups[group_key].values()
        )
        for group_key in group_keys
    ]
    group_labels = {group_key: list(groups[group_key]) for group_key in group_keys}
    sampled_group_keys = rng.choices(
        group_keys, weights=group_weights, k=synthetic_count
    )
    _log_multicod_progress(
        f"assembling {synthetic_count} synthetic rows from {len(group_keys)} groups."
    )

    text_values = dataframe[cfg.dataset_text_column].tolist()
    cod_value_list = cod_values.tolist()
    source_id_list = source_ids.tolist()
    source_path_list = source_paths.tolist()

    anchor_positions: list[int] = []
    merged_texts: list[str] = []
    sampled_label_sets: list[list[str]] = []
    synthetic_source_ids: list[str] = []
    synthetic_record_ids: list[str] = []
    synthetic_source_paths: list[str] = []
    synthetic_source_id_sets: list[str] = []
    synthetic_parent_uids: list[str] = []
    synthetic_languages: list[str] = []
    row_uids = _candidate_source_values(dataframe, "row_uid", "").tolist()
    language_values = _candidate_source_values(
        dataframe, "language_candidates", "und"
    ).tolist()
    progress_interval = max(50_000, synthetic_count // 10)

    for idx, group_key in enumerate(sampled_group_keys):
        if idx > 0 and idx % progress_interval == 0:
            _log_multicod_progress(
                f"assembled {idx}/{synthetic_count} requested synthetic rows."
            )
        labels, sampled_positions = _sample_distinct_label_positions(
            positions_by_label=groups[group_key],
            available_labels=group_labels[group_key],
            max_label_count=cfg.max_label_count,
            label_count_distribution=label_count_distribution,
            rng=rng,
        )
        if len(sampled_positions) < 2:
            continue

        row_text_separator = rng.choice(resolved_text_separators)
        merged_text = _merge_cod_values(
            anchor_text=text_values[sampled_positions[0]],
            cod_values=[cod_value_list[position] for position in sampled_positions],
            cfg=cfg,
            text_separator=row_text_separator,
        )
        if merged_text is None:
            continue

        source_ids = sorted(
            {source_id_list[position] for position in sampled_positions}
        )
        source_paths = sorted(
            {source_path_list[position] for position in sampled_positions}
        )
        synthetic_source_id = (
            f"{synthetic_source_prefix}:{group_key}"
            if resolved_source_scope == "within_source"
            else f"{synthetic_source_prefix}:any_source"
        )

        anchor_positions.append(sampled_positions[0])
        merged_texts.append(merged_text)
        sampled_label_sets.append(labels)
        synthetic_source_ids.append(synthetic_source_id)
        synthetic_record_ids.append(f"{synthetic_source_id}:{idx:06d}")
        synthetic_source_paths.append("; ".join(source_paths))
        synthetic_source_id_sets.append("; ".join(source_ids))
        synthetic_parent_uids.append(
            json.dumps([row_uids[position] for position in sampled_positions])
        )
        synthetic_languages.append(
            ",".join(
                sorted(
                    {
                        code
                        for position in sampled_positions
                        for code in language_values[position].split(",")
                    }
                )
            )
        )

    if not anchor_positions:
        _log_multicod_progress("no synthetic rows were assembled.")
        return dataframe.iloc[0:0].copy()

    _log_multicod_progress(f"materializing {len(anchor_positions)} synthetic rows.")
    synthetic_df = dataframe.iloc[anchor_positions].copy().reset_index(drop=True)
    synthetic_df["source_id"] = synthetic_source_ids
    synthetic_df["record_id"] = synthetic_record_ids
    synthetic_df["source_path"] = synthetic_source_paths
    synthetic_df[cfg.dataset_text_column] = merged_texts
    synthetic_df["y_codes"] = sampled_label_sets
    synthetic_df[cfg.dataset_label_column] = [
        _build_label(labels, separator=cfg.label_separator)
        for labels in sampled_label_sets
    ]
    synthetic_df["synthetic_source_ids"] = synthetic_source_id_sets
    if cfg.publication_eval_enabled:
        from codllm.evaluation.provenance import cod_from_input, normalize_cod

        synthetic_df["row_uid"] = synthetic_record_ids
        synthetic_df["synthetic_parent_uids"] = synthetic_parent_uids
        synthetic_df["language_candidates"] = synthetic_languages
        synthetic_df["language"] = [
            "mul" if "," in value else value for value in synthetic_languages
        ]
        synthetic_df["cod_text"] = [
            cod_from_input(value, cfg) for value in merged_texts
        ]
        synthetic_df["cod_key"] = synthetic_df["cod_text"].map(normalize_cod)
        synthetic_df["data_role"] = "synthetic"
    _log_multicod_progress(f"built {len(synthetic_df)} synthetic rows.")
    return synthetic_df


def shuffle_multicod_label_order(dataframe: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Randomize multi-label target order and rebuild target strings."""
    if dataframe.empty or cfg.max_label_count < 2 or not cfg.multicod_shuffle_labels:
        return dataframe.reset_index(drop=True)

    result = dataframe.reset_index(drop=True).copy()
    if cfg.label_separator:
        label_values = (
            result[cfg.dataset_label_column].fillna("").astype("string").str.strip()
        )
        multi_label_mask = label_values.str.contains(
            cfg.label_separator,
            regex=False,
            na=False,
        )
        if not bool(multi_label_mask.any()):
            return result
        row_indices = result.index[multi_label_mask]
    else:
        row_indices = result.index

    _log_multicod_progress(
        f"shuffling label order for {len(row_indices)}/{len(result)} rows."
    )
    rng = random.Random(cfg.resolved_data_seed())
    shuffled_codes: list[list[str]] = []
    labels: list[str] = []
    for raw_codes in result.loc[row_indices, "y_codes"].tolist():
        codes = _coerce_row_codes(raw_codes, cfg.label_separator)
        if len(codes) > 1:
            rng.shuffle(codes)
        shuffled_codes.append(codes)
        labels.append(_build_label(codes, separator=cfg.label_separator))

    result.loc[row_indices, "y_codes"] = pd.Series(
        shuffled_codes,
        index=row_indices,
        dtype=object,
    )
    result.loc[row_indices, cfg.dataset_label_column] = labels
    _log_multicod_progress(f"shuffled label order for {len(row_indices)} rows.")
    return result


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
