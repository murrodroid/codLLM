import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from codllm.config import Config
from codllm.data.splits import DataSplits


MAX_TABLE_ROWS = 50
MAX_EXAMPLE_TEXT_LENGTH = 180


def _active_wandb() -> Any | None:
    """Return the active W&B module when a run is initialized."""
    if os.getenv("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    if getattr(wandb, "run", None) is None:
        return None
    return wandb


def _log_wandb_payload(payload: Mapping[str, Any]) -> None:
    """Log a payload to W&B when possible."""
    if not payload:
        return
    wandb = _active_wandb()
    if wandb is None:
        return
    wandb.log(dict(payload))


def _wandb_table(rows: Sequence[Sequence[Any]], columns: Sequence[str]) -> Any | None:
    """Build a W&B table for rows and columns when W&B is active."""
    wandb = _active_wandb()
    if wandb is None:
        return None
    return wandb.Table(columns=list(columns), data=[list(row) for row in rows])


def _log_table_with_optional_bar(
    key: str,
    rows: Sequence[Sequence[Any]],
    columns: Sequence[str],
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    title: str | None = None,
) -> None:
    """Log a W&B table and, when requested, a simple bar plot from the same table."""
    table = _wandb_table(rows, columns)
    if table is None:
        return

    payload: dict[str, Any] = {key: table}
    if x_column is not None and y_column is not None and title is not None:
        wandb = _active_wandb()
        if wandb is not None:
            payload[f"{key}_bar"] = wandb.plot.bar(
                table,
                x_column,
                y_column,
                title=title,
            )
    _log_wandb_payload(payload)


def _log_bar_from_rows(
    key: str,
    rows: Sequence[Sequence[Any]],
    columns: Sequence[str],
    *,
    x_column: str,
    y_column: str,
    title: str,
) -> None:
    """Log a single W&B bar chart from in-memory rows."""
    table = _wandb_table(rows, columns)
    if table is None:
        return
    wandb = _active_wandb()
    if wandb is None:
        return
    _log_wandb_payload(
        {
            key: wandb.plot.bar(
                table,
                x_column,
                y_column,
                title=title,
            )
        }
    )


def _log_histogram_from_table(
    key: str,
    table: Any,
    *,
    value_column: str,
    title: str,
) -> None:
    """Log one W&B histogram from an existing table."""
    wandb = _active_wandb()
    if wandb is None:
        return
    _log_wandb_payload(
        {
            key: wandb.plot.histogram(
                table,
                value_column,
                title=title,
            )
        }
    )


def _truncate_text(value: Any, max_length: int = MAX_EXAMPLE_TEXT_LENGTH) -> str:
    """Render a compact single-line text preview for W&B tables."""
    text = " ".join(str(value).split())
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def _split_label_codes(value: Any, label_separator: str) -> list[str]:
    """Split a single label field into normalized code strings."""
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or pd.isna(value):
        return []
    rendered = str(value).strip()
    if not rendered:
        return []
    tokens = rendered.split(label_separator) if label_separator else [rendered]
    return [token.strip() for token in tokens if token.strip()]


def _chapter_block(code: str) -> str:
    """Return the chapter-block prefix used for compact visual summaries."""
    normalized = str(code).strip()
    return normalized[:3] if normalized else "unknown"


def _chapter_block_distribution(
    dataframe: pd.DataFrame | None,
    cfg: Config,
) -> Counter[str]:
    """Count label occurrences by first-three-character chapter-block prefix."""
    counts: Counter[str] = Counter()
    if dataframe is None or dataframe.empty:
        return counts
    if cfg.dataset_label_column not in dataframe.columns:
        return counts

    labels = dataframe[cfg.dataset_label_column].fillna("").astype(str)
    for value in labels.tolist():
        for code in _split_label_codes(value, cfg.label_separator):
            counts[_chapter_block(code)] += 1
    return counts


def _label_cardinality_distribution(
    dataframe: pd.DataFrame | None,
    cfg: Config,
) -> Counter[int]:
    """Count rows by number of labels in the configured label column."""
    counts: Counter[int] = Counter()
    if dataframe is None or dataframe.empty:
        return counts
    if cfg.dataset_label_column not in dataframe.columns:
        return counts

    labels = dataframe[cfg.dataset_label_column].fillna("").astype(str)
    for value in labels.tolist():
        counts[len(_split_label_codes(value, cfg.label_separator))] += 1
    return counts


def _cod_segment_value(text: Any, cfg: Config) -> str:
    """Extract the configured COD segment from processed text when present."""
    cod_prefix = cfg.input_field_prefix("cod")
    rendered = "" if text is None else str(text)
    if list(cfg.training_input) == ["cod"]:
        return rendered.strip()
    parts = (
        rendered.split(cfg.text_field_separator)
        if cfg.text_field_separator
        else [rendered]
    )
    for part in parts:
        stripped = part.strip()
        if stripped.startswith(cod_prefix):
            return stripped[len(cod_prefix) :].strip()
    return ""


def _split_frames(splits: DataSplits) -> dict[str, pd.DataFrame | None]:
    """Return named split frames in one stable mapping."""
    return {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
        "holdout": splits.holdout,
        "holdout_eval": splits.holdout_eval,
    }


def _source_distribution_rows(splits: DataSplits) -> list[list[Any]]:
    """Build rows for source distribution by split."""
    rows: list[list[Any]] = []
    for split_name, dataframe in _split_frames(splits).items():
        if dataframe is None or dataframe.empty or "source_id" not in dataframe.columns:
            continue
        counts = dataframe["source_id"].fillna("unknown").astype(str).value_counts()
        for source_id, count in counts.items():
            rows.append([split_name, str(source_id), int(count)])
    return rows


def _label_distribution_rows(
    splits: DataSplits,
    cfg: Config,
    max_rows_per_split: int,
) -> list[list[Any]]:
    """Build top chapter-block label distribution rows by split."""
    rows: list[list[Any]] = []
    for split_name, dataframe in _split_frames(splits).items():
        counts = _chapter_block_distribution(dataframe, cfg)
        total = sum(counts.values())
        for block, count in counts.most_common(max_rows_per_split):
            rows.append(
                [
                    split_name,
                    block,
                    int(count),
                    float(count / total) if total > 0 else 0.0,
                ]
            )
    return rows


def _label_cardinality_rows(splits: DataSplits, cfg: Config) -> list[list[Any]]:
    """Build label-count distribution rows by split."""
    rows: list[list[Any]] = []
    for split_name, dataframe in _split_frames(splits).items():
        counts = _label_cardinality_distribution(dataframe, cfg)
        for label_count, row_count in sorted(counts.items()):
            rows.append([split_name, int(label_count), int(row_count)])
    return rows


def _length_rows(splits: DataSplits, cfg: Config) -> list[list[Any]]:
    """Build text and COD length rows by split."""
    rows: list[list[Any]] = []
    for split_name, dataframe in _split_frames(splits).items():
        if (
            dataframe is None
            or dataframe.empty
            or cfg.dataset_text_column not in dataframe.columns
        ):
            continue
        for text in dataframe[cfg.dataset_text_column].fillna("").astype(str).tolist():
            cod_value = _cod_segment_value(text, cfg)
            rows.append([split_name, len(text), len(cod_value)])
    return rows


def _seen_unseen_rows(splits: DataSplits, cfg: Config) -> list[list[Any]]:
    """Build seen/unseen input-string coverage rows for validation-like splits."""
    if cfg.dataset_text_column not in splits.train.columns:
        return []
    train_strings = {
        " ".join(text.split())
        for text in splits.train[cfg.dataset_text_column]
        .fillna("")
        .astype(str)
        .tolist()
        if str(text).strip()
    }
    rows: list[list[Any]] = []
    for split_name, dataframe in _split_frames(splits).items():
        if split_name == "train":
            continue
        if (
            dataframe is None
            or dataframe.empty
            or cfg.dataset_text_column not in dataframe.columns
        ):
            continue
        texts = [
            " ".join(text.split())
            for text in dataframe[cfg.dataset_text_column]
            .fillna("")
            .astype(str)
            .tolist()
        ]
        seen_count = sum(1 for text in texts if text in train_strings)
        unseen_count = len(texts) - seen_count
        rows.append(
            [
                split_name,
                int(seen_count),
                int(unseen_count),
                float(seen_count / len(texts)) if texts else 0.0,
                float(unseen_count / len(texts)) if texts else 0.0,
            ]
        )
    return rows


def _split_rows(splits: DataSplits) -> list[list[Any]]:
    """Build row-count rows by split."""
    rows: list[list[Any]] = []
    for split_name, dataframe in _split_frames(splits).items():
        if dataframe is None:
            continue
        rows.append([split_name, int(len(dataframe))])
    return rows


def _filter_rows_by_split(
    rows: Sequence[Sequence[Any]],
    split_name: str,
) -> list[list[Any]]:
    """Return rows whose first column matches a split name."""
    return [list(row[1:]) for row in rows if row and row[0] == split_name]


def _metrics_summary_rows(
    prefix: str,
    metrics: Mapping[str, Any] | None,
    keys: Sequence[str],
) -> list[list[Any]]:
    """Build compact rows from scalar metadata metrics."""
    if metrics is None:
        return []
    rows: list[list[Any]] = []
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, bool | int | float | str):
            rows.append([prefix, key, _stringify_metric_value(value)])
    return rows


def _stringify_metric_value(value: bool | int | float | str) -> str:
    """Render heterogeneous scalar metrics as one W&B table column type."""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _balance_distribution_rows(metrics: Mapping[str, Any] | None) -> list[list[Any]]:
    """Build before/after balance distribution rows from balance diagnostics."""
    if metrics is None:
        return []

    before = metrics.get("label_distribution_before")
    after = metrics.get("label_distribution_after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return []

    blocks = sorted(set(before) | set(after))
    return [
        [
            str(block),
            int(before.get(block, 0)),
            int(after.get(block, 0)),
            int(after.get(block, 0)) - int(before.get(block, 0)),
        ]
        for block in blocks
    ]


def _train_preparation_rows(
    splits: DataSplits,
    balance_metrics: Mapping[str, Any] | None,
    masterlist_inject_metrics: Mapping[str, Any] | None,
    pretraining_upsampling_metrics: Mapping[str, Any] | None,
    pretraining_multicod_metrics: Mapping[str, Any] | None,
) -> list[list[Any]]:
    """Build rows summarizing real, synthetic, and perturbed data contributions."""
    rows: list[list[Any]] = [["final_train_rows", int(len(splits.train))]]
    if "source_id" in splits.train.columns:
        source_ids = splits.train["source_id"].fillna("").astype(str)
        rows.append(
            [
                "final_multicod_synthetic_rows",
                int(source_ids.str.startswith("synthetic_multicod").sum()),
            ]
        )
        rows.append(
            [
                "final_masterlist_source_rows",
                int((source_ids == "masterlist_pretrain").sum()),
            ]
        )

    if balance_metrics is not None:
        for key in ("rows_before", "rows_added", "rows_after", "base_perturbed_rows"):
            value = balance_metrics.get(key)
            if isinstance(value, int | float):
                rows.append([f"balance_{key}", value])

    if masterlist_inject_metrics is not None:
        for key in ("rows_injected", "labels_injected"):
            value = masterlist_inject_metrics.get(key)
            if isinstance(value, int | float):
                rows.append([f"masterlist_{key}", value])

    for prefix, metrics in (
        ("pretraining_upsampling", pretraining_upsampling_metrics),
        ("pretraining_multicod", pretraining_multicod_metrics),
    ):
        if metrics is None:
            continue
        for key in (
            "rows_before",
            "rows_added",
            "rows_after",
            "synthetic_rows",
            "perturbed_rows",
        ):
            value = metrics.get(key)
            if isinstance(value, int | float):
                rows.append([f"{prefix}_{key}", value])

    return rows


def log_data_visualizations(
    cfg: Config,
    splits: DataSplits,
    *,
    balance_metrics: Mapping[str, Any] | None = None,
    masterlist_inject_metrics: Mapping[str, Any] | None = None,
    pretraining_upsampling_metrics: Mapping[str, Any] | None = None,
    pretraining_multicod_metrics: Mapping[str, Any] | None = None,
    max_rows_per_split: int = 30,
) -> None:
    """Log compact split, label, provenance, and length visualizations to W&B."""
    if _active_wandb() is None:
        return

    _log_table_with_optional_bar(
        "data/split_rows",
        _split_rows(splits),
        ["split", "rows"],
        x_column="split",
        y_column="rows",
        title="Rows by split",
    )
    _log_table_with_optional_bar(
        "data/train_preparation_summary",
        _train_preparation_rows(
            splits,
            balance_metrics,
            masterlist_inject_metrics,
            pretraining_upsampling_metrics,
            pretraining_multicod_metrics,
        ),
        ["category", "rows"],
        x_column="category",
        y_column="rows",
        title="Training data preparation summary",
    )

    source_rows = _source_distribution_rows(splits)
    if source_rows:
        _log_table_with_optional_bar(
            "data/source_distribution",
            source_rows,
            ["split", "source_id", "rows"],
        )
        for split_name in ("train", "val", "test", "holdout", "holdout_eval"):
            split_rows = _filter_rows_by_split(source_rows, split_name)
            if not split_rows:
                continue
            _log_bar_from_rows(
                f"data/source_distribution/{split_name}_bar",
                split_rows[:max_rows_per_split],
                ["source_id", "rows"],
                x_column="source_id",
                y_column="rows",
                title=f"{split_name} rows by source",
            )

    label_rows = _label_distribution_rows(
        splits,
        cfg,
        max_rows_per_split=max_rows_per_split,
    )
    if label_rows:
        _log_table_with_optional_bar(
            "data/chapter_block_distribution",
            label_rows,
            ["split", "chapter_block", "count", "share"],
        )
        for split_name in ("train", "val", "test", "holdout", "holdout_eval"):
            split_rows = _filter_rows_by_split(label_rows, split_name)
            if not split_rows:
                continue
            _log_bar_from_rows(
                f"data/chapter_block_distribution/{split_name}_bar",
                split_rows,
                ["chapter_block", "count", "share"],
                x_column="chapter_block",
                y_column="count",
                title=f"{split_name} chapter-block distribution",
            )

    balance_rows = _balance_distribution_rows(balance_metrics)
    if balance_rows:
        _log_table_with_optional_bar(
            "data/balance_chapter_block_before_after",
            balance_rows,
            ["chapter_block", "before", "after", "delta"],
        )

    cardinality_rows = _label_cardinality_rows(splits, cfg)
    if cardinality_rows:
        _log_table_with_optional_bar(
            "data/label_cardinality_distribution",
            cardinality_rows,
            ["split", "label_count", "rows"],
        )
        for split_name in ("train", "val", "test", "holdout", "holdout_eval"):
            split_rows = _filter_rows_by_split(cardinality_rows, split_name)
            if not split_rows:
                continue
            _log_bar_from_rows(
                f"data/label_cardinality_distribution/{split_name}_bar",
                split_rows,
                ["label_count", "rows"],
                x_column="label_count",
                y_column="rows",
                title=f"{split_name} label cardinality",
            )

    coverage_rows = _seen_unseen_rows(splits, cfg)
    if coverage_rows:
        _log_table_with_optional_bar(
            "data/seen_unseen_string_coverage",
            coverage_rows,
            ["split", "seen_rows", "unseen_rows", "seen_rate", "unseen_rate"],
        )

    length_rows = _length_rows(splits, cfg)
    if length_rows:
        length_table = _wandb_table(
            length_rows,
            ["split", "text_length", "cod_length"],
        )
        if length_table is not None:
            _log_wandb_payload({"data/text_length_distribution": length_table})
            _log_histogram_from_table(
                "data/text_length_histogram",
                length_table,
                value_column="text_length",
                title="Processed text length",
            )
            _log_histogram_from_table(
                "data/cod_length_histogram",
                length_table,
                value_column="cod_length",
                title="COD segment length",
            )

    scalar_rows: list[list[Any]] = []
    scalar_rows.extend(
        _metrics_summary_rows(
            "balance",
            balance_metrics,
            ["enabled", "strategy", "rows_before", "rows_added", "rows_after"],
        )
    )
    scalar_rows.extend(
        _metrics_summary_rows(
            "pretraining_upsampling",
            pretraining_upsampling_metrics,
            ["enabled", "rows_before", "rows_added", "rows_after", "perturbed_rows"],
        )
    )
    scalar_rows.extend(
        _metrics_summary_rows(
            "pretraining_multicod",
            pretraining_multicod_metrics,
            ["enabled", "rows_before", "rows_added", "rows_after", "synthetic_rows"],
        )
    )
    scalar_rows.extend(
        _metrics_summary_rows(
            "masterlist_injection",
            masterlist_inject_metrics,
            ["enabled", "rows_injected", "labels_injected"],
        )
    )
    if scalar_rows:
        _log_table_with_optional_bar(
            "data/preparation_scalar_metrics",
            scalar_rows,
            ["scope", "metric", "value"],
        )


@dataclass
class MetricArtifactLogger:
    """Log compact prediction error tables during evaluation."""

    cfg: Config
    max_rows: int = MAX_TABLE_ROWS

    def __call__(
        self,
        *,
        metric_scope: str | None,
        input_strings: Sequence[str] | None,
        predictions: Sequence[str],
        labels: Sequence[str],
    ) -> None:
        """Log top chapter-block error tables for one evaluation scope."""
        if _active_wandb() is None:
            return
        if metric_scope is None:
            metric_scope = "eval"
        if not predictions or not labels:
            return

        true_block_sets = [
            {
                _chapter_block(code)
                for code in _split_label_codes(label, self.cfg.label_separator)
            }
            for label in labels
        ]
        pred_block_sets = [
            {
                _chapter_block(code)
                for code in _split_label_codes(prediction, self.cfg.label_separator)
            }
            for prediction in predictions
        ]
        input_values = list(input_strings or [""] * len(labels))

        pair_counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
        pair_examples: dict[
            tuple[tuple[str, ...], tuple[str, ...]], tuple[str, str, str]
        ] = {}
        false_negatives: Counter[str] = Counter()
        false_positives: Counter[str] = Counter()
        support: Counter[str] = Counter()

        for index, (true_blocks, pred_blocks) in enumerate(
            zip(true_block_sets, pred_block_sets)
        ):
            for block in true_blocks:
                support[block] += 1
            if true_blocks == pred_blocks:
                continue
            pair_key = (tuple(sorted(true_blocks)), tuple(sorted(pred_blocks)))
            pair_counts[pair_key] += 1
            pair_examples.setdefault(
                pair_key,
                (
                    _truncate_text(input_values[index]),
                    str(labels[index]),
                    str(predictions[index]),
                ),
            )
            for block in true_blocks.difference(pred_blocks):
                false_negatives[block] += 1
            for block in pred_blocks.difference(true_blocks):
                false_positives[block] += 1

        pair_rows = []
        for (true_blocks, pred_blocks), count in pair_counts.most_common(self.max_rows):
            example_input, example_label, example_prediction = pair_examples[
                (true_blocks, pred_blocks)
            ]
            pair_rows.append(
                [
                    ", ".join(true_blocks),
                    ", ".join(pred_blocks),
                    int(count),
                    example_label,
                    example_prediction,
                    example_input,
                ]
            )

        fn_rows = [
            [
                block,
                int(count),
                int(support[block]),
                float(count / support[block]) if support[block] > 0 else 0.0,
            ]
            for block, count in false_negatives.most_common(self.max_rows)
        ]
        fp_rows = [
            [block, int(count)]
            for block, count in false_positives.most_common(self.max_rows)
        ]

        payload: dict[str, Any] = {}
        pair_table = _wandb_table(
            pair_rows,
            [
                "true_chapter_blocks",
                "predicted_chapter_blocks",
                "count",
                "example_label",
                "example_prediction",
                "example_input",
            ],
        )
        if pair_table is not None:
            payload[f"{metric_scope}/errors/chapter_block_pairs"] = pair_table
        fn_table = _wandb_table(
            fn_rows,
            ["chapter_block", "misses", "support", "miss_rate"],
        )
        if fn_table is not None:
            payload[f"{metric_scope}/errors/chapter_block_false_negatives"] = fn_table
        fp_table = _wandb_table(
            fp_rows,
            ["chapter_block", "false_positives"],
        )
        if fp_table is not None:
            payload[f"{metric_scope}/errors/chapter_block_false_positives"] = fp_table

        _log_wandb_payload(payload)
