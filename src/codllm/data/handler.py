import hashlib
import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
import warnings

from filelock import FileLock, Timeout
import pandas as pd
from sklearn.model_selection import train_test_split

from codllm.config import Config, DataSourceConfig
from codllm.data.balancing import (
    _contains_cod_segment,
    _perturb_cod_segment,
    _resolve_perturbation_functions,
    _sample_upsample_rows,
    manipulate_classes,
    select_floor_upsample_targets,
    upsample,
)
from codllm.data.splits import DataSplits, resolve_training_frames
from codllm.data.storage import (
    DEFAULT_PROCESSED_LOCK_TIMEOUT_SECONDS,
    PROCESSING_METADATA_VERSION,
    _normalize_processing_metadata,
    build_and_save_processed_dataset,
    save_processed_dataset,
)
from codllm.data.tokenization import (
    prepare_sequence_classification_dataset,
    prepare_training_dataset,
)
from codllm.input import (
    DatasetMapping,
    MAPPING_REGISTRY,
    PERTURBATION_REGISTRY,
    _build_text,
    _build_y,
    build_synthetic_multicod_rows,
    build_processed_dataset,
    load_dataset,
    load_source_dataset,
    prepare_multicod_training_split,
    shuffle_multicod_label_order,
)
from codllm.input.harmonization import resolve_label_harmonization_workbook_path
from codllm.input.transform import _coerce_row_codes
from codllm.runtime.paths import resolve_source_path

PREPARED_SPLITS_METADATA_VERSION = 6


def _log_data_progress(message: str) -> None:
    """Print a timestamped data-preparation progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class DataHandler:
    """Handle processed-data lifecycle and train/val/test splitting."""

    def __init__(
        self,
        cfg: Config,
        mapping_registry: Mapping[str, DatasetMapping] | None = None,
    ) -> None:
        self.cfg = cfg
        self.mapping_registry = dict(MAPPING_REGISTRY)
        if mapping_registry is not None:
            self.mapping_registry.update(mapping_registry)
        self._pretraining_upsampling_metrics: dict[str, Any] | None = None
        self._pretraining_multicod_metrics: dict[str, Any] | None = None
        self._masterlist_inject_metrics: dict[str, Any] | None = None
        self._training_balance_metrics: dict[str, Any] | None = None

    @property
    def processed_path(self) -> Path:
        """Return the configured processed-data output path."""
        return Path(self.cfg.data_processed_dir) / self.cfg.processed_filename

    @property
    def processed_metadata_path(self) -> Path:
        """Return the sidecar metadata path used for processed-data validation."""
        suffix = self.processed_path.suffix
        return self.processed_path.with_suffix(f"{suffix}.meta.json")

    @property
    def processed_lock_path(self) -> Path:
        """Return lock path used to serialize processed-data cache access."""
        suffix = self.processed_path.suffix
        return self.processed_path.with_suffix(f"{suffix}.lock")

    @property
    def prepared_splits_root(self) -> Path:
        """Return directory used for prepared split caches."""
        return self.processed_path.parent / f"{self.processed_path.stem}.splits"

    def _processed_lock_timeout_seconds(self) -> float:
        """Return processed-cache lock timeout configured via environment."""
        raw_value = os.getenv("CODLLM_PROCESSED_LOCK_TIMEOUT_SECONDS")
        if raw_value is None or raw_value.strip() == "":
            return DEFAULT_PROCESSED_LOCK_TIMEOUT_SECONDS
        try:
            timeout_seconds = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                "CODLLM_PROCESSED_LOCK_TIMEOUT_SECONDS must be a positive float."
            ) from exc
        if timeout_seconds <= 0:
            raise ValueError(
                "CODLLM_PROCESSED_LOCK_TIMEOUT_SECONDS must be greater than 0."
            )
        return timeout_seconds

    def processed_exists(self) -> bool:
        """Return True when the configured processed file already exists."""
        return self.processed_path.exists()

    def _source_file_signature(self, path: Path) -> dict[str, Any]:
        """Return a lightweight file signature for cache invalidation checks."""
        resolved_path = path.resolve()
        signature: dict[str, Any] = {
            "path": str(resolved_path),
            "exists": path.exists(),
        }
        if path.exists():
            stats = path.stat()
            signature["size_bytes"] = stats.st_size
            signature["mtime_ns"] = stats.st_mtime_ns
        return signature

    def _build_processing_metadata(self) -> dict[str, Any]:
        """Build metadata that defines whether a processed file is still reusable."""
        source_metadata: list[dict[str, Any]] = []
        for source in self.cfg.data_sources:
            if not source.enabled:
                continue
            if source.mapping_id not in self.mapping_registry:
                raise KeyError(
                    f"Unknown mapping_id '{source.mapping_id}' for source '{source.source_id}'."
                )
            mapping = self.mapping_registry[source.mapping_id]
            source_path = resolve_source_path(source.path, self.cfg.data_raw_dir)
            source_metadata.append(
                {
                    "source": asdict(source),
                    "mapping": asdict(mapping),
                    "file": self._source_file_signature(source_path),
                }
            )

        return {
            "version": PROCESSING_METADATA_VERSION,
            "data_raw_dir": str(Path(self.cfg.data_raw_dir).resolve()),
            "training_input": list(self.cfg.training_input),
            "input_field_prefixes": dict(self.cfg.input_field_prefixes),
            "max_label_count": self.cfg.max_label_count,
            "label_separator": self.cfg.label_separator,
            "text_field_separator": self.cfg.text_field_separator,
            "label_harmonization": {
                "enabled": self.cfg.label_harmonization_enabled,
                "masterlist_path": self.cfg.label_harmonization_masterlist_path,
                "masterlist_sheet_name": (
                    self.cfg.label_harmonization_masterlist_sheet_name
                ),
                "transfer_sheet_name": (
                    self.cfg.label_harmonization_transfer_sheet_name
                ),
                "reference_file": self._source_file_signature(
                    resolve_label_harmonization_workbook_path(self.cfg)
                )
                if self.cfg.label_harmonization_enabled
                else None,
            },
            "sources": source_metadata,
        }

    def _load_saved_processing_metadata(self) -> dict[str, Any] | None:
        """Load saved processed metadata when available and parseable."""
        if not self.processed_metadata_path.exists():
            return None
        try:
            return json.loads(self.processed_metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_processing_metadata(self, metadata: dict[str, Any]) -> None:
        """Persist processed-data metadata next to the processed file."""
        metadata_path = self.processed_metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(metadata, sort_keys=True, indent=2)
        temp_path = metadata_path.parent / f".{metadata_path.name}.{uuid4().hex}.tmp"
        try:
            temp_path.write_text(f"{payload}\n")
            temp_path.replace(metadata_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _processed_cache_is_valid(self) -> bool:
        """Return True when the current setup matches saved processed metadata."""
        saved = self._load_saved_processing_metadata()
        if saved is None:
            return False
        current = self._build_processing_metadata()
        return _normalize_processing_metadata(saved) == _normalize_processing_metadata(
            current
        )

    def _build_prepared_splits_metadata(self) -> dict[str, Any]:
        """Build metadata that defines whether prepared split caches are reusable."""
        return {
            "version": PREPARED_SPLITS_METADATA_VERSION,
            "processing": self._build_processing_metadata(),
            "split_config": {
                "dataset_size": self.cfg.dataset_size,
                "train_size": self.cfg.train_size,
                "val_size": self.cfg.val_size,
                "test_size": self.cfg.test_size,
                "seed": self.cfg.seed,
                "data_seed": self.cfg.data_seed,
                "resolved_data_seed": self.cfg.resolved_data_seed(),
                "hold_out_dataset": self.cfg.hold_out_dataset,
                "train_excluded_source_ids": list(self.cfg.train_excluded_source_ids),
                "hold_out_evaluate_per": self.cfg.hold_out_evaluate_per,
                "hold_out_evaluate_ratio": self.cfg.hold_out_evaluate_ratio,
                "dataset_text_column": self.cfg.dataset_text_column,
                "dataset_label_column": self.cfg.dataset_label_column,
                "max_label_count": self.cfg.max_label_count,
                "label_separator": self.cfg.label_separator,
                "text_field_separator": self.cfg.text_field_separator,
                "training_input": list(self.cfg.training_input),
                "input_field_prefixes": dict(self.cfg.input_field_prefixes),
                "multicod_shuffle_labels": self.cfg.multicod_shuffle_labels,
                "multicod_synthetic_ratio": self.cfg.multicod_synthetic_ratio,
                "multicod_synthetic_source_scope": (
                    self.cfg.multicod_synthetic_source_scope
                ),
                "multicod_synthetic_text_separators": list(
                    self.cfg.multicod_synthetic_text_separators
                ),
                "balance_strategy": self.cfg.balance_strategy,
                "balance_perturbations": list(self.cfg.balance_perturbations),
                "balance_perturbation_mean": self.cfg.balance_perturbation_mean,
                "balance_perturbation_variance": self.cfg.balance_perturbation_variance,
                "balance_perturbation_loft": self.cfg.balance_perturbation_loft,
                "balance_floor": self.cfg.balance_floor,
                "balance_floor_decay": self.cfg.balance_floor_decay,
                "balance_floor_singlecod_only": (self.cfg.balance_floor_singlecod_only),
                "base_perturbations": list(self.cfg.base_perturbations),
                "base_perturbation_mean": self.cfg.base_perturbation_mean,
                "base_perturbation_variance": self.cfg.base_perturbation_variance,
                "base_perturbation_loft": self.cfg.base_perturbation_loft,
                "base_perturbation_rate": self.cfg.base_perturbation_rate,
                "masterlist_inject_enabled": self.cfg.masterlist_inject_enabled,
                "masterlist_inject_target_per_label": (
                    self.cfg.masterlist_inject_target_per_label
                ),
                "masterlist_inject_perturbations": list(
                    self.cfg.masterlist_inject_perturbations
                ),
                "masterlist_inject_perturbations_per_sample": (
                    self.cfg.masterlist_inject_perturbations_per_sample
                ),
                "pretrain_masterlist_path": self.cfg.pretrain_masterlist_path,
                "pretrain_masterlist_sheet_name": (
                    self.cfg.pretrain_masterlist_sheet_name
                ),
                "masterlist_inject_source": self._source_file_signature(
                    resolve_source_path(
                        self.cfg.pretrain_masterlist_path,
                        self.cfg.data_raw_dir,
                    )
                )
                if self.cfg.masterlist_inject_enabled
                else None,
            },
        }

    def _prepared_splits_cache_key(self, metadata: Mapping[str, Any]) -> str:
        """Return a stable cache key for prepared split metadata."""
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _prepared_splits_dir(self, metadata: Mapping[str, Any]) -> Path:
        """Return cache directory for one prepared split metadata payload."""
        return self.prepared_splits_root / self._prepared_splits_cache_key(metadata)

    def _prepared_splits_lock_path(self, metadata: Mapping[str, Any]) -> Path:
        """Return lock path used to serialize prepared split cache access."""
        return self.prepared_splits_root / (
            f"{self._prepared_splits_cache_key(metadata)}.lock"
        )

    def _write_dataframe_atomic(self, dataframe: pd.DataFrame, path: Path) -> None:
        """Write one cached dataframe atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.parent / f".{path.name}.{uuid4().hex}.tmp.parquet"
        try:
            dataframe.to_parquet(temp_path, index=False)
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _write_prepared_splits_cache(
        self,
        splits: DataSplits,
        metadata: dict[str, Any],
        cache_dir: Path,
    ) -> None:
        """Persist prepared data splits and split-generation side effects."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        frames = {
            "train": splits.train,
            "val": splits.val,
            "test": splits.test,
            "holdout": splits.holdout,
            "holdout_eval": splits.holdout_eval,
        }
        present_frames: dict[str, bool] = {}
        for name, dataframe in frames.items():
            present = dataframe is not None
            present_frames[name] = present
            if dataframe is not None:
                self._write_dataframe_atomic(dataframe, cache_dir / f"{name}.parquet")

        payload = {
            "metadata": metadata,
            "frames": present_frames,
            "masterlist_inject_metrics": self._masterlist_inject_metrics,
            "training_balance_metrics": self._training_balance_metrics,
        }
        payload_path = cache_dir / "metadata.json"
        temp_path = cache_dir / f".{payload_path.name}.{uuid4().hex}.tmp"
        try:
            temp_path.write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            temp_path.replace(payload_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _load_prepared_splits_cache(
        self,
        metadata: Mapping[str, Any],
        cache_dir: Path,
    ) -> DataSplits | None:
        """Load reusable prepared data splits when the cache is complete and current."""
        payload_path = cache_dir / "metadata.json"
        if not payload_path.exists():
            return None
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if payload.get("metadata") != metadata:
            return None
        frame_presence = payload.get("frames")
        if not isinstance(frame_presence, Mapping):
            return None

        loaded_frames: dict[str, pd.DataFrame | None] = {}
        for name in ("train", "val", "test", "holdout", "holdout_eval"):
            present = bool(frame_presence.get(name, False))
            if not present:
                loaded_frames[name] = None
                continue
            frame_path = cache_dir / f"{name}.parquet"
            if not frame_path.exists():
                return None
            loaded_frames[name] = pd.read_parquet(frame_path)

        for required_name in ("train", "val", "test"):
            if loaded_frames[required_name] is None:
                return None
        metrics = payload.get("masterlist_inject_metrics")
        self._masterlist_inject_metrics = metrics if isinstance(metrics, dict) else None
        balance_metrics = payload.get("training_balance_metrics")
        self._training_balance_metrics = (
            balance_metrics if isinstance(balance_metrics, dict) else None
        )
        return DataSplits(
            train=loaded_frames["train"],
            val=loaded_frames["val"],
            test=loaded_frames["test"],
            holdout=loaded_frames["holdout"],
            holdout_eval=loaded_frames["holdout_eval"],
        )

    def ensure_processed(self, force_reprocess: bool = False) -> pd.DataFrame:
        """Load processed data, or build and save it when missing."""
        lock_path = self.processed_lock_path
        lock_timeout_seconds = self._processed_lock_timeout_seconds()
        lock = FileLock(str(lock_path), timeout=lock_timeout_seconds)
        try:
            with lock:
                should_reprocess = force_reprocess or not self.processed_exists()
                if not should_reprocess:
                    should_reprocess = not self._processed_cache_is_valid()

                if should_reprocess:
                    _log_data_progress("Building processed dataset from raw sources.")
                    processed_df = build_processed_dataset(
                        cfg=self.cfg,
                        mapping_registry=self.mapping_registry,
                    )
                    save_processed_dataset(processed_df, str(self.processed_path))
                    self._write_processing_metadata(self._build_processing_metadata())
                    _log_data_progress(
                        f"Built processed dataset: rows={len(processed_df)}."
                    )
                    return processed_df
                _log_data_progress(
                    f"Loading processed dataset cache: {self.processed_path}"
                )
                processed_df = self._load_processed_dataset()
                _log_data_progress(
                    f"Loaded processed dataset cache: rows={len(processed_df)}."
                )
                return processed_df
        except Timeout as exc:
            raise TimeoutError(
                f"Timed out waiting for processed-data lock '{lock_path}'. "
                "Set CODLLM_PROCESSED_LOCK_TIMEOUT_SECONDS to a larger value."
            ) from exc

    def _prepare_splits_from_processed(self, processed_df: pd.DataFrame) -> DataSplits:
        """Build prepared split dataframes from processed rows."""
        _log_data_progress(f"Preparing split source rows: rows={len(processed_df)}.")
        processed_source_ids = (
            set(processed_df["source_id"].fillna("").astype(str).tolist())
            if "source_id" in processed_df.columns
            else set()
        )
        training_pool_df, holdout_df = self._partition_hold_out_dataset(processed_df)
        if holdout_df is not None:
            _log_data_progress(
                "Partitioned hold-out dataset: "
                f"training_pool={len(training_pool_df)}, holdout={len(holdout_df)}."
            )
        training_pool_df = self._exclude_training_sources(
            training_pool_df,
            processed_source_ids=processed_source_ids,
        )
        sampled_df = self._apply_dataset_size(training_pool_df)
        if len(sampled_df) != len(training_pool_df):
            _log_data_progress(
                f"Applied dataset_size={self.cfg.dataset_size}: rows={len(sampled_df)}."
            )
        _log_data_progress("Splitting train/validation/test dataframes.")
        splits = self.split_dataframe(sampled_df)
        _log_data_progress(
            "Initial split sizes: "
            f"train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}."
        )
        if self.cfg.max_label_count >= 2 and (
            self.cfg.multicod_synthetic_ratio > 0 or self.cfg.multicod_shuffle_labels
        ):
            _log_data_progress("Applying multi-COD split preparation.")
        splits.train = prepare_multicod_training_split(splits.train, self.cfg)
        splits.val = shuffle_multicod_label_order(splits.val, self.cfg)
        splits.test = shuffle_multicod_label_order(splits.test, self.cfg)
        _log_data_progress(
            "After multi-COD preparation: "
            f"train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}."
        )
        if holdout_df is not None:
            splits.holdout = shuffle_multicod_label_order(holdout_df, self.cfg)
            splits.holdout_eval = self._build_holdout_eval_dataframe(splits.holdout)
        if not splits.train.empty:
            if (
                self.cfg.balance_strategy != "none"
                or self.cfg.base_perturbation_rate > 0
            ):
                _log_data_progress("Applying training balance/perturbation policy.")
            splits.train = self._apply_balance_policy(splits.train)
            _log_data_progress(f"After balance policy: train={len(splits.train)}.")
        if self.cfg.masterlist_inject_enabled and not splits.train.empty:
            _log_data_progress("Injecting masterlist rows into training split.")
            splits.train = self._inject_masterlist(splits.train)
            _log_data_progress(
                f"After masterlist injection: train={len(splits.train)}."
            )
        return splits

    def get_splits(self, force_reprocess: bool = False) -> DataSplits:
        """Return train/validation/test splits from processed data."""
        metadata = self._build_prepared_splits_metadata()
        cache_dir = self._prepared_splits_dir(metadata)
        lock_path = self._prepared_splits_lock_path(metadata)
        self.prepared_splits_root.mkdir(parents=True, exist_ok=True)
        lock_timeout_seconds = self._processed_lock_timeout_seconds()
        lock = FileLock(str(lock_path), timeout=lock_timeout_seconds)
        _log_data_progress(f"Waiting for prepared data splits cache lock: {lock_path}")
        try:
            with lock:
                if not force_reprocess:
                    cached_splits = self._load_prepared_splits_cache(
                        metadata, cache_dir
                    )
                    if cached_splits is not None:
                        _log_data_progress(
                            f"Loaded prepared data splits cache: {cache_dir}",
                        )
                        return cached_splits

                processed_df = self.ensure_processed(force_reprocess=force_reprocess)
                _log_data_progress(f"Building prepared data splits cache: {cache_dir}")
                splits = self._prepare_splits_from_processed(processed_df)
                self._write_prepared_splits_cache(splits, metadata, cache_dir)
                _log_data_progress(f"Wrote prepared data splits cache: {cache_dir}")
                return splits
        except Timeout as exc:
            raise TimeoutError(
                f"Timed out waiting for prepared-splits lock '{lock_path}'. "
                "Set CODLLM_PROCESSED_LOCK_TIMEOUT_SECONDS to a larger value."
            ) from exc

    def get_pretraining_train_dataframe(self) -> pd.DataFrame | None:
        """Return optional masterlist dataframe used for pretraining."""
        if not self.cfg.pretrain_enabled:
            self._pretraining_upsampling_metrics = None
            self._pretraining_multicod_metrics = None
            return None

        pretrain_df = self._load_pretraining_source()
        if pretrain_df.empty:
            raise ValueError("Pretraining dataframe is empty.")
        upsampled_df = self._apply_pretraining_upsample_policy(pretrain_df)
        return self._apply_pretraining_multicod_policy(upsampled_df)

    def get_pretraining_upsampling_metrics(self) -> dict[str, Any] | None:
        """Return metrics from the latest pretraining upsampling pass."""
        if self._pretraining_upsampling_metrics is None:
            return None
        return dict(self._pretraining_upsampling_metrics)

    def get_pretraining_multicod_metrics(self) -> dict[str, Any] | None:
        """Return metrics from the latest pretraining multi-COD synthesis pass."""
        if self._pretraining_multicod_metrics is None:
            return None
        return dict(self._pretraining_multicod_metrics)

    def get_masterlist_label_vocabulary(self) -> list[str]:
        """Return sorted unique ICD10h label values from the configured masterlist."""
        masterlist_df = self._load_masterlist_source(
            source_id="masterlist_labels",
            path=self.cfg.label_harmonization_masterlist_path,
            sheet_name=self.cfg.label_harmonization_masterlist_sheet_name,
        )
        label_column = self.cfg.dataset_label_column
        if label_column not in masterlist_df.columns:
            raise KeyError(
                f"Masterlist dataframe is missing label column '{label_column}'."
            )
        labels = masterlist_df[label_column].fillna("").astype(str).str.strip()
        unique_labels = sorted(label for label in labels.unique().tolist() if label)
        if not unique_labels:
            raise ValueError("Masterlist label vocabulary is empty.")
        return unique_labels

    def _load_masterlist_source(
        self,
        source_id: str,
        path: str,
        sheet_name: int | str,
    ) -> pd.DataFrame:
        """Load rows from one configured ICD10h masterlist source."""
        source = DataSourceConfig(
            source_id=source_id,
            path=path,
            mapping_id="masterlist",
            sheet_name=sheet_name,
            enabled=True,
        )
        if source.mapping_id not in self.mapping_registry:
            raise KeyError(
                f"Unknown mapping_id '{source.mapping_id}' for masterlist source."
            )
        mapping = self.mapping_registry[source.mapping_id]
        masterlist_df = load_source_dataset(
            source=source,
            mapping=mapping,
            training_input=self.cfg.training_input,
            max_labels=self.cfg.max_label_count,
            label_separator=self.cfg.label_separator,
            text_field_separator=self.cfg.text_field_separator,
            input_field_prefixes=self.cfg.input_field_prefixes,
            data_raw_dir=self.cfg.data_raw_dir,
            text_column=self.cfg.dataset_text_column,
            label_column=self.cfg.dataset_label_column,
        )
        self._validate_required_columns(masterlist_df)
        self._validate_label_quality(masterlist_df)
        return masterlist_df.reset_index(drop=True)

    def _load_pretraining_source(self) -> pd.DataFrame:
        """Load pretraining rows from the configured ICD10h masterlist source."""
        return self._load_masterlist_source(
            source_id="masterlist_pretrain",
            path=self.cfg.pretrain_masterlist_path,
            sheet_name=self.cfg.pretrain_masterlist_sheet_name,
        )

    def _label_count_summary(self, counts: pd.Series) -> dict[str, float]:
        """Summarize label-count distribution for metadata and diagnostics."""
        if counts.empty:
            return {"min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0}
        return {
            "min": float(counts.min()),
            "max": float(counts.max()),
            "median": float(counts.median()),
            "mean": float(counts.mean()),
        }

    def _chapter_block_distribution(self, dataframe: pd.DataFrame) -> dict[str, int]:
        """Return label counts grouped by the first three characters of each code."""
        label_column = self.cfg.dataset_label_column
        if dataframe.empty or label_column not in dataframe.columns:
            return {}

        counts: dict[str, int] = {}
        labels = dataframe[label_column].fillna("").astype(str)
        for value in labels.tolist():
            tokens = (
                value.split(self.cfg.label_separator)
                if self.cfg.label_separator
                else [value]
            )
            for token in tokens:
                normalized = token.strip()
                if not normalized:
                    continue
                block = normalized[:3]
                counts[block] = counts.get(block, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def _label_counts(self, dataframe: pd.DataFrame) -> pd.Series:
        """Return per-row label cardinality using processed code lists when available."""
        if "y_codes" in dataframe.columns:
            return dataframe["y_codes"].map(
                lambda raw_codes: len(
                    _coerce_row_codes(raw_codes, self.cfg.label_separator)
                )
            )

        labels = (
            dataframe[self.cfg.dataset_label_column].fillna("").astype(str).str.strip()
        )
        if self.cfg.label_separator:
            return labels.map(
                lambda value: len(
                    [
                        part
                        for part in value.split(self.cfg.label_separator)
                        if part.strip()
                    ]
                )
            )
        return labels.map(lambda value: 1 if value else 0)

    def _balance_upsample_eligible_mask(self, train_df: pd.DataFrame) -> pd.Series:
        """Return rows eligible for floor upsampling."""
        label_counts = self._label_counts(train_df)
        eligible = label_counts.eq(1)
        if "source_id" in train_df.columns:
            source_ids = train_df["source_id"].fillna("").astype(str)
            eligible &= ~source_ids.str.startswith("synthetic_multicod")
        return eligible

    def _apply_pretraining_upsample_policy(
        self, pretrain_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Upsample pretraining rows per label and track perturbation diagnostics."""
        label_column = self.cfg.dataset_label_column
        text_column = self.cfg.dataset_text_column
        target_count = self.cfg.pretrain_upsample_target_per_label
        perturbations_per_sample = self.cfg.pretrain_upsample_perturbations_per_sample

        if target_count < 1:
            raise ValueError("pretrain_upsample_target_per_label must be at least 1.")
        if perturbations_per_sample < 1:
            raise ValueError(
                "pretrain_upsample_perturbations_per_sample must be at least 1."
            )

        rows_before = int(len(pretrain_df))
        before_counts = pretrain_df[label_column].value_counts()
        metrics: dict[str, Any] = {
            "enabled": bool(self.cfg.pretrain_upsample_enabled),
            "target_examples_per_label": int(target_count),
            "rows_before": rows_before,
            "rows_after": rows_before,
            "rows_added": 0,
            "synthetic_rows": 0,
            "perturbed_rows": 0,
            "perturbation_rate": 0.0,
            "perturbation_applications": 0,
            "perturbations_per_sample": int(perturbations_per_sample),
            "perturbations": list(self.cfg.pretrain_upsample_perturbations),
            "labels_before": int(before_counts.shape[0]),
            "labels_upsampled": 0,
            "labels_below_target_before": int((before_counts < target_count).sum()),
            "labels_below_target_after": int((before_counts < target_count).sum()),
            "label_count_summary_before": self._label_count_summary(before_counts),
            "label_count_summary_after": self._label_count_summary(before_counts),
        }

        if not self.cfg.pretrain_upsample_enabled:
            self._pretraining_upsampling_metrics = metrics
            return pretrain_df.reset_index(drop=True)

        perturbation_fns = _resolve_perturbation_functions(
            self.cfg.pretrain_upsample_perturbations
        )
        rng = random.Random(self.cfg.resolved_data_seed())

        synthetic_rows: list[dict[str, Any]] = []
        perturbed_rows = 0
        perturbation_applications = 0
        labels_upsampled = 0

        for label, current_count in before_counts.items():
            current_count_int = int(current_count)
            if current_count_int >= target_count:
                continue

            class_rows = pretrain_df[pretrain_df[label_column] == label]
            needed = target_count - current_count_int
            sampled_rows = _sample_upsample_rows(class_rows, needed=needed, rng=rng)
            if not sampled_rows:
                continue
            labels_upsampled += 1

            for row in sampled_rows:
                synthetic_row = dict(row)
                original_text = str(synthetic_row[text_column])
                perturbed_text = original_text

                if perturbation_fns:
                    perturbed_text = _perturb_cod_segment(
                        cfg=self.cfg,
                        text=original_text,
                        perturbation_fns=perturbation_fns,
                        perturbations_per_sample=perturbations_per_sample,
                        rng=rng,
                    )
                    if _contains_cod_segment(self.cfg, original_text):
                        perturbation_applications += perturbations_per_sample

                if perturbed_text != original_text:
                    perturbed_rows += 1
                synthetic_row[text_column] = perturbed_text
                synthetic_rows.append(synthetic_row)

        if synthetic_rows:
            synthetic_df = pd.DataFrame(synthetic_rows, columns=pretrain_df.columns)
            result_df = pd.concat([pretrain_df, synthetic_df], ignore_index=True)
        else:
            result_df = pretrain_df

        rows_after = int(len(result_df))
        synthetic_count = int(len(synthetic_rows))
        after_counts = result_df[label_column].value_counts()
        metrics["rows_after"] = rows_after
        metrics["rows_added"] = rows_after - rows_before
        metrics["synthetic_rows"] = synthetic_count
        metrics["labels_upsampled"] = labels_upsampled
        metrics["perturbed_rows"] = perturbed_rows
        metrics["perturbation_applications"] = perturbation_applications
        metrics["perturbation_rate"] = (
            float(perturbed_rows / synthetic_count) if synthetic_count > 0 else 0.0
        )
        metrics["labels_below_target_after"] = int((after_counts < target_count).sum())
        metrics["label_count_summary_after"] = self._label_count_summary(after_counts)
        self._pretraining_upsampling_metrics = metrics
        return result_df.reset_index(drop=True)

    def _apply_pretraining_multicod_policy(
        self, pretrain_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create optional synthetic multi-COD rows for pretraining."""
        ratio = self.cfg.pretrain_multicod_synthetic_ratio
        if ratio <= 0:
            self._pretraining_multicod_metrics = None
            return pretrain_df.reset_index(drop=True)
        if self.cfg.max_label_count < 2:
            raise ValueError(
                "pretrain_multicod_synthetic_ratio requires max_label_count >= 2."
            )
        if "cod" not in self.cfg.training_input:
            raise ValueError(
                "Synthetic pretraining multi-COD rows require 'cod' in training_input."
            )
        if any(
            separator == ""
            for separator in self.cfg.pretrain_multicod_synthetic_text_separators
        ):
            raise ValueError(
                "pretrain_multicod_synthetic_text_separators must not contain empty values."
            )

        rows_before = int(len(pretrain_df))
        synthetic_rows = build_synthetic_multicod_rows(
            pretrain_df,
            self.cfg,
            synthetic_ratio=ratio,
            source_scope="any_source",
            text_separators=self.cfg.pretrain_multicod_synthetic_text_separators,
            synthetic_source_prefix="synthetic_pretrain_multicod",
            seed_offset=29,
        )
        if synthetic_rows.empty:
            result_df = pretrain_df
        else:
            result_df = pd.concat([pretrain_df, synthetic_rows], ignore_index=True)
        result_df = shuffle_multicod_label_order(result_df, self.cfg)

        synthetic_count = int(len(synthetic_rows))
        label_lengths = (
            result_df["y_codes"]
            .map(lambda codes: len(codes) if isinstance(codes, list) else 0)
            .value_counts()
            .sort_index()
        )
        self._pretraining_multicod_metrics = {
            "enabled": True,
            "ratio": float(ratio),
            "text_separators": list(
                self.cfg.pretrain_multicod_synthetic_text_separators
            ),
            "rows_before": rows_before,
            "rows_after": int(len(result_df)),
            "rows_added": synthetic_count,
            "synthetic_rows": synthetic_count,
            "max_label_count": int(self.cfg.max_label_count),
            "label_count_distribution": {
                str(int(label_count)): int(row_count)
                for label_count, row_count in label_lengths.items()
            },
        }
        return result_df.reset_index(drop=True)

    def _apply_balance_policy(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Apply optional upsampling and manipulation rules to the training split."""
        balanced_train_df = train_df
        text_column = self.cfg.dataset_text_column
        label_column = self.cfg.dataset_label_column
        metrics: dict[str, Any] = {
            "enabled": bool(
                self.cfg.balance_strategy != "none"
                or self.cfg.base_perturbation_rate > 0
            ),
            "strategy": self.cfg.balance_strategy,
            "base_perturbation_rate": float(self.cfg.base_perturbation_rate),
            "rows_before": int(len(train_df)),
            "rows_after_upsample": int(len(train_df)),
            "rows_after": int(len(train_df)),
            "rows_added": 0,
            "upsample_eligible_rows": int(len(train_df)),
            "upsample_excluded_multicod_rows": 0,
            "upsample_excluded_synthetic_multicod_rows": 0,
            "base_perturbed_rows": 0,
            "label_distribution_before": self._chapter_block_distribution(train_df),
            "label_distribution_after": self._chapter_block_distribution(train_df),
        }

        if self.cfg.balance_strategy == "floor":
            eligible_mask = self._balance_upsample_eligible_mask(train_df)
            label_counts = self._label_counts(train_df)
            metrics["upsample_eligible_rows"] = int(eligible_mask.sum())
            metrics["upsample_excluded_multicod_rows"] = int(label_counts.gt(1).sum())
            if "source_id" in train_df.columns:
                source_ids = train_df["source_id"].fillna("").astype(str)
                metrics["upsample_excluded_synthetic_multicod_rows"] = int(
                    source_ids.str.startswith("synthetic_multicod").sum()
                )
            balance_source_df = train_df.loc[eligible_mask]
            target_counts = select_floor_upsample_targets(
                df=balance_source_df,
                label_column=label_column,
                floor=self.cfg.balance_floor,
                decay=self.cfg.balance_floor_decay,
                singlecod_only=self.cfg.balance_floor_singlecod_only,
                label_separator=self.cfg.label_separator,
            )
            perturbation_fns = _resolve_perturbation_functions(
                self.cfg.balance_perturbations
            )
            balanced_train_df = upsample(
                df=balanced_train_df,
                label_column=self.cfg.dataset_label_column,
                target_counts=target_counts,
                seed=self.cfg.resolved_data_seed(),
                cfg=self.cfg,
                text_column=text_column,
                perturbation_fns=perturbation_fns,
                perturbation_mean=self.cfg.balance_perturbation_mean,
                perturbation_variance=self.cfg.balance_perturbation_variance,
                perturbation_loft=self.cfg.balance_perturbation_loft,
                text_field_separator=self.cfg.text_field_separator,
            )

        metrics["rows_after_upsample"] = int(len(balanced_train_df))
        metrics["rows_added"] = int(len(balanced_train_df) - len(train_df))
        if self.cfg.base_perturbation_rate > 0:
            before_perturbation_texts = (
                balanced_train_df[text_column].fillna("").astype(str).tolist()
                if text_column in balanced_train_df.columns
                else []
            )
            balanced_train_df = manipulate_classes(
                cfg=self.cfg,
                label_column=label_column,
                df=balanced_train_df,
                text_column=text_column,
                target_labels=None,
                perturbation_names=self.cfg.base_perturbations,
                perturbation_mean=self.cfg.base_perturbation_mean,
                perturbation_variance=self.cfg.base_perturbation_variance,
                perturbation_loft=self.cfg.base_perturbation_loft,
                sample_fraction=self.cfg.base_perturbation_rate,
                seed=self.cfg.resolved_data_seed() + 1,
            )
            after_perturbation_texts = (
                balanced_train_df[text_column].fillna("").astype(str).tolist()
                if text_column in balanced_train_df.columns
                else []
            )
            metrics["base_perturbed_rows"] = sum(
                1
                for before_text, after_text in zip(
                    before_perturbation_texts,
                    after_perturbation_texts,
                )
                if before_text != after_text
            )

        metrics["rows_after"] = int(len(balanced_train_df))
        metrics["label_distribution_after"] = self._chapter_block_distribution(
            balanced_train_df
        )
        self._training_balance_metrics = metrics
        return balanced_train_df

    def get_training_balance_metrics(self) -> dict[str, Any] | None:
        """Return metrics from the latest training balance and perturbation pass."""
        if self._training_balance_metrics is None:
            return None
        return dict(self._training_balance_metrics)

    def _inject_masterlist(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Load the masterlist, upsample with perturbations, and inject into training split."""
        masterlist_df = self._load_pretraining_source()
        if masterlist_df.empty:
            warnings.warn("Masterlist is empty; skipping injection.")
            return train_df

        target_count = self.cfg.masterlist_inject_target_per_label
        perturbations_per_sample = self.cfg.masterlist_inject_perturbations_per_sample
        label_column = self.cfg.dataset_label_column
        text_column = self.cfg.dataset_text_column

        if target_count < 1:
            raise ValueError("masterlist_inject_target_per_label must be at least 1.")

        perturbation_fns = _resolve_perturbation_functions(
            self.cfg.masterlist_inject_perturbations
        )
        rng = random.Random(self.cfg.resolved_data_seed() + 2)

        before_counts = masterlist_df[label_column].value_counts()
        synthetic_rows: list[dict[str, Any]] = []

        for label, current_count in before_counts.items():
            current_count_int = int(current_count)
            needed = max(0, target_count - current_count_int)
            class_rows = masterlist_df[masterlist_df[label_column] == label]

            for _, row in class_rows.iterrows():
                synthetic_rows.append(dict(row))

            if needed > 0:
                sampled = _sample_upsample_rows(class_rows, needed=needed, rng=rng)
                for row in sampled:
                    new_row = dict(row)
                    original_text = str(new_row[text_column])
                    if perturbation_fns:
                        new_row[text_column] = _perturb_cod_segment(
                            cfg=self.cfg,
                            text=original_text,
                            perturbation_fns=perturbation_fns,
                            perturbations_per_sample=perturbations_per_sample,
                            rng=rng,
                        )
                    synthetic_rows.append(new_row)

        inject_df = pd.DataFrame(synthetic_rows, columns=train_df.columns)
        rows_injected = len(inject_df)
        labels_injected = inject_df[label_column].nunique()
        print(
            f"Masterlist injection: {rows_injected} rows "
            f"({labels_injected} labels) injected into training split."
        )
        self._masterlist_inject_metrics = {
            "enabled": True,
            "target_per_label": target_count,
            "rows_injected": rows_injected,
            "labels_injected": labels_injected,
            "perturbations": list(self.cfg.masterlist_inject_perturbations),
            "perturbations_per_sample": perturbations_per_sample,
        }
        return pd.concat([train_df, inject_df], ignore_index=True)

    def get_masterlist_inject_metrics(self) -> dict[str, Any] | None:
        """Return metrics from the latest masterlist injection pass."""
        metrics = getattr(self, "_masterlist_inject_metrics", None)
        if metrics is None:
            return None
        return dict(metrics)

    def split_dataframe(self, df: pd.DataFrame) -> DataSplits:
        """Split a dataframe into train, validation, and test sets."""
        self._validate_split_sizes()
        self._validate_required_columns(df)
        self._validate_label_quality(df)

        if df.empty:
            raise ValueError("Cannot split an empty dataframe.")

        holdout_size = round(self.cfg.val_size + self.cfg.test_size, 10)
        empty_df = df.iloc[0:0].copy()
        data_seed = self.cfg.resolved_data_seed()

        if holdout_size == 0:
            shuffled = df.sample(frac=1.0, random_state=data_seed).reset_index(
                drop=True
            )
            return DataSplits(train=shuffled, val=empty_df.copy(), test=empty_df.copy())

        try:
            train_df, holdout_df = train_test_split(
                df,
                test_size=holdout_size,
                random_state=data_seed,
                shuffle=True,
            )
        except ValueError as exc:
            raise ValueError(
                "Unable to split data with the configured train/val/test sizes."
            ) from exc

        if self.cfg.val_size == 0:
            val_df = empty_df.copy()
            test_df = holdout_df
        elif self.cfg.test_size == 0:
            val_df = holdout_df
            test_df = empty_df.copy()
        else:
            test_ratio = self.cfg.test_size / holdout_size
            try:
                val_df, test_df = train_test_split(
                    holdout_df,
                    test_size=test_ratio,
                    random_state=data_seed,
                    shuffle=True,
                )
            except ValueError as exc:
                raise ValueError(
                    "Unable to split holdout data into validation and test sets."
                ) from exc

        return DataSplits(
            train=train_df.reset_index(drop=True),
            val=val_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
        )

    def _load_processed_dataset(self) -> pd.DataFrame:
        """Load processed data from CSV or Parquet."""
        path = self.processed_path
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported processed file format. Use .csv or .parquet.")
        self._validate_required_columns(df)
        self._validate_label_quality(df)
        return df

    def _sample_dataframe_by_fraction(
        self,
        df: pd.DataFrame,
        size: float,
        setting_name: str,
    ) -> pd.DataFrame:
        """Subsample dataframe rows according to a configured fractional size."""
        if size <= 0 or size > 1:
            raise ValueError(f"{setting_name} must be in the interval (0, 1].")
        if size == 1:
            return df.reset_index(drop=True)

        sample_count = max(1, int(round(len(df) * size)))
        if sample_count >= len(df):
            return df.reset_index(drop=True)

        data_seed = self.cfg.resolved_data_seed()
        return df.sample(
            n=sample_count, random_state=data_seed, replace=False
        ).reset_index(drop=True)

    def _apply_dataset_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subsample dataframe for pilot runs according to cfg.dataset_size."""
        return self._sample_dataframe_by_fraction(
            df=df,
            size=self.cfg.dataset_size,
            setting_name="dataset_size",
        )

    def _exclude_training_sources(
        self,
        df: pd.DataFrame,
        *,
        processed_source_ids: set[str],
    ) -> pd.DataFrame:
        """Remove configured non-training sources before train/val/test splitting."""
        excluded_sources = {
            source_id.strip()
            for source_id in self.cfg.train_excluded_source_ids
            if source_id.strip()
        }
        if not excluded_sources:
            return df
        if "source_id" not in df.columns:
            raise KeyError(
                "Processed data is missing required column 'source_id' for "
                "train_excluded_source_ids filtering."
            )

        missing_sources = sorted(excluded_sources.difference(processed_source_ids))
        if missing_sources:
            available_sources = ", ".join(
                sorted(source for source in processed_source_ids if source)
            )
            missing = ", ".join(missing_sources)
            raise ValueError(
                f"train_excluded_source_ids did not match processed rows: {missing}. "
                f"Available source_id values: {available_sources or '<none>'}."
            )

        source_ids = df["source_id"].fillna("").astype(str)
        excluded_mask = source_ids.isin(excluded_sources)
        if not excluded_mask.any():
            return df.reset_index(drop=True)

        filtered_df = df.loc[~excluded_mask].reset_index(drop=True)
        if filtered_df.empty:
            excluded = ", ".join(sorted(excluded_sources))
            raise ValueError(
                f"train_excluded_source_ids '{excluded}' would leave no rows for "
                "train/val/test splits."
            )
        _log_data_progress(
            "Excluded configured non-training sources: "
            f"sources={','.join(sorted(excluded_sources))}, "
            f"rows_removed={int(excluded_mask.sum())}, rows_remaining={len(filtered_df)}."
        )
        self._validate_label_quality(filtered_df)
        return filtered_df

    def _build_holdout_eval_dataframe(
        self, holdout_df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """Return optional sampled hold-out rows for during-training evaluation."""
        if self.cfg.hold_out_evaluate_per is None:
            return None
        return self._sample_dataframe_by_fraction(
            df=holdout_df,
            size=self.cfg.hold_out_evaluate_ratio,
            setting_name="hold_out_evaluate_ratio",
        )

    def _partition_hold_out_dataset(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Remove the configured source dataset from train/val/test preparation."""
        hold_out_dataset = self.cfg.hold_out_dataset
        if hold_out_dataset is None:
            return df, None
        if "source_id" not in df.columns:
            raise KeyError(
                "Processed data is missing required column 'source_id' for "
                "hold-out dataset selection."
            )

        source_ids = df["source_id"].fillna("").astype(str)
        holdout_mask = source_ids == hold_out_dataset
        if not holdout_mask.any():
            available_sources = ", ".join(sorted(source_ids[source_ids != ""].unique()))
            raise ValueError(
                f"hold_out_dataset '{hold_out_dataset}' did not match any processed rows. "
                f"Available source_id values: {available_sources or '<none>'}."
            )

        training_pool_df = df.loc[~holdout_mask].reset_index(drop=True)
        if training_pool_df.empty:
            raise ValueError(
                f"hold_out_dataset '{hold_out_dataset}' would leave no rows for train/val/test splits."
            )
        holdout_df = df.loc[holdout_mask].reset_index(drop=True)
        self._validate_label_quality(training_pool_df)
        self._validate_label_quality(holdout_df)
        return training_pool_df, holdout_df

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Ensure configured text/label columns exist in the dataframe."""
        required = {self.cfg.dataset_text_column, self.cfg.dataset_label_column}
        missing = required.difference(df.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise KeyError(
                f"Processed data is missing required columns: {missing_columns}."
            )

    def _validate_label_quality(self, df: pd.DataFrame) -> None:
        """Validate that labels contain trainable targets."""
        labels = df[self.cfg.dataset_label_column].fillna("").astype(str).str.strip()
        non_empty_labels = labels[labels != ""]
        if non_empty_labels.empty:
            raise ValueError(
                f"Processed data has no non-empty values in '{self.cfg.dataset_label_column}'."
            )
        if non_empty_labels.nunique() == 1:
            warnings.warn(
                (
                    f"Processed data contains only one unique label in "
                    f"'{self.cfg.dataset_label_column}'. Training may collapse to trivial loss."
                ),
                stacklevel=2,
            )

    def _validate_split_sizes(self) -> None:
        """Validate train, validation, and test split percentages."""
        split_sizes = {
            "train_size": self.cfg.train_size,
            "val_size": self.cfg.val_size,
            "test_size": self.cfg.test_size,
        }
        for name, size in split_sizes.items():
            if size < 0 or size > 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.cfg.train_size <= 0:
            raise ValueError("train_size must be greater than 0.")

        total = self.cfg.train_size + self.cfg.val_size + self.cfg.test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train_size, val_size, and test_size must sum to 1.0.")


__all__ = [
    "DataHandler",
    "DataSplits",
    "DatasetMapping",
    "MAPPING_REGISTRY",
    "PERTURBATION_REGISTRY",
    "_build_text",
    "_build_y",
    "build_and_save_processed_dataset",
    "build_processed_dataset",
    "load_dataset",
    "load_source_dataset",
    "manipulate_classes",
    "prepare_sequence_classification_dataset",
    "prepare_training_dataset",
    "resolve_training_frames",
    "save_processed_dataset",
    "upsample",
]
