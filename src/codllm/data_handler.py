import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4
import warnings

from filelock import FileLock, Timeout
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from codllm.config import Config, DataSourceConfig
import codllm.dataset_input as dataset_input
from codllm.path_utils import resolve_source_path
from codllm.preprocess import build_preprocess_fn

DatasetMapping = dataset_input.DatasetMapping
MAPPING_REGISTRY = dataset_input.MAPPING_REGISTRY
PERTURBATION_REGISTRY = dataset_input.PERTURBATION_REGISTRY
_build_text = dataset_input._build_text
_build_y = dataset_input._build_y
load_source_dataset = dataset_input.load_source_dataset
build_processed_dataset = dataset_input.build_processed_dataset
load_dataset = dataset_input.load_dataset

PROCESSING_METADATA_VERSION = 7
DEFAULT_PROCESSED_LOCK_TIMEOUT_SECONDS = 900.0
NON_PROCESSING_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "balance_strategy",
        "balance_target_quantile",
        "balance_perturbations",
        "balance_perturbations_per_sample",
        "balance_upsample_labels",
        "balance_upsample_perturbation_rate",
        "balance_upsample_inverse_power",
        "balance_upsample_budget_ratio",
        "balance_base_perturbation_rate",
    }
)


def _normalize_processing_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Drop metadata keys that do not affect processed dataset content."""
    normalized = dict(metadata)
    for key in NON_PROCESSING_METADATA_KEYS:
        normalized.pop(key, None)
    return normalized


@dataclass
class DataSplits:
    """Container for train, validation, and test dataframe splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class TokenizedSeq2SeqDataset(Dataset):
    """Simple torch dataset wrapper for tokenized seq2seq features."""

    def __init__(self, features: dict[str, list[Any]]) -> None:
        if not features:
            raise ValueError("Tokenized features must not be empty.")
        lengths = {len(values) for values in features.values()}
        if len(lengths) != 1:
            raise ValueError("Tokenized feature lengths are inconsistent.")
        self.features = features
        self.length = lengths.pop()

    def __len__(self) -> int:
        """Return number of rows."""
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one tokenized training sample."""
        return {key: values[idx] for key, values in self.features.items()}


def _tokenize_dataframe(
    cfg: Config,
    tokenizer: Any,
    dataframe: pd.DataFrame,
    target_max_length: int,
) -> TokenizedSeq2SeqDataset:
    """Convert a pandas dataframe into a tokenized seq2seq dataset."""
    if dataframe.empty:
        raise ValueError("Training dataframe is empty.")
    if cfg.dataset_text_column not in dataframe.columns:
        raise KeyError(
            f"Missing source column '{cfg.dataset_text_column}' in dataframe."
        )
    if cfg.dataset_label_column not in dataframe.columns:
        raise KeyError(
            f"Missing label column '{cfg.dataset_label_column}' in dataframe."
        )

    sources = dataframe[cfg.dataset_text_column].fillna("").astype(str).tolist()
    targets = dataframe[cfg.dataset_label_column].fillna("").astype(str).tolist()

    model_inputs = tokenizer(
        sources,
        max_length=cfg.max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=target_max_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return TokenizedSeq2SeqDataset(model_inputs)


def prepare_training_dataset(
    cfg: Config, tokenizer: Any, dataset: Any, target_max_length: int
) -> Any:
    """Convert a dataset into the tokenized format expected by Seq2SeqTrainer."""
    if isinstance(dataset, pd.DataFrame):
        return _tokenize_dataframe(cfg, tokenizer, dataset, target_max_length)

    if hasattr(dataset, "map") and hasattr(dataset, "column_names"):
        preprocess = build_preprocess_fn(
            cfg, tokenizer, max_target_length=target_max_length
        )
        return dataset.map(
            preprocess, batched=True, remove_columns=dataset.column_names
        )

    raise TypeError(
        "Unsupported dataset type. Expected pandas.DataFrame or a dataset with map/column_names."
    )


def _normalize_label_value(value: Any) -> str:
    """Normalize one label value to a stripped string."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _tokenize_dataframe_for_sequence_classification(
    cfg: Config,
    tokenizer: Any,
    dataframe: pd.DataFrame,
    label2id: Mapping[str, int],
) -> TokenizedSeq2SeqDataset:
    """Convert a pandas dataframe into a tokenized sequence-classification dataset."""
    if dataframe.empty:
        raise ValueError("Training dataframe is empty.")
    if cfg.dataset_text_column not in dataframe.columns:
        raise KeyError(
            f"Missing source column '{cfg.dataset_text_column}' in dataframe."
        )
    if cfg.dataset_label_column not in dataframe.columns:
        raise KeyError(
            f"Missing label column '{cfg.dataset_label_column}' in dataframe."
        )

    sources = dataframe[cfg.dataset_text_column].fillna("").astype(str).tolist()
    labels = [_normalize_label_value(value) for value in dataframe[cfg.dataset_label_column]]

    unique_labels = set(labels)
    unknown_labels = sorted(label for label in unique_labels if label not in label2id)
    if unknown_labels:
        preview = ", ".join(unknown_labels[:10])
        raise ValueError(
            "Found labels missing from classifier label space: "
            f"{preview}."
        )

    model_inputs = tokenizer(
        sources,
        max_length=cfg.max_source_length,
        truncation=True,
    )
    model_inputs["labels"] = [int(label2id[label]) for label in labels]
    return TokenizedSeq2SeqDataset(model_inputs)


def prepare_sequence_classification_dataset(
    cfg: Config,
    tokenizer: Any,
    dataset: Any,
    label2id: Mapping[str, int],
) -> Any:
    """Convert a dataset into tokenized format expected by Trainer classification."""
    if isinstance(dataset, pd.DataFrame):
        return _tokenize_dataframe_for_sequence_classification(
            cfg=cfg,
            tokenizer=tokenizer,
            dataframe=dataset,
            label2id=label2id,
        )

    if hasattr(dataset, "map") and hasattr(dataset, "column_names"):

        def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
            if cfg.dataset_text_column not in batch:
                raise KeyError(
                    f"Missing source column '{cfg.dataset_text_column}' in batch."
                )
            if cfg.dataset_label_column not in batch:
                raise KeyError(
                    f"Missing label column '{cfg.dataset_label_column}' in batch."
                )
            sources = batch[cfg.dataset_text_column]
            raw_labels = batch[cfg.dataset_label_column]
            normalized_labels = [_normalize_label_value(value) for value in raw_labels]
            unknown_labels = sorted(
                label for label in set(normalized_labels) if label not in label2id
            )
            if unknown_labels:
                preview = ", ".join(unknown_labels[:10])
                raise ValueError(
                    "Found labels missing from classifier label space: "
                    f"{preview}."
                )
            model_inputs = tokenizer(
                sources,
                max_length=cfg.max_source_length,
                truncation=True,
            )
            model_inputs["labels"] = [int(label2id[label]) for label in normalized_labels]
            return model_inputs

        return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    raise TypeError(
        "Unsupported dataset type. Expected pandas.DataFrame or a dataset with map/column_names."
    )


def resolve_training_frames(
    splits: DataSplits,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return train and optional validation dataframes from split output."""
    if splits.val.empty:
        return splits.train, None
    return splits.train, splits.val


PerturbationFn = Callable[[str], str]


def _resolve_perturbation_functions(
    perturbation_names: Sequence[str] | None,
) -> list[PerturbationFn]:
    """Resolve configured perturbation names into callable functions."""
    names = (
        list(PERTURBATION_REGISTRY.keys())
        if perturbation_names is None
        else list(perturbation_names)
    )
    perturbation_fns: list[PerturbationFn] = []
    for name in names:
        if name not in PERTURBATION_REGISTRY:
            available = ", ".join(sorted(PERTURBATION_REGISTRY.keys()))
            raise ValueError(f"Unknown perturbation '{name}'. Available: {available}.")
        perturbation_fns.append(PERTURBATION_REGISTRY[name])
    return perturbation_fns


def _apply_perturbation_with_seed(
    perturbation_fn: PerturbationFn, text: str, seed: int
) -> str:
    """Apply one perturbation deterministically without leaking global RNG state."""
    previous_state = random.getstate()
    try:
        random.seed(seed)
        return perturbation_fn(text)
    finally:
        random.setstate(previous_state)


def _perturb_cod_segment(
    cfg: Config,
    text: str,
    perturbation_fns: Sequence[PerturbationFn],
    perturbations_per_sample: int,
    rng: random.Random,
) -> str:
    """Perturb only the cod: segment inside a configured training text."""
    if perturbations_per_sample < 1 or not perturbation_fns:
        return text

    parts = text.split(cfg.text_field_separator) if cfg.text_field_separator else [text]
    cod_idx = next(
        (idx for idx, part in enumerate(parts) if part.startswith("cod: ")), None
    )
    if cod_idx is None:
        return text

    cod_value = parts[cod_idx][len("cod: ") :]
    for _ in range(perturbations_per_sample):
        perturbation_fn = rng.choice(perturbation_fns)
        cod_value = _apply_perturbation_with_seed(
            perturbation_fn,
            cod_value,
            seed=rng.randint(0, 2_147_483_647),
        )

    parts[cod_idx] = f"cod: {cod_value}"
    return (
        cfg.text_field_separator.join(parts) if cfg.text_field_separator else parts[0]
    )


def _quantile_target_count(class_counts: pd.Series, target_quantile: float) -> int:
    """Compute the class-count target used for quantile-based balancing."""
    if target_quantile < 0 or target_quantile > 1:
        raise ValueError("target_quantile must be between 0 and 1.")
    if class_counts.empty:
        return 0
    return int(class_counts.quantile(target_quantile))


def select_upsample_targets(
    df: pd.DataFrame,
    label_column: str,
    target_quantile: float = 0.5,
    candidate_labels: Sequence[str] | None = None,
    inverse_power: float = 0.5,
    budget_ratio: float = 0.1,
) -> dict[Any, int]:
    """Select per-class target counts for upsampling."""
    if inverse_power <= 0 or inverse_power > 1:
        raise ValueError("inverse_power must be in the interval (0, 1].")
    if budget_ratio < 0 or budget_ratio > 1:
        raise ValueError("budget_ratio must be between 0 and 1.")

    class_counts = df[label_column].value_counts()
    q_count = _quantile_target_count(class_counts, target_quantile)
    if q_count <= 0 or budget_ratio == 0:
        return {}

    selected_labels: list[Any] = []
    if candidate_labels is None:
        for label, count in class_counts.items():
            if count < q_count:
                selected_labels.append(label)
    else:
        available_labels = set(class_counts.index.tolist())
        for label in candidate_labels:
            if label not in available_labels:
                continue
            if int(class_counts[label]) < q_count:
                selected_labels.append(label)

    if not selected_labels:
        return {}

    total_rows = int(len(df))
    budget = int(round(budget_ratio * total_rows))
    if budget <= 0:
        return {}

    per_label_pressure: dict[Any, float] = {}
    pressure_denominator = 0.0
    for label in selected_labels:
        current_count = int(class_counts[label])
        pressure = (q_count / current_count) ** inverse_power - 1.0
        per_label_pressure[label] = pressure
        pressure_denominator += current_count * pressure

    if pressure_denominator <= 0:
        return {}

    lambda_scale = min(1.0, budget / pressure_denominator)
    selected_targets: dict[Any, int] = {}
    for label in selected_labels:
        current_count = int(class_counts[label])
        pressure = per_label_pressure[label]
        target_float = current_count * (1.0 + lambda_scale * pressure)
        target_count = int(round(target_float))
        target_count = min(target_count, q_count)
        target_count = max(current_count, target_count)
        if target_count > current_count:
            selected_targets[label] = target_count
    return selected_targets


def _sample_upsample_rows(
    class_rows: pd.DataFrame, needed: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Sample class rows with shuffled cycles to avoid repeatedly duplicating one row."""
    if needed <= 0 or class_rows.empty:
        return []

    source_rows = class_rows.to_dict(orient="records")
    sampled_rows: list[dict[str, Any]] = []
    while len(sampled_rows) < needed:
        shuffled_rows = list(source_rows)
        rng.shuffle(shuffled_rows)
        take = min(needed - len(sampled_rows), len(shuffled_rows))
        sampled_rows.extend(dict(row) for row in shuffled_rows[:take])
    return sampled_rows


def upsample(
    df: pd.DataFrame,
    label_column: str,
    target_counts: Mapping[Any, int],
    seed: int = 42,
) -> pd.DataFrame:
    """Upsample classes to target counts by appending sampled rows without perturbation."""
    if not target_counts:
        return df

    rng = random.Random(seed)
    class_counts = df[label_column].value_counts()

    synthetic_rows: list[dict[str, Any]] = []
    for label, target_count in target_counts.items():
        if target_count <= 0:
            continue

        current_count = int(class_counts.get(label, 0))
        if current_count == 0 or current_count >= target_count:
            continue

        class_rows = df[df[label_column] == label]
        needed = target_count - current_count
        synthetic_rows.extend(_sample_upsample_rows(class_rows, needed=needed, rng=rng))

    if not synthetic_rows:
        return df

    synthetic_df = pd.DataFrame(synthetic_rows, columns=df.columns)
    return pd.concat([df, synthetic_df], ignore_index=True)


def upsample_minority_classes(
    df: pd.DataFrame,
    label_column: str,
    target_quantile: float = 0.5,
    inverse_power: float = 0.5,
    budget_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Upsample classes selected from quantile-based minority detection."""
    target_counts = select_upsample_targets(
        df=df,
        label_column=label_column,
        target_quantile=target_quantile,
        inverse_power=inverse_power,
        budget_ratio=budget_ratio,
    )
    return upsample(
        df=df,
        label_column=label_column,
        target_counts=target_counts,
        seed=seed,
    )


def manipulate_classes(
    cfg: Config,
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    target_labels: Sequence[str] | None = None,
    perturbation_names: Sequence[str] | None = None,
    perturbations_per_sample: int = 1,
    sample_fraction: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Perturb selected rows in-place without changing class counts."""
    if sample_fraction <= 0 or sample_fraction > 1:
        raise ValueError("sample_fraction must be in the interval (0, 1].")
    if target_labels is not None and not target_labels:
        return df

    perturbation_fns = _resolve_perturbation_functions(perturbation_names)
    if target_labels is None:
        selected_indices = df.index.tolist()
    else:
        selected_mask = df[label_column].isin(set(target_labels))
        selected_indices = df.index[selected_mask].tolist()
    if not selected_indices:
        return df

    sample_size = max(1, int(round(len(selected_indices) * sample_fraction)))
    sample_size = min(sample_size, len(selected_indices))
    rng = random.Random(seed)
    indices_to_perturb = rng.sample(selected_indices, sample_size)

    manipulated_df = df.copy()
    for index in indices_to_perturb:
        text = str(manipulated_df.at[index, text_column])
        manipulated_df.at[index, text_column] = _perturb_cod_segment(
            cfg=cfg,
            text=text,
            perturbation_fns=perturbation_fns,
            perturbations_per_sample=perturbations_per_sample,
            rng=rng,
        )
    return manipulated_df


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
            "max_label_count": self.cfg.max_label_count,
            "label_separator": self.cfg.label_separator,
            "text_field_separator": self.cfg.text_field_separator,
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
                    processed_df = build_processed_dataset(
                        cfg=self.cfg, mapping_registry=self.mapping_registry
                    )
                    save_processed_dataset(processed_df, str(self.processed_path))
                    self._write_processing_metadata(self._build_processing_metadata())
                    return processed_df
                return self._load_processed_dataset()
        except Timeout as exc:
            raise TimeoutError(
                f"Timed out waiting for processed-data lock '{lock_path}'. "
                "Set CODLLM_PROCESSED_LOCK_TIMEOUT_SECONDS to a larger value."
            ) from exc

    def get_splits(self, force_reprocess: bool = False) -> DataSplits:
        """Return train/validation/test splits from processed data."""
        processed_df = self.ensure_processed(force_reprocess=force_reprocess)
        sampled_df = self._apply_dataset_size(processed_df)
        splits = self.split_dataframe(sampled_df)
        if not splits.train.empty:
            splits.train = self._apply_balance_policy(splits.train)
        return splits

    def get_pretraining_train_dataframe(self) -> pd.DataFrame | None:
        """Return optional masterlist dataframe used for pretraining."""
        if not self.cfg.pretrain_enabled:
            return None

        pretrain_df = self._load_pretraining_source()
        if pretrain_df.empty:
            raise ValueError("Pretraining dataframe is empty.")
        return pretrain_df

    def get_masterlist_label_vocabulary(self) -> list[str]:
        """Return sorted unique ICD10h label values from the configured masterlist."""
        masterlist_df = self._load_pretraining_source()
        label_column = self.cfg.dataset_label_column
        if label_column not in masterlist_df.columns:
            raise KeyError(
                f"Masterlist dataframe is missing label column '{label_column}'."
            )
        labels = (
            masterlist_df[label_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        unique_labels = sorted(label for label in labels.unique().tolist() if label)
        if not unique_labels:
            raise ValueError("Masterlist label vocabulary is empty.")
        return unique_labels

    def _load_pretraining_source(self) -> pd.DataFrame:
        """Load pretraining rows from the configured ICD10h masterlist source."""
        source = DataSourceConfig(
            source_id="masterlist_pretrain",
            path=self.cfg.pretrain_masterlist_path,
            mapping_id="masterlist",
            sheet_name=self.cfg.pretrain_masterlist_sheet_name,
            enabled=True,
        )
        if source.mapping_id not in self.mapping_registry:
            raise KeyError(
                f"Unknown mapping_id '{source.mapping_id}' for pretraining source."
            )
        mapping = self.mapping_registry[source.mapping_id]
        pretrain_df = load_source_dataset(
            source=source,
            mapping=mapping,
            training_input=self.cfg.training_input,
            max_labels=self.cfg.max_label_count,
            label_separator=self.cfg.label_separator,
            text_field_separator=self.cfg.text_field_separator,
            data_raw_dir=self.cfg.data_raw_dir,
            text_column=self.cfg.dataset_text_column,
            label_column=self.cfg.dataset_label_column,
        )
        self._validate_required_columns(pretrain_df)
        self._validate_label_quality(pretrain_df)
        return pretrain_df.reset_index(drop=True)

    def _apply_balance_policy(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Apply optional upsampling and manipulation rules to the training split."""
        balanced_train_df = train_df

        if self.cfg.balance_strategy == "upsample":
            upsample_candidates = self.cfg.balance_upsample_labels or None
            target_counts = select_upsample_targets(
                df=train_df,
                label_column=self.cfg.dataset_label_column,
                target_quantile=self.cfg.balance_target_quantile,
                candidate_labels=upsample_candidates,
                inverse_power=self.cfg.balance_upsample_inverse_power,
                budget_ratio=self.cfg.balance_upsample_budget_ratio,
            )
            balanced_train_df = upsample(
                df=balanced_train_df,
                label_column=self.cfg.dataset_label_column,
                target_counts=target_counts,
                seed=self.cfg.resolved_data_seed(),
            )

        if self.cfg.balance_base_perturbation_rate > 0:
            balanced_train_df = manipulate_classes(
                cfg=self.cfg,
                label_column=self.cfg.dataset_label_column,
                df=balanced_train_df,
                text_column=self.cfg.dataset_text_column,
                target_labels=None,
                perturbation_names=self.cfg.balance_perturbations,
                perturbations_per_sample=self.cfg.balance_perturbations_per_sample,
                sample_fraction=self.cfg.balance_base_perturbation_rate,
                seed=self.cfg.resolved_data_seed() + 1,
            )

        return balanced_train_df

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
        self, df: pd.DataFrame, size: float, setting_name: str
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


def save_processed_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Persist processed data as CSV or Parquet based on file extension."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    temporary_output = output.parent / f".{output.name}.{uuid4().hex}.tmp{suffix}"
    try:
        if suffix == ".csv":
            df.to_csv(temporary_output, index=False)
        elif suffix == ".parquet":
            df.to_parquet(temporary_output, index=False)
        else:
            raise ValueError("Unsupported processed file format. Use .csv or .parquet.")
        temporary_output.replace(output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()


def build_and_save_processed_dataset(
    cfg: Config,
    mapping_registry: Mapping[str, DatasetMapping] | None = None,
) -> pd.DataFrame:
    """Build and persist processed data using config output settings."""
    processed_df = build_processed_dataset(cfg, mapping_registry=mapping_registry)
    output_path = str(Path(cfg.data_processed_dir) / cfg.processed_filename)
    save_processed_dataset(processed_df, output_path)
    return processed_df
