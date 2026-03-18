import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast
from uuid import uuid4
import warnings

from filelock import FileLock, Timeout
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from codllm.config import (
    Config,
    DataSourceConfig,
    TrainingInput,
)
from codllm.data_augmentation import (
    accent_random_vowel,
    delete_random_char,
    insert_random_whitespace,
    qwerty_misspell,
    swap_adjacent_chars,
)
from codllm.path_utils import resolve_source_path
from codllm.preprocess import build_preprocess_fn

UNKNOWN_VALUE = "unknown"
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


def _processed_columns(text_column: str, label_column: str) -> list[str]:
    """Return canonical processed-data column order for configured text/label names."""
    return [
        "source_id",
        "record_id",
        "source_path",
        text_column,
        "y_codes",
        label_column,
    ]


def _normalize_processing_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Drop metadata keys that do not affect processed dataset content."""
    normalized = dict(metadata)
    for key in NON_PROCESSING_METADATA_KEYS:
        normalized.pop(key, None)
    return normalized


@dataclass
class DatasetMapping:
    """Describe how to map one source dataset into the canonical training schema."""

    text_col: int
    single_code_col: int
    multi_code_cols: list[int] = field(default_factory=list)
    sex_col: int | None = None
    sex_map: dict[str, str] = field(default_factory=dict)
    age_col: int | None = None
    record_id_col: int | None = None
    skip_rows: list[int] = field(default_factory=list)


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


def resolve_training_frames(
    splits: DataSplits,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return train and optional validation dataframes from split output."""
    if splits.val.empty:
        return splits.train, None
    return splits.train, splits.val


BELGIUM_MAPPING = DatasetMapping(
    text_col=5,
    single_code_col=11,
    multi_code_cols=[12, 13, 14, 15, 16],
    sex_col=2,
    sex_map={"1": "male", "2": "female"},
    age_col=3,
    record_id_col=0,
    skip_rows=[0],
)

AMSTERDAM_MAPPING = DatasetMapping(
    text_col=5,
    single_code_col=7,
    multi_code_cols=[7, 9, 11, 13, 15, 17],
    sex_col=2,
    sex_map={
        "man": "male",
        "vrouw": "female",
        "m": "male",
        "v": "female",
        "1": "male",
        "2": "female",
    },
    age_col=3,
    record_id_col=0,
)

MAPPING_REGISTRY: dict[str, DatasetMapping] = {
    "belgium": BELGIUM_MAPPING,
    "amsterdam": AMSTERDAM_MAPPING,
}

PERTURBATION_REGISTRY: dict[str, Any] = {
    "swap_adjacent_chars": swap_adjacent_chars,
    "delete_random_char": delete_random_char,
    "insert_random_whitespace": insert_random_whitespace,
    "accent_random_vowel": accent_random_vowel,
    "qwerty_misspell": qwerty_misspell,
}


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

    def _apply_dataset_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subsample dataframe for pilot runs according to cfg.dataset_size."""
        size = self.cfg.dataset_size
        if size <= 0 or size > 1:
            raise ValueError("dataset_size must be in the interval (0, 1].")
        if size == 1:
            return df.reset_index(drop=True)

        sample_count = max(1, int(round(len(df) * size)))
        if sample_count >= len(df):
            return df.reset_index(drop=True)

        data_seed = self.cfg.resolved_data_seed()
        return df.sample(
            n=sample_count, random_state=data_seed, replace=False
        ).reset_index(drop=True)

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


def _normalize_training_input(training_input: Sequence[str]) -> list[TrainingInput]:
    """Validate and normalize requested training input fields."""
    normalized: list[TrainingInput] = []
    supported_inputs = Config.SUPPORTED_TRAINING_INPUTS
    for feature in training_input:
        cleaned = feature.strip().lower()
        if cleaned not in supported_inputs:
            supported = ", ".join(supported_inputs)
            raise ValueError(
                f"Unsupported training input '{feature}'. Supported values are: {supported}."
            )
        normalized_feature = cast(TrainingInput, cleaned)
        if normalized_feature not in normalized:
            normalized.append(normalized_feature)
    if not normalized:
        raise ValueError("training_input must contain at least one field.")
    return normalized


def _get(row: pd.Series, col: int) -> str | None:
    """Get a value from a row by positional index."""
    if col < 0 or col >= len(row):
        return None
    value = row.iloc[col]
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _format_age(raw_age: str | None) -> str:
    """Normalize age into a compact numeric string."""
    if raw_age is None:
        return UNKNOWN_VALUE
    try:
        numeric_age = round(float(raw_age), 2)
    except ValueError:
        return UNKNOWN_VALUE
    if numeric_age.is_integer():
        return str(int(numeric_age))
    return f"{numeric_age:.2f}".rstrip("0").rstrip(".")


def _format_sex(raw_sex: str | None, sex_map: Mapping[str, str]) -> str:
    """Map raw sex values into canonical values."""
    if raw_sex is None:
        return UNKNOWN_VALUE
    raw = raw_sex.strip()
    normalized_raw = raw.lower()
    if raw in sex_map:
        return sex_map[raw]
    if normalized_raw in sex_map:
        return sex_map[normalized_raw]
    return UNKNOWN_VALUE


def _build_text(
    row: pd.Series,
    mapping: DatasetMapping,
    training_input: Sequence[str],
    field_separator: str = Config.DEFAULT_TEXT_FIELD_SEPARATOR,
) -> str:
    """Build text input from configured training input fields."""
    normalized_training_input = _normalize_training_input(training_input)
    parts: list[str] = []
    for feature in normalized_training_input:
        if feature == "cod":
            cod_text = _get(row, mapping.text_col) or UNKNOWN_VALUE
            parts.append(f"cod: {cod_text}")
        elif feature == "age":
            raw_age = (
                _get(row, mapping.age_col) if mapping.age_col is not None else None
            )
            parts.append(f"age: {_format_age(raw_age)}")
        else:
            raw_sex = (
                _get(row, mapping.sex_col) if mapping.sex_col is not None else None
            )
            parts.append(f"sex: {_format_sex(raw_sex, mapping.sex_map)}")
    return field_separator.join(parts)


def _collect_codes(row: pd.Series, mapping: DatasetMapping) -> list[str]:
    """Collect code values from multi-code columns first, then single-code fallback."""
    codes: list[str] = []
    seen_codes: set[str] = set()
    for col in mapping.multi_code_cols:
        code = _get(row, col)
        if code and code not in seen_codes:
            codes.append(code)
            seen_codes.add(code)
    if not codes:
        single_code = _get(row, mapping.single_code_col)
        if single_code:
            codes.append(single_code)
    return codes


def _build_y(row: pd.Series, mapping: DatasetMapping) -> list[str]:
    """Build complete target code list for one source row."""
    return _collect_codes(row, mapping)


def _build_label(
    codes: list[str], separator: str = Config.DEFAULT_LABEL_SEPARATOR
) -> str:
    """Convert code labels into a single seq2seq target string."""
    return separator.join(codes)


def _detect_file_type(path: Path, source: DataSourceConfig) -> str:
    """Determine file type from source config or file extension."""
    if source.file_type is not None:
        return source.file_type.lower().lstrip(".")
    return path.suffix.lower().lstrip(".")


def _read_raw_dataframe(path: Path, source: DataSourceConfig) -> pd.DataFrame:
    """Read one raw source file into a dataframe."""
    file_type = _detect_file_type(path, source)
    if file_type == "csv":
        return pd.read_csv(path, header=source.header, dtype=str, sep=source.sep)
    if file_type in {"xlsx", "xls"}:
        return pd.read_excel(
            path, header=source.header, dtype=str, sheet_name=source.sheet_name
        )
    raise ValueError(
        f"Unsupported file type '{file_type}' for source '{source.source_id}'."
    )


def load_source_dataset(
    source: DataSourceConfig,
    mapping: DatasetMapping,
    training_input: Sequence[str],
    max_labels: int = 1,
    label_separator: str = Config.DEFAULT_LABEL_SEPARATOR,
    text_field_separator: str = Config.DEFAULT_TEXT_FIELD_SEPARATOR,
    data_raw_dir: str = Config.DEFAULT_DATA_RAW_DIR,
    text_column: str = Config.DEFAULT_DATASET_TEXT_COLUMN,
    label_column: str = Config.DEFAULT_DATASET_LABEL_COLUMN,
    drop_missing_label: bool = True,
) -> pd.DataFrame:
    """Load and process one source dataset into the canonical schema."""
    if max_labels < 1:
        raise ValueError("max_labels must be at least 1.")
    normalized_training_input = _normalize_training_input(training_input)
    source_path = resolve_source_path(source.path, data_raw_dir)
    raw_df = _read_raw_dataframe(source_path, source)
    combined_skip_rows = sorted(set(mapping.skip_rows + source.skip_rows))
    if combined_skip_rows:
        raw_df = raw_df.drop(index=combined_skip_rows, errors="ignore").reset_index(
            drop=True
        )

    result = pd.DataFrame()
    result["source_id"] = [source.source_id] * len(raw_df)
    if mapping.record_id_col is None:
        result["record_id"] = [
            f"{source.source_id}:{idx}" for idx in range(len(raw_df))
        ]
    else:
        extracted_ids = raw_df.apply(
            lambda row: _get(row, mapping.record_id_col), axis=1
        )
        result["record_id"] = [
            record_id if record_id is not None else f"{source.source_id}:{idx}"
            for idx, record_id in enumerate(extracted_ids.tolist())
        ]
    result["source_path"] = [str(source_path)] * len(raw_df)
    result[text_column] = raw_df.apply(
        lambda row: _build_text(
            row,
            mapping,
            normalized_training_input,
            field_separator=text_field_separator,
        ),
        axis=1,
    )
    result["y_codes"] = raw_df.apply(lambda row: _build_y(row, mapping), axis=1)
    result["label_count"] = result["y_codes"].apply(len)
    result = result[result["label_count"] <= max_labels].reset_index(drop=True)
    result = result.drop(columns=["label_count"])
    result[label_column] = result["y_codes"].apply(
        lambda codes: _build_label(codes, separator=label_separator)
    )
    if drop_missing_label:
        result = result[result[label_column] != ""].reset_index(drop=True)
    return result


def build_processed_dataset(
    cfg: Config,
    mapping_registry: Mapping[str, DatasetMapping] | None = None,
) -> pd.DataFrame:
    """Build one processed dataframe from all configured raw data sources."""
    registry = dict(MAPPING_REGISTRY)
    if mapping_registry is not None:
        registry.update(mapping_registry)

    processed_frames: list[pd.DataFrame] = []
    for source in cfg.data_sources:
        if not source.enabled:
            continue
        if source.mapping_id not in registry:
            raise KeyError(
                f"Unknown mapping_id '{source.mapping_id}' for source '{source.source_id}'."
            )
        mapping = registry[source.mapping_id]
        processed_frames.append(
            load_source_dataset(
                source=source,
                mapping=mapping,
                training_input=cfg.training_input,
                max_labels=cfg.max_label_count,
                label_separator=cfg.label_separator,
                text_field_separator=cfg.text_field_separator,
                data_raw_dir=cfg.data_raw_dir,
                text_column=cfg.dataset_text_column,
                label_column=cfg.dataset_label_column,
            )
        )

    if not processed_frames:
        return pd.DataFrame(
            columns=_processed_columns(
                cfg.dataset_text_column,
                cfg.dataset_label_column,
            )
        )
    return pd.concat(processed_frames, ignore_index=True)


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


def load_dataset(
    path: str,
    mapping: DatasetMapping,
    training_input: Sequence[str] | None = None,
    max_labels: int = 1,
    sep: str = ",",
) -> pd.DataFrame:
    """Load one dataset into legacy text/y output format."""
    source = DataSourceConfig(
        source_id="inline_source",
        path=path,
        mapping_id="inline_mapping",
        sep=sep,
    )
    processed = load_source_dataset(
        source=source,
        mapping=mapping,
        training_input=training_input or list(Config.SUPPORTED_TRAINING_INPUTS),
        max_labels=max_labels,
        data_raw_dir="",
        text_column=Config.DEFAULT_DATASET_TEXT_COLUMN,
        label_column=Config.DEFAULT_DATASET_LABEL_COLUMN,
        drop_missing_label=False,
    )
    return pd.DataFrame(
        {
            Config.DEFAULT_DATASET_TEXT_COLUMN: processed[
                Config.DEFAULT_DATASET_TEXT_COLUMN
            ],
            "y": processed["y_codes"],
        }
    )
