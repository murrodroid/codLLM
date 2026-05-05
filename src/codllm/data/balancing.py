import math
import random
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from codllm.config import Config
from codllm.input import PERTURBATION_REGISTRY

PerturbationFn = Callable[[str], str]
RNG_AWARE_PERTURBATION_NAMES = frozenset(
    {
        "swap_adjacent_chars",
        "delete_random_char",
        "insert_random_whitespace",
        "accent_random_vowel",
        "qwerty_misspell",
    }
)


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
    perturbation_fn: PerturbationFn,
    text: str,
    seed: int,
) -> str:
    """Apply one perturbation deterministically without leaking global RNG state."""
    previous_state = random.getstate()
    try:
        random.seed(seed)
        return perturbation_fn(text)
    finally:
        random.setstate(previous_state)


def _apply_perturbation_with_rng(
    perturbation_fn: PerturbationFn,
    text: str,
    rng: random.Random,
) -> str:
    """Apply one perturbation using the local RNG when the function supports it."""
    if (
        getattr(perturbation_fn, "__module__", "") == "codllm.data.augmentation"
        and getattr(perturbation_fn, "__name__", "") in RNG_AWARE_PERTURBATION_NAMES
    ):
        return perturbation_fn(text, rng=rng)
    return _apply_perturbation_with_seed(
        perturbation_fn,
        text,
        seed=rng.randint(0, 2_147_483_647),
    )


def _perturb_cod_segment(
    cfg: Config,
    text: str,
    perturbation_fns: Sequence[PerturbationFn],
    rng: random.Random,
    perturbations_per_sample: int | None = None,
    perturbation_mean: float | None = None,
    perturbation_variance: float = 0.0,
) -> str:
    """Perturb only the configured COD segment inside a training text."""
    if not perturbation_fns:
        return text

    if list(cfg.training_input) == ["cod"]:
        cod_value = text
        if perturbations_per_sample is not None:
            perturbation_count = perturbations_per_sample
        else:
            perturbation_count = _sample_perturbation_count(
                text=cod_value,
                perturbation_mean=0.0
                if perturbation_mean is None
                else perturbation_mean,
                perturbation_variance=perturbation_variance,
                rng=rng,
            )
        if perturbation_count < 1:
            return text
        for _ in range(perturbation_count):
            perturbation_fn = rng.choice(perturbation_fns)
            cod_value = _apply_perturbation_with_rng(
                perturbation_fn,
                cod_value,
                rng=rng,
            )
        return cod_value

    cod_prefix = cfg.input_field_prefix("cod")
    parts = text.split(cfg.text_field_separator) if cfg.text_field_separator else [text]
    cod_idx = next(
        (idx for idx, part in enumerate(parts) if part.strip().startswith(cod_prefix)),
        None,
    )
    if cod_idx is None:
        return text

    cod_value = parts[cod_idx].strip()[len(cod_prefix) :]
    if perturbations_per_sample is not None:
        perturbation_count = perturbations_per_sample
    else:
        perturbation_count = _sample_perturbation_count(
            text=cod_value,
            perturbation_mean=0.0 if perturbation_mean is None else perturbation_mean,
            perturbation_variance=perturbation_variance,
            rng=rng,
        )
    if perturbation_count < 1:
        return text

    for _ in range(perturbation_count):
        perturbation_fn = rng.choice(perturbation_fns)
        cod_value = _apply_perturbation_with_rng(
            perturbation_fn,
            cod_value,
            rng=rng,
        )

    parts[cod_idx] = f"{cod_prefix}{cod_value}"
    return (
        cfg.text_field_separator.join(parts) if cfg.text_field_separator else parts[0]
    )


def _sample_perturbation_count(
    text: str,
    perturbation_mean: float,
    perturbation_variance: float,
    rng: random.Random,
) -> int:
    """Sample a length-scaled perturbation count for one text segment."""
    if perturbation_mean < 0:
        raise ValueError("perturbation_mean must be non-negative.")
    if perturbation_variance < 0:
        raise ValueError("perturbation_variance must be non-negative.")
    if perturbation_mean == 0 and perturbation_variance == 0:
        return 0

    text_length = max(1, len(text))
    expected_count = perturbation_mean * text_length
    if perturbation_variance == 0:
        sampled_count = int(round(expected_count))
    else:
        standard_deviation = math.sqrt(perturbation_variance * text_length)
        sampled_count = int(round(rng.gauss(expected_count, standard_deviation)))
    sampled_count = max(1, sampled_count)
    return min(sampled_count, text_length)


def _contains_cod_segment(cfg: Config, text: str) -> bool:
    """Return whether a training text contains the configured COD segment."""
    if list(cfg.training_input) == ["cod"]:
        return text.strip() != ""
    cod_prefix = cfg.input_field_prefix("cod")
    parts = text.split(cfg.text_field_separator) if cfg.text_field_separator else [text]
    return any(part.strip().startswith(cod_prefix) for part in parts)


def select_floor_upsample_targets(
    df: pd.DataFrame,
    label_column: str,
    floor: int = 10,
    decay: float = 0.1,
) -> dict[Any, int]:
    """Select per-class target counts by enforcing a minimum sample floor."""
    if floor < 0:
        raise ValueError("floor must be non-negative.")
    if floor == 0:
        return {}
    if decay < 0 or decay > 1:
        raise ValueError("decay must be between 0 and 1.")

    class_counts = df[label_column].value_counts()
    targets: dict[Any, int] = {}
    for label, count in class_counts.items():
        original = int(count)
        if original >= floor:
            continue
        t = 1.0 - (original - 1) / max(floor - 1, 1)
        scale = 1.0 - decay * (math.log1p(t) / math.log(2))
        target = max(original, round(floor * scale))
        if target > original:
            targets[label] = target
    return targets


def _sample_upsample_rows(
    class_rows: pd.DataFrame,
    needed: int,
    rng: random.Random,
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
    cfg: Config | None = None,
    text_column: str | None = None,
    perturbation_fns: Sequence[Any] | None = None,
    perturbation_mean: float = 0.05,
    perturbation_variance: float = 0.0,
    text_field_separator: str = " | ",
) -> pd.DataFrame:
    """Upsample classes to target counts, perturbing synthetic rows proportionally."""
    del text_field_separator
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
        new_rows = _sample_upsample_rows(class_rows, needed=needed, rng=rng)

        if text_column and perturbation_fns and new_rows:
            perturb_rate = 1.0 - (current_count / target_count)
            for row in new_rows:
                if rng.random() < perturb_rate:
                    text = str(row[text_column])
                    if cfg is not None:
                        text = _perturb_cod_segment(
                            cfg=cfg,
                            text=text,
                            perturbation_fns=perturbation_fns,
                            perturbation_mean=perturbation_mean,
                            perturbation_variance=perturbation_variance,
                            rng=rng,
                        )
                    else:
                        perturbation_count = _sample_perturbation_count(
                            text=text,
                            perturbation_mean=perturbation_mean,
                            perturbation_variance=perturbation_variance,
                            rng=rng,
                        )
                        for _ in range(perturbation_count):
                            fn = rng.choice(perturbation_fns)
                            text = _apply_perturbation_with_rng(
                                fn,
                                text,
                                rng=rng,
                            )
                    row[text_column] = text

        synthetic_rows.extend(new_rows)

    if not synthetic_rows:
        return df

    synthetic_df = pd.DataFrame(synthetic_rows, columns=df.columns)
    return pd.concat([df, synthetic_df], ignore_index=True)


def manipulate_classes(
    cfg: Config,
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    target_labels: Sequence[str] | None = None,
    perturbation_names: Sequence[str] | None = None,
    perturbation_mean: float = 0.05,
    perturbation_variance: float = 0.0,
    sample_fraction: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Perturb selected rows in-place without changing class counts."""
    if sample_fraction <= 0 or sample_fraction > 1:
        raise ValueError("sample_fraction must be in the interval (0, 1].")
    if target_labels is not None and not target_labels:
        return df

    perturbation_fns = _resolve_perturbation_functions(perturbation_names)
    if not perturbation_fns:
        return df
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
    perturbed_texts = [
        _perturb_cod_segment(
            cfg=cfg,
            text=str(text),
            perturbation_fns=perturbation_fns,
            perturbation_mean=perturbation_mean,
            perturbation_variance=perturbation_variance,
            rng=rng,
        )
        for text in manipulated_df.loc[indices_to_perturb, text_column].tolist()
    ]
    manipulated_df.loc[indices_to_perturb, text_column] = perturbed_texts
    return manipulated_df
