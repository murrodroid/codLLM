import math
import random
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from codllm.config import Config
from codllm.input import PERTURBATION_REGISTRY

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
        (idx for idx, part in enumerate(parts) if part.startswith("cod: ")),
        None,
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


def _contains_cod_segment(cfg: Config, text: str) -> bool:
    """Return whether a training text contains a cod: segment."""
    parts = text.split(cfg.text_field_separator) if cfg.text_field_separator else [text]
    return any(part.startswith("cod: ") for part in parts)


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
    text_column: str | None = None,
    perturbation_fns: Sequence[Any] | None = None,
    perturbations_per_sample: int = 1,
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
                    for _ in range(perturbations_per_sample):
                        fn = rng.choice(perturbation_fns)
                        text = _apply_perturbation_with_seed(
                            fn,
                            text,
                            seed=rng.randint(0, 2_147_483_647),
                        )
                    row[text_column] = text

        synthetic_rows.extend(new_rows)

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
