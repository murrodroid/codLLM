"""Group-disjoint publication splits and integrity checks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from codllm.config import Config
from codllm.data.splits import DataSplits


def linked_groups(frame: pd.DataFrame, *, by_cod: bool) -> np.ndarray:
    """Connect identical CODs and repeated source-record identities without quadratic comparisons."""
    parents = np.arange(len(frame))

    def find(index: int) -> int:
        """Resolve one connected-component root with path compression."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    identities: dict[tuple[str, ...], int] = {}
    for index, (source, record, cod) in enumerate(
        frame[["source_id", "record_id", "cod_key"]]
        .fillna("")
        .astype(str)
        .itertuples(index=False, name=None)
    ):
        keys = [("record", source, record)] if record else []
        if by_cod and cod:
            keys.append(("cod", cod))
        for key in keys:
            if key in identities:
                parents[find(index)] = find(identities[key])
            else:
                identities[key] = index
    return np.asarray([find(index) for index in range(len(frame))])


def grouped_split(frame: pd.DataFrame, cfg: Config) -> DataSplits:
    """Assign whole connected groups to approximate row-count targets deterministically."""
    groups = linked_groups(frame, by_cod=cfg.evaluation_protocol == "cod")
    identities, sizes = np.unique(groups, return_counts=True)
    fractions = np.asarray([cfg.train_size, cfg.val_size, cfg.test_size])
    active = np.flatnonzero(fractions > 0)
    if len(identities) < len(active):
        raise ValueError(
            "Too few independent groups for the requested non-empty publication splits."
        )
    rng = np.random.default_rng(cfg.resolved_data_seed())
    order = rng.permutation(len(identities))
    targets = fractions * len(frame)
    counts = np.zeros(3)
    assignments: dict[int, int] = {}
    for position, offset in enumerate(order):
        remaining = len(order) - position
        empty = active[counts[active] == 0]
        candidates = empty if remaining == len(empty) else active
        split = int(candidates[np.argmax((targets - counts)[candidates])])
        assignments[int(identities[offset])] = split
        counts[split] += sizes[offset]
    membership = np.asarray([assignments[int(group)] for group in groups])
    pieces = [
        frame.loc[membership == index].reset_index(drop=True) for index in range(3)
    ]
    return DataSplits(train=pieces[0], val=pieces[1], test=pieces[2])


def validate_group_integrity(splits: DataSplits, cfg: Config) -> None:
    """Fail when original historical partitions share a forbidden identity."""
    train = splits.original_train if splits.original_train is not None else splits.train
    seen_records: set[tuple[str, str]] = set()
    seen_cods: set[str] = set()
    for name, frame in (("train", train), ("val", splits.val), ("test", splits.test)):
        records = set(
            zip(
                frame.loc[frame["record_id"].fillna("").ne(""), "source_id"].astype(
                    str
                ),
                frame.loc[frame["record_id"].fillna("").ne(""), "record_id"].astype(
                    str
                ),
                strict=True,
            )
        )
        if seen_records.intersection(records):
            raise ValueError(f"Linked records cross the {name} split boundary.")
        seen_records.update(records)
        if cfg.evaluation_protocol == "cod":
            cods = set(frame["cod_key"].astype(str)) - {""}
            if seen_cods.intersection(cods):
                raise ValueError(f"Normalized CODs cross the {name} split boundary.")
            seen_cods.update(cods)
    permitted = set(train["row_uid"])
    if "synthetic_parent_uids" in splits.train:
        for value in splits.train["synthetic_parent_uids"].dropna().unique():
            if value and not set(json.loads(value)).issubset(permitted):
                raise ValueError(
                    "Synthetic constituents include rows outside original training."
                )
    if cfg.hold_out_dataset:
        for name, frame in (
            ("train", splits.train),
            ("val", splits.val),
            ("test", splits.test),
        ):
            if frame["source_id"].eq(cfg.hold_out_dataset).any():
                raise ValueError(f"Held-out source leaked into {name}.")
