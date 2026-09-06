"""Offline selected-checkpoint aggregation and paired group-bootstrap analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from codllm.evaluation.artifacts import write_json
from codllm.evaluation.scoring import reference_macro_f1
from codllm.evaluation.splitting import linked_groups


def summarize_runs(root: str, output: str) -> pd.DataFrame:
    """Collect selected-checkpoint scalar summaries without mixing epoch maxima."""
    records = []
    for path in sorted(Path(root).rglob("selected.json")):
        payload = json.loads(path.read_text())
        identity = payload.get("identity", {})
        records.append({"summary_path": str(path), **identity, **payload["metrics"]})
    frame = pd.DataFrame(records)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    destination.chmod(0o600)
    return frame


def paired_bootstrap(
    first: str, second: str, output: str, replicates: int = 2000, seed: int = 777
) -> dict:
    """Bootstrap paired macro-F1 differences using shared global COD groups, conditional on the models."""
    if replicates < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    a, b = pd.read_parquet(first), pd.read_parquet(second)
    if a["row_uid"].duplicated().any() or b["row_uid"].duplicated().any():
        raise ValueError("Paired artifacts must contain unique row identities.")
    if set(a["row_uid"]) != set(b["row_uid"]):
        raise ValueError("Bootstrap requires predictions on exactly the same rows.")
    b = b.set_index("row_uid").loc[a["row_uid"]].reset_index()
    targets = [set(value) for value in a["labels"]]
    if (
        targets != [set(value) for value in b["labels"]]
        or a["cod_hash"].tolist() != b["cod_hash"].tolist()
        or a["source_id"].tolist() != b["source_id"].tolist()
    ):
        raise ValueError("Paired row labels, sources, or COD identities differ.")
    p = [set(value) for value in a["predictions"]]
    q = [set(value) for value in b["predictions"]]
    if "record_id" not in a:
        a["record_id"] = ""
    group_ids = linked_groups(a.rename(columns={"cod_hash": "cod_key"}), by_cod=True)
    grouped = list(
        pd.Series(np.arange(len(a))).groupby(group_ids, sort=True).apply(np.asarray)
    )
    if len(grouped) < 2:
        raise ValueError("At least two independent COD groups are required.")
    source_values = a["source_id"].to_numpy()
    sources = sorted(set(source_values))
    rng = np.random.default_rng(seed)

    def difference(indices: np.ndarray) -> float:
        """Recompute equal-archive macro F1 within a group-resampled panel."""
        values = []
        for source in sources:
            selected = indices[source_values[indices] == source]
            if not len(selected):
                continue
            labels = [targets[i] for i in selected]
            values.append(
                reference_macro_f1([p[i] for i in selected], labels)
                - reference_macro_f1([q[i] for i in selected], labels)
            )
        return float(np.mean(values))

    samples = [
        difference(
            np.concatenate(
                [
                    grouped[index]
                    for index in rng.integers(len(grouped), size=len(grouped))
                ]
            )
        )
        for _ in range(replicates)
    ]
    payload = {
        "difference_first_minus_second": difference(np.arange(len(a))),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "groups": len(grouped),
        "rows": len(a),
        "replicates": replicates,
        "seed": seed,
        "cluster_unit": "Connected COD and source-record groups",
        "estimand": "Conditional equal-source reference-supported macro F1; not training-seed or future-archive variance.",
    }
    write_json(Path(output), payload)
    return payload
