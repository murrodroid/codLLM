"""Compute per-source accuracy + macro F1 from an inference JSONL."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from codllm.metrics import _macro_precision_recall_f1


def main(jsonl_path: Path, output_path: Path) -> int:
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source_id"]].append(r)

    def metrics_block(records: list[dict]) -> dict:
        n = len(records)
        n_correct = sum(1 for r in records if r.get("correct"))
        preds = [{str(r["prediction"]).strip()} for r in records]
        labels = [{str(r["gold"]).strip()} for r in records]
        f1 = _macro_precision_recall_f1(preds, labels).get("macro_f1", 0.0)
        return {
            "n": n,
            "accuracy": n_correct / n if n else 0.0,
            "macro_f1": f1,
        }

    out = {
        "overall": metrics_block(rows),
        "per_source": {
            src: metrics_block(recs) for src, recs in sorted(by_source.items())
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Pretty print table
    print(f"\nPer-source accuracy ({jsonl_path.name})")
    print("-" * 72)
    print(f"{'source_id':<30} {'n':>7} {'accuracy':>10} {'macro_f1':>10}")
    print("-" * 72)
    for src, m in out["per_source"].items():
        print(f"{src:<30} {m['n']:>7d} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")
    print("-" * 72)
    o = out["overall"]
    print(f"{'OVERALL':<30} {o['n']:>7d} {o['accuracy']:>10.4f} {o['macro_f1']:>10.4f}")
    print(f"\nWrote: {output_path}")
    return 0


if __name__ == "__main__":
    jsonl = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "experiments/uncertainty/results/test_n13829_seed333.jsonl"
    )
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "experiments/uncertainty/results/per_source_accuracy.json"
    )
    sys.exit(main(jsonl, out))
