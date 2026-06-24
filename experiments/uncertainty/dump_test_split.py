"""Dump the locked seed-333 TEST split (text, gold, source_id) to a JSONL.

Run on the HPC login node (CPU only; no model needed):
    uv run python -m experiments.uncertainty.dump_test_split

Produces experiments/uncertainty/results/seed333_fixed/test_split_texts.jsonl, the
full ~76k test split inputs, so a standalone local GPU run can score it without
needing the processed dataset or the codllm package on the local machine. Reuses
the exact split logic from inference._load_eval_split (Config + DataHandler,
cod+age+sex, seed 333, 90/5/5).
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.uncertainty.inference import _load_eval_split

pairs, sep = _load_eval_split("test", None, 333)
out = Path("experiments/uncertainty/results/seed333_fixed/test_split_texts.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for text, gold, src in pairs:
        f.write(json.dumps({"text": text, "gold": gold, "source_id": src},
                           ensure_ascii=False) + "\n")
print(f"wrote {len(pairs)} test rows; label_separator={sep!r} -> {out}")
