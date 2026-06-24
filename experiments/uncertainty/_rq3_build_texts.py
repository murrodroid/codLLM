"""STEP 2: build the held-out texts files (10% W&B-aligned sample per source, random_state=333)."""
import json
import pathlib

import pandas as pd

df = pd.read_parquet("data/processed/data.parquet")
SRCS = {
    "amsterdam": "amsterdam_1854_1926",
    "copenhagen": "copenhagen_may2025",
    "madrid": "madrid_1905_1927",
    "belgium": "belgium_1920_1930",
    "ipswich": "ipswich_1871_1911",
}
for short, sid in SRCS.items():
    sub = df[df.source_id == sid].reset_index(drop=True)
    s = sub.sample(n=round(len(sub) * 0.10), random_state=333, replace=False).reset_index(drop=True)
    out = pathlib.Path(f"experiments/uncertainty/results/loso/texts_{short}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for _, r in s.iterrows():
            f.write(json.dumps(
                {"text": str(r.text), "gold": str(r.label), "source_id": str(r.source_id)},
                ensure_ascii=False) + "\n")
    print(short, len(s), "->", out, flush=True)
print("BUILD TEXTS DONE", flush=True)
