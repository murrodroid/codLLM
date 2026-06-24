"""Build EXACT-N scored prediction files for the agentic baseline.
For each config: drop historic_strings_en_2024 rows, order by canonical seed-333
position, take the first N non-ref records. Write *.scored3500/.scored1000.jsonl.
"""
import json, sys
from pathlib import Path

R = Path("experiments/agentic_baseline_v2/results")
CANON = json.load(open("canon_order.json"))          # list of parquet_idx in seed-333 order
POS = {v:i for i,v in enumerate(CANON)}
REF = "historic_strings_en_2024"

def build(src_name, out_name, target):
    src = R / f"{src_name}.predictions.jsonl"
    rows = []
    seen = set()
    with src.open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            if d.get("source_id")==REF:
                continue
            idx=int(d["parquet_idx"])
            if idx in seen:   # dedupe (resume can double-append on rare crash)
                continue
            seen.add(idx)
            rows.append((POS.get(idx, 10**9), d))
    rows.sort(key=lambda t:t[0])
    nonref_total = len(rows)
    chosen = [d for _,d in rows[:target]]
    out = R / f"{out_name}.predictions.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for d in chosen:
            f.write(json.dumps(d, ensure_ascii=False)+"\n")
    # verify
    nref = sum(1 for d in chosen if d.get("source_id")==REF)
    print(f"{out_name}: wrote n={len(chosen)} (target {target}) ref_rows={nref} "
          f"nonref_available={nonref_total} -> {'OK' if len(chosen)==target and nref==0 else 'SHORT/BAD'}",
          file=sys.stderr)
    return len(chosen), nonref_total

if __name__=="__main__":
    name=sys.argv[1]; out=sys.argv[2]; tgt=int(sys.argv[3])
    build(name,out,tgt)
