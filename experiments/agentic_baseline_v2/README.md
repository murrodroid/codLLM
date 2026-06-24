# Agentic baseline v2

Claude agentic baseline for RQ1 ("how does our approach compare to existing methods"), built to be directly comparable to the size_sweep flan-t5 models on the locked test split (`data_seed=333`, n=76,418).

## What this is

For each test record, a fresh isolated `claude -p` subprocess receives the CoD string and the prompt at [prompt.md](prompt.md). The agent gets full agentic access — `Bash, Read, Grep, Glob, WebSearch, WebFetch` — and must return 1–3 ICD-10h codes as a JSON code block.

**Differences vs. v1** (`experiments/agentic_baseline_claudecode/`):

- Multi-label (max_label_count=3, no comma filter), matching the size_sweep test split exactly
- Per-record JSONL append for crash-resilient logging + post-hoc analysis
- Multi-label parse and per-record set-based F1
- Uses `codllm.metrics` primitives so macro/micro/sample F1 are bit-identical to the flan-t5 evaluator
- Logs `source_id` per record for per-source breakdowns
- Cost cap option to abort runaway runs

## Tool access (decided 2026-06-03)

| Tool | Access | Rationale |
|---|---|---|
| Masterlist (`data/raw/ICD10h_Masterlist_2024.tsv`) | ✅ | Codebook is what a human coder would also use |
| Open internet (WebSearch, WebFetch) | ✅ | What a 2026 historian with an AI assistant would have. Overrides the multi-agent committee's leakage-risk concern |
| ICD-10h hierarchy navigation | ✅ via Bash | Agent can `grep` the masterlist's hierarchy columns |
| Bash / file IO | ✅ | Lets the agent explore freely |

## Run commands

```bash
# Smoke test (5 records, ~$0.50, ~80s)
uv run python -m experiments.agentic_baseline_v2.run --n 5 --workers 2 --batch-label smoke

# Full run (1000 records, ~$100, ~30min with workers=4)
uv run python -m experiments.agentic_baseline_v2.run \
    --n 1000 --workers 4 --model sonnet \
    --batch-label n1000_seed333_sonnet \
    --cost-cap-usd 150

# Resume a partial run
uv run python -m experiments.agentic_baseline_v2.run \
    --n 1000 --batch-label n1000_seed333_sonnet --resume
```

Each record is processed in its own isolated subprocess. `--workers` only controls how many subprocesses run concurrently; there is zero shared state between records.

## Post-hoc analysis

```bash
uv run python -m experiments.agentic_baseline_v2.eval_posthoc \
    experiments/agentic_baseline_v2/results/n1000_seed333_sonnet.predictions.jsonl \
    --rare-threshold 5 \
    --save-table experiments/agentic_baseline_v2/results/n1000_summary.json
```

The post-hoc script produces:

- **Headline**: macro/micro/sample F1, exact-set-match rate, invalid-code rate, no-answer rate
- **Per-source table**: macro/sample/micro F1 per historical source (Amsterdam, Copenhagen, Madrid, Belgium, Ipswich, historic_strings)
- **Rare-codes slice**: macro_f1 on records whose gold contains a code with <K training occurrences (default K=5)
- **Confidence calibration**: empirical accuracy by predicted-confidence bucket (high/medium/low)
- **Multi-label breakdown**: performance by gold cardinality (1 / 2 / 3 codes); over- and under-prediction counts
- **Cost + latency stats**
- **Error breakdown** (timeouts, parse failures, etc.)
- **Sample of 10 misclassifications** for spot-checking

## Output schema (per-record JSONL row)

```json
{
  "parquet_idx": 46687,
  "source_id": "amsterdam_1854_1926",
  "cod": "placenta praevia",
  "gold_str": "O44.100",
  "gold_codes": ["O44.100"],
  "predicted_codes": ["O44.100"],
  "predicted_codes_invalid": [],
  "raw_codes_field": ["O44.100"],
  "exact_set_match": true,
  "precision": 1.0,
  "recall": 1.0,
  "f1": 1.0,
  "reasoning": "Placenta praevia is a well-defined obstetric condition; ...",
  "confidence": "high",
  "alternatives": ["O44.000"],
  "num_turns": 2,
  "elapsed_s": 14.82,
  "cost_usd": 0.0962,
  "error": null,
  "raw_response": "..."
}
```

The `predictions.jsonl` file is append-only, so it survives crashes — `--resume` reads it back to skip already-completed records.

## Comparability with the size_sweep models

This baseline is designed to be **directly comparable** to the flan-t5 size_sweep:

- Same locked test split (`data_seed=333`, n=76,418), no model-specific subsetting
- Same multi-label regime (max 3 codes, comma-separated)
- Same metric primitives (`codllm.metrics`), so macro_f1 and sample_f1 are bit-identical to the trainer's `test/macro_f1` and `test/sample_f1`

A subsample (n=1000) is used for cost reasons; the per-source breakdown will be noisy for Belgium/Ipswich (~28 records each). To get tighter per-source CIs, run a stratified n=200/source variant separately (use `--seed` to vary).

## Project memory link

See [project_agentic_baseline_v2.md](../../../.claude/projects/c--Users-edlun-Desktop-DTU-Bachelor-codLLM/memory/project_agentic_baseline_v2.md) for the access-policy decision record.
