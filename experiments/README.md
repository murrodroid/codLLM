# Experiments

Research code for the thesis. Each subdirectory is a self-contained study. Heavy
outputs (`results/`) are gitignored and regenerated from the commands in the
[project guide](../docs/guide.md).

| Directory | What it is | Research question |
|---|---|---|
| `baselines/` | Classical baselines: TF-IDF + linear classifier, multilingual sentence-embedding retrieval, majority class | RQ1 |
| `agentic_baseline_v2/` | Agentic baseline: isolated Claude agents with shell, web, and masterlist access, returning a fixed JSON schema | RQ1 |
| `offline_test_eval/` | Authoritative held-out test evaluation. The W&B `test/*` metric is aliased to validation; this recomputes the real test number from a checkpoint | RQ1 / RQ2 |
| `uncertainty/` | Uncertainty-informed selective prediction, plus leave-one-source-out (LOSO) transfer analysis | RQ2, RQ3 |
| `tree_search/` | ICD10h hierarchy utilities (with tests) and the tree-search decoding experiment | background |
| `notebooks/` | Exploratory data analysis: class imbalance, perturbation, upsampling, ICD10h tree | background |
| `exploratory/` | Superseded directions kept for the record: RAG retrieval and the v1 agentic baseline | background |
| `LOGBOOK.md` | Running notes on methodology decisions | - |
