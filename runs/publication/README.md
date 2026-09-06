# Publication experiments

Design: [publication_plan.md](../../docs/publication/publication_plan.md).
Exact preparation/submission commands: [publication_runs.md](../../docs/publication/publication_runs.md).
Results: [publication_progress.md](../../docs/publication/publication_progress.md).
Open ideas: [publication_thoughts.md](../../docs/publication/publication_thoughts.md).
Completed Phase 0a: [audit findings and decisions](../../docs/publication/publication_data_audit.md).

Both aggregate and full-size split audits are complete. Phase 0c passed 14 integrity checks;
[its review](../../docs/publication/publication_split_audit_review.md) quantifies concentrated description
groups and sparse cross-language support. Decisions remain with Lucas before full screening.
The source TOMLs and candidate levels are unchanged. The audit did not build the four augmented split
caches; keep the normal `hpc.build` step. Exact audit rerun and submission commands are in the launch guide.

## Legacy versus v1

The three completed response curves (7 floor + 7 synthesis + 6 pretraining runs) and `base.toml`
retain their original settings and output roots. Do not rerun or relabel them as grouped-COD evidence.
The new campaign inherits `protocol_v1.toml` and writes below
`$RUN_STORAGE_DIR/checkpoints/publication-v1/`.

V1 defaults: COD-only input, globally grouped full COD descriptions, linked source-record grouping,
90/5/5 approximate natural partitions, seeds 777, and exclusion of `historic_strings_en_2024`.
Missing CODs are dropped before splitting. Normalization uses Unicode NFC, case folding, and collapsed
whitespace; it preserves punctuation and accents. Group integrity takes priority over exact split
proportions or label stratification. Audit resulting source/label coverage.

Original rows are partitioned before synthesis, balancing, perturbation, or masterlist injection.
Source folds retain the full held-out archive; during-training holdout monitoring is disabled.
Tuning saves the best internal-validation legacy `macro_f1` checkpoint and suppresses final test evaluation.
The 120-epoch ceiling/patience 10 remain unchanged; ceiling-limited improving runs need review.

## Launchable steps

| Phase | TOML | Cells |
|---|---|---:|
| 0 | `smoke.toml` | 1 small smoke |
| 1 | `interaction_confirmation.toml` | 8 |
| 1 | `screening_controls.toml` | 2 |
| 2 | `source_transfer_pilot.toml` | 4 |
| 2 | `source_transfer_validation.toml` | 6 |
| 3 | `row_split_confirmation.toml` | 1 |
| 3 | `source_ablation_controls.toml` | 4 |
| 3 | `baselines_grouped.toml`, `baselines_row.toml` | 3 each |
| 3 | `baselines_copenhagen.toml`, `baselines_belgium.toml` | 3 each |
| 4 | `reduced_data_calibration.toml` | 4 |
| 4 | `reduced_split_sensitivity.toml` | 8 |
| 4 | `reduced_training_seed_sensitivity.toml` | 4 |
| Optional 3 | `metadata_confirmation.toml`, `perturbation_confirmation.toml` | 3 each |
| Optional 5 | `model_scale_confirmation.toml` | 2 large |
| Optional 5 | `model_scale_xl.toml` | 1 XL |
| Optional 5 | `scale_source_confirmation.toml` | 2 large source folds |
| 6 | `frozen_test_evaluation.toml`, `external_evaluation.toml` | 1 inference job each |
| 6 | `baselines_final.toml` | 3 baseline fits + inference |
| Optional 6 | `final_model.toml` | 1 fresh training |

Training cells may span multiple scheduler allocations. Baseline and frozen-evaluation jobs are
single-allocation jobs without automatic training resumption; do not attach a duration campaign.

## Candidate and phase decisions

`candidate_winner.toml` currently holds floor 0, fine-tuning synthesis .30, and 48 full-code pretraining
epochs; `candidate_runner_up.toml` is the matched 4-epoch alternative. Both are provisional.
Completed conditional curves motivate these levels but do not establish their joint optimality.
Keep both doses through the paired source panel; the longer dose is a product preference in a supported
near-tie, not an observed generalization improvement.

The source pilots are Copenhagen and Belgium; the remaining panel covers Amsterdam, Madrid, and Ipswich.
Choose structural settings after Phase 1, then freeze winner/runner-up after all five paired source folds.
Update candidate files centrally; downstream specs inherit them. Do not submit the candidate files as
extra runs when the required cell already exists in the screening grid.

`invoke publication.approve --stage <source_pair|recipe|reduced|final> --note "..."` records a reviewed
decision in `decisions.json`. Submission and runtime verify fingerprints of inherited candidate settings,
language curation, and the reduced protocol. Edits invalidate older approvals. These checks do not
inspect results or choose winners: review the evidence before recording approval.
Use new output roots if changing a previously executed recipe; training-contract guards reject
incompatible resume attempts.

## Reduced data and scale

Calibration and sensitivity studies all use whole-cohort sampling, not shorter training.
`CODLLM_DATASET_SAMPLE_SEED=777` fixes the cohort independently of `CODLLM_DATA_SEED`.
After comparing both candidates at 25% and 40%, set the chosen fraction in `reduced_protocol.toml`
before approving the reduced phase. Reuse only that fraction's matching seed-777 calibration anchors.
Split seeds alter preparation/evaluation membership; training seeds leave the split fixed and rerun the
full pretraining/fine-tuning pipeline. Neither estimates full-data seed variance directly.

Large uses batch 96 × accumulation 2; XL uses 32 × 6, matching base's effective batch 192.
If promoting XL to the source-fold check, update its profiles, accumulation, and output roots together.
Architecture alternatives beyond the prespecified scale checks require a separately reviewed extension.

## Languages and metrics

Reviewed source metadata lives in [source_languages.toml](../../data/curation/source_languages.toml).
Belgium is Flemish/Belgian Dutch (`nl`), following Lucas's clarification; Amsterdam is also `nl`.
Copenhagen is `da`, Madrid `es`, Ipswich `en`; masterlist supervision is English.
Do not label Amsterdam↔Belgium transfer as cross-lingual.
Mixed/unknown language is excluded from strict eligibility, with support reported explicitly.
Optional reviewed row overrides use `source_id,record_id,language` CSV columns.

Legacy metrics remain unchanged. New `pub_v1_*` metrics use original historical training exposure:
COD novelty, source/language slices, source transfer, cross-lingual eligible targets, stricter
all-adaptation transfer, single/multi-COD slices, novel known-code combinations, and frequency strata.
Historical cross-lingual transfer means a target code was taught only in other verified historical
languages. Strict transfer additionally excludes target/unknown-language supervision in augmentation
and the masterlist. Neither claims absence from the base model's unknown pretraining corpus.

`pub_v1_macro_f1_ref` averages over reference-supported target codes in that evaluation slice,
independent of a model's predicted-label set. Pair it with exact match/micro F1 and invalid-code counts:
prediction-only codes still penalize those metrics. Primary recipe screening uses
`pub_v1_source_mean_macro_f1_ref`; checkpoint selection remains legacy `macro_f1`.
Eligible-target recall and complete-row accuracy/F1 answer different questions. Empty slices are
unavailable, not zero-performing. Slice row/code/target counts accompany scores.

## Private outputs and recovery

Each training run stores:

- `publication/reference.json`: original historical, augmentation, and masterlist exposure.
- `publication/contract.json` and `effective_config.json`: recipe and partition fingerprints.
- `publication/manifests/`: full original-row manifests, including raw input text and labels.
- `publication/predictions/<scope>/selected.json`: selected-model identity and same-checkpoint metrics.
- Content-addressed JSON/Parquet prediction files with row identities, full predictions/targets, COD
  hashes, and transfer eligibility.

Exports are local private files, independent of W&B sync; do not publish them without data permission.
The new writer does not upload raw records. Existing W&B error-table behavior is unchanged.
W&B scalar metrics use `codllmdev/codllm`; baseline/frozen-evaluation scalars sync after local export.

`invoke publication.report --root <campaign-root>` collects selected results, not independent epoch maxima.
`invoke publication.bootstrap --first <predictions.parquet> --second <predictions.parquet>` requires
identical evaluation rows and computes conditional paired group-bootstrap differences; it is not a
training-seed interval or an estimate across all possible archives.

Frozen evaluation requires an explicit local selected checkpoint and its matching original
`publication/` directory. Internal evaluation reads checksum-verified saved manifests.
External inference accepts an explicitly curated CSV/Parquet table; no UniCin source ID, label mapping,
or dataset is invented. Freeze those resources before the final approval. The evaluator records a
checkpoint SHA256 and calibrates uncertainty on the original validation partition, not external labels.

## Optional Bayesian search

The previous `bayesian_base.toml` / `bayesian_sweep.yaml` remain legacy, shortened-budget candidate
generation. They are outside this launch sequence and require v1 migration plus a fidelity check before use.
No Bayesian sweep is created automatically.
