# Publication commands

Run from the repository root. [Plan](../publication_plan.md) and [decisions](../publication_decisions.md)
define the stage to run; the table below is not an instruction to submit every stage.

## HPC setup

```shell
cd ~/codLLM
source hpc/env.sh
uv sync
```

Sync after dependency changes; otherwise use `uv run --no-sync`. From the laptop,
`uv run connect --lucas` opens the shared connection; `--status` checks it.

## Build and submit one stage

After the required decision and approval, choose one specification from the table:

```shell
PUBLICATION_SPEC=runs/publication/source_transfer_pilot.toml
uv run --no-sync invoke hpc.build --config "$PUBLICATION_SPEC" --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config "$PUBLICATION_SPEC" --profile h100 --lucas --duration 1w --dry-run
uv run --no-sync invoke hpc.submit --config "$PUBLICATION_SPEC" --profile h100 --lucas --duration 1w
```

Training uses one-week resumable campaigns. **Omit `--duration 1w` for baselines and frozen evaluation**;
they use single allocations. Output roots are defined in the TOMLs under `$RUN_STORAGE_DIR`.
Use new roots when changing a completed recipe.

All specification paths below are relative to `runs/publication/`.

| Phase | Specification | Cells | Approval |
|---|---|---:|---|
| 2 pilots | `source_transfer_pilot.toml` | 4 | `source_pair` |
| 2 remaining folds | `source_transfer_validation.toml` | 6 | `source_pair` |
| 3 row comparison | `row_split_confirmation.toml` | 1 | `recipe` |
| 3 source ablations | `source_ablation_controls.toml` | 4 | `recipe` |
| 3 baselines | `baselines_grouped.toml`, `baselines_row.toml`, `baselines_copenhagen.toml`, `baselines_belgium.toml` | 3 each | `recipe` |
| 4 calibration | `reduced_data_calibration.toml` | 4 | `recipe` |
| 4 split seeds | `reduced_split_sensitivity.toml` | 8 | `reduced` |
| 4 training seeds | `reduced_training_seed_sensitivity.toml` | 4 | `reduced` |
| Optional metadata / perturbation | `metadata_confirmation.toml`, `perturbation_confirmation.toml` | 3 each | `recipe` |
| Optional large / XL | `model_scale_confirmation.toml`, `model_scale_xl.toml` | 2 / 1 | `recipe` |
| Optional scale source folds | `scale_source_confirmation.toml` | 2 | `recipe` |
| 6 internal / external | `frozen_test_evaluation.toml`, `external_evaluation.toml` | 1 each | `final` |
| 6 final baselines | `baselines_final.toml` | 3 | `final` |
| Optional fresh final training | `final_model.toml` | 1 | `final` |

Completed specifications: `smoke.toml`, `interaction_confirmation.toml`, `interaction_patience20.toml`,
`interaction_metadata.toml`, and `interaction_metadata_patience20.toml`. Preserve their outputs; do not resubmit.
`screening_controls.toml` defines two Phase 1 controls; no separate completion record is in the current ledger.

Candidate settings belong in `candidate_winner.toml` and `candidate_runner_up.toml`.
The reduced fraction belongs in `reduced_protocol.toml`; sample seed stays 777.
Scale source confirmation defaults to large. Selecting XL requires matching profile, accumulation,
and new output roots. Later stopping/input policies require review before submission.

## Record an approved decision

Only after Lucas has made the decision:

```shell
read -r -p "Decision and supporting run IDs: " PUBLICATION_REVIEW_NOTE
uv run --no-sync invoke publication.approve --stage source_pair --note "$PUBLICATION_REVIEW_NOTE"
```

Use `source_pair` before Phase 2, `recipe` after the paired source panel, `reduced` after cohort calibration,
and `final` after freezing assessment. Candidate, language-curation, or reduced-protocol edits invalidate
affected approvals. Mirror the actual choice in [decisions](../publication_decisions.md).

## Frozen evaluation

Use a selected fine-tuning checkpoint and its matching original `publication/` directory:

```shell
read -r -p "Absolute selected checkpoint directory: " CODLLM_EVALUATION_CHECKPOINT
read -r -p "Absolute original run publication directory: " CODLLM_EVALUATION_REFERENCE_DIR
export CODLLM_EVALUATION_CHECKPOINT CODLLM_EVALUATION_REFERENCE_DIR
unset CODLLM_EVALUATION_DATA_PATH
PUBLICATION_SPEC=runs/publication/frozen_test_evaluation.toml
uv run --no-sync invoke hpc.build --config "$PUBLICATION_SPEC" --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config "$PUBLICATION_SPEC" --profile h100 --lucas
```

For external assessment, also export `CODLLM_EVALUATION_DATA_PATH` to the curated CSV/Parquet and use
`external_evaluation.toml`. Required columns: `source_id`, `record_id`, `text`, `label`;
optional `cod_text` retains the original description. Text must match the model's input format,
and labels use full ICD10h codes. Freeze reviewed language metadata before final approval.
External data and label mapping are not supplied by the specification.

Use `baselines_final.toml` for the prespecified final baseline fits. Rebuilding a split or retraining
the transformer is not required for frozen evaluation. Calibration uses the original validation split.

## Recover selected results

```shell
uv run --no-sync invoke publication.report --root "$RUN_STORAGE_DIR/checkpoints/publication-v1"
```

Output: `logs/publication/selected_results.csv`. Verify completion markers, checkpoint identity,
and prediction exports before updating [progress](../publication_progress.md).
Private predictions and text-bearing manifests stay under the run's `publication/` directory.

For aligned prediction files, `publication.bootstrap --first <parquet> --second <parquet>` computes
paired COD/linked-record bootstrap intervals; it does not estimate training-seed variance.

## Checkpoint consistency

Run locally; no HPC or training is needed:

```shell
uv run --no-sync python -m experiments.checkpoint_consistency \
  --input docs/publication/reference/checkpoint_consistency_input.json \
  --output-json logs/publication/checkpoint_consistency.json \
  --output-md logs/publication/checkpoint_consistency.md
uv run --no-sync python -m experiments.checkpoint_consistency \
  --input docs/publication/reference/metadata_checkpoint_consistency_input.json \
  --output-json logs/publication/metadata_checkpoint_consistency.json \
  --output-md logs/publication/metadata_checkpoint_consistency.md
```

## Audits and continuations

The completed [audit snapshots](audits.md) are retained. Re-audit when relevant inputs change:
`publication.audit` inventories processed data; `publication.audit-splits` checks original partitions;
`publication.audit-metadata-pairing` requires an exact match to frozen COD-only manifests before a new metadata comparison.
Passing an integrity audit does not approve a scientific choice or build augmented caches.

Both patience-20 campaigns are complete. Any separately requested future continuation must use
`publication.prepare-continuation` before build/submission: dry-run first, `--yes` for independent,
checksum-verified checkpoint copies and fresh W&B identities. All parents must be complete.
Preserve parent outputs and frozen lineage; never hardlink mutable checkpoint copies.
