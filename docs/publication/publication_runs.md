# Publication runs

Results: [publication_progress.md](publication_progress.md). Design: [publication_plan.md](publication_plan.md).
Run one section at a time. Review completed results before each approval.
Inherited candidate/protocol files are not separate jobs.

## Setup on HPC

Copy/sync this repository revision to `~/codLLM` first.

```shell
cd ~/codLLM
source hpc/env.sh
uv sync
```

Run `uv sync` once after updating code/dependencies, or after maintenance removed the environment.
For later terminals, repeat only `cd` and `source`.
Training submissions below use resumable one-week campaigns; check for completion or budget-censoring.

## Completed exploratory curves — do not resubmit

- Balance floor: 7/7.
- Multi-COD synthesis: 7/7.
- Pretraining dose: 6/6.

Their original TOMLs and results remain unchanged. All new jobs use `checkpoints/publication-v1/`.

## Phase 0a. Data audit

```shell
uv run --no-sync invoke publication.audit --config runs/publication/protocol_v1.toml
```

Review `logs/publication/data_audit.json` before training. Copenhagen and Belgium are the prespecified pilot sources.
Belgian Flemish and Amsterdam both use language `nl`.

## Phase 0b. Smoke test — 1 run

Confirm checkpointing, local prediction exports, and W&B logging before Phase 1.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/smoke.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/smoke.toml --profile h100 --lucas --duration 1w
```

## Phase 1a. Joint recipe screening — 8 runs

Floor 0/450 × synthesis 0.30/0.60 × pretraining 4/48.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/interaction_confirmation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/interaction_confirmation.toml --profile h100 --lucas --duration 1w
```

## Phase 1b. Screening controls — 2 runs

No pretraining; no fine-tuning synthesis.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/screening_controls.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/screening_controls.toml --profile h100 --lucas --duration 1w
```

## Review 1. Choose the paired recipes

Update `candidate_winner.toml` and `candidate_runner_up.toml` with the selected shared floor/synthesis settings. Keep the matched 48- and 4-epoch doses. They currently contain provisional floor 0 / synthesis 0.30.

```shell
read -r -p "Decision and supporting run IDs: " PUBLICATION_REVIEW_NOTE
uv run --no-sync invoke publication.approve --stage source_pair --note "$PUBLICATION_REVIEW_NOTE"
```

## Phase 2a. Paired source pilots — 4 runs

Both candidates: Copenhagen and Belgium.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/source_transfer_pilot.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/source_transfer_pilot.toml --profile h100 --lucas --duration 1w
```

## Phase 2b. Remaining paired source folds — 6 runs

Both candidates: Amsterdam, Madrid, and Ipswich. Reuse the four pilot runs.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/source_transfer_validation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/source_transfer_validation.toml --profile h100 --lucas --duration 1w
```

## Review 2. Freeze winner and runner-up

Review all five paired source folds, then update the two candidate files if needed. Do not choose from test/external results. Candidate or protocol edits invalidate previous approvals; use new output roots when changing a completed recipe.

```shell
read -r -p "Decision and supporting run IDs: " PUBLICATION_REVIEW_NOTE
uv run --no-sync invoke publication.approve --stage recipe --note "$PUBLICATION_REVIEW_NOTE"
```

## Phase 3a. Matched row-split model — 1 run

```shell
uv run --no-sync invoke hpc.build --config runs/publication/row_split_confirmation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/row_split_confirmation.toml --profile h100 --lucas --duration 1w
```

## Phase 3b. Source ablations — 4 runs

No pretraining / no fine-tuning synthesis, each on Copenhagen and Belgium.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/source_ablation_controls.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/source_ablation_controls.toml --profile h100 --lucas --duration 1w
```

## Phase 3c. Grouped-COD baselines — 3 fits

Lookup, character-linear, masterlist retrieval. Baseline jobs use one allocation and do not auto-resume.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/baselines_grouped.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/baselines_grouped.toml --profile h100 --lucas
```

## Phase 3d. Row-split baselines — 3 fits

Same baseline methods under the matched row protocol.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/baselines_row.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/baselines_row.toml --profile h100 --lucas
```

## Phase 3e. Copenhagen baselines — 3 fits

Same baseline methods; full held-out Copenhagen assessment.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/baselines_copenhagen.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/baselines_copenhagen.toml --profile h100 --lucas
```

## Phase 3f. Belgium baselines — 3 fits

Same baseline methods; full held-out Belgium assessment.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/baselines_belgium.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/baselines_belgium.toml --profile h100 --lucas
```

## Phase 4a. Reduced-cohort calibration — 4 runs

Both candidates at 25% and 40%, to convergence. This samples the whole cohort; evaluation populations differ across fractions.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/reduced_data_calibration.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/reduced_data_calibration.toml --profile h100 --lucas --duration 1w
```

## Review 3. Choose the reduced cohort

Set `CODLLM_DATASET_SIZE` in `runs/publication/reduced_protocol.toml` to 0.25 or 0.40 after calibration. Keep sample seed 777. Reuse only matching seed-777 calibration runs as anchors.

```shell
read -r -p "Decision and supporting run IDs: " PUBLICATION_REVIEW_NOTE
uv run --no-sync invoke publication.approve --stage reduced --note "$PUBLICATION_REVIEW_NOTE"
```

## Phase 4b. Split/preparation sensitivity — 8 runs

Both candidates; split seeds 101/202/303/404; fixed cohort and training seed 777.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/reduced_split_sensitivity.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/reduced_split_sensitivity.toml --profile h100 --lucas --duration 1w
```

## Phase 4c. Training-pipeline seed sensitivity — 4 runs

Winner; training seeds 111/222/333/444; fixed cohort and split seed 777. Includes rerunning pretraining.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/reduced_training_seed_sensitivity.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/reduced_training_seed_sensitivity.toml --profile h100 --lucas --duration 1w
```

## Optional 3g. Metadata portability — 3 runs

COD+age+sex on grouped-COD, Copenhagen, and Belgium; matches the frozen winner otherwise.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/metadata_confirmation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/metadata_confirmation.toml --profile h100 --lucas --duration 1w
```

## Optional 3h. Base-perturbation ablation — 3 runs

No base perturbation on grouped-COD, Copenhagen, and Belgium; other augmentation remains unchanged.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/perturbation_confirmation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/perturbation_confirmation.toml --profile h100 --lucas --duration 1w
```

## Optional Phase 5a. FLAN-T5-large — 2 runs

Both frozen recipes. If reduced-cohort edits invalidated the recipe approval, record Review 2 again first.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/model_scale_confirmation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/model_scale_confirmation.toml --profile h100 --lucas --duration 1w
```

## Optional Phase 5b. FLAN-T5-XL — 1 run

Only after the large-model comparison justifies the cost.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/model_scale_xl.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/model_scale_xl.toml --profile h100 --lucas --duration 1w
```

## Optional Phase 5c. Larger-model source confirmation — 2 runs

Defaults to large on Copenhagen and Belgium. If selecting XL, change both variant profiles to `h100-xl.toml`, gradient accumulation to 6, and use new XL output roots before submission.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/scale_source_confirmation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/scale_source_confirmation.toml --profile h100 --lucas --duration 1w
```

## Before final assessment

Freeze the release checkpoint, decoding, baseline settings, label mapping, and external cohort first.
An external table must have `source_id`, `record_id`, `text`, and `label` columns.
`text` must follow the frozen model's input format; `label` uses full ICD10h codes separated by the configured separator.
Optional `cod_text` preserves the original full description.
Add reviewed external language information to `data/curation/source_languages.toml` before final approval.
Unknown language remains unscored for language transfer. UniCin data/mapping are not supplied by these TOMLs.

## Review 4. Freeze final assessment

Complete this only after selection and external-data curation. Later evaluation reads the original training exposure and split manifests; it does not rebuild a new test split.

```shell
read -r -p "Decision and supporting run IDs: " PUBLICATION_REVIEW_NOTE
uv run --no-sync invoke publication.approve --stage final --note "$PUBLICATION_REVIEW_NOTE"
```

## Phase 6a. Select the existing checkpoint

Use the selected fine-tuning checkpoint, not a pretraining checkpoint.
Use its matching `publication/` directory from the same original run.

```shell
read -r -p "Absolute selected checkpoint directory: " CODLLM_EVALUATION_CHECKPOINT
read -r -p "Absolute original run publication directory: " CODLLM_EVALUATION_REFERENCE_DIR
export CODLLM_EVALUATION_CHECKPOINT CODLLM_EVALUATION_REFERENCE_DIR
unset CODLLM_EVALUATION_DATA_PATH
```

## Phase 6b. Frozen internal test — inference only

`hpc.build` validates the checkpoint and frozen artifacts. Includes uncertainty analysis calibrated on the original validation partition.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/frozen_test_evaluation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/frozen_test_evaluation.toml --profile h100 --lucas
```

## Phase 6c. Select the frozen external table

Keep the same checkpoint and reference-directory variables from Phase 6a.

```shell
read -r -p "Absolute curated external CSV or Parquet: " CODLLM_EVALUATION_DATA_PATH
export CODLLM_EVALUATION_DATA_PATH
```

## Phase 6d. External assessment — inference only

Run only when the curated external data exists. No training or external threshold selection.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/external_evaluation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/external_evaluation.toml --profile h100 --lucas
```

## Phase 6e. Prespecified final baselines — 3 fits

Keep the external-data variable set to include that frozen table. The same baseline recipes are refitted on their original internal training partition, with thresholds selected only on internal validation.

```shell
uv run --no-sync invoke hpc.build --config runs/publication/baselines_final.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/baselines_final.toml --profile h100 --lucas
```

## Optional. Fresh final training — 1 run

Not required to evaluate an existing checkpoint. If deliberately retraining, set the selected architecture
in `final_model.toml` first and retain new checkpoint-specific results.

```shell
unset CODLLM_EVALUATION_DATA_PATH
uv run --no-sync invoke hpc.build --config runs/publication/final_model.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/final_model.toml --profile h100 --lucas --duration 1w
```

## Collect selected-checkpoint results

```shell
uv run --no-sync invoke publication.report --root "$RUN_STORAGE_DIR/checkpoints/publication-v1"
```

Output: `logs/publication/selected_results.csv`. Copy verified results into `publication_progress.md`.

## Optional. Paired prediction interval

Choose comparable `.parquet` prediction files with identical evaluation rows; not training manifests.

```shell
read -r -p "First model prediction Parquet: " PUBLICATION_FIRST_PREDICTIONS
read -r -p "Second model prediction Parquet: " PUBLICATION_SECOND_PREDICTIONS
uv run --no-sync invoke publication.bootstrap --first "$PUBLICATION_FIRST_PREDICTIONS" --second "$PUBLICATION_SECOND_PREDICTIONS"
```

## Bayesian search

Legacy Bayesian specifications are not part of this launch sequence. Their short-budget objective needs
a separate fidelity check and v1 migration before use.
