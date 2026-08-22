# codLLM publication experiment plan

## Goal and selection principle

The goal is the strongest defensible codLLM model, with enough controlled evidence
to explain:

1. where the balance-floor benefit saturates;
2. how much fine-tuning synthetic multi-COD data is beneficial;
3. how much masterlist pretraining is beneficial;
4. whether the selected recipe is stable to split/preparation and training seeds;
5. whether it transfers across historical sources and benefits from model scaling.

Model selection uses validation macro F1. Exact match, micro F1, sample F1,
Jaccard, seen/unseen-string performance, single/multi-COD subsets, rare-label
support buckets, per-source results, runtime, and generated-row counts remain
secondary outcomes. The test set stays disabled until the final frozen model.

The historical test set has influenced earlier development and must be described
as a legacy test set. UniCin or another genuinely external corpus is the preferred
final generalization assessment when it becomes available in the pipeline.

## Frozen controls

Unless a specification explicitly overrides a setting, all experiments inherit
`runs/publication/base.toml`:

- `google/flan-t5-base` during recipe selection;
- full 120-epoch ceiling with validation macro-F1 early stopping;
- training seed 777 and data seed 777;
- COD, age, and sex inputs;
- fixed harmonization, standardization, scheduler, perturbation, and label-count
  policy;
- online W&B logging with final test and uncertainty evaluation disabled;
- exact declared pretraining epochs, without pretraining early stopping.

Each cell has an isolated output directory and may resume across scheduler slots.
Dataset, curation, code commit, effective configuration, cache key, stopping reason,
and W&B run ID must be retained.

## Stage 1: full-data response curves

Run the three curves on the same full-data split at seed 777:

| Question | File | Values |
|---|---|---|
| Balance floor | `balance_floor_curve.toml` | 0, 100, 200, 300, 450, 600, 900 |
| Fine-tuning synthetic ratio | `multicod_synthetic_curve.toml` | 0, .15, .30, .45, .60, .80, 1.00 |
| Pretraining epochs | `pretraining_dose_curve.toml` | off, 4, 8, 16, 32, 48 |

The pretraining curve fixes target-per-label at 12 and pretraining synthetic ratio
at .30. Report its actual pretraining rows, optimizer updates, and examples
presented; epochs alone do not describe exposure when the generated dataset changes.

Plot every cell and do not infer a plateau from the selected maximum alone. A
practical saturation point is the smallest setting within .002 validation macro F1
of the best observed value when the next tested increase also improves by less than
.002. Because this stage has one seed, describe that point as a provisional
saturation point rather than a confidence-bound conclusion.

## Stage 2: full-fidelity interaction confirmation

After all curves finish, set the low/high levels in
`interaction_confirmation.toml` to the two scientifically plausible neighboring
values for each of the three variables. Run the resulting 2x2x2 design at full
fidelity (8 runs). The checked-in provisional levels are:

- floor: 300 and 450;
- fine-tuning synthetic ratio: .30 and .60;
- pretraining: 16 and 32 epochs.

Estimate the three main effects and all pairwise interactions. Select a provisional
winner and runner-up using validation macro F1, with paired secondary-metric checks.
If candidates differ by less than .002 macro F1, prefer the lower floor, lower
synthetic ratio, or shorter pretraining dose.

Copy the frozen recipes into `candidate_winner.toml` and
`candidate_runner_up.toml`. Every later specification inherits these two files, so
candidate settings must be changed only there.

The broad 24-epoch Bayesian search is not part of the primary publication sequence:
shortened training may reorder configurations, while full-fidelity Bayesian trials
cannot safely span the current single-allocation W&B agent lifecycle. The existing
Bayesian implementation remains available for optional hypothesis generation, but
it must not replace the full-fidelity curves or confirmation design.

## Stage 3: calibrate the reduced-data robustness study

Run winner and runner-up at 25% and 40% of the data using
`reduced_data_calibration.toml` (4 runs). Both fractions keep the full 120-epoch
ceiling and the same early-stopping policy; epochs are not shortened.

Use 25% only if it retains adequate support for the rare-label buckets used in the
paper and preserves the winner/runner-up ordering and approximate paired difference
seen at 40%. Otherwise conduct the sensitivity study at 40% by updating its TOMLs
before submission.

`CODLLM_DATASET_SAMPLE_SEED=777` fixes the sampled cohort independently of
`CODLLM_DATA_SEED`. This prevents split-seed changes from silently selecting a
different 25% cohort. Reduced-data runs still repeat full masterlist pretraining, so
their total compute reduction is less than the fine-tuning fraction suggests.

## Stage 4: estimate seed sensitivity economically

### Split/preparation sensitivity

Use `reduced_split_sensitivity.toml` for paired winner/runner-up comparisons on
data seeds 101, 202, 303, and 404, while fixing training seed and cohort-sampling
seed at 777 (8 new runs). Combine these with the seed-777 25% calibration cells to
obtain five paired split/preparation observations per candidate.

Changing `CODLLM_DATA_SEED` changes the train/validation/test allocation and
seeded split-time preparation such as synthetic generation and label shuffling.
Therefore report this as split/preparation sensitivity, not pure split variance.

### Training-pipeline sensitivity

Use `reduced_training_seed_sensitivity.toml` for the winner at training seeds 111,
222, 333, and 444, with the sampled cohort and data seed fixed at 777 (4 new runs).
Combine these with the seed-777 winner calibration cell for five observations.
Because pretraining is rerun, this estimates end-to-end training-pipeline seed
sensitivity rather than fine-tuning-only variance.

For both analyses publish every seed result, mean, standard deviation, range, and
paired winner-minus-runner differences. A paired interval across five splits may be
shown with an explicit small-sample warning. Do not call these reduced-data
intervals confidence intervals for the full-data model: scale can change both the
mean and variance.

Test-sample uncertainty is a separate quantity. On final predictions, compute a
paired cluster bootstrap grouped by normalized COD text so duplicate causes are
not treated as independent observations.

## Stage 5: model scale

After freezing the recipe, run `model_scale_confirmation.toml`:

1. winner on `flan-t5-large`;
2. runner-up on `flan-t5-large`;
3. winner on `flan-t5-xl`.

All three use full data, seed 777, full-fidelity early stopping, and validation-only
selection. This separates recipe selection from model-scale selection while still
testing whether the runner-up overtakes the winner at the next scale.

If XL is clearly still improving and affordable, it is the final scale. Otherwise
prefer the smaller model when the macro-F1 difference is below .002 or the runtime
and deployment cost are disproportionate to the gain.

## Stage 6: source transfer and external validation

Run `source_transfer_validation.toml` for the frozen base-model winner with each of
Amsterdam, Copenhagen, Madrid, Belgium, and Ipswich held out in turn (5 runs).
Report each source, mean, standard deviation, and worst-source result. This measures
cross-source transfer and is not a substitute for seed sensitivity.

Evaluate UniCin once, without retraining or tuning on UniCin, after its dataset is
available as a configured source or inference dataset. No runnable UniCin TOML is
included yet because the repository currently has no UniCin source definition or
source ID; inventing one would create a job that fails or, worse, evaluates the
wrong data.

## Stage 7: final frozen evaluation

Choose the scale in `final_model.toml`, freeze all settings, and then run it once.
This is the only active publication specification with final test and uncertainty
evaluation enabled. Do not change hyperparameters after viewing these results.

Report:

- exact accuracy, macro/micro/sample F1, Jaccard, hamming metrics;
- seen/unseen-string and single/multi-COD subsets;
- per-source and rare-label support buckets;
- label-count error and false multi-label rate;
- paired normalized-COD cluster-bootstrap intervals;
- the reduced-data seed sensitivity results, clearly labeled as reduced-data;
- parameter count, training exposure, stopping epoch, and compute/runtime.

Release one prespecified checkpoint. An ensemble may be a separate secondary
system, but the main reported model must not be chosen by taking the best seed.

## Compute budget

The active plan contains:

- 20 primary curve runs;
- 8 full-data interaction runs;
- 4 reduced-data calibration runs;
- 8 additional paired split/preparation runs;
- 4 additional training-seed runs;
- 3 scale-confirmation runs;
- 5 leave-one-source-out runs;
- 1 final frozen run.

This is 53 runs, of which 16 use reduced fine-tuning data. Stages are gated: do not
queue interaction jobs before curve review, reduced-data jobs before candidates are
frozen, scale/source-transfer jobs before robustness review, or final evaluation
before every selection decision is locked.
