# Publication plan

Updated: 2026-10-19. Phase 1 is complete.
[Progress](publication_progress.md) · [Decisions](publication_decisions.md) · [Commands](reference/commands.md)

Study how training choices affect historical cause-of-death coding on new descriptions and new archives.
The target is full ICD10h code sets. Current experiments use FLAN-T5-base.

## Evaluation

| Protocol | Split | Use |
|---|---|---|
| Row | Historical rows, with linked duplicate records kept together. | Matched comparison with familiar-description performance. |
| Grouped COD | Globally group normalized full COD descriptions and linked records. | Internal development and recipe screening. |
| Source held out | Hold out one whole archive; grouped-COD validation within the remaining archives. | Paired development comparison across five archives. |
| External | Separately curated archive, frozen before evaluation. | Final assessment; no tuning on its results. |

Use NFC normalization, case folding, and collapsed whitespace for COD grouping; retain punctuation and accents.
Grouping precedes tokenization, age/sex formatting, and augmentation. Target approximately 90/5/5
train/validation/test by rows without breaking groups. Validation/test/holdout data remain natural.
Build augmentation from training rows only. Exclude missing CODs and `historic_strings_en_2024`.

Primary input is COD-only; the completed metadata comparison uses COD+age+sex.
The five archives are Amsterdam, Copenhagen, Madrid, Belgium, and Ipswich. Reviewed languages are
`nl`, `da`, `es`, `nl`, and `en`, respectively; masterlist pretraining is English.

## Metrics and selection

- Select each checkpoint by internal-validation `macro_f1`.
- Screen grouped-COD recipes using equal-source reference macro F1
  (`pub_v1_source_mean_macro_f1_ref`), with pooled and per-source results alongside it.
- Select the final recipe using the paired five-source panel's equal-archive reference macro F1.
  Near-tie margins and transfer-support rules still require a decision.
- Report exact match, micro/sample F1, reference-supported macro F1, block metrics, and invalid codes.
  Include per-source/language, natural single/multi-COD, novel code-combination, unseen-COD,
  and original-training-frequency slices with support counts.
- Keep source transfer separate from language transfer. Historical cross-language eligibility requires
  the target code in other verified training languages and absent from the target language.
  The strict variant also excludes target/unknown-language supervision from adaptation resources.
  Empty slices are unavailable; legacy metric definitions remain unchanged.
- Disable final test evaluation during tuning. For source folds, also disable during-training holdout
  monitoring and evaluate the full held-out source only after selecting the checkpoint.

## Remaining experiments

| Phase | Work | Planned size |
|---|---|---:|
| 2 | Matched 4-/48-epoch pretraining recipes on five held-out archives. Confirm Copenhagen/Belgium pilots, then Amsterdam/Madrid/Ipswich. | 4 pilot + 6 remaining runs |
| 3 | Selected recipe under the matched row split; no-pretraining and no-synthesis controls on two confirmed source folds. | 1 + 4 runs |
| 3 | Normalized-COD lookup, character-linear multilabel, and masterlist-retrieval baselines on grouped COD, row, and two source folds. | 12 fits |
| 4 | Both recipes at 25% and 40% of the historical cohort, trained to convergence. | 4 runs |
| 4 | Both recipes at split seeds 101/202/303/404; training seed 777. | 8 runs |
| 4 | Winner at training seeds 111/222/333/444; split seed 777; rerun pretraining. | 4 runs |
| 5, optional | Both recipes at large scale; winner at XL; selected scale on two source folds. | 2 + 1 + 2 runs |
| 6 | Frozen internal/external evaluation and prespecified final baselines. | Inference + 3 baseline fits |

Existing optional specifications cover metadata portability and removal of base perturbation
(grouped COD and two source folds, three runs each). They require a separate decision; reuse a completed
grouped run when its recipe and stopping policy match. Bayesian search is outside the current sequence.

### Phase 2

Choose a shared floor/synthesis pair and retain both pretraining doses. The candidate TOMLs still use
provisional floor 0 / synthesis .30 with 4 and 48 pretraining epochs. Confirm input fields and stopping
policy before submission; the Phase 1 continuation policy has not been applied to later TOMLs.

All five source folds use internal validation from the training archives for checkpoint selection.
Freeze winner and runner-up after the complete paired panel. Development folds remain development evidence.

### Phase 3

Match the selected recipe in every setting except the declared ablation. The existing source-ablation
TOML uses Copenhagen and Belgium; confirm that choice before launch. Baseline thresholds are selected
on internal validation, with their access to historical labels and the masterlist recorded.

### Phase 4

Sample the whole cohort, not just training rows. Choose 25% or 40% after calibration and keep
`CODLLM_DATASET_SAMPLE_SEED=777` fixed while varying the split seed. Reuse only matching seed-777
calibration anchors. Report individual scores, mean, SD, range, and paired differences.
These studies measure reduced-cohort split and training sensitivity, not full-data uncertainty.

### Phases 5 and 6

Scale is optional. Record quality, training cost, inference latency, memory, and throughput.
A claim across all archives requires the larger model's full source panel.

Freeze the checkpoint, decoding, calibration, label mapping, external cohort, and baseline settings.
Evaluate with the selected checkpoint's original manifests; a new training run is not required.
Report uncertainty/abstention with thresholds calibrated on validation, and paired prediction differences
with 95% COD/linked-record cluster-bootstrap intervals. These intervals are separate from seed variability.
Release the inference interface, input contract, model/data documentation, and applicable licenses.
A later all-data refit has its own checkpoint and evaluation record.

## Checkpoint consistency

For each selected epoch B, use ten completed evaluations B-9 through B, anchored to the actual
macro-F1-selected checkpoint for every metric. Exclude partial epochs and restored-best reevaluations.
When a continuation retains its parent checkpoint, label its window `inherited_parent`.
Never splice parent/child trajectories or fill missing histories.

Report selected score, mean, median, population variance/SD, min/max, endpoint minus preceding-nine
median, and linear slope. Reverse the uplift sign for loss. These are descriptive checkpoint statistics,
not seed uncertainty, confidence intervals, or an alternative selection rule.
[Inputs and reproduction](reference/commands.md#checkpoint-consistency) are retained.

## Execution

Experiment settings live in `runs/publication/*.toml`; editing this plan does not migrate them.
Proceed one stage at a time. Record Lucas's decisions before `source_pair`, `recipe`, `reduced`,
and `final` approvals. Keep completed output roots, private manifests, and prediction exports intact.
