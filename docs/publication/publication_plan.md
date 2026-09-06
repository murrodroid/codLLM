# codLLM publication plan: training for historical generalization

Updated: 2026-09-06. Agreed direction: a broader engineering/generalization study, without a deadline-driven scope.

## 1. Central question and contribution

> How should we train and evaluate a multilingual historical cause-of-death coding system so that it
> remains useful on new descriptions and new archives?

The intended contribution is a reproducible system and an evidence-backed training/evaluation recipe,
not just a maximum validation score or a single synthetic-data ablation. Organize the paper around:

1. **Evaluation validity:** how performance changes between familiar descriptions, unseen COD descriptions,
   and entirely held-out archives.
2. **Training choices:** which combinations of full-code masterlist pretraining, balancing, and multi-COD
   synthesis improve those settings, and where the benefits stop.
3. **Transfer:** whether a code learned from other sources or languages can be applied to a new description,
   distinguishing this from codes absent from historical fine-tuning.
4. **Research usefulness:** robustness to seeds, model size, missing metadata, and uncertainty; a usable,
   documented checkpoint with clear limits.

A broad study still needs this common argument. Include positive and negative findings that answer it;
put the full experiment history and secondary grids in supplementary material. Novel architecture is
not a prerequisite for an engineering contribution: ARR explicitly welcomes engineering experiments,
released models, and informative negative results.
[ARR contribution types](https://aclrollingreview.org/cfp)

Do not select a conference deadline first. Choose a venue after the evidence and release are ready.
Neither this plan nor the number of experiments guarantees acceptance.

## 2. What the completed curves establish

Keep the 20 completed response-curve runs as exploratory evidence; do not rerun them wholesale.
The detailed values remain in [publication_progress.md](publication_progress.md).

| Curve | Current evidence | Implication for the new study |
|---|---|---|
| Floor | Floor 0 leads at 0.744945; 450 leads the positive floors at 0.731604. | Compare 0 and 450; do not assume positive balancing helps. |
| Fine-tuning synthesis | Ratio .30 leads at 0.735801; .60 is close at 0.733740. | Compare .30 and .60 jointly; retain a no-synthesis control. |
| Full-code pretraining | 4 epochs: 0.733186; 48: 0.732698. | Keep both: the difference is only 0.000488, not evidence that long pretraining is inferior. |

These are conditional, single-seed comparisons with different other settings. They do not establish
that their individually best settings form the best joint recipe. The floor and pretraining values
are provisional history peaks until matched to retained checkpoints; compare all secondary metrics
at the same selected checkpoint, not their independent maxima.

The old unseen-string metric is not the proposed unseen-COD metric. The old split, input fields,
reference-data policy, and metric version must remain attached to these results. A new evaluation
protocol cannot be retroactively established by renaming columns.

## 3. Freeze the evaluation contract before more expensive training

### 3.1 Four complementary protocols

| ID | Construction | Question and role |
|---|---|---|
| P-row | Row-level split within the historical source pool, with real duplicate-record leakage removed. | Familiar-archive performance, including recurring descriptions. Secondary continuity with earlier work. |
| P-cod | Group by normalized COD text globally across sources before splitting. | Performance on new descriptions from the represented archive mixture. Primary internal development protocol. |
| P-source | Leave one entire historical archive out; use P-cod train/validation splits on the remaining archives. | New-archive transfer. Central development comparison across five folds, not a late winner-only check. |
| P-external | A separately acquired/frozen archive such as UniCin, never used for tuning. | Final external assessment of the frozen system and prespecified baselines. |

Source shift and description novelty are different axes. A held-out archive may contain COD strings
seen in another archive; report its seen-COD and unseen-COD slices rather than assuming all rows are novel.
Likewise, a grouped split is not a new-language or new-archive evaluation.

Natural distribution shifts can expose failures hidden by in-distribution evaluation; that motivates
making P-source central, not an assumption that any particular recipe will generalize.
[WILDS](https://proceedings.mlr.press/v139/koh21a.html)

### 3.2 Data and split rules

- Version source files, curation rules, ICD10h vocabulary, row IDs, exclusions, and split manifests.
  Publish source/language counts, label support, natural multi-COD rates, missing fields, and overlap counts.
- Keep a versioned, conservative COD normalization: Unicode normalization, case folding, and whitespace
  normalization; preserve meaningful punctuation and diacritics. Do not translate, stem, or merge clinical
  synonyms into the grouping key. Audit normalization collisions and near-duplicate examples.
- Derive the key from the full COD field before tokenization, augmentation, and age/sex formatting.
  Group globally, not separately within each source. Where repeated persons/records can be identified,
  keep those linked records together too. Never split a group to force a rare code into training.
  Exclude genuinely missing CODs under a fixed documented rule and report their counts.
- Use approximately 90/5/5 train/validation/test proportions for P-cod, accepting deviations required by
  groups. The implemented deterministic allocator targets row counts without breaking groups; it does not
  promise label stratification. Audit source/label coverage and publish unsupported-code counts before launch.
- Build split-specific vocabularies/statistics and all synthetic rows from permitted training data only.
  Perturbed copies and synthetic constituents must not originate in validation, test, or the held-out source.
- Exclude `historic_strings_en_2024` from the primary historical split pool; it is an auxiliary reference,
  not a sixth independent archive. Any later inclusion is a declared resource ablation.
- A public masterlist is an allowed domain resource, but log its text/code/language overlap with evaluation.
  Separate masterlist pretraining from historical supervision. Do not claim its covered codes were never taught.
- Use COD-only as the primary portable input proposed for the new campaign; compare COD+age+sex as a
  matched deployment ablation. This intentionally differs from the old curves' COD+age+sex input:
  new archives may lack reliable metadata. Do not attribute differences between campaigns to one setting.
- Validation/test/holdout rows remain natural, unbalanced, and unsynthesized. Report natural multi-COD
  performance separately from any artificial composition challenge.

### 3.3 Development versus final evidence

All five existing archives may inform development through P-source. Consequently the mean of those
folds is a development comparison, not an unbiased external estimate of the selected recipe.
Prespecify Copenhagen and Belgium as the two pilots before seeing new model scores, subject to the
score-blind support audit. Copenhagen tests Danish without another Danish historical archive; Belgium
tests a new Dutch-language archive with Amsterdam still available. Then extend the same paired
comparison to all five. Do not choose folds by model scores.

The completed audit supports this as a useful transfer contrast, but the pilot choice remains for Lucas
to confirm. Copenhagen has zero recorded natural multi-COD targets; do not reuse the pilot rationale
as evidence that these are two multi-COD benefit tests. Phase 3 mechanism-fold choices are separate.

For each P-source run, select its checkpoint using only validation from the training archives.
Explicitly disable during-training holdout monitoring with `CODLLM_HOLD_OUT_EVALUATE_PER=none`;
use final `holdout/full/*` predictions for the comparison. Disabling final test evaluation alone
does not disable this separate holdout evaluation path.

Repeated validation-based selection can itself overfit. Additional seeds quantify variability but do
not repair that selection bias.
[Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)

Keep final test evaluation disabled during tuning. Previously consulted test data remain legacy evidence,
even if repartitioned. Freeze UniCin's inclusion/label-mapping rules, decoding, calibration, baselines,
and analysis before examining its results; use a separate external development partition if adaptation
is later desired. Check for shared provenance or duplicates with existing sources.

If no genuinely external archive becomes available, report P-cod/P-source evidence honestly and narrow
the external-validity claim. Acquiring one is preferable to calling tuned folds independent tests;
nested source-wise model selection is a more expensive alternative, not something seed averaging replaces.

### 3.4 Completed Phase 0a audit and decisions still open

The 2026-09-06 HPC audit is preserved in [data_audit.json](data_audit.json). Detailed counts,
limitations, and decision options are in [publication_data_audit.md](publication_data_audit.md).
These are score-blind processed-data observations, not model results or final split statistics.

- There are 1,525,598 historical rows after excluding the 3,295-row English reference. Removing the
  886 missing-COD rows leaves 1,524,712 before any further identity deduplication or sampling.
- Copenhagen contributes 501,855 rows but only 2,525 distinct CODs and 389 codes. Inspect full-size
  grouped-split composition and group-size concentration before full screening; a 0.2% smoke test
  cannot establish that the seed-777 full-data partitions have adequate source/code coverage.
- Natural multi-COD prevalence is 0% in Copenhagen, 16.05% in Belgium, 18.52% in Amsterdam,
  42.28% in Ipswich, and 5.08% in Madrid, before missing-COD removal. The source's annotation
  convention may contribute to these differences; the audit does not establish why.
- Amsterdam, Copenhagen, and Madrid account for 94.67% of rows after the two stated exclusions.
  Retain pooled and per-source reporting alongside the already planned equal-source summary.
- Between 56.2% and 68.0% of codes in the four non-Copenhagen archives have fewer than ten
  source-local rows. Global, grouped, and cross-lingual support cannot be inferred from those counts.
- Amsterdam and Copenhagen share 459 distinct CODs (18.18% of Copenhagen's distinct descriptions
  before splitting). Pairwise text overlap is not an actual training-exposure rate or label overlap.

Readiness adjustment: obtain a score-blind full-size split-support review before committing to the
eight full-data screening cells. Include source/row/group/code coverage, largest connected groups,
historically unsupported targets, and historical versus all-adaptation language-transfer eligibility.
Record raw-source/processed-cache identities as well: the current JSON includes a language-curation
hash but does not provide those data fingerprints. The new Phase 0c `publication.audit-splits` command
collects these measurements using training's exact original-partition construction, once for the eight
Phase 1a cells. Run it with the training profile/user to use the same processed-cache paths. Its real-data
execution and review remain pending; implementation is not a completed validation.

The report distinguishes source transfer, historical cross-language transfer, and strict planned
all-adaptation transfer after masterlist exposure. Support is reported as rows, target occurrences,
distinct codes, CODs, connected groups, and code-language pairs, overall and by source/language.
Existing prepared-cache original partitions are compared when present. The audit neither trains models
nor generates augmented datasets; lexical overlap introduced by perturbation/synthesis still requires
the smoke/runtime provenance checks. It does not choose a support threshold, change a split, or approve a phase.

Lucas retains the choices: confirm the two Phase 2 pilots; choose whether Phase 3 retains a single-COD
control or uses two multi-COD-rich sources; consider an additional equal-description sensitivity analysis;
and agree a minimum-support rule for transfer-based tie-breaks after the counts exist. Retain all sparse
slices descriptively rather than converting unavailable evidence into zero performance. No audit finding
changes the candidate hyperparameters, sample fraction, or approved phases automatically.

## 4. Make the transfer metric precise

### 4.1 Existing source-transfer metric: retain and report it

The current implementation in [metrics.py](../../src/codllm/metrics.py) indexes each historical training
code by `source_id`, not language. It is already wired into evaluation; it has not been central to the
publication reports. Its three target-code buckets are:

- same-source label: the code occurs in training from the evaluation row's source;
- source-transfer label: the code occurs in training, but only in other sources;
- unseen label: the code is absent from that training-source index.

A source-transfer label could have been taught in the same language by another archive. These values
must therefore retain the name **source transfer**, not be presented as cross-lingual accuracy.

For multi-COD rows, `source_transfer_label_accuracy` is full-code-set exact match on rows containing
at least one source-transfer target. `source_transfer_label_macro_f1` also scores the complete labels
and predictions of those rows. `source_transfer_label_recall` measures recovery of the eligible
target-code occurrences themselves. Rows containing different target types can enter multiple buckets.

Report these metrics with their eligible row, target-occurrence, and distinct-code counts. In a
leave-one-source-out fold, all historically seen codes necessarily come from other sources; this
slice separates known-code transfer from historically unseen codes, not languages.

### 4.2 Add the actual cross-lingual label-transfer assessment

For an evaluation target code `c` in verified language `L`, define historical cross-lingual eligibility as:

1. `c` occurs in the real historical training partition in at least one verified language other than `L`;
2. `c` has no historical training occurrence in `L`;
3. training rows with unresolved language do not make that absence claim uncertain.

Construct a label-to-language index independently of the label-to-source index. Use curated source
language metadata only where the archive is demonstrably monolingual; otherwise use reviewed row-level
language annotations. City/country is not a language label. Mark mixed/unknown cases unresolved and
report their coverage; do not silently place them in the strict cross-lingual bucket.

The reviewed inventory is [source_languages.toml](../../data/curation/source_languages.toml): Amsterdam
Dutch (`nl`), Copenhagen Danish (`da`), Madrid Spanish (`es`), Belgium Flemish/Belgian Dutch (`nl`),
and Ipswich English (`en`). The masterlist and English historical reference are `en`.
Lucas's clarification that this Belgian dataset is Flemish supersedes the thesis's French/Dutch entry.
Flemish is treated as a Dutch variety, not a mixture assigned two language labels. Amsterdam↔Belgium
is therefore source transfer, not cross-lingual transfer. Reviewed row overrides can represent actual
exceptions; country names and automatic language guesses are not substituted for annotation.

Keep separate exposure inventories for original historical training rows, fine-tuning augmentation,
auxiliary references, and masterlist pretraining. The current source-index filter excludes several
synthetic/masterlist prefixes but does not exclude `historic_strings_en_2024`; correct that provenance
policy in the new metric version rather than treating the reference list as an archive.

Report both **historical cross-lingual transfer** and, where supported, the stricter slice with no
target-language supervision in any of our adaptation stages. A target-language masterlist example
invalidates the latter, even if historical training contains the code only in another language.
Base-model pretraining exposure is unknown; neither definition implies never-before-seen knowledge.

For this assessment report:

- exact-match accuracy on eligible single-COD rows;
- complete-set exact match and macro/micro/sample F1 on multi-COD rows containing eligible targets;
- eligible-target recall, including a macro average over eligible code-language pairs;
- support by target language, source, code, row, and target occurrence;
- intersection with unseen COD, historical code frequency, and masterlist exposure.

Do not remove unrelated predictions before computing row accuracy/F1: that would hide false positives.
Use N/A, not zero, for empty slices. Keep legacy metric keys unchanged and version the new keys.
Archive and language effects remain partly confounded when only one archive represents a language.

If natural eligible support is too small, first report that limitation. A later controlled
code-language withholding experiment can remove all training rows containing selected code-language
pairs, preserve other-language examples, and rebuild augmentation. It is a separate intervention,
not a replacement for natural transfer. Do not translate evaluation descriptions to manufacture evidence.

### 4.3 Replace unseen-string reporting with unseen-COD reporting

Use the versioned COD-only key against unaugmented historical training CODs for the primary
seen/unseen split. Also flag matches to augmented fine-tuning text and masterlist/reference text,
so “unseen historically” is not confused with “never presented during adaptation.”

The current implementation compares decoded tokenized inputs, potentially including age/sex and
truncation. Preserve its old results as legacy unseen-string values. Rescore old retained predictions
only when original row IDs and training manifests can be recovered; otherwise mark the new metric
unavailable. Rescoring an old row-split model does not turn it into a P-cod-trained model.

### 4.4 Metric and selection hierarchy

Use full ICD10h codes as the primary target. Three-character category scores are useful diagnostics,
not a pretraining-target change or a substitute for full-code accuracy.

For all protocols report overall, per-source, and per-language macro F1; exact match; micro/sample F1;
natural single/multi-COD slices; unseen-COD results; both transfer assessments; and training-frequency
buckets defined from original historical rows before upsampling. Always provide support.
For natural multi-COD rows, distinguish novel combinations of historically known codes from targets
containing historically unseen codes; these test different forms of compositional generalization.

Version the macro-F1 label policy before training. Keep the existing `macro_f1` for continuity and
checkpointing; add a clearly named reference-supported macro F1 with a fixed set of reference codes
per evaluation slice for cross-model comparison. Report prediction-only/invalid-code errors and
micro F1/exact match alongside it. Do not change the label universe silently or average model-specific
sets as if they were identical.

Selection order:

1. Within each training run, retain the best internal-validation `macro_f1` checkpoint with the same
   stopping policy. Never select an epoch using the held-out archive or a sparse transfer bucket.
2. Screen P-cod recipes by equal-source mean reference-supported macro F1; report pooled results too.
3. Choose the final recipe using equal-archive mean reference-supported macro F1 across the five
   development P-source folds. Inspect every archive, not only that mean.
4. Prespecify .002 absolute macro F1 as a practical near-tie band, not a significance test.
   Within it, prioritize adequately supported cross-lingual eligible-target macro recall, then
   unseen-COD macro F1, and check worst-source and natural multi-COD performance.
5. Flag any source loss exceeding .01 absolute macro F1 for explicit review before freezing; this is a
   proposed practical guardrail, not a proven noninferiority margin. If tradeoffs remain unresolved,
   present the alternatives rather than inventing a universally best model.

Finalize these margins and minimum support rules after the score-blind split-support review, before comparison.
The aggregate Phase 0a audit does not contain transfer-eligible support counts and cannot settle that rule.
Do not collapse all metrics into an arbitrary weighted score or change the primary outcome after seeing
which one a favorite configuration wins.

## 5. Experiment sequence and gates

The phases below replace the previous numbered training sequence. Counts are training cells, not
scheduler allocations; resumptions are not new experiments. The v1 implementation and TOMLs are now
prepared. Follow [publication_runs.md](publication_runs.md) one stage at a time; no new HPC jobs have
been submitted by this implementation update.

### Phase 0 — evaluation readiness and recovery

No full training campaign yet.

1. Recover selected-checkpoint identities, prediction/row alignment, effective configs, and manifests for
   completed curves where possible. Retrieve existing source-transfer metrics before paying to retrain.
2. The source/language/label/overlap inventory is complete; see Section 3.4. Run Phase 0c's
   `publication.audit-splits` and review full-size grouped-split/source/transfer support before freezing
   the remaining choices and data versions. Retain its JSON, Markdown review, and data/code fingerprints.
3. Verify the implemented grouped splitting, COD-only novelty, language/provenance-aware transfer,
   and content-addressed selected-model prediction exports against the real-data audit.
4. Test duplicate COD with different age/sex, same-language different archives, other-language-only
   codes, unknown-language exposure, masterlist-only codes, mixed multi-COD buckets, empty slices,
   and absence of held-out rows/constituents from training.
5. Verify a tiny CPU-capable end-to-end fixture and a short HPC smoke test, including checkpoint resume,
   metric persistence, and W&B logging to `codllmdev/codllm`.

Gate: matching manifests and metrics can be audited independently of the W&B run page.
The CPU fixture is implemented and tested, and Phase 0a's aggregate audit is complete. The HPC smoke
and full-size split-support review remain separate requirements; neither is certified by that JSON.

### Phase 1 — joint recipe screening on unseen descriptions

Use full-data P-cod, FLAN-T5-base, training/data seeds 777, and the primary COD-only input.

Replace the placeholder grid with:

| Factor | Levels |
|---|---|
| Balance floor | 0, 450 |
| Fine-tuning synthetic ratio | .30, .60 |
| Full-code pretraining epochs | 4, 48 |

This is eight converged training runs. Add two controls at floor 0:

- no masterlist pretraining, fine-tuning synthesis .30;
- 48-epoch masterlist pretraining, no fine-tuning synthesis.

Total: 10 runs. Keep base perturbation and pretraining synthesis fixed so each control isolates its
declared factor. A floor contrast includes the pipeline's upsample-copy perturbations; do not describe
it as a pure class-weighting experiment.
Floor 450 is the strongest observed positive-floor alternative; floor 300 remains a legacy anchor,
not the assumed optimum. Revisit intermediate floors only if the new positive-floor comparison helps.

Keep the 120-epoch ceiling and patience 10. If a run ends at the ceiling while still improving, mark it
budget-censored and extend the relevant comparison consistently before claiming peak performance.
Retain full ICD10h pretraining targets, target-per-label 12, pretraining synthetic ratio .30, and
declared pretraining doses without early stopping/best-checkpoint substitution.
Record warmup and scheduler exposure too: varying epochs with a fixed warmup ratio changes the number
of warmup updates. This tests the full dose recipe; a matched-schedule follow-up is needed before
attributing any difference exclusively to longer representation learning.

Long pretraining is a serious hypothesis: repeated domain supervision may help rare/unseen descriptions
and codes absent from historical training. It does not introduce new medical information merely by
adding epochs, and flat accuracy does not demonstrate deeper representations. Test the hypothesis
through P-source, cross-lingual, and masterlist-exposure slices.

Estimate conditional effects and pairwise interactions descriptively; one seed does not support
precise population-level interaction claims. A saturation point from the old .002 rule remains
provisional, and a positive-floor benefit under transfer would justify a targeted additional floor,
not an automatic repetition of all seven levels.

Gate: select a promising floor/synthesis pair, retaining both its matched 4- and 48-epoch recipes.
Use performance averaged across the two doses to screen structural settings; retain a materially
better differently configured contender for the optional challenge below.

### Phase 2 — early, paired new-archive evaluation

For the matched short/long pair, hold out each of:

- `amsterdam_1854_1926`;
- `copenhagen_may2025`;
- `madrid_1905_1927`;
- `belgium_1920_1930`;
- `ipswich_1871_1911`.

Start with the proposed Copenhagen+Belgium pilot archives after Lucas confirms the source choice:
two recipes x two folds = 4 runs. Copenhagen tests a single-COD Danish archive; Belgium tests a
multi-COD-containing Dutch archive with another Dutch archive still in training.
Then complete the remaining three archives: 6 more runs, reusing the pilot results.
Total: 10 runs, with internal P-cod checkpoint selection and full natural holdout evaluation.

If Phase 1 identifies a substantially better structural contender, challenge the pair on the two pilot
folds first (+2); complete its remaining folds (+3) only if it remains competitive. Include that search
history in the report. Do not compare a two-fold candidate mean against another candidate's five-fold mean.

This phase directly tests whether the long dose improves new-archive/cross-lingual performance while
preserving within-source quality. Keep 48 epochs when it earns that choice on the selection hierarchy.
If supported outcomes remain practically tied, 48 may be the declared product preference given the
user's training budget; explicitly call it a preference, not an established improvement. It does not
increase the deployed architecture's size or per-example inference cost.

Gate: freeze the winner and a meaningful runner-up using the complete comparable development panel,
not just the largest archive or the most favorable language. Add >48-epoch dose tests only if transfer
evidence or an unresolved learning curve gives a concrete reason; no automatic escalation.

### Phase 3 — explain the recipe and compare systems

Train the selected recipe once under P-row, with the same source pool, resource policy, input fields,
and stopping rules as its P-cod counterpart: 1 additional run. Report split sizes and label/source
composition alongside the protocol contrast. The old curves alone cannot isolate the effect of grouping
because other controls also changed. Do not select a new recipe from this secondary row-split result.

Repeat the no-pretraining and no-fine-tuning-synthesis ablations on the selected full recipe for two
prespecified development source folds: 4 additional runs. Phase 1 controls were anchored at floor 0;
these ablations must instead match the selected recipe in every other setting.

The current TOML uses Copenhagen+Belgium, but the audit changes the interpretation: Copenhagen can
test single-COD harm or unnecessary extra predictions, not natural multi-COD benefit. Lucas can retain
that control, use Belgium+Ipswich for two multi-COD-rich folds (still 4 ablation runs), or add Ipswich
to the existing panel (6 ablation runs). No option is selected by this plan update. Adjust the relevant
TOML and launch guide only after that choice. These are minimum mechanism checks; use all five folds
for any claim that the effect holds across archives generally.

Fit inexpensive training-only baselines on the same manifests:

- normalized-COD lookup with a documented fallback;
- a character n-gram linear multilabel model;
- a label-description retrieval baseline with access to the same permitted masterlist.

The runnable baseline panel contains P-cod, P-row, and the two pilot source folds (12 fits).
Lookup chooses the most frequent complete label set per normalized COD, falling back to the global
training-mode set. The linear baseline uses character 2–5-grams and one-vs-rest logistic SGD, selecting
its threshold from .20/.35/.50/.65 on internal validation. Retrieval uses nearest character-TFIDF
masterlist descriptions; it does not fit on held-out descriptions. Each method exports the same
versioned metrics and row identities. These CPU estimators currently use the existing H100 allocation
profile for submission compatibility; their runtime has not been measured on the full dataset.

Tune thresholds/label counts on internal validation only. Report historical-label-only and
masterlist-assisted resource access explicitly, including codes a classifier cannot emit.
Lookup success on P-row but failure on P-cod is informative, not a reason to omit it.

Prioritized extensions, each tied to a remaining question:

1. **Metadata:** matched COD-only versus COD+age+sex on P-cod and two source folds; measure missing-field
   behavior without retraining on evaluation data. This tests archive portability.
2. **Perturbation:** remove base perturbation while holding synthesis/floor fixed; if necessary separately
   distinguish floor copies from copy perturbations. Do not call a combined switch a single mechanism.
3. **Architecture:** a credible multilingual encoder classifier or multilingual text-to-text alternative,
   trained with the same resource and evaluation contract. Choose after reviewing relevant baselines,
   not because it is newest; avoid a large architecture sweep.

Report compute-matched sensitivity only where claiming augmentation itself is superior independently
of extra training exposure. Record actual original/augmented rows, optimizer updates, tokens/examples
presented, GPU-hours, and stopping epochs. Convergence comparisons alone do not isolate extra-compute effects.

### Phase 4 — affordable seed sensitivity without shortening convergence

Use the fixed winner/runner-up from Phase 2. Calibrate both at 25% and 40% historical data:
4 runs, full training ceiling and the same stopping policy. The prepared TOMLs explicitly use
whole-cohort sampling for calibration and both seed studies: sample seed 777 selects the historical
cohort before grouped splitting. This makes calibration seed-777 cells reusable as exact anchors for
the chosen fraction. Evaluation populations change between 25% and 40%; this is a feasibility/fidelity
check, not a pure learning curve. A separate training-only learning curve is supported through
`CODLLM_TRAIN_SAMPLE_FRACTION`, but is not mixed into this seed study.

Choose the smaller fraction only if its supported rare/transfer slices and paired recipe behavior are
reasonably consistent with 40%. One matching ranking is a feasibility check, not proof of fidelity.
If rankings reverse, retain both candidates and use 40% or defer the full-data superiority claim.

The audit implies approximately 381k/610k cohort rows at 25%/40% before augmentation, not uniformly
distributed independent descriptions. Inspect rare-code, unique-COD, and transfer support per source
at both fractions. The high source-local rare-code fractions make total row count an insufficient
criterion. The checked-in 25% seed-study default remains provisional until Lucas reviews calibration.

On the chosen fixed cohort, pin `CODLLM_DATASET_SAMPLE_SEED=777` and perform:

- split/preparation sensitivity: both recipes at data seeds 101, 202, 303, 404, training seed 777;
- training-pipeline sensitivity: winner at training seeds 111, 222, 333, 444, data seed 777.

These are 8 + 4 new runs. Reuse calibration seed 777 only if the chosen cohort, manifest construction,
and training policy match exactly; otherwise create matching seed-777 anchors. Do not reuse 25% anchors
for a 40% study or a training-only subsample as an anchor for a different whole-cohort protocol.
The split study changes grouping/preparation and evaluation membership, so call it split/preparation
sensitivity; keep the cohort constant. The training study keeps the split fixed.

Publish all values, mean, standard deviation, range, and paired recipe differences.
Five observations give a limited variance estimate, not a precise stability guarantee. Reduced-data
intervals are not confidence intervals for full-data performance. Bootstrap uncertainty from fixed
predictions is a different quantity.

If each training seed reruns pretraining, this is end-to-end pipeline sensitivity. Reusing a single
pretrained checkpoint saves compute but changes the estimand to fine-tuning-only sensitivity; declare
that change. One additional full-data seed is useful if affordable, not a requirement to repeat the
entire full-data search or a basis for picking the luckiest seed.

### Phase 5 — scale only after the transfer evidence is credible

Compare the frozen winner and runner-up on FLAN-T5-large under P-cod (2 runs).
Test the winner on XL only if large gives a meaningful gain and deployment is feasible (+1).
Check the selected larger scale on the two prespecified source folds (+2); complete all five if
claiming an across-archive scale improvement. Smaller-model transfer results do not establish that
the larger released model transfers equally well.

Report quality versus training and inference cost, latency, memory, and throughput. Choose one primary
release checkpoint before final assessment; a smaller practical variant may also be released.
Scaling is not a substitute for resolving leakage, sparse transfer support, or an unstable recipe.

### Phase 6 — frozen assessment and a useful release

Freeze recipe, architecture, seed policy, decoding, curation, and thresholds. Evaluate the selected
checkpoint and prespecified baselines on reserved internal data and P-external without selecting among
them using external scores. Reuse a valid selected checkpoint when possible: final evaluation does not
inherently require another training run.

Report per-archive/per-language results and all prespecified slices, including invalid codes,
false multi-COD predictions, missed codes, and code-count errors. Review a stratified, blinded sample
of errors with a domain expert where feasible; separate coding-policy ambiguity from model mistakes.
Do not repair labels after seeing errors and report the repaired score as the original frozen result.

For researcher-facing usefulness:

- evaluate whether a held-out-calibrated confidence score identifies errors; raw sequence likelihood
  is not automatically a calibrated probability of a correct code set;
- report risk/coverage and the retained-label/source mix under abstention, with thresholds selected
  on development/calibration data only;
- retain row IDs, predictions, labels, novelty/transfer eligibility, scores, and checkpoint hashes for
  reproducible paired analysis, subject to data-sharing permissions;
- provide an inference interface, input/missing-field contract, model/data documentation, licensing,
  and explicit language/archive limitations. This is historical coding support, not clinical diagnosis.

Compute paired 95% cluster-bootstrap intervals on final prediction differences using normalized COD
groups, or larger linked-record units where available. Preserve cross-archive group links when present
and recompute equal-archive summaries within each replicate. Report the number of
independent groups and sparse-label limitations. These intervals are conditional on trained models
and observed archives, not estimates of training-seed variance or variability across all future archives.

Keep the evaluation checkpoint distinct from any later all-data deployment refit. A refit may be useful
but cannot inherit exact performance numbers from a different checkpoint. An external failure is a
finding; further development creates a new version and requires a fresh final assessment.

## 6. Compute discipline and optional search

| Block | Planned new training cells | Gate / scope |
|---|---:|---|
| Evaluation readiness | Smoke tests only | No full sweep until metrics/manifests pass. |
| P-cod screening and two controls | 10 full-data | Shared base-model protocol. |
| Matched short/long source comparison | 10 full-data | Submit 4 pilot cells, then 6. |
| Matched row-split comparison | 1 full-data | Compare against the selected P-cod recipe. |
| Core source ablations | 4 full-data | Two controls x two sources. |
| Reduced calibration and seed studies | 16 reduced-data | May need matching anchors; full convergence. |
| Scale and initial transfer confirmation | Up to 5 full-data | Optional if gains/cost warrant it. |
| Final assessment | Usually inference only | Additional refit is separate and optional. |

The core before scale is 25 new full-data and 16 reduced-data runs, plus the 20 completed exploratory
runs. This is a staged envelope, not permission or a request to queue everything. Extra challengers,
metadata/perturbation baselines, broader ablations, matching seed anchors, and full scale-transfer panels
are additional costs. Measure actual GPU-hours at Phase 1 before committing to the envelope.

Prioritize correct evaluation, paired source transfer, controls, and seed evidence over XL or a large
hyperparameter search. Preserve selected checkpoints and compact prediction/provenance artifacts.
Share data/pretraining caches only when manifests, seeds, resources, and training policies match;
do not share a pretrained state that used a held-out source. Plan disk headroom before each wave.

Bayesian optimization remains optional candidate generation after the evaluation contract is stable.
Prefer a small interpretable confirmation design for the three scientific questions. The current
short-budget, single-allocation W&B agent cannot substitute for convergence: reduced-fidelity ranking
must first be checked, every promoted recipe must be confirmed at full fidelity, and all search trials
remain development evidence. Do not optimize on the final external archive.

## 7. Implementation and launch contract

The implementation lives in [src/codllm/evaluation](../../src/codllm/evaluation/), with Config/env,
split-cache, tokenization, Trainer, and HPC integration. Legacy metric names remain unchanged;
new measurements carry `pub_v1_`. Training logs include `val/pub_v1_*` and final `holdout/full/pub_v1_*`.
Important suffixes include `macro_f1_ref`, `source_mean_macro_f1_ref`, `unseen_cod_macro_f1_ref`,
`crosslingual_eligible_recall`, and `strict_crosslingual_eligible_recall`.
Empty slices are NaN in runtime metrics and null in JSON, accompanied by zero support counts.

The original training reference, effective configuration, and checksummed train/validation/test/holdout
manifests are saved under each run's `publication/` directory. Final selected-model predictions and
same-checkpoint scalar metrics are stored in `publication/predictions/<scope>/`, with a `selected.json`
pointer. Resume rejects changes to the training exposure, partitions, or training recipe. Manifests
contain archival text and are private; prediction exports omit raw text but retain sensitive labels
and pseudonymous identifiers, so they are not automatically public data. Neither is uploaded by the
new artifact writer. Existing W&B error tables still follow existing logging settings.

| Artifact | Implemented treatment |
|---|---|
| Three curves and `base.toml` | Preserved legacy campaign; do not overwrite or resubmit. |
| `protocol_v1.toml` | COD-only, global grouped splitting, reference exclusion, versioned metrics, seed 777. |
| `interaction_confirmation.toml` | Eight cells: floor 0/450, synthesis .30/.60, pretraining 4/48. |
| Candidate files | Provisional shared floor 0 / synthesis .30; 48 epochs and matched 4-epoch alternative. |
| Source pilot / remaining panel | Four pilot and six remaining paired cells; no during-training holdout monitoring. |
| Reduced-data files | Four calibration cells; shared `reduced_protocol.toml`; fixed-cohort split and training seeds. |
| Baseline / ablation / scale files | Separate launchable steps; optional extensions explicitly labeled. |
| Frozen test / external evaluation | Inference from an explicit checkpoint and its original manifests; no mandatory retraining. |
| `final_model.toml` | Optional fresh training, not the default final-evaluation path. |
| UniCin | Canonical CSV/Parquet entrypoint exists; actual licensed data, label mapping, and provenance remain required. |

`invoke publication.approve` records explicit reviewed decisions (`source_pair`, `recipe`, `reduced`,
`final`) tied to inherited candidate settings, language-curation content, and the reduced protocol.
Submission and execution reject missing or stale approvals. These are reproducibility guards, not
automatic scientific judgment: inspect run results before approving, and choose new output roots when
changing already-run recipes. Later stages are prepared but cannot be scientifically frozen in advance.

`invoke publication.audit` writes aggregate score-blind data summaries; `publication.audit-splits`
checks original split integrity/support and planned masterlist exposure without training. It writes
`logs/publication/split_audit.json` and `.md`; integrity failures return exit code 2, while passing
checks remain explicitly subject to scientific review. `publication.report` recovers selected
results directly from local/HPC artifacts; `publication.bootstrap` compares aligned prediction files.
The frozen evaluator validates manifest checksums, records the evaluated checkpoint's SHA256, and uses
the original validation partition for uncertainty calibration. Final baseline jobs deterministically
refit the prespecified methods on unchanged training partitions, never on external labels.

[publication_runs.md](publication_runs.md) contains exact, separate preparation and submission commands
for every step and review gate. [runs/publication/README.md](../../runs/publication/README.md) describes
configuration ownership and artifact interpretation. Keep [publication_progress.md](publication_progress.md)
limited to real training results and short interpretations; keep unresolved ideas in
[publication_thoughts.md](publication_thoughts.md).

## 8. Readiness to write and submit

The study is ready when the metric/provenance audit passes, paired new-description and new-archive
comparisons support bounded conclusions, key controls and economical seed evidence are complete,
external evaluation is completed or its absence is explicitly limiting the claims, and the release is
reproducible and usable.

The main paper should lead with the evaluation gap and the lessons that survive it, then explain the
training recipe, transfer analysis, robustness, and practical system. It need not narrate every job.
Long pretraining, balancing, and synthesis may help different slices or fail to help: those are useful
engineering results when controls and evaluation make the interpretation credible.
