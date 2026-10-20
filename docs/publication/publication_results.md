# Publication results

Updated: 2026-10-20. Phase 1 is complete. Scores below are percentages; differences are percentage points (pp).
Recipe notation is **balance floor / synthetic ratio / pretraining epochs**.

## Phase 0 — data and split checks

All 14 original-split integrity checks passed. The grouped split contains 1,372,247 training,
75,536 validation, and 76,929 test rows, with all five archives represented and no cross-partition
normalized-COD overlap. This establishes original-split integrity, not model performance or augmented-data integrity.
[Audit details](reference/audits.md).

Frozen-manifest verification on 20 October found **1,262 / 1,297 test labels (97.30%)** in original
historical training, covering **84,371 / 84,414 label occurrences (99.949%)**. Only 43 of 76,929 test
rows contain any historically unseen label. All 35 such distinct labels appear in the recorded
masterlist inventory. The split holds out descriptions, not label identities.
[Coverage by archive](reference/audits.md#frozen-training-label-coverage--2026-10-20).

## Exploratory response curves — 20/20 complete

These earlier runs used row splits, COD+age+sex, and seed 777.

| Comparison | Finding on pooled macro F1 |
|---|---|
| Balance floor | Floor 0 scored 74.49%; the best positive floor, 450, scored 73.16% (−1.33 pp). |
| Synthetic ratio | .30 scored highest at 73.58%, versus 72.03% without synthesis (+1.55 pp) and 73.37% at .60. |
| Pretraining dose | 4 epochs scored 73.32%, versus 72.42% without pretraining (+0.90 pp); 48 epochs was close at 73.27%. |

Floor and dose scores retain provisional history-peak provenance; synthesis uses selected checkpoints.
These curves are separate from the grouped-COD comparisons below.
Paired intervals are unavailable: original prediction exports and frozen partition manifests were not
found in their retained HPC output locations. Scalar scores alone cannot recover them.

![Legacy row-split response curves](reference/figures/legacy.png)

## Phase 1 — grouped-COD recipe and metadata comparisons

Eight recipes completed for each input regime at patience 10, followed by all 16 patience-20 continuations.
All use seed 777. Results are validation scores at the pooled-macro-F1-selected checkpoint; final test
evaluation was disabled. Equal-source reference macro F1 is a validation metric, not held-out-source performance.

### Patience 10 → 20

Pooled macro F1 improved in **6/8 COD-only recipes** (largest gain +0.99 pp) and **4/8 metadata recipes**
(largest gain +1.47 pp). The remaining recipes retained their parent checkpoints. All continuations
stopped below the unchanged 120-epoch ceiling; these are continuations, not independent seed repeats.

### Final patience-20 results

| Highest observed score | COD-only: recipe → score | COD+age+sex: recipe → score |
|---|---|---|
| Pooled macro F1 | 0 / .30 / 48 → **59.77%** | 0 / .30 / 4 → **59.85%** |
| Equal-source reference macro F1 | 0 / .60 / 48 → **71.16%** | 0 / .60 / 48 → **71.60%** |
| Micro F1 | 0 / .30 / 4 → **91.01%** | 0 / .30 / 48 → **90.40%** |

These are metric-specific leaders, not a reviewed overall recipe selection.

![Selected validation scores for all eight matched recipes](reference/figures/selected.png)

- **Floor 0 beats floor 450 on pooled macro F1 in all eight matched comparisons:** +0.41–2.26 pp
  for COD-only and +0.76–2.76 pp for metadata. This comparison includes the floor's upsample-copy perturbations.
- **Age+sex has mixed effects:** higher equal-source macro F1 in 6/8 recipes, but higher pooled macro F1
  in only 1/8 and higher micro F1 in 2/8. Age and sex were tested together.
- **More synthesis favors equal-source macro F1:** .60 beats .30 in 3/4 COD-only and 4/4 metadata comparisons.
  Pooled macro F1 improves in 3/4 and 2/4, respectively.
- **Longer pretraining is not consistently better:** 48 beats 4 epochs on pooled macro F1 in 2/4 COD-only
  and 3/4 metadata comparisons. Its equal-source macro F1 improves in 2/4 and 4/4, respectively.

![Matched changes in floor, synthesis, and pretraining](reference/figures/differences.png)

### Paired uncertainty

All 32 selected Phase 1 models were analyzed on the same 75,536 validation rows and 8,217 connected
COD/source-record groups. The analysis covers 144 pairs: 112 recipe pairs within an input/patience
campaign, 16 matched input comparisons, and 16 parent/continuation comparisons. Each method uses
10,000 shared group-weight draws, seed 777.

Paired-difference plots show **solid 95% Bayesian cluster-bootstrap intervals** and **dashed ordinary
bootstrap confidence intervals**. These are individual, exploratory intervals conditional on the
trained models and observed archive panel. They do not account for training-seed variability,
validation-based selection, or multiple comparisons.

The Bayesian analysis uses positive Dirichlet group weights to retain observed label support.
Ordinary resampling can omit rare labels: 1,143 of 2,533 source–label pairs occur in only one validation
group. Both methods and a fixed-label-support sensitivity are retained in the
[full interval tables and methods](reference/paired_intervals.md). Sparse-label generalization
uncertainty cannot be recovered from these observed groups alone.

The top two COD-only pooled-macro-F1 scores differ by **+0.236 pp**, with a Bayesian paired interval
of **[−0.245, +0.697] pp** and an ordinary-bootstrap interval of **[−0.966, +1.218] pp**.
Their ordering is unresolved by either interval.
Floor 0 has a positive pooled-macro-F1 Bayesian interval in seven of eight matched patience-20
comparisons and a positive ordinary-bootstrap interval in five of eight; the others cross zero.
All eight metadata pooled-macro-F1 ordinary-bootstrap intervals cross zero. These are conditional comparisons,
not evidence of seed robustness or an overall recipe-selection decision.

![Paired metadata differences](reference/figures/metadata_differences.png)

![Paired patience-continuation differences](reference/figures/patience_differences.png)

### Checkpoint consistency

The preceding ten-epoch mean and selected-checkpoint rankings differ. For COD-only, 0 / .60 / 4 has the
highest mean pooled macro F1 (59.15%); for metadata, 0 / .30 / 48 leads (58.98%).
These B−9…B windows describe checkpoint behavior, not seed uncertainty, and do not replace selected scores.

![Ten completed checkpoints preceding and including each selected checkpoint](reference/figures/consistency.png)

## Phases 2–6 — no results recorded

| Phase | Comparison or assessment | Status |
|---|---|---|
| 2 | Plain fine-tuning versus the selected enhanced recipe across five held-out archives | Four pilots submitted 20 October; six remaining runs not submitted. No results recorded. |
| 3 | Ablations and baselines | No results. |
| 4 | Reduced-cohort split and training-seed sensitivity | No results. |
| 5 | Optional model scaling | No results. |
| 6 | Frozen internal/external evaluation and release | No results. |

## Evidence and figures

[Current scores](publication_progress.md), [earlier scores and scalar ledgers](reference/results.md),
and [study plan](publication_plan.md). All comparisons above describe this seed and these tested settings.

Regenerate the eight PNG/SVG pairs from the local aggregate evidence with [create.py](reference/figures/create.py):

```sh
uv run --no-sync python docs/publication/reference/figures/create.py
```
