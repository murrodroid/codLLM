# Publication split audit review

Phase 0c, completed on HPC on 2026-09-06. This is a data-support review, not model evaluation or phase approval.
Original outputs: [JSON](split_audit.json) and [generated summary](split_audit.md), preserved byte-for-byte.

## Verdict

All 14 integrity checks passed. The full-size seed-777 grouped split is technically usable for Phase 1a;
there is no detected integrity reason to rebuild it or change the seed. All five historical archives occur
in train, validation, and test. However, repeated descriptions concentrate case-weighted evaluation,
and natural cross-language support is sparse in distinct codes/descriptions despite substantial row counts.

My recommendation is to retain this split and the planned eight-cell comparison, subject to Lucas accepting
the interpretation limits below. Keep equal-source reference-supported macro F1 as the screening primary.
Treat the current cross-language slices as supporting diagnostics, not a decisive tie-break, until a
support rule is agreed. No TOML, selection rule, approval record, or job queue was changed by this review.

## What is now established

| Partition | Original rows | Distinct COD / connected groups | Codes | Natural multi-COD rows |
|---|---:|---:|---:|---:|
| Train | 1,372,247 | 88,519 | 3,321 | 130,505 |
| Validation | 75,536 | 8,217 | 1,493 | 11,447 |
| Test | 76,929 | 4,859 | 1,297 | 6,710 |

- The 1,528,893 processed rows lose 886 missing-COD rows and the 3,295-row English reference.
  Identity deduplication removes no additional rows. The retained pool is exactly 1,524,712 rows.
- All 148,662 natural multi-COD rows remain: the missing-COD exclusions were single-target rows.
- Actual global allocation is 90.0004% / 4.9541% / 5.0455% by rows, not equal proportions of distinct CODs.
- Train/validation/test share no row IDs, source-record identities, or normalized CODs.
  No conflicting reused row UID was detected. Source-based language metadata has no unknown/mixed rows;
  this does not independently validate the language of every description.
- The same original split covers all eight Phase 1a cells. The four augmentation-specific caches were
  **not built** at audit time. The cache check means no existing mismatch was found, not four caches verified.
  The normal `hpc.build` step is still necessary.
- Enabled data resources were present; processed data remained unchanged during the audit.
  Raw data, curation, processed-cache, specification, and source-code fingerprints are recorded.

## Findings that matter for the study

### 1. Case counts overstate description diversity

The ten largest COD groups contain **37.31% of validation rows** and **66.25% of test rows**.
The largest single test group contains 11,635 cases, or 15.12% of the pooled test set.

| Source | Validation rows / CODs | Test rows / CODs | Largest validation group | Largest test group |
|---|---:|---:|---:|---:|
| Amsterdam | 25,373 / 4,207 | 28,384 / 2,485 | 12.45% | 25.24% |
| Belgium | 5,564 / 676 | 2,448 / 387 | 27.23% | 14.71% |
| Copenhagen | 21,268 / 212 | 22,544 / 114 | 12.64% | 35.76% |
| Ipswich | 2,400 / 1,263 | 967 / 737 | 5.79% | 2.38% |
| Madrid | 20,931 / 1,931 | 22,586 / 1,191 | 26.82% | 24.49% |

Largest-group percentages are within the indicated source/partition. These can be legitimate repeated
historical cases; they are not automatically erroneous duplicates. Their weights directly affect
case-weighted accuracy/micro metrics. Macro F1 reduces class-frequency dominance but does not make
repeated descriptions independent or equally weighted within classes.

Keep case-weighted scores to represent archival workload. A secondary equal-description analysis and
paired COD-group bootstrap would make sensitivity to frequent descriptions visible. The bootstrap is
already implemented for aligned prediction exports; equal-description metric weighting is not.
Neither supplies a full-data training-seed confidence interval or guarantees coverage of future archives.

The allocator targets global row ratios, not per-source stratification: Belgium assigns 13.24% of its
rows to validation, while Ipswich assigns only 2.46% to test. Natural multi-COD prevalence also differs:
9.51% in training, 15.15% in validation, and 8.72% in test. Report these differences; do not reroll seeds
to seek favorable performance. If a different allocation policy is desired, define it before full runs.

### 2. Cross-language transfer exists, but its effective coverage is narrow

| Validation source | Historical cross-language rows / codes / CODs | Strict planned-adaptation rows / codes / CODs |
|---|---:|---:|
| Amsterdam | 8 / 5 / 5 | 8 / 5 / 5 |
| Belgium | 4 / 3 / 4 | 4 / 3 / 4 |
| Copenhagen | 1,352 / 11 / 12 | 1,352 / 11 / 12 |
| Ipswich | 20 / 20 / 20 | 0 / 0 / 0 |
| Madrid | 28 / 21 / 21 | 28 / 21 / 21 |
| Overall | 1,412 / 60 / 62 | 1,392 / 40 / 42 |

Copenhagen supplies **95.75% of historical cross-language validation rows**, but only 11 of its 60
eligible codes. Thus row-weighted transfer and macro-averaged eligible-code recall answer different
questions; Danish row dominance alone does not imply equal dominance of a macro score.
Across the historical slice, **44 of 60 eligible codes have one target occurrence**, and 50 have fewer
than ten. In the strict slice, 25 of 40 codes are singletons. Only 28 historical-transfer rows and ten
strict-transfer rows are natural multi-COD examples. These are not broad multilingual or multi-COD panels.

Ipswich's strict slice is unavailable because the English masterlist supplies target-language code
supervision. It is not zero performance. Amsterdam and Belgium remain the same language (`nl`).
Test support is also limited: 690 historical-transfer rows over 39 codes / 41 CODs; 676 strict-transfer
rows over 28 codes / 29 CODs. No test model scores were inspected for this audit.

Keep source holdouts central. This audit does not measure their actual held-out support. If those folds
also lack diverse cross-language support, controlled code-language withholding is an optional separate
experiment, not something to simulate by relabeling source transfer or treating repeated rows as replicates.

### 3. Historically unseen labels are not labels absent from all adaptation

There are 127 validation rows involving 72 codes absent from original historical training; all those
codes occur in the planned pretraining masterlist. Test has 43 such rows over 35 codes, also all supplied
by the masterlist. No evaluation targets are absent from all planned adaptation resources.

This provides a small masterlist-to-history transfer diagnostic, not a test of codes never supervised
anywhere. Both 4- and 48-epoch recipes access the same 14,088-row/code English masterlist; extra epochs
change training duration, not available code/language support. The no-pretraining control needs its own
exposure interpretation and is not covered by the audit's shared pretraining-resource assumption.

All validation/test CODs are unseen in original historical training, but 1,137 validation rows (1.51%)
and 332 test rows (0.43%) exactly match normalized masterlist descriptions. State the reference for
"unseen": historical novelty is established; absence from every adaptation resource is not.
Augmented lexical overlap remains outside this audit.

### 4. Rare codes and inconsistent target sets deserve separate treatment

Original training has 1,932 of 3,321 codes with fewer than ten rows (58.18%); 887 codes occur in only
one connected COD group. Validation has 1,050 of 1,493 codes below ten rows and 578 confined to one
group. A large total case count therefore does not remove rare-code uncertainty.

There are 449 training COD groups with multiple target sets, versus 36 in validation and 26 in test.
This is distinct from conflicting row IDs, of which there are none. Shared terms can be ambiguous,
language-dependent, or coded differently across archives; the count alone does not diagnose annotation
error or quantify an accuracy ceiling. If curation is considered, inspect original training cases first,
use the tracked overlay workflow, and do not silently correct evaluation labels after seeing predictions.

## Decisions for Lucas

| Decision | Evidence-backed options | Recommendation, not an applied change |
|---|---|---|
| Keep the seed-777 grouped split? | All sources are present and integrity passes; source ratios and group concentration vary. A new source-aware group allocator would require implementation and a fresh audit. | Keep this split for Phase 1a; do not search seeds. |
| Add equal-description sensitivity? | Ten CODs cover 37.31% of validation and 66.25% of test rows. It targets description diversity rather than natural case frequency. | Add it as secondary, retaining the current primary; implementation still required. |
| Let natural cross-language results decide a near-tie? | Validation has only 62 historical / 42 strict COD groups, with many singleton codes. Actual source-fold support is not measured here. | Use current slices descriptively; agree support requirements before employing a transfer tie-break. |
| Change the source pilots or reduced-data fraction? | This audit covers the pooled full-size split, not source-held-out or 25%/40% cohorts. | Leave existing proposals unchanged pending their own evidence and your review. |

No new training or curation is launched by this review. Candidate settings and scientific approvals remain unchanged.

## Snapshot identity

- HPC code revision: `c72912a5c5d12b24cfaf05c5ef88812cae8dcecc`; recorded source-code hashes match this local revision.
- Generated at `2026-09-06T18:15:04.643970+00:00` (20:15 Copenhagen time).
- JSON SHA256: `01d8c035ab78da0e31ef8ed4f6fafbb85021a321c168521d825fe726dbea0aba`.
- Generated Markdown SHA256: `10f49d482009fd8e51e416c199af0797a39cb687762925b4aacbe51388095952`.
- Processed Parquet SHA256: `a44b05075f21f399ce235c324e11fb9b81c02da365a069a1c537093eaa39f6e9`.

The JSON includes source/code/language aggregate counts, including small cells, and provenance paths/hashes.
It contains no archival descriptions, row identifiers, predictions, or credentials. The requested repository
copy preserves the HPC bytes; copying it to Git makes these aggregates available to repository readers.
