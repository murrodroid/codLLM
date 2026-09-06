# Publication data audit

Audit snapshot: 2026-09-06. Model scores were not examined for this review.
Source: [data_audit.json](data_audit.json), copied byte-for-byte from HPC
`~/codLLM/logs/publication/data_audit.json`.

This page preserves the Phase 0a inventory review. The completed Phase 0c full-size split findings and
current readiness decisions are in [publication_split_audit_review.md](publication_split_audit_review.md).

## Provenance and scope

- HPC repository revision: `755cb4d88acbe10b5964cde997bc293a196de367`.
- Audit SHA256: `87d67dcae7fb618f0fa31ee4a7ce9a600bbff13fe00a29e1881d958212d15746`.
- Reviewed language-inventory SHA256:
  `1baeaf350ac42d3bfeb76b75b2b99d6ef81ee5c906714917fe62c2a1bcb871cd`.
- The file contains aggregate processed-data counts, not archival text, individual predictions, or model results.
- Counts are before publication reference-source exclusion, missing-COD removal, cohort sampling, and splitting.
  The audit does not identify the processed cache fingerprint or hash raw files; the repository revision alone
  is not a complete data-version record.

The copy is preserved unchanged. Derived percentages below use the reported processed-row counts,
including missing-COD rows. Source code inventories overlap and must not be summed into a global class count.

## Dataset inventory

| Historical source | Language | Rows | Distinct CODs | Missing COD | Natural multi-COD rows | Multi-COD rate | Codes | Codes with <10 rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Amsterdam | Dutch | 576,163 | 51,122 | 0 | 106,711 | 18.52% | 2,207 | 1,241 (56.2%) |
| Belgium | Flemish / Dutch | 42,037 | 8,345 | 0 | 6,747 | 16.05% | 986 | 656 (66.5%) |
| Copenhagen | Danish | 501,855 | 2,525 | 0 | 0 | 0.00% | 389 | 24 (6.2%) |
| Ipswich | English | 39,261 | 15,815 | 24 | 16,601 | 42.28% | 1,168 | 794 (68.0%) |
| Madrid | Spanish | 366,282 | 24,875 | 862 | 18,603 | 5.08% | 1,705 | 1,034 (60.6%) |

Distinct-COD counts include the empty normalized key where missing CODs exist. Nonempty distinct counts
are therefore 15,814 for Ipswich and 24,874 for Madrid. These are text identities, not counts of independent people.

The auxiliary English reference contains 3,295 rows, 3,295 distinct CODs, 1,382 codes, no missing CODs,
and no multi-COD rows. It is correctly visible in this pre-exclusion inventory; that does not mean it
will enter historical training. It is not the masterlist-pretraining dataset.

Arithmetic under the already configured exclusion rules:

| Stage | Rows |
|---|---:|
| All six processed sources | 1,528,893 |
| Exclude the 3,295-row English reference | 1,525,598 |
| Exclude 886 historical rows with missing COD | 1,524,712 |

The last number is the pool after these two exclusions only, before any additional identity deduplication
or cohort sampling; it is not the final training size. There are 148,662 natural multi-COD rows before
missing-COD removal (9.74% of historical rows). The JSON does not show how many missing-COD rows are
also multi-COD, so the exact post-exclusion multi-COD count is unavailable.

## Findings that change how the study should be interpreted

### 1. Copenhagen has many cases but limited description diversity

501,855 rows / 2,525 CODs = approximately 199 rows per distinct description, compared with 11.3 in
Amsterdam, 5.0 in Belgium, 2.5 in Ipswich, and 14.7 in Madrid.

This makes the P-row versus P-cod distinction particularly important. A large row count does not
establish a large number of distinct generalization challenges. It does not imply that repeated
historical cases are erroneous duplicates or should be deleted.

Before full screening, inspect the actual grouped split's source counts, group-size distribution,
largest connected record/COD groups, code support, and number of distinct descriptions per partition.
The current allocator targets global row proportions, not per-source or per-code stratification.
The audit does not establish whether seed 777 produces representative source coverage.
The 0.2% smoke fixture checks execution, not the representativeness of the full-size grouped split.

### 2. The archives test different aspects of multi-COD behavior

Copenhagen has no multi-COD targets in this processed inventory. It can test single-COD accuracy and
unwanted extra predicted codes after synthetic training, but cannot demonstrate improved natural
multi-COD recovery. Zero targets could reflect the source's annotation convention; the audit alone
cannot determine whether the underlying historical descriptions ever contain multiple causes.

Belgium has 6,747 multi-COD rows; Ipswich has 16,601 and the highest proportion (42.28%).
Amsterdam has the largest absolute multi-COD set (106,711). Madrid contributes a lower-prevalence
setting (5.08%). Consequently a Copenhagen+Belgium ablation panel has one positive multi-COD source
and one single-COD control, not two independent multi-COD benefit tests.

### 3. A pooled result is dominated by three archives

After the two stated exclusions, Amsterdam, Copenhagen, and Madrid supply 94.67% of the historical
rows. Belgium and Ipswich supply only 2.76% and 2.57%, respectively.

Keep the existing equal-source selection summary and report pooled and individual-source results too.
This is an evaluation-weighting distinction, not a decision to resample training uniformly.
Ipswich is small in rows but important for multi-COD behavior; Copenhagen is large but has only
389 observed codes. Source macro-F1 differences should not be interpreted as differences on an
identical label inventory.

### 4. Rare-code support needs inspection before choosing the reduced cohort

More than half the codes in Amsterdam, Belgium, Ipswich, and Madrid have fewer than ten source-local
rows. This does not show that they are globally rare: another archive may provide many examples.
It also does not show whether each code spans one description group or many.

The 25% and 40% options correspond to roughly 381,000 and 610,000 historical cohort rows before
augmentation, respectively, using the exclusion arithmetic above. Large total cohorts do not guarantee
usable rare-code or transfer support. Compare source/code/group coverage before choosing a fraction,
then use the planned converged paired calibration. Neither fraction is selected by this audit.

### 5. Source holdout is not automatically unseen-description or cross-lingual evaluation

| Source pair | Shared distinct nonempty CODs |
|---|---:|
| Amsterdam–Copenhagen | 459 |
| Amsterdam–Belgium | 279 |
| Amsterdam–Madrid | 152 |
| Amsterdam–Ipswich | 128 |

All 15 source-pair intersections, including the reference source, are retained in the JSON.
The 459 shared strings equal 18.18% of Copenhagen's distinct CODs before splitting—not 18.18% of
its rows or its eventual held-out rows with training exposure. Multi-way overlaps and actual
training membership are absent, so neither the global union nor final seen-COD rates can be derived
exactly from these pair counts.

Belgium and Amsterdam both use `nl`; their transfer is between archives, not between languages.
Copenhagen provides the only Danish historical archive. Nevertheless, eligible cross-lingual target
counts require actual code-by-language training exposure, which this audit does not contain.
English masterlist exposure can exclude Ipswich targets from the stricter all-adaptation definition;
the historical English reference and the pretraining masterlist must remain separate inventories.

All language counts come from the reviewed source mapping, not automated row-level language validation.
Exact cross-language text overlap does not, by itself, prove an incorrect language label.

## Decisions for Lucas — no settings changed

| Decision | Evidence and options | When needed |
|---|---|---|
| Keep Copenhagen+Belgium as the Phase 2 pilots? | Keep for Danish-without-same-language-history versus Dutch-with-Amsterdam transfer. They do not provide two natural multi-COD benefit tests. My recommendation is to keep this transfer contrast, subject to split-support checks. | Before approving the source pair. |
| Which sources should test the Phase 3 mechanisms? | Keep Copenhagen+Belgium to test single-COD harm and multi-COD benefit (4 runs). Alternatively use Belgium+Ipswich for two multi-COD-rich archives (still 4 runs), or add Ipswich to the existing pair (6 runs total). Ipswich's strict language-transfer support may be limited by English masterlist exposure. | Before Phase 3 ablation submission. |
| Keep case-weighted scores only within each source, or add an equal-description sensitivity analysis? | Case weighting represents archival workload; equal-description weighting tests whether frequent strings dominate conclusions. Recommendation: retain the primary scores and add the latter as a secondary analysis if group-size concentration is substantial. It is not implemented/selected by this audit. | Freeze analysis rules before comparing new scores. |
| What support is sufficient for a transfer-based tie-break? | The file lacks eligible rows, unique CODs, distinct codes, and code-language pairs per split. Obtain these counts first. Until a support rule is agreed, show sparse slices descriptively; do not let them determine a winner. | Before candidate selection. |
| Use 25% or 40% for seed sensitivity? | About 381k versus 610k cohort rows; rare-code and group support, then paired converged calibration, determine feasibility. Current TOML default stays 25%, not an approved choice. | After the planned calibration and before the reduced-stage approval. |

Changing Phase 3 sources would require matching edits to the ablation TOML and launch documentation;
change baseline/metadata/perturbation panels only if you also want those comparisons on the new sources.
Do not automatically change the already paired five-source Phase 2 panel.

## What remains before expensive selection

1. Keep the completed Phase 0a snapshot; no audit rerun is needed solely to obtain this same report.
2. The smoke execution was verified; it is not scientific approval of seed 777 or a test of automatic resumption.
3. The full-size split-support report is now complete: [snapshot](split_audit.json) and
   [interpretation](publication_split_audit_review.md). All 14 integrity checks passed; data/code
   fingerprints and actual support counts are recorded. Augmentation caches were not built by the audit.
4. Review the current split-audit decisions and the source choices above. Do not repair a split after seeing model scores
   or assume group disjointness alone establishes good coverage.

The aggregate `publication.audit` command does not generate that split-support report. A second command,
`publication.audit-splits`, completed Phase 0c in [the launch guide](publication_runs.md#phase-0c-full-size-split-audit--no-training).
It checked the full-size original partitions shared by the eight Phase 1a recipes without model training.
The findings earlier on this page still refer only to Phase 0a; the separate split review supersedes
its unresolved split-count questions, not Lucas's unmade scientific decisions.

Its JSON and Markdown outputs include sequential exclusion counts; source allocations; per-code,
per-source, and connected-group support/concentration; cross-partition identity/COD overlap; historical
versus planned masterlist language exposure; and data/code fingerprints. Cached original partitions are
compared if present. Missing/sparse metric slices are counts to review, not zero performance or an
automatic reason to change the seed. The report cannot determine an acceptable support threshold,
annotation quality, whether a different pilot panel is preferable, or actual trained-model performance.
Augmented lexical overlap, future holdout folds, and reduced cohorts are outside this Phase 1a audit.

Candidate levels, TOMLs, scientific approvals, and HPC job queues are unchanged. Training-only results
remain in [publication_progress.md](publication_progress.md).
