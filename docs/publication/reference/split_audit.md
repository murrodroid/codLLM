# Publication split audit

Status: review_required. No model scores or scientific approval.

Cells covered: 8; shared original partitions: 1.

## Natural partition inventory

| Split / source | Rows | Distinct CODs | Connected groups | Codes | Multi-COD rows | Largest-group share |
|---|---:|---:|---:|---:|---:|---:|
| original_train / all | 1,372,247 | 88,519 | 88,519 | 3,321 | 130,505 | 2.50% |
| original_train / amsterdam_1854_1926 | 522,406 | 44,430 | 44,430 | 2,142 | 93,450 | 4.80% |
| original_train / belgium_1920_1930 | 34,025 | 7,282 | 7,282 | 952 | 5,955 | 4.92% |
| original_train / copenhagen_may2025 | 458,043 | 2,199 | 2,199 | 371 | 0 | 4.60% |
| original_train / ipswich_1871_1911 | 35,870 | 13,814 | 13,814 | 1,123 | 14,522 | 4.62% |
| original_train / madrid_1905_1927 | 321,903 | 21,752 | 21,752 | 1,637 | 16,578 | 9.02% |
| val / all | 75,536 | 8,217 | 8,217 | 1,493 | 11,447 | 7.75% |
| val / amsterdam_1854_1926 | 25,373 | 4,207 | 4,207 | 980 | 8,228 | 12.45% |
| val / belgium_1920_1930 | 5,564 | 676 | 676 | 323 | 522 | 27.23% |
| val / copenhagen_may2025 | 21,268 | 212 | 212 | 109 | 0 | 12.64% |
| val / ipswich_1871_1911 | 2,400 | 1,263 | 1,263 | 464 | 1,398 | 5.79% |
| val / madrid_1905_1927 | 20,931 | 1,931 | 1,931 | 657 | 1,299 | 26.82% |
| test / all | 76,929 | 4,859 | 4,859 | 1,297 | 6,710 | 15.12% |
| test / amsterdam_1854_1926 | 28,384 | 2,485 | 2,485 | 878 | 5,033 | 25.24% |
| test / belgium_1920_1930 | 2,448 | 387 | 387 | 261 | 270 | 14.71% |
| test / copenhagen_may2025 | 22,544 | 114 | 114 | 76 | 0 | 35.76% |
| test / ipswich_1871_1911 | 967 | 737 | 737 | 361 | 681 | 2.38% |
| test / madrid_1905_1927 | 22,586 | 1,191 | 1,191 | 522 | 726 | 24.49% |

## Source allocation

Fractions use each source's retained train/validation/test pool. Per-source stratification is not guaranteed.

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| amsterdam_1854_1926 | 90.67% | 4.40% | 4.93% |
| belgium_1920_1930 | 80.94% | 13.24% | 5.82% |
| copenhagen_may2025 | 91.27% | 4.24% | 4.49% |
| ipswich_1871_1911 | 91.42% | 6.12% | 2.46% |
| madrid_1905_1927 | 88.09% | 5.73% | 6.18% |

## Eligible targets — counts, not performance

| Split | Bucket | Rows | Targets | Codes | CODs | Connected groups |
|---|---|---:|---:|---:|---:|---:|
| val | source_transfer | 1,451 | 1,452 | 77 | 81 | 81 |
| val | crosslingual | 1,412 | 1,413 | 60 | 62 | 62 |
| val | strict_crosslingual | 1,392 | 1,392 | 40 | 42 | 42 |
| val | historically_unseen_label | 127 | 127 | 72 | 77 | 77 |
| val | masterlist_only_label | 127 | 127 | 72 | 77 | 77 |
| val | absent_from_all_adaptation | 0 | 0 | 0 | 0 | 0 |
| val | seen_cod | 0 | 0 | 0 | 0 | 0 |
| val | unseen_cod | 75,536 | 88,159 | 1,493 | 8,217 | 8,217 |
| val | masterlist_seen_cod | 1,137 | 1,137 | 13 | 13 | 13 |
| val | novel_known_code_combination | 2,422 | 5,740 | 992 | 2,031 | 2,031 |
| test | source_transfer | 699 | 699 | 46 | 48 | 48 |
| test | crosslingual | 690 | 690 | 39 | 41 | 41 |
| test | strict_crosslingual | 676 | 676 | 28 | 29 | 29 |
| test | historically_unseen_label | 43 | 43 | 35 | 36 | 36 |
| test | masterlist_only_label | 43 | 43 | 35 | 36 | 36 |
| test | absent_from_all_adaptation | 0 | 0 | 0 | 0 | 0 |
| test | seen_cod | 0 | 0 | 0 | 0 | 0 |
| test | unseen_cod | 76,929 | 84,414 | 1,297 | 4,859 | 4,859 |
| test | masterlist_seen_cod | 332 | 337 | 12 | 10 | 10 |
| test | novel_known_code_combination | 1,532 | 3,658 | 902 | 1,315 | 1,315 |

## Transfer support by evaluation source

Zero targets means unavailable, not zero performance. Detailed single/multi-COD support is in the JSON.

| Split / source | Definition | Rows | Targets | Codes | CODs | Groups |
|---|---|---:|---:|---:|---:|---:|
| val / amsterdam_1854_1926 | source_transfer | 28 | 28 | 9 | 10 | 10 |
| val / amsterdam_1854_1926 | crosslingual | 8 | 8 | 5 | 5 | 5 |
| val / amsterdam_1854_1926 | strict_crosslingual | 8 | 8 | 5 | 5 | 5 |
| val / belgium_1920_1930 | source_transfer | 23 | 23 | 17 | 19 | 19 |
| val / belgium_1920_1930 | crosslingual | 4 | 4 | 3 | 4 | 4 |
| val / belgium_1920_1930 | strict_crosslingual | 4 | 4 | 3 | 4 | 4 |
| val / copenhagen_may2025 | source_transfer | 1,352 | 1,352 | 11 | 12 | 12 |
| val / copenhagen_may2025 | crosslingual | 1,352 | 1,352 | 11 | 12 | 12 |
| val / copenhagen_may2025 | strict_crosslingual | 1,352 | 1,352 | 11 | 12 | 12 |
| val / ipswich_1871_1911 | source_transfer | 20 | 21 | 20 | 20 | 20 |
| val / ipswich_1871_1911 | crosslingual | 20 | 21 | 20 | 20 | 20 |
| val / ipswich_1871_1911 | strict_crosslingual | 0 | 0 | 0 | 0 | 0 |
| val / madrid_1905_1927 | source_transfer | 28 | 28 | 21 | 21 | 21 |
| val / madrid_1905_1927 | crosslingual | 28 | 28 | 21 | 21 | 21 |
| val / madrid_1905_1927 | strict_crosslingual | 28 | 28 | 21 | 21 | 21 |
| test / amsterdam_1854_1926 | source_transfer | 4 | 4 | 4 | 4 | 4 |
| test / amsterdam_1854_1926 | crosslingual | 2 | 2 | 2 | 2 | 2 |
| test / amsterdam_1854_1926 | strict_crosslingual | 2 | 2 | 2 | 2 | 2 |
| test / belgium_1920_1930 | source_transfer | 13 | 13 | 11 | 11 | 11 |
| test / belgium_1920_1930 | crosslingual | 6 | 6 | 6 | 6 | 6 |
| test / belgium_1920_1930 | strict_crosslingual | 6 | 6 | 6 | 6 | 6 |
| test / copenhagen_may2025 | source_transfer | 643 | 643 | 6 | 6 | 6 |
| test / copenhagen_may2025 | crosslingual | 643 | 643 | 6 | 6 | 6 |
| test / copenhagen_may2025 | strict_crosslingual | 643 | 643 | 6 | 6 | 6 |
| test / ipswich_1871_1911 | source_transfer | 14 | 14 | 12 | 12 | 12 |
| test / ipswich_1871_1911 | crosslingual | 14 | 14 | 12 | 12 | 12 |
| test / ipswich_1871_1911 | strict_crosslingual | 0 | 0 | 0 | 0 | 0 |
| test / madrid_1905_1927 | source_transfer | 25 | 25 | 15 | 15 | 15 |
| test / madrid_1905_1927 | crosslingual | 25 | 25 | 15 | 15 | 15 |
| test / madrid_1905_1927 | strict_crosslingual | 25 | 25 | 15 | 15 | 15 |

## Integrity checks

- PASS: original_train_unique_row_ids
- PASS: original_train_no_excluded_source
- PASS: val_unique_row_ids
- PASS: val_no_excluded_source
- PASS: test_unique_row_ids
- PASS: test_no_excluded_source
- PASS: original_train__val
- PASS: original_train__test
- PASS: val__test
- PASS: duplicate_uid_content_consistent
- PASS: training_integrity_validator
- PASS: existing_cached_partitions_match
- PASS: processed_file_unchanged
- PASS: all_data_resource_files_present

## Review notes

- Review source-allocation fractions and largest groups before choosing any split policy.
- Review per-source/language eligibility and per-code support in the JSON before choosing support thresholds.
- Do not equate a clean integrity check with representative coverage or model quality.

## Scope and limits

- No scientific pass/fail threshold is selected; coverage and sparse support require user review.
- This audits the selected original partition, not other seeds, fractions, or unconfigured source folds.
- Strict support uses planned full-code exposure, not measured trained-model knowledge or epoch completion.
- Checked source-homogeneous within-source synthesis and floor copies retain code/language support.
- No augmented datasets are generated: their lexical overlaps and constituent files are not checked.
- Planned PT support uses the original masterlist; checked homogeneous synthesis adds no language pairs.
- Connected COD/record groups are dependence units, not verified independent people or archives.
- Group counts are partition-local; source-level counts use those same partition group identities.
- Transfer rows contain eligible targets; target counts exclude ineligible co-occurring codes.
- Per-code/source/language summaries may expose small counts; outputs remain private until reviewed.
