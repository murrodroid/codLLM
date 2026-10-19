# Publication decisions

Updated: 2026-10-19. Dates marked “recorded” refer to the existing documentation, not a recovered approval date.

## Decisions made

| Date | Decision |
|---|---|
| Recorded 2026-09-06 | Study historical coding across new descriptions and archives; choose a venue after the study, without a deadline-driven scope. |
| Recorded 2026-09-06 | Keep full ICD10h pretraining targets. Category-only/A00-only pretraining was rejected. Retain both 4- and 48-epoch doses. |
| Recorded 2026-09-06 | Separate row, grouped-COD, source-held-out, and external evaluation. Use COD-only as the primary portable input and exclude the English historical reference from the historical source pool. |
| Recorded 2026-09-06 | Treat the Belgian dataset as Flemish/Belgian Dutch (`nl`), like Amsterdam; their transfer is between sources, not languages. |
| Recorded 2026-09-06 | Estimate seed sensitivity on a fixed reduced cohort, with full convergence. Compare 25% and 40% before choosing the fraction. |
| 2026-09-16 | Continue all eight COD-only cells from their best checkpoints with patience 20. Keep the original 120-epoch ceiling and LR schedule; 160 epochs was rejected. Preserve the parents. |
| 2026-10-05 | Use ten completed epochs B-9 through B for checkpoint consistency, anchored to the actual macro-F1-selected checkpoint. This replaced the final-ten-epoch proposal. |
| 2026-10-05 | Repeat all eight cells with COD+age+sex as fresh patience-10 runs; submitted that day. Keep COD-only grouping and the original recipe. Synthetic records retain their first constituent's age/sex. |
| 2026-10-13 | Continue all eight completed metadata cells with patience 20, the original 120-epoch ceiling and LR schedule, isolated outputs, and fresh W&B IDs. |
| 2026-10-19 | Lucas confirmed Phase 1 complete. This closes the phase; no final recipe selection was supplied. |
| 2026-10-19 | Keep publication documentation focused on the plan, factual progress, and decisions. Remove unsolicited speculative interpretation and duplicate narratives. |

## Decisions pending

| When | Choice |
|---|---|
| Before Phase 2 | Select the shared floor/synthesis settings for the matched 4-/48-epoch pair; confirm input fields and stopping policy. Candidate TOMLs still contain provisional floor 0 / synthesis .30 settings. |
| Before Phase 2 | Confirm Copenhagen and Belgium as the pilot sources; finalize any near-tie margins and minimum transfer-support rule used for selection. |
| After the five-source panel | Freeze the winner and runner-up. |
| Before Phase 3 ablations | Confirm the two source folds; the existing specification uses Copenhagen and Belgium. |
| After reduced-cohort calibration | Choose 25% or 40% for seed studies. The current 25% TOML value is provisional. |
| Before final evaluation | Freeze the checkpoint, decoding/calibration, baseline settings, and external data/label mapping. |

Record each resolved choice here once. Runtime approvals use `publication.approve` only after Lucas's decision;
candidate, language-curation, or reduced-protocol edits invalidate affected approvals.
