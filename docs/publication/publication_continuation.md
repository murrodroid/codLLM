# Interaction continuation: patience 20

Decision: 2026-09-16. Increase fine-tuning early-stopping patience from 10 to 20 for all eight
interaction cells. Lucas explicitly retained the **120-epoch ceiling and original learning-rate
schedule**; the briefly considered 160-epoch extension is not adopted.

Specification: `runs/publication/interaction_patience20.toml`.
Exact commands: [publication_runs.md](publication_runs.md#phase-1a-continuation-patience-20--8-runs-ceiling-still-120).
**Status: 8/8 complete, verified 2026-10-05.** Preparation and launch instructions are retained for provenance;
do not resubmit this completed campaign. [Selected-checkpoint results](publication_progress.md#phase-1a-continuation-patience-20).

## What changes, and what does not

- Only the stopping patience changes in the scientific recipe. The eight cells, data/splits,
  augmentation, seeds, pretraining doses, optimizer, learning rate, scheduler horizon/warmup,
  checkpoint metric (`macro_f1`), and locked test policy stay fixed.
- Preparation requires every named parent to have both completion markers and no resume request.
  It validates all parents before copying any cell. Wait for the original campaign to finish.
- Resume from each parent's retained **best checkpoint**, not its last training epoch. For example,
  the epoch-80 parent resumes from epoch 70, replaying roughly ten epochs. Execution after a restart
  is not guaranteed to be bit-identical. Pretraining is not repeated.
- Patience 20 means 20 validation evaluations without a new best macro F1, subject to the unchanged
  absolute ceiling of 120. A new best resets the counter. A ceiling-limited run is still budget-censored;
  this amendment does not guarantee every run will exhaust its full patience allowance.
- These are same-seed continuation branches, not eight new independent replicates and not a claim
  that the original campaign was run with patience 20 from the beginning. This is an exploratory
  stopping-policy sensitivity study motivated by observed curves.
- Later phases and completed legacy curves are not silently migrated. Review future matched controls
  and source comparisons before deciding whether to adopt patience 20 more broadly.

## Safety and provenance

Parents remain unchanged. Children live in
`$RUN_STORAGE_DIR/checkpoints/publication-v1/interaction-patience20/<cell>/run-0001/`.
Preparation makes independent copies, not hardlinks, verifies checkpoint SHA256 hashes, rewrites only
the copied best-checkpoint pointer, and starts a clean child log history. It copies frozen manifests
and the exposure reference; it does not copy parent completion markers, W&B identity, or prediction reports.

Each child has a fingerprinted `continuation.json`: parent W&B path, parent checkpoint and hashes,
fork epoch/update, old/new patience, unchanged epoch/scheduler horizon, child recipe digest, and new
W&B ID. The recipe guard rejects changing any other scientific setting, including increasing the ceiling.
Training retains the original partition/exposure checks and fails rather than starting from scratch
if a prepared checkpoint is missing. Do not delete the parent data or frozen manifests.

An exclusive per-child training lock rejects overlapping writers. Completion markers reject duplicate
training after a child finishes. New submissions reuse the prepared child's identity, not the parent.
Patience state is restored across wall-time allocations without reintroducing the parent's limit of 10.
Allocation pauses retain both the latest resumable checkpoint and the best checkpoint; they do not
reload best weights, delete the latest checkpoint, or count a partial-epoch validation as a new epoch.
The original update-based cosine schedule is retained, with no restart, stretching, or extra warmup.

Preparation is dry-run unless `--yes` is supplied. Existing valid preparations are reused; foreign or
incompatible destinations are refused. An interrupted copy leaves a uniquely named `.run-0001.prepare-*`
staging directory for inspection, not a launchable partial run. Do not remove it blindly.
Initial copies need about 24 GB; allow about 75 GB additional headroom including best/latest retention
and transient concurrent checkpoint writes, plus manifests or regenerated caches. Generated continuation stdout logs and W&B staging use work3,
not the home quota. No maintenance cleanup is part of this workflow.

## W&B and reporting

Use **new LSF jobs and new W&B runs**, grouped as `publication_v1_interaction_patience20` in
`codllmdev/codllm`. The generated IDs are stored during preparation; W&B runs are created only when
training starts. Each child logs `continuation.parent_wandb_run`, fork epoch/update, and stopping settings.
HPC resubmissions of that child reuse its new ID. Parent identities supplied through the environment
or sidecars are rejected.

Do not overwrite, delete, or splice the originals. Compare the two groups as initial versus extended
stopping policies. Child epoch numbers retain their original meaning and start at the fork, not zero;
the replayed interval is a separate trajectory, not a correction to old measurements.

During-training curves remain `val/*`. The restored-best checkpoint's final reevaluation is logged
separately under `selected/val/*` and exported privately under `publication/predictions/selected/val/`.
It cannot create a misleading upward jump at the stopping epoch in the new training curve.
The consistency report uses epochs **B-9 through B**, anchored to the retained macro-F1-selected
checkpoint, excluding restored-model reevaluations. This replaces the earlier final-ten-epoch proposal
(decision 2026-10-05). Use the verified parent window, explicitly labeled inherited, when the child
retains its parent's checkpoint; otherwise use the child's own history, without splicing replayed
trajectories. See [the reporting plan](publication_plan.md#45-selected-checkpoint-consistency-best-epoch-minus-nine-through-best-epoch).
Keep original and continuation results in separate progress tables.

## Launch record

Submitted **2026-09-22 at 00:27 CEST** from HPC commit `97415de34c645143e10dbddec390fcb4881d03c5`.
LSF H100 array: **`29456470[1–8]`**. All eight were confirmed pending at 00:29 CEST,
waiting for available memory, job slots, or GPU resources.
Patience 20, ceiling 120, original LR schedule; 24-hour allocations with the documented one-week
auto-resume campaign. This campaign is now complete; do not submit duplicate jobs.

All eight parent checks, independent checkpoint copies/checksums, and data-preparation checks passed.
The originals remain untouched. Submission directory relative to the HPC repository:
`jobs/generated/publication_v1_interaction_patience20-20260922-002745-993630/`.

| Array index | Floor | Synthesis | Pretraining epochs | Resume epoch | New W&B run |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | .30 | 4 | 70 | [c9fec67e2ecf4e8085fa82f84723f408](https://wandb.ai/codllmdev/codllm/runs/c9fec67e2ecf4e8085fa82f84723f408) |
| 2 | 0 | .30 | 48 | 65 | [fd1a4c7714ad476f952a9ef75ddb0021](https://wandb.ai/codllmdev/codllm/runs/fd1a4c7714ad476f952a9ef75ddb0021) |
| 3 | 0 | .60 | 4 | 46 | [515792acbfed43be993f85f82eb13f91](https://wandb.ai/codllmdev/codllm/runs/515792acbfed43be993f85f82eb13f91) |
| 4 | 0 | .60 | 48 | 57 | [0e4a8c648c974c1ea1ef48847c0fd6d3](https://wandb.ai/codllmdev/codllm/runs/0e4a8c648c974c1ea1ef48847c0fd6d3) |
| 5 | 450 | .30 | 4 | 49 | [3f52710979ac46d099984ccc578724af](https://wandb.ai/codllmdev/codllm/runs/3f52710979ac46d099984ccc578724af) |
| 6 | 450 | .30 | 48 | 36 | [63b5465c4d8244e5a3f51c1d92f78de7](https://wandb.ai/codllmdev/codllm/runs/63b5465c4d8244e5a3f51c1d92f78de7) |
| 7 | 450 | .60 | 4 | 60 | [732fc79a61564f0b9cc1264b675d5727](https://wandb.ai/codllmdev/codllm/runs/732fc79a61564f0b9cc1264b675d5727) |
| 8 | 450 | .60 | 48 | 57 | [3cf6ecc34f254de0b43a029d5a15b473](https://wandb.ai/codllmdev/codllm/runs/3cf6ecc34f254de0b43a029d5a15b473) |

## Completion verification

Checked 2026-10-05: all eight W&B runs are `finished`; all eight HPC children have `.training_complete`
and `.final_eval_done`, with no `.resume_needed`. Selected model weights and prediction Parquet files exist,
and selected scalar reports agree with W&B. Final scheduler logs report successful completion and no further
resubmission required. The HPC queue was empty at verification.

All times below are CEST. Macro F1 changes are child minus parent, in percentage points, calculated before rounding.

| Array index | Resume epoch | Best epoch | Stop epoch | Epochs trained after fork | Macro F1 change (pp) | Completed |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 70 | 70 | 90 | 20 | +0.000 | 2026-09-22 12:51 |
| 2 | 65 | 89 | 109 | 44 | +0.390 | 2026-09-23 14:35 |
| 3 | 46 | 83 | 103 | 57 | +0.992 | 2026-09-27 16:50 |
| 4 | 57 | 57 | 77 | 20 | +0.000 | 2026-09-25 19:18 |
| 5 | 49 | 60 | 80 | 31 | +0.177 | 2026-09-26 16:56 |
| 6 | 36 | 78 | 98 | 62 | +0.529 | 2026-09-28 22:03 |
| 7 | 60 | 79 | 99 | 39 | +0.273 | 2026-09-28 12:41 |
| 8 | 57 | 76 | 96 | 39 | +0.243 | 2026-09-28 15:27 |

Every stop epoch equals best epoch + 20. None hit the 120-epoch ceiling. The shorter campaign is expected:
pretraining was reused and only 20–62 fine-tuning epochs were run after each fork, not a fresh full training run.

Indices 3, 6, 7, and 8 each reached their allocation time budget and resumed once, under jobs
`29484162`, `29499514`, `29503979`, and `29504589` respectively. Their final logs confirm successful
completion; these were normal wall-time resumptions, not incomplete or abandoned runs.
No recovery or resubmission is needed. The [selected-checkpoint consistency report](publication_checkpoint_consistency.md)
was completed on 2026-10-05. Scientific candidate selection remains separate from this execution-completion check.
