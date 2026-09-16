# Interaction continuation: patience 20

Decision: 2026-09-16. Increase fine-tuning early-stopping patience from 10 to 20 for all eight
interaction cells. Lucas explicitly retained the **120-epoch ceiling and original learning-rate
schedule**; the briefly considered 160-epoch extension is not adopted.

Specification: `runs/publication/interaction_patience20.toml`.
Exact commands: [publication_runs.md](publication_runs.md#phase-1a-continuation-patience-20--8-runs-ceiling-still-120).
This prepares a new campaign; it does not submit jobs or change already-running allocations.

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
The ten-epoch stability report uses the child's actual final ten training-epoch evaluations, excluding
that restored-model reevaluation. Keep original and continuation results in separate progress tables.
