# Onboarding: Belgium + Ipswich LOSO holdout runs (RQ3)

Goal: finish the 5-city leave-one-source-out (LOSO) set for RQ3 by running the two
remaining cities, **Belgium** and **Ipswich**, with the exact same optimal deployment
config as the three already running (Amsterdam, Copenhagen, Madrid).

## What "optimal config" means
Both specs (`runs/single/base_holdout_belgium.toml`, `base_holdout_ipswich.toml`)
inherit `size_sweep_base_final.toml`:
- upsampling floor 200, synthetic multi-CoD ratio 0.30, constant-with-warmup LR
  schedule, masterlist pretraining ON, 80 max epochs, patience-10 early-stop,
  input = cause text + age + sex.
- LOSO: the held-out city is removed from the pool **before** the 90/5/5 split,
  so the model trains on the other four and is evaluated on the held-out city.
  No leak (validation breaks down into the four trained-on sources only).

## Prereqs
- DTU VPN up (Cisco Secure Client) + SSH to the HPC.
- A repo checkout on whichever account you run on.
- The critical code fix below must be present in that checkout.

## 1. The one critical code fix (do not skip)
Without it, best-checkpoint selection and early-stopping silently break and the run
keeps the LAST (overfit-to-source) checkpoint. It is already applied on the
s234854 checkout. If you run on your own account, apply this to
`src/codllm/trainer_logging.py`, in BOTH `HoldoutEvaluationCallback.on_epoch_end`
and `on_step_end`:

```python
# replace:
        self._evaluate(include_baseline=not control.should_evaluate)
# with:
        native_eval_scheduled = control.should_evaluate
        self._evaluate(include_baseline=not native_eval_scheduled)
        if native_eval_scheduled:
            # our manual evaluate() flips should_evaluate to False
            # (HF CallbackHandler.on_evaluate); re-arm so the Trainer's own
            # eval still runs and updates best_metric + early stopping.
            control.should_evaluate = True
```
(Committed locally as `fd484bc` on the codLLM repo.)

## 2. TOML settings (already pre-staged on s234854; replicate if on your own account)
Each `[env]` block must contain:
```
CODLLM_HOLD_OUT_DATASET = "belgium_1920_1930"     # or "ipswich_1871_1911"
CODLLM_HOLD_OUT_EVALUATE_PER = "epoch"
CODLLM_HOLD_OUT_EVALUATE_RATIO = 0.10
CODLLM_OUTPUT_DIR = "/dtu/blackhole/<YOUR-ALLOCATION>/codllm/runs_holdout_belgium_fixed"
CODLLM_DATALOADER_PERSISTENT_WORKERS = false   # avoids a dataloader-cache bug
CODLLM_SAVE_TOTAL_LIMIT = 10                    # keep enough checkpoints
CODLLM_AUTO_RESUME = true                       # resume the latest run-* dir on resubmit
CODLLM_UNCERTAINTY_EVAL = false                 # skip the SIGFPE-prone end-of-training pass
```
If you are NOT on the 205502 allocation, change the `205502` in `CODLLM_OUTPUT_DIR`
to your own blackhole allocation.

## 3. Submit
```
bash .claude/skills/dtu-hpc/driver.sh submit runs/single/base_holdout_belgium.toml
bash .claude/skills/dtu-hpc/driver.sh submit runs/single/base_holdout_ipswich.toml
bash .claude/skills/dtu-hpc/driver.sh bjobs          # confirm RUN/PEND
```

## 4. The known SIGFPE crash (expected; not a leak, not the wall)
These runs hit a systematic native SIGFPE (exit code 136) in a per-epoch eval
metric once the model gets confident (typically ep35-50). The val-selected peak
checkpoint is ALWAYS saved before the crash, and the runs save every epoch, so a
crash loses under one epoch.
- The mid-training crash: just resubmit the same TOML. `AUTO_RESUME` continues from
  the latest per-epoch checkpoint and advances toward the early-stop. Usually ~1
  resubmit per city is enough to reach the 10-no-improvement stop.
- Hands-off option: copy `_holdout_autoresubmit.sh`, change `SRCS="madrid amsterdam
  copenhagen"` to `SRCS="belgium ipswich"` and the blackhole path, and run it in the
  background. It resubmits on a real mid-training crash and marks a city done once its
  best checkpoint is >= 10 epochs behind the current epoch (patience fired).

Per-city progress check (epoch / best / is-it-early-stopped):
```
ssh dtu "bash -lc 'for s in belgium ipswich; do D=/dtu/blackhole/<ALLOC>/codllm/runs_holdout_\${s}_fixed; L=\$(ls -1dt \$D/run-*/finetune/checkpoint-* | head -1); python3 -c \"import json;s=json.load(open(\\\"\$L/trainer_state.json\\\"));ep=s.get(\\\"epoch\\\");g=s.get(\\\"global_step\\\");b=float((s.get(\\\"best_model_checkpoint\\\") or \\\"-0\\\").split(\\\"-\\\")[-1]);print(\\\"$s ep\\\",round(ep,1),\\\"best\\\",round(s.get(\\\"best_metric\\\") or 0,4),\\\"epochs_since_best\\\",round(ep-b*ep/g,1))\"; done'"
```
A city is DONE when `epochs_since_best >= 10` (patience-10 fired) and the run exits.

## 5. After both early-stop: the RQ3 numbers
On each city's val-selected best checkpoint, run the post-hoc uncertainty inference
(`experiments/uncertainty/local_infer.py` + the analysis step) to get the per-record
confidence-vs-accuracy / risk-coverage deferral numbers. These feed
`07_experiments_results.tex` (`sec:results-transfer-uncertainty`, `tab:transfer`).
Report the transfer at the val-selected (no-peek) checkpoint; do not cherry-pick an
earlier holdout-eval peak (that would be test-set tuning).

## Gotchas
- W&B may flag a run "crashed" during a long eval (5-min heartbeat) while it is
  actually training. Cross-check `bjobs` before believing it.
- Belgium (~2.1k) and Ipswich (~2k) are small as held-out sets, but the TRAIN pool is
  the other four (large), so runtime is similar to the three already running.
- 24h wall: `AUTO_RESUME` + a resubmit handles it; nothing is lost.
- VPN must be connected for every HPC command.
