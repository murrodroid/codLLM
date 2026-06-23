# Onboarding: Belgium + Ipswich LOSO holdout runs (RQ3)

Run the two remaining cities (Belgium, Ipswich) with the same optimal deployment config
as the three already running (Amsterdam, Copenhagen, Madrid), using the standard
invoke submission path (`uv run invoke hpc.submit`).

Both specs inherit `size_sweep_base_final.toml`: upsampling floor 200, synthetic
multi-CoD 0.30, constant-with-warmup LR schedule, masterlist pretraining on, input =
cod + age + sex, 80 epochs, patience-10 early-stop, select on macro-F1. LOSO removes
the held-out city before the 90/5/5 split, so the model trains on the other four and is
evaluated on the held-out city (no leak).

## 1. On the HPC login node (s234805), in the repo
```bash
source hpc/env.sh                 # storage bootstrap (caches under $RUN_STORAGE_DIR)
uv sync --frozen --no-dev
git checkout main && git pull origin main    # brings the holdout fix + the ready TOMLs
grep -c "should_evaluate = True" src/codllm/trainer_logging.py   # MUST print 2
```
The last line confirms the one critical fix is present. Without it, best-checkpoint
selection and early-stopping silently break (the run keeps the last, overfit checkpoint).
The fix and the specs are on `main` (merged from `elias`). If `grep` does not print 2, apply this to
`src/codllm/trainer_logging.py` in both `HoldoutEvaluationCallback.on_epoch_end` and
`on_step_end`:
```python
# replace:  self._evaluate(include_baseline=not control.should_evaluate)
# with:
        native_eval_scheduled = control.should_evaluate
        self._evaluate(include_baseline=not native_eval_scheduled)
        if native_eval_scheduled:
            control.should_evaluate = True
```

## 2. Output dir (one-time, only if needed)
The two TOMLs set `CODLLM_OUTPUT_DIR = /dtu/blackhole/1e/205502/codllm/...`. If `205502`
is the shared project allocation you can write to, leave it. Otherwise change `205502`
to your own blackhole allocation in `runs/single/base_holdout_belgium.toml` and
`base_holdout_ipswich.toml`.

## 3. Submit (24h H100 profile)
```bash
uv run invoke hpc.submit --config runs/single/base_holdout_belgium.toml --profile h100-24h
uv run invoke hpc.submit --config runs/single/base_holdout_ipswich.toml --profile h100-24h
bjobs
```
`h100-24h` = gpuh100, 24:00 wall, 17 cores, 1 GPU exclusive. The TOMLs carry a 23h
internal runtime budget (`MAX_RUNTIME_SECONDS=82800`), so each run saves and stops ~1h
before the wall. Add `--dry-run` to inspect the generated LSF file without submitting.

## 4. The expected SIGFPE crash (~ep35-50)
These runs hit a systematic native floating-point fault in a per-epoch eval metric once
predictions sharpen. It is NOT a leak and NOT the wall; the val-selected peak checkpoint
is always saved before the crash, and a checkpoint is written every epoch.
- On a crash, just re-run the same `uv run invoke hpc.submit ... --profile h100-24h`
  line for that city. `CODLLM_AUTO_RESUME = true` continues from the latest checkpoint
  (under one epoch lost) and advances toward the early-stop. Usually ~1 resubmit per city.
- `CODLLM_UNCERTAINTY_EVAL = false` (already set) removes the end-of-training instance so
  a finished run exits cleanly.

## 5. Done = the patience-10 early-stop
A city is finished when its best checkpoint is >= 10 epochs behind the current epoch
(patience-10 fired) and the run exits. Quick check:
```bash
uv run invoke hpc.storage    # or inspect runs_holdout_<city>_fixed/run-*/finetune/checkpoint-*/trainer_state.json
```

## 6. After both early-stop: the RQ3 numbers
On each city's val-selected best checkpoint, run the post-hoc uncertainty inference
(`experiments/uncertainty/local_infer.py` + the analysis step) to get the per-record
confidence-vs-accuracy / risk-coverage deferral numbers. These feed
`07_experiments_results.tex` (`sec:results-transfer-uncertainty`, `tab:transfer`).
Report transfer at the val-selected (no-peek) checkpoint, not an earlier holdout-eval peak.

## Notes
- W&B may show a run "crashed" during a long eval (5-min heartbeat) while it is training;
  cross-check `bjobs`.
- Belgium (~2.1k) and Ipswich (~2k) are small held-out sets, but the train pool is the
  other four (large), so runtime is similar to the three already running.
- VPN must be connected for HPC access.
