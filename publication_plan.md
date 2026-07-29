# codLLM publication optimization plan

## Objective

The objective is to select the strongest publishable codLLM model while producing
defensible evidence for three scientific questions:

1. At what balance floor do further increases stop improving macro F1?
2. How much synthetic multi-COD training data is beneficial before it distorts the
   observed single-COD/multi-COD distribution?
3. How much masterlist pretraining is useful, and does its benefit transfer across
   historical sources?

The recommended strategy is a staged hybrid: controlled response curves for
interpretability, Bayesian optimization for interactions, factorial confirmation,
grouped cross-validation and leave-one-source-out validation for finalists, and
model-scale optimization last. Pure Bayesian optimization could locate a strong
configuration, but it would not by itself answer the three scientific questions.

## Evidence from the thesis and latest completed run

The thesis response curves ended before a clear plateau:

| Setting | Thesis values | Best observed |
|---|---:|---:|
| Balance floor | 0, 25, 50, 100, 200 | 200 |
| Fine-tuning synthetic ratio | 0, 0.15, 0.30 | 0.30 |
| Pretraining | Primarily on/off | Mixed source-transfer benefit |

Balance-floor macro F1 rose from 0.441 at zero to 0.520 at 200. Synthetic
multi-COD macro F1 rose from 0.467 at zero to 0.545 at 0.30. Both sweeps ended
while performance was still rising, so they establish promising directions but
not optima.

The thesis deployment and latest completed run differ as follows:

| Parameter/result | Thesis deployment | Latest run |
|---|---:|---:|
| Balance floor | 200 | 300 |
| Fine-tuning synthetic ratio | 0.30 | 0.50 |
| Pretraining epochs | 16 | 32 |
| Pretraining target per label | 8 | 12 |
| Pretraining synthetic ratio | 0.15 | 0.50 |
| Approximate pretraining examples presented | 2.1M | 8.1M |
| Maximum target labels | 3 | 5 |
| Fine-tuning scheduler | constant with warmup | cosine |
| Test macro F1 | 0.749 | 0.737 |
| Unseen-string macro F1 | 0.625 | 0.723 |
| Exact accuracy | 0.985 | 0.959 |

This is not a controlled comparison. The scheduler, label cardinality, curation,
data preparation, and test composition changed. It nevertheless shows that
maximizing augmentation and pretraining does not automatically maximize aggregate
macro F1. The latest run is a useful search seed, not proof that floor 300,
synthetic ratio 0.50, and 32 pretraining epochs are optimal.

## Stage 0: freeze the publication protocol

Before using results for model selection, freeze:

- Dataset and curation hashes.
- Label harmonization and standardization.
- `max_label_count=5`.
- COD, age, and sex as training inputs.
- The data and training seeds for each declared repetition.
- Split manifests.
- Metrics and checkpoint-selection rules.
- Explicit scheduler, warmup, augmentation, and pretraining controls.

The historical test set has already influenced development. It must be described
as a legacy test set rather than an untouched publication test. For an unbiased
final estimate, either create a new locked grouped outer test split that remains
unobserved until selection is complete, or use nested grouped cross-validation if
no untouched data can be reserved.

Group splits by normalized COD text so identical causes cannot cross folds.
Approximately stratify folds by source, label frequency, and label cardinality.
Changing `data_seed` is not a substitute for cross-validation.

The response curves and Bayesian search must not evaluate the test split. Selection
uses validation macro F1 only. Test, holdout, and uncertainty evaluation are restored
only for declared finalists.

## Stage 1: measure the three response curves

Use `google/flan-t5-base`, the full training corpus, seed and data seed 777, a
fixed validation split, and identical controls apart from the parameter being
measured.

| Question | Values |
|---|---|
| Balance floor | 0, 100, 200, 300, 450, 600, 900 |
| Fine-tuning synthetic ratio | 0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00 |
| Pretraining epochs | off, 4, 8, 16, 32, 48 |

For the first pretraining curve, hold target-per-label at 12 and pretraining
synthetic ratio at 0.30. Follow it with secondary curves over:

- Pretraining synthetic ratio: 0, 0.15, 0.30, 0.50.
- Pretraining target per label: 8, 12, 16.

Report pretraining dose as:

```text
pretraining rows × epochs
optimizer updates
```

Epochs alone are not comparable when pretraining upsampling and synthetic ratios
change the pretraining dataset size.

For every curve cell, record:

- Overall macro, micro, and sample F1.
- Exact-match accuracy.
- Seen- and unseen-string macro F1.
- Single-COD and multi-COD subset metrics.
- Per-source macro F1.
- Macro F1 by label-support bucket.
- Mean predicted label count and false multi-label rate.
- Prepared training rows and generated rows by augmentation type.
- Pretraining rows, epochs, examples presented, and optimizer updates.

Use a saturation rule defined before inspecting the curve: stop increasing a
parameter after two consecutive increases improve mean macro F1 by less than
0.002 and the upper bound of the paired 95% confidence interval is below 0.005.
Select the smallest setting within one standard error of the best result.

The checked-in response curves are single-seed discovery experiments. Repeat the
plateau neighborhood with at least three training seeds before making a scientific
saturation claim.

## Stage 2: Bayesian interaction search

After locating sensible response-curve ranges, run 40 to 60 W&B Bayesian trials on
the base model. Use full data and 24 fine-tuning epochs for medium-fidelity
screening. Each Bayesian trial must fit inside one scheduler slot; it must not use
the multi-slot codLLM auto-resume lifecycle.

Recommended search space:

| Parameter | Search space |
|---|---|
| Balance floor | quantized 100–900 |
| Fine-tuning synthetic ratio | 0.10–0.90 |
| Pretraining epochs | off, 4, 8, 16, 32, 48 |
| Pretraining synthetic ratio | 0–0.50 |
| Pretraining target per label | 8, 12, 16 |
| Learning rate | log-uniform 1e-5–5e-5 |
| Fine-tuning scheduler | cosine, constant |
| Warmup ratio | 0.05–0.30 |
| Base perturbation rate | 0, 0.25, 0.50 |

Seed or compare the optimizer with:

- The thesis recipe: floor 200, ratio 0.30, pretraining 16.
- The latest recipe: floor 300, ratio 0.50, pretraining 32.
- A no-pretraining baseline.
- A low-augmentation baseline.

Optimize a single end-of-run metric,
`optimization/val_macro_f1`, generated from the trainer's best validation
`macro_f1`. Retain accuracy, unseen-string macro F1, worst-source macro F1,
multi-COD performance, training rows, and runtime for offline Pareto analysis.
Do not hide these tradeoffs inside an arbitrary composite score.

Run W&B agents in waves of four, with exactly one trial per LSF allocation. This
lets Bayesian search incorporate completed observations before assigning most of
the budget. Do not enable Hyperband until codLLM has an explicit pruned-run
lifecycle; scheduler termination currently cannot be safely distinguished from
normal completion.

Promote the best five Bayesian configurations to full 100–120 epoch training with
fine-tuning early stopping. Treat the Bayesian outcome as a shortlist, not as the
final scientific estimate.

## Stage 3: confirm interactions

To determine whether the three key settings work together, perform a local
factorial confirmation around the Bayesian optimum:

- Floor: one level below, the optimum, and one level above.
- Synthetic ratio: optimum minus 0.15, optimum, and optimum plus 0.15.
- Pretraining dose: lower, optimum, and higher.

Run the resulting 27 cells once, then repeat the nine most informative cells with
three seeds. Estimate main effects and pairwise interactions. This provides a more
interpretable interaction result than the optimizer's selected point alone.

## Stage 4: grouped cross-validation and source transfer

Do not run cross-validation for every Bayesian trial. Use this promotion sequence:

1. Fixed development fold for curves and Bayesian screening.
2. Top five configurations at full fidelity with three seeds.
3. Top three configurations across five grouped folds.
4. Winning configuration across five grouped folds with three seeds.
5. Winner and runner-up across all leave-one-source-out evaluations.

General grouped cross-validation and leave-one-source-out evaluation answer
different questions:

- Grouped CV measures robustness to cause strings and ordinary split composition.
- Leave-one-source-out measures transfer to a distinct historical collection.

Run leave-one-source-out evaluation for Amsterdam, Belgium, Copenhagen, Ipswich,
and Madrid. Report the mean, standard deviation, worst source, and every individual
source result.

General fold support requires a committed manifest and configuration such as:

```text
CODLLM_CV_MANIFEST_PATH
CODLLM_CV_FOLD
```

Both fields and the manifest hash must be part of prepared-split cache metadata.

## Stage 5: optimize model scale

The thesis indicated that large and XL models could still improve macro F1. After
optimizing the recipe on base:

1. Train the top two configurations on `flan-t5-large`.
2. Train the winning large configuration on `flan-t5-xl`.
3. Use full-fidelity early stopping.
4. Repeat the winning final scale with at least three seeds.

Without this stage, the project identifies the best base-model recipe rather than
the best codLLM model.

For release, publish one reproducible single checkpoint selected by a prespecified
rule rather than cherry-picking the best seed. Optionally release a three-seed
ensemble as the maximum-performance system.

## Final evaluation and statistical reporting

Open the locked outer test only after every hyperparameter, seed policy, and model
scale decision is frozen. Report:

- Overall exact accuracy, macro F1, micro F1, sample F1, and Jaccard metrics.
- Seen- and unseen-string performance.
- Single-COD and multi-COD performance.
- Per-source results.
- Rare-label support buckets.
- Chapter and chapter-block results.
- Label-count error and false multi-label rate.
- Paired 95% cluster-bootstrap confidence intervals, clustered by normalized COD
  text.
- Mean and standard deviation across declared seeds and folds.

If two configurations differ by less than 0.002 macro F1 or their paired interval
includes zero, prefer the smaller floor, lower synthetic ratio, or shorter
pretraining dose. This yields a simpler and less expensive model when evidence does
not support the added data generation.

After final evaluation, train the release checkpoint on all eligible development
data using the frozen recipe. Do not present that all-data model's training
performance as an unbiased generalization estimate.

## Compute budget and promotion gates

A thorough programme is approximately:

- 20 response-curve jobs.
- 40–60 Bayesian screening jobs.
- 15 full-fidelity finalist jobs.
- 15 initial grouped-CV jobs.
- 15 winner fold/seed repetitions.
- 5–10 source-holdout jobs.
- 4–8 large/XL jobs.

This is roughly 115–140 base-equivalent runs plus larger-model cost. Promotion
gates and early stopping keep weak regions from consuming the full budget.

Stop Bayesian screening when the 60-run cap is reached, or when the best
cross-validation candidate has not improved by more than 0.002 across 15 completed
trials and the explored high-probability region is stable.

## Reproducibility and storage

For every run, preserve:

- Git commit.
- Full effective configuration.
- W&B sweep and group identifiers.
- Data-source and curation hashes.
- Prepared-split cache key and fold-manifest hash.
- Generated-row diagnostics.
- Training and pretraining exposure.
- Checkpoint selected and stopping reason.
- Failed, interrupted, and rejected trials.

Use full W&B configuration and metrics for response curves and final confirmation.
Use standard W&B configuration and metrics for Bayesian screening to control run
volume. Never synthesize `WANDB_SWEEP_ID`; set it only to the real identifier
created by W&B.

Curve cells and Bayesian trials require isolated output roots. This prevents
parallel runs and W&B run-ID sidecars from sharing checkpoints. Prebuild explicit
response-curve caches with `hpc.build`; build Bayesian candidates on demand to
avoid materializing the entire Cartesian data-cache space.

Reusable pretraining checkpoints should eventually be keyed by model, data hash,
seed, pretraining dose, target-per-label, and pretraining synthetic ratio. This
would prevent floor and fine-tuning synthetic-ratio trials from repeating
identical pretraining work.

## Checked-in implementation

The initial implementation lives under `runs/publication/`:

```text
runs/publication/base.toml
runs/publication/balance_floor_curve.toml
runs/publication/multicod_synthetic_curve.toml
runs/publication/pretraining_dose_curve.toml
runs/publication/bayesian_base.toml
runs/publication/bayesian_sweep.yaml
```

The response curves use seed and data seed 777 and isolated checkpoint roots.
Bayesian trials use the allowlisted wrapper in
`src/codllm/experiments/bayesian.py`. Operational commands and safeguards are
documented in `runs/publication/README.md`.
