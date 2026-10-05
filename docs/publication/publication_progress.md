# Publication progress

Updated: 2026-10-05. Launch steps: [publication_runs.md](publication_runs.md).

Validation results, seed 777; test evaluation disabled. Metrics within each row describe one selected evaluation,
not independently maximized scores.
Floor and dose curves use provisional W&B history peaks; synthesis uses recovered selected-checkpoint metrics.

## Completed exploratory curve: balance floor

**Complete: 7/7 runs.** Fixed: 32 pretraining epochs, synthetic ratio 0.50.

| Floor | Macro F1 | Exact-match accuracy | Micro F1 | Sample F1 | Unseen-string macro F1 |
|---:|---:|---:|---:|---:|---:|
| [0](https://wandb.ai/codllmdev/codllm/runs/hfpg86ty) | **0.744945** | 0.960207 | 0.963972 | 0.964892 | 0.734134 |
| [100](https://wandb.ai/codllmdev/codllm/runs/u01po1qj) | 0.727998 | 0.958990 | 0.962712 | 0.963903 | 0.713292 |
| [200](https://wandb.ai/codllmdev/codllm/runs/qzz4cj80) | 0.729236 | 0.959814 | 0.963492 | 0.964639 | 0.714743 |
| [300](https://wandb.ai/codllmdev/codllm/runs/xzir6v5j) | 0.725513 | 0.959147 | 0.962941 | 0.964079 | 0.711659 |
| [450](https://wandb.ai/codllmdev/codllm/runs/1q6kfgnl) | 0.731604 | 0.959945 | 0.963565 | 0.964493 | 0.716200 |
| [600](https://wandb.ai/codllmdev/codllm/runs/45drv26d) | 0.728460 | 0.958768 | 0.962510 | 0.963776 | 0.712912 |
| [900](https://wandb.ai/codllmdev/codllm/runs/1y5t040t) | 0.722265 | 0.958362 | 0.962248 | 0.963526 | 0.704882 |

Provisional interpretation: **floor 0 wins**. No tested positive floor improves macro F1 under this recipe.

## Completed exploratory curve: multi-COD synthesis

**Complete: 7/7 runs.** Results recovered from HPC checkpoints.

| Synthetic ratio | Macro F1 | Exact-match accuracy | Micro F1 | Sample F1 | Unseen-string macro F1 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.720257 | 0.958663 | 0.962226 | 0.963451 | 0.705909 |
| 0.15 | 0.725130 | 0.958833 | 0.962696 | 0.963858 | 0.709559 |
| 0.30 | **0.735801** | 0.959579 | 0.963328 | 0.964334 | 0.719855 |
| 0.45 | 0.732696 | 0.959330 | 0.963127 | 0.964188 | 0.718728 |
| 0.60 | 0.733740 | 0.959173 | 0.963042 | 0.964042 | 0.718112 |
| 0.80 | 0.732242 | 0.959487 | 0.963369 | 0.964416 | 0.715001 |
| 1.00 | 0.721462 | 0.958519 | 0.962445 | 0.963588 | 0.705950 |

Provisional interpretation: **ratio 0.30 wins**. Ratios 0.45–0.80 are close, but increasing synthesis gives no improvement.

## Completed exploratory curve: pretraining dose

**Complete: 6/6 runs.** Fixed: floor 300, fine-tuning synthetic ratio 0.50; full ICD10h pretraining targets.

| Pretraining epochs | Macro F1 | Exact-match accuracy | Micro F1 | Sample F1 | Unseen-string macro F1 |
|---:|---:|---:|---:|---:|---:|
| [Off](https://wandb.ai/codllmdev/codllm/runs/rteaq2lk) | 0.724188 | 0.959173 | 0.962906 | 0.964126 | 0.709637 |
| [4](https://wandb.ai/codllmdev/codllm/runs/k7y91n4g) | **0.733186** | 0.959814 | 0.963403 | 0.964464 | 0.717039 |
| [8](https://wandb.ai/codllmdev/codllm/runs/7h6a2o1i) | 0.726152 | 0.959330 | 0.963183 | 0.964323 | 0.712252 |
| [16](https://wandb.ai/codllmdev/codllm/runs/uoces8ry) | 0.720516 | 0.958663 | 0.962332 | 0.963572 | 0.702699 |
| [32](https://wandb.ai/codllmdev/codllm/runs/l5knr2o7) | 0.725513 | 0.959147 | 0.962941 | 0.964079 | 0.711659 |
| [48](https://wandb.ai/codllmdev/codllm/runs/uju97ibn) | 0.732698 | 0.959448 | 0.963366 | 0.964349 | 0.717854 |

Provisional interpretation: 4 and 48 epochs differ by only 0.000488 macro F1. Retain both for paired
source-transfer evaluation; longer pretraining is not yet an established improvement.

## Publication v1

The curves above used the legacy row split, COD+age+sex input, and unseen-string metric.
New v1 results must remain separate: COD-only, grouped-COD, reference exclusion, versioned metrics.
Candidate files still contain provisional floor 0 / synthesis .30 with 48 versus 4 pretraining epochs;
the completed interaction results below do not automatically approve a recipe.

| Launch phase | Status | Brief interpretation |
|---|---|---|
| 0b: HPC smoke | [Complete](https://wandb.ai/codllmdev/codllm/runs/9bhbzqz8) | 1 pretraining + 1 fine-tuning epoch; 138 validation rows. Accuracy/micro F1/macro F1 = 0; exports and metrics verified. Execution check, not performance evidence; resumption not exercised. |
| 1a: interaction screening | 8/8 complete | Final two cells finished on 17 September. Original patience 10; results below remain separate from continuations. |
| 1a continuation: patience 20 | 8/8 complete; verified 5 October | Last finished 28 September. All exhausted patience 20 before the 120-epoch ceiling; six improved macro F1. |
| 1a metadata: COD+age+sex | 8 cells prepared; not submitted | Matched original patience-10 grid; frozen historical-partition preflight required. No results yet. |
| 1b: controls | Not verified in this update | Keep separate from the eight interaction cells. |
| 2a–b: paired five-source panel | Not started | Determines the transfer-supported recipe. |
| 3a–f: row comparison, source ablations, baselines | Not started | Awaiting candidate selection. |
| 4a: 25%/40% cohort calibration | Not started | Choose feasible seed-study fraction. |
| 4b–c: split and training seeds | Not started | Awaiting cohort calibration. |
| Optional 3g–h: metadata / perturbation | Not started | Conditional mechanism checks. |
| Optional 5a–c: larger models + source confirmation | Not started | Conditional on benefit and compute. |
| 6: frozen internal/external assessment | Not started | Awaiting final model and curated external data. |

### Phase 1a: completed interaction cells

First six verified 2026-09-16; final two completion checks on 17 September, with their selected macro/micro
scores added in the 2026-10-05 update. Evidence: HPC completion markers, retained checkpoint trainer states,
selected-checkpoint reports, and W&B. Ceiling: 120 fine-tuning epochs; stopping metric:
`val/macro_f1`; patience: 10. All scores below belong to the retained best-macro-F1 checkpoint.

| Floor | Synthesis | Pretrain epochs | Best epoch | Stopped at epoch | Macro F1 | Micro F1 | Block micro F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| [0](https://wandb.ai/codllmdev/codllm/runs/cvbcfp8q) | .30 | 4 | 70 | 80 | 0.593941 | 0.910057 | 0.945571 |
| [0](https://wandb.ai/codllmdev/codllm/runs/cfezy52r) | .30 | 48 | 65 | 75 | 0.593832 | 0.896024 | 0.937179 |
| [0](https://wandb.ai/codllmdev/codllm/runs/ytasoth7) | .60 | 4 | 46 | 56 | 0.585455 | 0.892143 | 0.928218 |
| [0](https://wandb.ai/codllmdev/codllm/runs/49wttgfv) | .60 | 48 | 57 | 67 | 0.592092 | 0.895769 | 0.935972 |
| [450](https://wandb.ai/codllmdev/codllm/runs/4d28c62y) | .30 | 4 | 49 | 59 | 0.578496 | 0.900891 | 0.934892 |
| [450](https://wandb.ai/codllmdev/codllm/runs/xjwaxnye) | .30 | 48 | 36 | 46 | 0.569893 | 0.825357 | 0.935400 |
| [450](https://wandb.ai/codllmdev/codllm/runs/wgfhan5j) | .60 | 4 | 60 | 70 | 0.582953 | 0.887947 | — |
| [450](https://wandb.ai/codllmdev/codllm/runs/gtqoizxs) | .60 | 48 | 57 | 67 | 0.585538 | 0.886960 | — |

The final two cells finished on 17 September. Their block micro F1 values are not transcribed in this update (—).

The final W&B validation point re-scores the restored best checkpoint at the stopping epoch. For the
epoch-80 run, the actual epoch-80 macro F1 was 0.590198; the final point repeats epoch 70's 0.593941.
Exclude restored-checkpoint reevaluations from the planned B-9 through B consistency summaries.

Provisional interpretation: floor 0 / synthesis .30 leads completed cells in macro F1, with nearly tied
pretraining doses. The floor-450/.30/48 selected checkpoint's low micro F1 illustrates the selection
tradeoff; assess sustained performance and other metrics before choosing a recipe. The continuation
consistency analysis below is complete; no separate all-parent consistency panel is reported.

### Phase 1a continuation: patience 20

**Complete: 8/8 runs. Verified 2026-10-05 against W&B and HPC.** Last completion: 28 September,
22:03 CEST. Each run has saved model weights, prediction exports, and training/final-evaluation completion
markers. W&B final selected metrics match HPC reports.

These are same-seed branches from retained best checkpoints, not independent replicates or repeated pretraining.
Patience 20; original 120-epoch ceiling and LR schedule. Scores are restored-best `selected/val/*` metrics,
not the final training-epoch scores or independent metric peaks.

| Floor | Synthesis | Pretrain epochs | Best epoch | Stopped at epoch | Macro F1 | Micro F1 | Equal-source reference macro F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| [0](https://wandb.ai/codllmdev/codllm/runs/c9fec67e2ecf4e8085fa82f84723f408) | .30 | 4 | 70 | 90 | 0.593941 | **0.910057** | 0.702627 |
| [0](https://wandb.ai/codllmdev/codllm/runs/fd1a4c7714ad476f952a9ef75ddb0021) | .30 | 48 | 89 | 109 | **0.597734** | 0.892177 | 0.701027 |
| [0](https://wandb.ai/codllmdev/codllm/runs/515792acbfed43be993f85f82eb13f91) | .60 | 4 | 83 | 103 | 0.595371 | 0.895525 | 0.697404 |
| [0](https://wandb.ai/codllmdev/codllm/runs/0e4a8c648c974c1ea1ef48847c0fd6d3) | .60 | 48 | 57 | 77 | 0.592092 | 0.895769 | **0.711639** |
| [450](https://wandb.ai/codllmdev/codllm/runs/3f52710979ac46d099984ccc578724af) | .30 | 4 | 60 | 80 | 0.580265 | 0.902555 | 0.695257 |
| [450](https://wandb.ai/codllmdev/codllm/runs/63b5465c4d8244e5a3f51c1d92f78de7) | .30 | 48 | 78 | 98 | 0.575179 | 0.891374 | 0.686982 |
| [450](https://wandb.ai/codllmdev/codllm/runs/732fc79a61564f0b9cc1264b675d5727) | .60 | 4 | 79 | 99 | 0.585678 | 0.887966 | 0.704681 |
| [450](https://wandb.ai/codllmdev/codllm/runs/3cf6ecc34f254de0b43a029d5a15b473) | .60 | 48 | 76 | 96 | 0.587967 | 0.892903 | 0.706692 |

Equal-source reference macro F1 is `pub_v1_source_mean_macro_f1_ref`, the prespecified screening metric;
checkpoint selection still uses pooled `macro_f1`. This is not held-out-source performance.
Every run stopped exactly 20 epochs after its best checkpoint; none reached the ceiling.

Provisional interpretation: six cells improved pooled macro F1; two retained their parent checkpoint.
The largest gain was floor 0 / synthesis .60 / pretraining 4: 0.585455 → 0.595371 (+0.992 percentage points).
Pooled macro F1 favors 0/.30/48, micro F1 favors 0/.30/4, and equal-source reference macro F1 favors 0/.60/48.
No overall recipe is approved. Best-ending ten-epoch consistency summaries are below; seed uncertainty
remains outstanding.

[Continuation timing, parent comparisons, and verification](publication_continuation.md#completion-verification).

### Phase 1a continuation: selected-checkpoint consistency

**Complete: 8/8 windows, calculated 2026-10-05.** Ten original full-epoch evaluations B-9 through B;
R1 (0/.30/4) and R4 (0/.60/48) use inherited parent histories. All metrics share the macro-selected
window; no restored-best reevaluations or mixed parent/child trajectories are included.

Scores are percentages; SD is descriptive checkpoint standard deviation in percentage points, not a CI.

| Floor / synthesis / pretrain | Mean macro F1 | SD | Mean equal-source ref. macro F1 | SD | Mean micro F1 | SD |
|---|---:|---:|---:|---:|---:|---:|
| 0 / .30 / 4 | 58.7378 | 0.3616 | 70.1533 | 0.1737 | 89.8415 | 0.7179 |
| 0 / .30 / 48 | 59.1326 | 0.3609 | 69.6027 | 0.2294 | 89.1565 | 0.0611 |
| 0 / .60 / 4 | 59.1531 | 0.1974 | 69.8144 | 0.2801 | 89.4233 | 0.1037 |
| 0 / .60 / 48 | 57.9316 | 0.5449 | 69.9081 | 0.5391 | 88.9445 | 0.3241 |
| 450 / .30 / 4 | 57.5314 | 0.2115 | 69.7715 | 0.2691 | 89.3524 | 1.9862 |
| 450 / .30 / 48 | 57.1225 | 0.2681 | 68.4767 | 0.2422 | 83.2298 | 2.0596 |
| 450 / .60 / 4 | 57.9037 | 0.3114 | 69.9568 | 0.2733 | 88.7153 | 0.1118 |
| 450 / .60 / 48 | 58.5126 | 0.2437 | 70.5245 | 0.2786 | 89.1268 | 0.8637 |

Provisional interpretation: floor 0 leads pooled macro F1 in 3/4 matched window-mean comparisons,
versus 4/4 selected-checkpoint comparisons. At .60/48, floor 450 instead leads the window mean by
0.5810 pp. The highest mean macro F1 is 0/.60/4, only 0.0205 pp above 0/.30/48; the highest mean
equal-source score is 450/.60/48, not the selected-score leader 0/.60/48.
For 450/.30/48, selected micro F1 (89.1374%) exceeds the preceding-nine median by 6.3252 pp;
its nearby checkpoints are substantially weaker. These selection-conditioned observations qualify
the main comparison, not prove superiority or change candidate approval.

[Full consistency report](publication_checkpoint_consistency.md): all windows, medians, variance/SD,
extremes, endpoint uplift, trends, block metrics, matched contrasts, and reproducible aggregate inputs.
