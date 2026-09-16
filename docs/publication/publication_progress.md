# Publication progress

Updated: 2026-09-16. Launch steps: [publication_runs.md](publication_runs.md).

Validation results, seed 777; test evaluation disabled. Each row uses the same peak-macro-F1 evaluation.
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
Floor 0 / synthesis .30 with 48 versus 4 pretraining epochs is provisional, not a tested joint winner.

| Launch phase | Status | Brief interpretation |
|---|---|---|
| 0b: HPC smoke | [Complete](https://wandb.ai/codllmdev/codllm/runs/9bhbzqz8) | 1 pretraining + 1 fine-tuning epoch; 138 validation rows. Accuracy/micro F1/macro F1 = 0; exports and metrics verified. Execution check, not performance evidence; resumption not exercised. |
| 1a: interaction screening | 6/8 complete; 2 running | Completed cells stopped after 10 epochs without a new best validation macro F1. See below. |
| 1a continuation: patience 20 | Setup prepared; not launched | Same 120-epoch ceiling/LR schedule. Wait for all parents; preserve original results and report child runs separately. |
| 1b: controls | Not verified in this update | Keep separate from the eight interaction cells. |
| 2a–b: paired five-source panel | Not started | Determines the transfer-supported recipe. |
| 3a–f: row comparison, source ablations, baselines | Not started | Awaiting candidate selection. |
| 4a: 25%/40% cohort calibration | Not started | Choose feasible seed-study fraction. |
| 4b–c: split and training seeds | Not started | Awaiting cohort calibration. |
| Optional 3g–h: metadata / perturbation | Not started | Conditional mechanism checks. |
| Optional 5a–c: larger models + source confirmation | Not started | Conditional on benefit and compute. |
| 6: frozen internal/external assessment | Not started | Awaiting final model and curated external data. |

### Phase 1a: completed interaction cells

Verified 2026-09-16 against HPC completion markers, retained checkpoint trainer states, generated
job settings, and complete W&B validation histories. Ceiling: 120 fine-tuning epochs; stopping metric:
`val/macro_f1`; patience: 10. All scores below belong to the retained best-macro-F1 checkpoint.

| Floor | Synthesis | Pretrain epochs | Best epoch | Stopped at epoch | Macro F1 | Micro F1 | Block micro F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| [0](https://wandb.ai/codllmdev/codllm/runs/cvbcfp8q) | .30 | 4 | 70 | 80 | 0.593941 | 0.910057 | 0.945571 |
| [0](https://wandb.ai/codllmdev/codllm/runs/cfezy52r) | .30 | 48 | 65 | 75 | 0.593832 | 0.896024 | 0.937179 |
| [0](https://wandb.ai/codllmdev/codllm/runs/ytasoth7) | .60 | 4 | 46 | 56 | 0.585455 | 0.892143 | 0.928218 |
| [0](https://wandb.ai/codllmdev/codllm/runs/49wttgfv) | .60 | 48 | 57 | 67 | 0.592092 | 0.895769 | 0.935972 |
| [450](https://wandb.ai/codllmdev/codllm/runs/4d28c62y) | .30 | 4 | 49 | 59 | 0.578496 | 0.900891 | 0.934892 |
| [450](https://wandb.ai/codllmdev/codllm/runs/xjwaxnye) | .30 | 48 | 36 | 46 | 0.569893 | 0.825357 | 0.935400 |

Floor 450 / synthesis .60 / pretraining [4](https://wandb.ai/codllmdev/codllm/runs/wgfhan5j)
and [48](https://wandb.ai/codllmdev/codllm/runs/gtqoizxs) remain running, at approximately epochs
51.7 and 45.3 when checked at 13:38 CEST. Do not rank the full interaction yet.

The final W&B validation point re-scores the restored best checkpoint at the stopping epoch. For the
epoch-80 run, the actual epoch-80 macro F1 was 0.590198; the final point repeats epoch 70's 0.593941.
Exclude restored-checkpoint reevaluations from the planned ten-epoch stability summaries.

Provisional interpretation: floor 0 / synthesis .30 leads completed cells in macro F1, with nearly tied
pretraining doses. The floor-450/.30/48 selected checkpoint's low micro F1 illustrates the selection
tradeoff; assess sustained performance and other metrics before choosing a recipe. Ten-epoch summaries
remain to be computed after the interaction completes.
