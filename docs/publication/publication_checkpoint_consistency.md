# Phase 1a patience-20 checkpoint consistency

Calculated 2026-10-05. Method: [publication plan §4.5](publication_plan.md#45-selected-checkpoint-consistency-best-epoch-minus-nine-through-best-epoch).
Companion selected-checkpoint results: [publication progress](publication_progress.md#phase-1a-continuation-patience-20).

## Main findings

- **The pooled-macro result broadly persists, with one important exception.** Floor 0 leads all four
  matched floor comparisons at the selected checkpoint, but only three of four by window mean and median.
  At synthesis .60 / pretraining 48, floor 450 has mean macro F1 **58.5126%** versus **57.9316%**
  for floor 0 (+0.5810 pp), despite a lower selected score.
- **The strongest pooled-macro window is 0/.60/4 (R3), narrowly ahead of 0/.30/48 (R2).**
  Means are **59.1531% versus 59.1326%** (only 0.0205 pp apart); SDs are **0.1974 versus 0.3609 pp**.
  R2 still has the better selected macro F1 (59.7734% versus 59.5371%). This is not evidence of a
  statistically reliable superiority for either recipe.
- **The equal-source leader depends on selected versus sustained performance.** R4 (0/.60/48) leads
  selected equal-source reference macro F1 at **71.1639%**, but its window mean is **69.9081%**,
  median **69.6866%**, SD **0.5391 pp**, and endpoint uplift **1.4799 pp**. R8 (450/.60/48) leads
  window mean at **70.5245%**, followed by R1 (0/.30/4) at **70.1533%**. R4 improves across its
  window (source-score slope +0.0916 pp/epoch), so its high endpoint is not automatically a lucky spike.
- **Micro/block stability and macro stability are different.** R2 has micro SD **0.0611 pp** and
  block-micro SD **0.0493 pp**. R1 leads their window means (**89.8415%** and **94.1212%**), but
  fluctuates more (SD **0.7179** and **0.5203 pp**). A smoother trajectory is not automatically better.
- **R6 (450/.30/48) needs particular caution.** Its selected micro F1 is **89.1374%**, but its
  ten-epoch mean is **83.2298%**, median **82.8972%**, and endpoint uplift **6.3252 pp**.
  Epochs 69–77 remain between 81.5876% and 83.2816%; epoch 78 jumps to 89.1374%.
  Block micro F1 remains much steadier (SD **0.1516 pp**) and is lower at the selected endpoint than
  its preceding-nine median. This documents a fine-code/block-level discrepancy, not its cause.
- **Synthesis and pretraining remain conditional choices.** Moving synthesis .30 → .60 improves
  window-mean pooled macro F1 for both floor-450 doses and at floor 0/PT4, but worsens it at
  floor 0/PT48. More pretraining improves the mean at 0/.30 and 450/.60, and worsens it at the other
  two settings. The matched-contrast table below preserves the numerical tradeoffs.

**Phase 1a takeaway:** selected-checkpoint results are not equally representative of nearby checkpoints.
Floor 0 remains promising for pooled macro F1, but no recipe wins every criterion. Preserve R3 as a
pooled-macro consistency contender and R8 as an equal-source consistency contender alongside the existing
selected-score leaders. This is a reporting recommendation, not approval to change candidate TOMLs,
launch additional jobs, or replace the prespecified selection metric.

## Evidence and limitations

- All **8/8 windows**, **80 full-epoch records**, and **8 scalar histories per window** were recovered
  from retained HPC `trainer_state.json` files. Accuracy and exact match coincide in every recovered row,
  so there are seven distinct reported metric series, not eight independent outcomes.
- All windows have exactly ten consecutive integer epochs with the expected optimizer-step boundaries,
  ending at the saved best checkpoint and its recorded macro F1. No interpolation or smoothing was used.
- R1 and R4 use explicitly inherited parent windows. Their child checkpoint retained the original best
  model; these are not new continuation observations. The other six use child histories without parent
  splicing. R6 includes epochs across an ordinary allocation resume, with consecutive epoch/step coverage.
- All runs share the frozen validation manifest: **75,536 rows**, digest
  `545a6160134898e3519d8e32c4088f0148bd973d211abd2420ee840bb2ceb3e1`.
  Per-recipe exposure-reference fingerprints differ and are preserved in the ledger; identical validation
  membership does not imply identical augmented training exposure.
- For all eight runs, endpoint macro F1, micro F1, exact match, and equal-source reference macro F1
  agree with the separate restored-best reports (32 checks, no discrepancy). Block micro F1, sample F1,
  and loss here are original epoch-B observations; separate restored-best values were not retrieved for them.
- Source-state SHA256 hashes and exact epoch/step/metric values are recorded in the aggregate input ledger.
  Parent R1 and resumed-child R6 windows were also inspected directly. R4's rising endpoint trajectory
  was inspected before interpreting its large uplift.
- These are seed-777, COD-only, grouped-COD validation findings. Windows are correlated,
  selection-conditioned, and located at different epochs. They do not establish seed uncertainty,
  statistical significance, equal-compute efficiency, post-selection persistence, or source-held-out performance.
- No new training, test evaluation, checkpoint selection, remote edits, or W&B writes were performed.
  Supplementary demographic/rare-code/per-source histories were not part of this extraction; no consistency
  conclusions about those slices are made. This does not claim those histories are unavailable remotely.

## Reproduction

The checked-in [aggregate input ledger](checkpoint_consistency_input.json) contains scalars and provenance only,
not individual descriptions, labels, or predictions. Its SHA256 is
`60b6199e94667aecfcbebc011bc2132d7b6cac7c1b33a899abeca9b0a55bfaa2`.

From the laptop repository, reproduce the full-precision JSON (including variance) and numerical tables:

```shell
uv run --no-sync python -m experiments.checkpoint_consistency \
  --input docs/publication/checkpoint_consistency_input.json \
  --output-json logs/publication/checkpoint_consistency.json \
  --output-md logs/publication/checkpoint_consistency.md
uv run --no-sync pytest tests/test_checkpoint_consistency.py -q
```

The calculator has no network access or training side effects. The test suite checks exact window bounds,
divisor-10 variance, slope/uplift direction, missing metrics, duplicate conflicts, fork boundaries,
partial epochs, restored-best exclusions, inherited windows, and validation identity.
The narrative and matched contrasts below interpret the reproducible numerical tables; they do not alter them.

## Computed metric tables

Window: B-9 through B. Scores are percentages; SD/uplift are percentage points; slope is pp/epoch.
Loss uses native units. Variance and full-precision observations are retained in the JSON report.
Endpoint uplift compares B with the preceding nine-epoch median. SD is descriptive, not a CI.
Selected is the original epoch-B score; available restored-best scores are checked separately in JSON.

| ID | Floor | Synthesis | Pretraining | Window | History |
|---|---:|---:|---:|---|---|
| [R1](https://wandb.ai/codllmdev/codllm/runs/c9fec67e2ecf4e8085fa82f84723f408) | 0 | 0.3 | 4 | 61–70 | inherited_parent |
| [R2](https://wandb.ai/codllmdev/codllm/runs/fd1a4c7714ad476f952a9ef75ddb0021) | 0 | 0.3 | 48 | 80–89 | child |
| [R3](https://wandb.ai/codllmdev/codllm/runs/515792acbfed43be993f85f82eb13f91) | 0 | 0.6 | 4 | 74–83 | child |
| [R4](https://wandb.ai/codllmdev/codllm/runs/0e4a8c648c974c1ea1ef48847c0fd6d3) | 0 | 0.6 | 48 | 48–57 | inherited_parent |
| [R5](https://wandb.ai/codllmdev/codllm/runs/3f52710979ac46d099984ccc578724af) | 450 | 0.3 | 4 | 51–60 | child |
| [R6](https://wandb.ai/codllmdev/codllm/runs/63b5465c4d8244e5a3f51c1d92f78de7) | 450 | 0.3 | 48 | 69–78 | child |
| [R7](https://wandb.ai/codllmdev/codllm/runs/732fc79a61564f0b9cc1264b675d5727) | 450 | 0.6 | 4 | 70–79 | child |
| [R8](https://wandb.ai/codllmdev/codllm/runs/3cf6ecc34f254de0b43a029d5a15b473) | 450 | 0.6 | 48 | 67–76 | child |

### macro_f1

| ID | Selected | Mean | Median | SD | Min | Max | Uplift | Slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 59.3941 | 58.7378 | 58.6995 | 0.3616 | 58.2816 | 59.3941 | 0.7516 | 0.0355 |
| R2 | 59.7734 | 59.1326 | 58.9588 | 0.3609 | 58.7341 | 59.7734 | 0.8258 | 0.1081 |
| R3 | 59.5371 | 59.1531 | 59.1396 | 0.1974 | 58.8004 | 59.5371 | 0.4054 | 0.0202 |
| R4 | 59.2092 | 57.9316 | 57.8651 | 0.5449 | 57.2028 | 59.2092 | 1.3948 | 0.1131 |
| R5 | 58.0265 | 57.5314 | 57.4831 | 0.2115 | 57.2141 | 58.0265 | 0.6009 | 0.0169 |
| R6 | 57.5179 | 57.1225 | 57.1119 | 0.2681 | 56.6281 | 57.5179 | 0.4472 | 0.0206 |
| R7 | 58.5678 | 57.9037 | 57.8389 | 0.3114 | 57.3267 | 58.5678 | 0.7604 | 0.0377 |
| R8 | 58.7967 | 58.5126 | 58.5620 | 0.2437 | 58.0070 | 58.7967 | 0.2626 | 0.0495 |

### micro_f1

| ID | Selected | Mean | Median | SD | Min | Max | Uplift | Slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 91.0057 | 89.8415 | 89.4994 | 0.7179 | 89.0355 | 91.0342 | 1.5074 | -0.0219 |
| R2 | 89.2177 | 89.1565 | 89.1712 | 0.0611 | 89.0564 | 89.2471 | 0.0535 | 0.0110 |
| R3 | 89.5525 | 89.4233 | 89.4688 | 0.1037 | 89.2287 | 89.5525 | 0.0865 | 0.0209 |
| R4 | 89.5769 | 88.9445 | 88.7951 | 0.3241 | 88.6659 | 89.5769 | 0.8079 | 0.0909 |
| R5 | 90.2555 | 89.3524 | 90.0716 | 1.9862 | 83.5064 | 90.3989 | 0.2646 | 0.3705 |
| R6 | 89.1374 | 83.2298 | 82.8972 | 2.0596 | 81.5876 | 89.1374 | 6.3252 | 0.4462 |
| R7 | 88.7966 | 88.7153 | 88.7444 | 0.1118 | 88.4588 | 88.8319 | 0.0542 | 0.0083 |
| R8 | 89.2903 | 89.1268 | 89.3227 | 0.8637 | 87.1527 | 90.4040 | -0.0648 | 0.1179 |

### block_micro_f1

| ID | Selected | Mean | Median | SD | Min | Max | Uplift | Slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 94.5571 | 94.1212 | 94.3536 | 0.5203 | 93.0677 | 94.6136 | 0.2411 | -0.0838 |
| R2 | 93.6448 | 93.5782 | 93.5895 | 0.0493 | 93.4832 | 93.6448 | 0.0649 | 0.0128 |
| R3 | 93.0074 | 92.9304 | 92.9046 | 0.0574 | 92.8648 | 93.0190 | 0.1094 | 0.0135 |
| R4 | 93.5972 | 93.5077 | 93.4888 | 0.1127 | 93.3826 | 93.7283 | 0.1245 | 0.0256 |
| R5 | 93.7110 | 93.5339 | 93.5900 | 0.1934 | 93.2191 | 93.8174 | 0.1338 | -0.0018 |
| R6 | 93.3430 | 93.5538 | 93.5429 | 0.1516 | 93.3430 | 93.7961 | -0.2013 | -0.0445 |
| R7 | 93.7118 | 93.4593 | 93.4719 | 0.1368 | 93.1660 | 93.7118 | 0.2675 | 0.0205 |
| R8 | 93.3717 | 93.3853 | 93.3434 | 0.1246 | 93.2975 | 93.7482 | 0.0315 | -0.0045 |

### exact_match

| ID | Selected | Mean | Median | SD | Min | Max | Uplift | Slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 89.4289 | 88.0787 | 87.6648 | 0.8297 | 87.1386 | 89.4368 | 1.7660 | -0.0247 |
| R2 | 87.5741 | 87.5240 | 87.5371 | 0.0619 | 87.4153 | 87.6244 | 0.0397 | 0.0090 |
| R3 | 87.7105 | 87.5692 | 87.6278 | 0.1171 | 87.3570 | 87.7105 | 0.0861 | 0.0231 |
| R4 | 88.0256 | 87.2767 | 87.1419 | 0.3655 | 86.9678 | 88.0256 | 0.9373 | 0.1004 |
| R5 | 88.6147 | 87.4417 | 88.1732 | 2.3585 | 80.4755 | 88.6941 | 0.5494 | 0.4315 |
| R6 | 87.4987 | 80.3494 | 79.8963 | 2.4905 | 78.2739 | 87.4987 | 7.7605 | 0.5265 |
| R7 | 86.8963 | 86.8018 | 86.8533 | 0.1402 | 86.5309 | 86.9519 | 0.0543 | 0.0139 |
| R8 | 87.6642 | 87.4699 | 87.7224 | 0.9808 | 85.1554 | 88.7206 | -0.1165 | 0.1458 |

### accuracy

| ID | Selected | Mean | Median | SD | Min | Max | Uplift | Slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 89.4289 | 88.0787 | 87.6648 | 0.8297 | 87.1386 | 89.4368 | 1.7660 | -0.0247 |
| R2 | 87.5741 | 87.5240 | 87.5371 | 0.0619 | 87.4153 | 87.6244 | 0.0397 | 0.0090 |
| R3 | 87.7105 | 87.5692 | 87.6278 | 0.1171 | 87.3570 | 87.7105 | 0.0861 | 0.0231 |
| R4 | 88.0256 | 87.2767 | 87.1419 | 0.3655 | 86.9678 | 88.0256 | 0.9373 | 0.1004 |
| R5 | 88.6147 | 87.4417 | 88.1732 | 2.3585 | 80.4755 | 88.6941 | 0.5494 | 0.4315 |
| R6 | 87.4987 | 80.3494 | 79.8963 | 2.4905 | 78.2739 | 87.4987 | 7.7605 | 0.5265 |
| R7 | 86.8963 | 86.8018 | 86.8533 | 0.1402 | 86.5309 | 86.9519 | 0.0543 | 0.0139 |
| R8 | 87.6642 | 87.4699 | 87.7224 | 0.9808 | 85.1554 | 88.7206 | -0.1165 | 0.1458 |

### sample_f1

| ID | Selected | Mean | Median | SD | Min | Max | Uplift | Slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 90.5893 | 89.2190 | 88.8145 | 0.8393 | 88.2778 | 90.6112 | 1.7850 | -0.0249 |
| R2 | 88.7136 | 88.6481 | 88.6652 | 0.0676 | 88.5314 | 88.7536 | 0.0536 | 0.0116 |
| R3 | 88.8513 | 88.7207 | 88.7751 | 0.1117 | 88.5322 | 88.8513 | 0.0907 | 0.0226 |
| R4 | 89.1721 | 88.4260 | 88.2851 | 0.3586 | 88.1152 | 89.1721 | 0.9282 | 0.0991 |
| R5 | 89.7939 | 88.7337 | 89.5611 | 2.3156 | 81.9005 | 89.9401 | 0.3307 | 0.4380 |
| R6 | 88.7067 | 81.5882 | 81.1971 | 2.4760 | 79.6618 | 88.7067 | 7.5968 | 0.5368 |
| R7 | 88.0862 | 87.9751 | 88.0196 | 0.1397 | 87.7005 | 88.1132 | 0.0852 | 0.0124 |
| R8 | 88.7874 | 88.6075 | 88.8542 | 0.9765 | 86.2972 | 89.8525 | -0.1337 | 0.1443 |

### loss

| ID | Selected | Mean | Median | SD | Min | Max | Uplift | Slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 0.2854 | 0.2855 | 0.2849 | 0.0047 | 0.2767 | 0.2944 | -0.0010 | 0.0009 |
| R2 | 0.3265 | 0.3199 | 0.3191 | 0.0052 | 0.3086 | 0.3270 | -0.0082 | 0.0013 |
| R3 | 0.3072 | 0.3011 | 0.3022 | 0.0065 | 0.2918 | 0.3094 | -0.0091 | 0.0014 |
| R4 | 0.2676 | 0.2679 | 0.2671 | 0.0047 | 0.2617 | 0.2762 | -0.0010 | 0.0005 |
| R5 | 0.3103 | 0.3012 | 0.3036 | 0.0078 | 0.2877 | 0.3136 | -0.0069 | 0.0001 |
| R6 | 0.3669 | 0.3936 | 0.3917 | 0.0137 | 0.3669 | 0.4174 | 0.0255 | -0.0039 |
| R7 | 0.3356 | 0.3233 | 0.3224 | 0.0068 | 0.3107 | 0.3356 | -0.0135 | 0.0014 |
| R8 | 0.2950 | 0.2893 | 0.2916 | 0.0098 | 0.2744 | 0.3021 | -0.0068 | 0.0026 |

### pub_v1_source_mean_macro_f1_ref

| ID | Selected | Mean | Median | SD | Min | Max | Uplift | Slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 70.2627 | 70.1533 | 70.1872 | 0.1737 | 69.7708 | 70.3680 | 0.1129 | 0.0002 |
| R2 | 70.1027 | 69.6027 | 69.6019 | 0.2294 | 69.2107 | 70.1027 | 0.5198 | 0.0230 |
| R3 | 69.7404 | 69.8144 | 69.8108 | 0.2801 | 69.4058 | 70.2780 | -0.1407 | 0.0443 |
| R4 | 71.1639 | 69.9081 | 69.6866 | 0.5391 | 69.4091 | 71.1639 | 1.4799 | 0.0916 |
| R5 | 69.5257 | 69.7715 | 69.7050 | 0.2691 | 69.3877 | 70.1645 | -0.2029 | -0.0120 |
| R6 | 68.6982 | 68.4767 | 68.5668 | 0.2422 | 67.9210 | 68.7558 | 0.1528 | 0.0327 |
| R7 | 70.4681 | 69.9568 | 69.9733 | 0.2733 | 69.5126 | 70.4681 | 0.5533 | 0.0236 |
| R8 | 70.6692 | 70.5245 | 70.5472 | 0.2786 | 70.1276 | 71.1140 | 0.2093 | 0.0297 |

## Matched contrasts

Differences are second recipe minus first, in percentage points. Each recipe uses its own selected window;
these are descriptive contrasts, not causal effects or independent-epoch significance tests.

| Change (fixed factors) | Recipes | Macro selected | Macro mean | Macro median | Source selected | Source mean | Source median |
|---|---|---:|---:|---:|---:|---:|---:|
| Floor 0 → 450 (.30, PT4) | R1 → R5 | -1.3676 | -1.2063 | -1.2164 | -0.7370 | -0.3818 | -0.4822 |
| Floor 0 → 450 (.30, PT48) | R2 → R6 | -2.2555 | -2.0101 | -1.8469 | -1.4045 | -1.1261 | -1.0350 |
| Floor 0 → 450 (.60, PT4) | R3 → R7 | -0.9693 | -1.2493 | -1.3007 | 0.7276 | 0.1424 | 0.1626 |
| Floor 0 → 450 (.60, PT48) | R4 → R8 | -0.4126 | 0.5810 | 0.6969 | -0.4947 | 0.6164 | 0.8606 |
| Synthesis .30 → .60 (floor 0, PT4) | R1 → R3 | 0.1430 | 0.4153 | 0.4401 | -0.5223 | -0.3389 | -0.3765 |
| Synthesis .30 → .60 (floor 0, PT48) | R2 → R4 | -0.5641 | -1.2010 | -1.0937 | 1.0612 | 0.3054 | 0.0848 |
| Synthesis .30 → .60 (floor 450, PT4) | R5 → R7 | 0.5413 | 0.3723 | 0.3558 | 0.9424 | 0.1853 | 0.2683 |
| Synthesis .30 → .60 (floor 450, PT48) | R6 → R8 | 1.2788 | 1.3901 | 1.4501 | 1.9711 | 2.0479 | 1.9804 |
| Pretraining 4 → 48 (floor 0, .30) | R1 → R2 | 0.3792 | 0.3949 | 0.2593 | -0.1600 | -0.5506 | -0.5854 |
| Pretraining 4 → 48 (floor 0, .60) | R3 → R4 | -0.3279 | -1.2215 | -1.2745 | 1.4235 | 0.0937 | -0.1242 |
| Pretraining 4 → 48 (floor 450, .30) | R5 → R6 | -0.5086 | -0.4089 | -0.3712 | -0.8275 | -1.2948 | -1.1382 |
| Pretraining 4 → 48 (floor 450, .60) | R7 → R8 | 0.2288 | 0.6089 | 0.7230 | 0.2012 | 0.5677 | 0.5739 |
