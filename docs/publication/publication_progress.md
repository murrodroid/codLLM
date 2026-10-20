# Publication progress

Updated: 2026-10-20. **Phase 1 complete**, confirmed by Lucas.

| Work | Status |
|---|---|
| Exploratory curves | 20/20 complete: 7 floor, 7 synthesis, 6 pretraining-dose runs. |
| Phase 0 | Data/split audits and HPC smoke completed. |
| Phase 1: COD-only, patience 10 | 8/8 complete; last finished 17 September. |
| Phase 1: COD-only, patience 20 | 8/8 complete; last finished 28 September. |
| Phase 1: COD+age+sex, patience 10 | 8/8 complete; last finished 12 October. |
| Phase 1: COD+age+sex, patience 20 | 8/8 complete; last finished 17 October, 19:53 CEST. |
| Checkpoint consistency | All eight continuation windows recovered for each input setting. |
| Phase 2a: Copenhagen/Belgium pilots | Submitted 20 October as LSF `29760677[1-4]`, patience 20 / max 120. Three running, one queued at the post-submission check. No results recorded. |
| Phase 2b: remaining source folds | Six runs configured with the same recipe; not submitted. |
| Phase 3 onward | No results recorded. |

Next: complete and review the four pilot runs before requesting the remaining six submissions.

Pilot array mapping: [`[1]` selected Copenhagen](https://wandb.ai/codllmdev/codllm/runs/oen5w7vh),
[`[2]` plain Copenhagen](https://wandb.ai/codllmdev/codllm/runs/ymfxna72), `[3]` selected Belgium (queued),
[`[4]` plain Belgium](https://wandb.ai/codllmdev/codllm/runs/xq7n4zto).
Elements `[2]` and `[3]` initially exited before training during concurrent
shared-environment setup; only these two were requeued under the same IDs. Recipe settings were unchanged.

## Phase 1 results

Grouped-COD validation, seed 777; test evaluation disabled. Each row reports one macro-F1-selected
checkpoint. Continuations resume their parent's best checkpoint without repeating pretraining.
All 16 continuations stopped at best epoch + 20, before the unchanged 120-epoch ceiling.

Equal-source reference macro F1 is `pub_v1_source_mean_macro_f1_ref`.
Checkpoint selection uses pooled `macro_f1`. These are validation results, not source-held-out results.

### COD-only, patience 20

Verified against HPC and W&B on 5 October.

| Floor | Synthesis | Pretrain epochs | Best epoch | Stopped at epoch | Macro F1 | Micro F1 | Equal-source reference macro F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| [0](https://wandb.ai/codllmdev/codllm/runs/c9fec67e2ecf4e8085fa82f84723f408) | .30 | 4 | 70 | 90 | 0.593941 | 0.910057 | 0.702627 |
| [0](https://wandb.ai/codllmdev/codllm/runs/fd1a4c7714ad476f952a9ef75ddb0021) | .30 | 48 | 89 | 109 | 0.597734 | 0.892177 | 0.701027 |
| [0](https://wandb.ai/codllmdev/codllm/runs/515792acbfed43be993f85f82eb13f91) | .60 | 4 | 83 | 103 | 0.595371 | 0.895525 | 0.697404 |
| [0](https://wandb.ai/codllmdev/codllm/runs/0e4a8c648c974c1ea1ef48847c0fd6d3) | .60 | 48 | 57 | 77 | 0.592092 | 0.895769 | 0.711639 |
| [450](https://wandb.ai/codllmdev/codllm/runs/3f52710979ac46d099984ccc578724af) | .30 | 4 | 60 | 80 | 0.580265 | 0.902555 | 0.695257 |
| [450](https://wandb.ai/codllmdev/codllm/runs/63b5465c4d8244e5a3f51c1d92f78de7) | .30 | 48 | 78 | 98 | 0.575179 | 0.891374 | 0.686982 |
| [450](https://wandb.ai/codllmdev/codllm/runs/732fc79a61564f0b9cc1264b675d5727) | .60 | 4 | 79 | 99 | 0.585678 | 0.887966 | 0.704681 |
| [450](https://wandb.ai/codllmdev/codllm/runs/3cf6ecc34f254de0b43a029d5a15b473) | .60 | 48 | 76 | 96 | 0.587967 | 0.892903 | 0.706692 |

### COD+age+sex, patience 20

Verified on HPC on 19 October: completion markers, saved weights, and prediction exports exist;
no resume requests remain. W&B status was not independently queried.

| Floor | Synthesis | Pretrain epochs | Best epoch | Stopped at epoch | Macro F1 | Micro F1 | Equal-source reference macro F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| [0](https://wandb.ai/codllmdev/codllm/runs/71f254d6dc3b41c48edde22997c3e0a6) | .30 | 4 | 66 | 86 | 0.598531 | 0.895114 | 0.702160 |
| [0](https://wandb.ai/codllmdev/codllm/runs/057bc431939b4fc3b00d198ea28248ef) | .30 | 48 | 57 | 77 | 0.597307 | 0.903982 | 0.710900 |
| [0](https://wandb.ai/codllmdev/codllm/runs/93e827e228964114bc68e397f87c481d) | .60 | 4 | 70 | 90 | 0.589545 | 0.895687 | 0.705274 |
| [0](https://wandb.ai/codllmdev/codllm/runs/fb162d347fa54aeeb8a97870bc28b0c5) | .60 | 48 | 61 | 81 | 0.591692 | 0.868511 | 0.716040 |
| [450](https://wandb.ai/codllmdev/codllm/runs/1cf920de8bd046b491880b6d287fef69) | .30 | 4 | 41 | 61 | 0.570964 | 0.884850 | 0.690118 |
| [450](https://wandb.ai/codllmdev/codllm/runs/37b977ac5a7d4ddcbf3e21ae33540b3b) | .30 | 48 | 52 | 72 | 0.574810 | 0.883142 | 0.702997 |
| [450](https://wandb.ai/codllmdev/codllm/runs/309672e55ac54afd8cf60ab4b40b918a) | .60 | 4 | 54 | 74 | 0.576963 | 0.885845 | 0.706766 |
| [450](https://wandb.ai/codllmdev/codllm/runs/fe3073f8875848739f8b473cdb605214) | .60 | 48 | 79 | 99 | 0.584081 | 0.884893 | 0.712204 |

[Earlier results, completion times, and consistency summaries](reference/results.md).
The [decision log](publication_decisions.md) records selection separately from run completion.
