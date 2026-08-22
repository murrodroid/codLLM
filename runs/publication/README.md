# Publication experiments

The scientific design is in [`publication_plan.md`](../../publication_plan.md).
The submission sequence is in [`publication_runs.md`](../../publication_runs.md).

## Active specifications

| Stage | Specification | Runs |
|---|---|---:|
| Response curves | `balance_floor_curve.toml` | 7 |
| Response curves | `multicod_synthetic_curve.toml` | 7 |
| Response curves | `pretraining_dose_curve.toml` | 6 |
| Interaction confirmation | `interaction_confirmation.toml` | 8 |
| Reduced-data calibration | `reduced_data_calibration.toml` | 4 |
| Split/preparation sensitivity | `reduced_split_sensitivity.toml` | 8 |
| Training-pipeline sensitivity | `reduced_training_seed_sensitivity.toml` | 4 |
| Scale confirmation | `model_scale_confirmation.toml` | 3 |
| Source transfer | `source_transfer_validation.toml` | 5 |
| Final evaluation | `final_model.toml` | 1 |

`candidate_winner.toml` and `candidate_runner_up.toml` are inherited recipe files,
not independent submission steps. Their checked-in settings are provisional.
Freeze them after the curves and interaction confirmation.

Reduced-data specifications pin `CODLLM_DATASET_SAMPLE_SEED=777` so changing
`CODLLM_DATA_SEED` does not change the sampled cohort. The seed-777 25% calibration
cells provide the fifth observation for both seed studies.

The legacy Bayesian files remain available for optional hypothesis generation but
are excluded from the active sequence because their shortened-fidelity objective is
not a reliable replacement for full-fidelity selection.
