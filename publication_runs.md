# Publication runs

## Setup

```shell
cd ~/codLLM
source hpc/env.sh
```

## 1. Balance floor curve

```shell
uv run --no-sync invoke hpc.build --config runs/publication/balance_floor_curve.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/balance_floor_curve.toml --profile h100 --lucas --duration 1w
```

## 2. Multi-COD synthetic curve

```shell
uv run --no-sync invoke hpc.build --config runs/publication/multicod_synthetic_curve.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/multicod_synthetic_curve.toml --profile h100 --lucas --duration 1w
```

## 3. Pretraining dose curve

```shell
uv run --no-sync invoke hpc.build --config runs/publication/pretraining_dose_curve.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/pretraining_dose_curve.toml --profile h100 --lucas --duration 1w
```

Before step 4, update the low/high levels in `runs/publication/interaction_confirmation.toml` from steps 1–3.

## 4. Interaction confirmation

```shell
uv run --no-sync invoke hpc.build --config runs/publication/interaction_confirmation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/interaction_confirmation.toml --profile h100 --lucas --duration 1w
```

Before step 5, freeze `candidate_winner.toml` and `candidate_runner_up.toml` from step 4.

## 5. Reduced-data calibration

```shell
uv run --no-sync invoke hpc.build --config runs/publication/reduced_data_calibration.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/reduced_data_calibration.toml --profile h100 --lucas --duration 1w
```

## 6. Reduced-data split sensitivity

```shell
uv run --no-sync invoke hpc.build --config runs/publication/reduced_split_sensitivity.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/reduced_split_sensitivity.toml --profile h100 --lucas --duration 1w
```

## 7. Reduced-data training-seed sensitivity

```shell
uv run --no-sync invoke hpc.build --config runs/publication/reduced_training_seed_sensitivity.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/reduced_training_seed_sensitivity.toml --profile h100 --lucas --duration 1w
```

## 8. Model-scale confirmation

```shell
uv run --no-sync invoke hpc.build --config runs/publication/model_scale_confirmation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/model_scale_confirmation.toml --profile h100 --lucas --duration 1w
```

## 9. Source-transfer validation

```shell
uv run --no-sync invoke hpc.build --config runs/publication/source_transfer_validation.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/source_transfer_validation.toml --profile h100 --lucas --duration 1w
```

Before step 10, set the selected model profile in `runs/publication/final_model.toml`.

## 10. Final model

```shell
uv run --no-sync invoke hpc.build --config runs/publication/final_model.toml --profile h100 --lucas
uv run --no-sync invoke hpc.submit --config runs/publication/final_model.toml --profile h100 --lucas --duration 1w
```
