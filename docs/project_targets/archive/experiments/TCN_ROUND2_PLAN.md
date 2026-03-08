# TCN Round-2 Plan (CAUCAFall Mainline)

Date: 2026-03-04

## Goal

Improve deployment robustness for CAUCAFall TCN under real-world noise (occlusion/pose jitter) while keeping event-level quality and low false alerts.

## Why Round-2

Round-1 showed TCN is already strong on current test split, but frontend replay/live still exhibits domain-shift instability in some scenes.  
Round-2 focuses on train-side robustness with strict experiment recording.

## Protocol (leakage-safe)

1. Hard negatives must be mined from `train` windows only.
2. `fit_ops` uses `val` only.
3. Final `eval_metrics` uses `test` only.
4. Every run is appended to:
   - `artifacts/registry/overfit_experiment_registry.csv`

## Experiments

1. `exp_tcn_caucafall_r2_train_hneg_seed33724876`
   - Build hard negatives from `data/processed/caucafall/windows_W48_S12/train`
   - Retrain with:
     - `resume=outputs/caucafall_tcn_W48S12_r1_augreg/best.pt`
     - `hard_neg_mult=2`
   - Keep Round-1 AugReg defaults (dropout/mask/wd/smoothing).

2. `exp_tcn_caucafall_r2_train_hneg_plus_seed33724876`
   - Same as above + slightly stronger regularization:
     - `dropout=0.45`
     - `mask_joint_p=0.16`
     - `mask_frame_p=0.10`
     - `weight_decay=0.0015`
     - `label_smoothing=0.04`

## Acceptance

- Required artifacts per experiment:
  - `best.pt`
  - `configs/ops/tcn_caucafall_<tag>.yaml`
  - `outputs/metrics/tcn_caucafall_<tag>.json`
- Deployment-facing pass criterion:
  - keep `Recall >= baseline recall`
  - no increase in `FA/24h`
  - improved stability evidence in replay/live follow-up.

## Execution outcomes (2026-03-04)

- `exp_tcn_caucafall_r2_train_hneg_seed33724876`
  - AP `0.9680`, `TP=4`, `FP=0`, Recall `0.8`, Precision `1.0`, F1 `0.8889`, FA/24h `0.0`
- `exp_tcn_caucafall_r2_train_hneg_plus_seed33724876`
  - AP `0.9693`, `TP=4`, `FP=0`, Recall `0.8`, Precision `1.0`, F1 `0.8889`, FA/24h `0.0`

Baseline for comparison (`r1_augreg`):
- AP `0.9691`, `TP=5`, `FP=0`, Recall `1.0`, Precision `1.0`, F1 `1.0`, FA/24h `0.0`

Decision:
- Both Round-2 candidates are **rejected for promotion** (recall regression).
- Keep `r1_augreg` as the main TCN deployment checkpoint/policy.
