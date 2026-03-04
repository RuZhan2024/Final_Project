# Overfit Mitigation Round-1 (CAUCAFall Priority)

Objective: reduce early overfitting and improve deployment-facing metrics stability without changing model families.

## Scope

- Dataset priority: `caucafall` (main target)
- Architectures: `TCN`, `GCN`
- Strategy: A/B with control vs stronger regularization/mask augmentation

## Experiments

1. `exp_tcn_caucafall_r1_ctrl_seed33724876`
2. `exp_tcn_caucafall_r1_augreg_seed33724876`
3. `exp_gcn_caucafall_r1_ctrl_seed33724876`
4. `exp_gcn_caucafall_r1_augreg_seed33724876`

## Parameter deltas

- Control:
  - keep current defaults (`dropout=0.30`, `mask_joint_p=0.05`, `mask_frame_p=0.05`)
- AugReg:
  - `dropout=0.40`
  - `mask_joint_p=0.12`
  - `mask_frame_p=0.08`
  - `label_smoothing=0.03`
  - `weight_decay=1e-3`

## Run commands

- Main script:
  - `artifacts/reports/tuning/overfit_round1_commands.sh`
- Run step-by-step (recommended), not full batch in one shot.

## Recording requirements

- Must use:
  - `tools/track_experiment.py`
  - `artifacts/registry/overfit_experiment_registry.csv`
- Record `planned -> running -> done/failed` for each `EXP_ID`.

## Acceptance criteria

- For each experiment:
  - `best.pt` exists
  - `configs/ops/*.yaml` exists
  - `outputs/metrics/*.json` exists
  - registry has final status row with metrics/artifacts paths
- Round-1 success condition:
  - at least one AugReg config improves event-level behavior over its control
  - no unacceptable regression in FA/24h for selected OP profile

## Round-1 outcomes (executed 2026-03-04)

Source summary:
- `artifacts/reports/tuning/overfit_round1_summary.csv`
- `artifacts/reports/tuning/overfit_round1_summary.md`

Observed:
- TCN control:
  - AP `0.9676`, Event F1 `1.0000`, Recall `1.0000`, Precision `1.0000`, FA/24h `0.0000`
- TCN AugReg:
  - AP `0.9691`, Event F1 `1.0000`, Recall `1.0000`, Precision `1.0000`, FA/24h `0.0000`
- GCN control:
  - AP `0.9595`, Event F1 `0.6667`, Recall `1.0000`, Precision `0.5000`, FA/24h `4704.5455`
- GCN AugReg:
  - AP `0.9702`, Event F1 `0.6667`, Recall `1.0000`, Precision `0.5000`, FA/24h `4704.5455`

Conclusion:
- `TCN`: AugReg produced a small AP uplift without event-level regression.
- `GCN`: AugReg improved AP but did not fix deployment-critical false-alert behavior (FA/24h unchanged and very high).

Decision for next round:
- Keep `TCN` as main deploy path for CAUCAFall.
- For `GCN`, prioritize policy/routing constraints (or stronger train-side changes) before enabling autonomous alerts.
