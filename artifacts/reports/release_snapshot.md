# Release Snapshot

- Timestamp (UTC): 2026-03-04T23:12:00Z
- Branch: `exp_overfit_augmentation_ablation`
- Commit: `677fe8d`
- Working tree: `dirty` (contains experiment/report artifacts)

## Locked Deployment Profile

- Dataset: `caucafall`
- Model: `TCN`
- Operating Point: `OP-2`
- Thresholds: `tau_high=0.7099999785`, `tau_low=0.5537999868`
- Canonical ops: `configs/ops/tcn_caucafall.yaml`
- Canonical checkpoint: `outputs/caucafall_tcn_W48S12_r1_augreg/best.pt`

## Validation Status

- Deployment lock validation report: `artifacts/reports/deployment_lock_validation.md`
- Auto checks: PASS (`health=yes`, `predict endpoint=yes`, `active model=TCN`, `dataset=caucafall`, `op=OP-2`)
- Manual replay checks: PASS (non-fall no alert, fall clip triggers event)

## Experiment Evidence Included

- Round-1 overfit comparison:
  - `artifacts/reports/tuning/overfit_round1_summary.md`
- GCN policy Round-2:
  - `docs/project_targets/GCN_POLICY_ROUND2_RESULTS.md`
- TCN train-side Round-2:
  - `artifacts/reports/tuning/tcn_round2_results.md`
- Registry log:
  - `artifacts/registry/overfit_experiment_registry.csv`

## Promotion Decision

- Promoted for demo/release:
  - `TCN + CAUCAFall + OP-2` (locked profile)
- Not promoted:
  - GCN autonomous alert path (precision/FA tradeoff not acceptable)
  - TCN Round-2 hard-negative variants (event recall regression vs baseline)
