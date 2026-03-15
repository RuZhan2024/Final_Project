# Experiment Evidence Index

This page is the shortest navigation guide for finding experimental evidence when writing the report or thesis.

For direct config-to-result-to-document tracing, also use:

- `docs/reports/runbooks/CONFIG_RESULT_EVIDENCE_MAP.md`

## Runtime / Delivery Evidence

Use these when describing what the deployed system currently does:

- `configs/ops/tcn_caucafall.yaml`
- `configs/ops/gcn_caucafall.yaml`
- `configs/ops/tcn_le2i.yaml`
- `configs/ops/gcn_le2i.yaml`
- `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`
- `docs/reports/runbooks/ONLINE_FRONTEND_SMOKE_CHECKLIST.md`
- `docs/reports/runbooks/DELIVERY_RELEASE_BOUNDARY.md`

## 24-Video Custom Delivery Evidence

Use these for the corridor/kitchen four-folder evaluation story:

- `configs/delivery/tcn_caucafall_r2_train_hneg_four_video.yaml`
- `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md`
- `artifacts/fall_test_eval_20260315/`
- `artifacts/fall_test_eval_20260315_online_reverify_20260315/`

## Online Ops Fitting and Recovery Evidence

Use these when explaining how online behavior was tuned and repaired:

- `artifacts/online_ops_fit_20260315/`
- `artifacts/online_ops_fit_20260315_verify/`
- `artifacts/online_ops_fit_20260315_verify_le2i_bypass/`
- `artifacts/ops_reverify_20260315/`
- `artifacts/ops_reverify_20260315_after_gatefix/`
- `artifacts/ops_reverify_20260315_after_motionfix/`
- `artifacts/reports/tuning/`

## Cross-Dataset / Comparative Evidence

Use these for broader model comparison or transfer claims:

- `artifacts/figures/cross_dataset/`
- `artifacts/reports/cross_dataset_*`
- `docs/project_targets/CLAIM_TABLE.md`
- `docs/project_targets/THESIS_EVIDENCE_MAP.md`

## Stability / Reliability Evidence

Use these when discussing repeatability, stability, or operational robustness:

- `artifacts/figures/stability/`
- `artifacts/reports/stability_*`
- `artifacts/reports/op123_*`
- `docs/project_targets/LATENCY_REPORT.md`
- `docs/reports/readiness/READINESS_REPORT.md`

## Model-Tuning Evidence

Use these when describing iterative optimization:

- `configs/ops/README.md`
- `configs/ops/archive/`
- `configs/ops/grid_midplateau/`
- `configs/ops/grid_midplateau_temporal/`
- `configs/ops/grid_startguard_midplateau/`
- `artifacts/reports/tuning/`
- `docs/project_targets/archive/experiments/`

## Non-Core Archived Materials

These are preserved, but they are not part of the project/report/thesis core argument:

- `docs/archive/tutorial_materials/`

## Suggested Writing Workflow

1. Start with `docs/project_targets/CLAIM_TABLE.md`.
2. Map each claim to a concrete artifact or config.
3. Use this index to jump to the matching evidence directory.
4. Only then pull in detailed tables, figures, or sweep logs.
