# Config Result Evidence Map

This map links configuration files, generated result artifacts, and narrative documents.

Use it when writing the report or thesis so each claim can be traced across:

1. the config used
2. the result files produced
3. the document that interprets the result

## A. Current Online Runtime Profiles

### `caucafall + TCN`

Config:

- `configs/ops/tcn_caucafall.yaml`

Result artifacts:

- `artifacts/online_ops_fit_20260315/caucafall_tcn_refit.json`
- `artifacts/ops_reverify_20260315/caucafall_tcn.json`
- `artifacts/ops_reverify_20260315/offline_summary.json`

Interpretation docs:

- `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`
- `docs/project_targets/DEPLOYMENT_DEFAULT_PROFILE.md`
- `docs/project_targets/OPS_POLICY_REPORT.md`

### `caucafall + GCN`

Config:

- `configs/ops/gcn_caucafall.yaml`

Result artifacts:

- `artifacts/online_ops_fit_20260315/caucafall_gcn.json`
- `artifacts/online_ops_fit_20260315_verify/caucafall_gcn.json`
- `artifacts/online_ops_fit_20260315_verify/caucafall_gcn_after_motion016.json`
- `artifacts/online_ops_fit_20260315_verify/caucafall_gcn_final.json`
- `artifacts/ops_reverify_20260315/caucafall_gcn.json`

Interpretation docs:

- `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`
- `artifacts/reports/tuning/caucafall_gcn_hneg_ablation_table.md`
- `artifacts/reports/tuning/gcn_round1_recovery_summary.md`
- `artifacts/reports/tuning/gcn_round2_summary.md`

### `le2i + TCN`

Config:

- `configs/ops/tcn_le2i.yaml`

Result artifacts:

- `artifacts/online_ops_fit_20260315/le2i_tcn.json`
- `artifacts/online_ops_fit_20260315_verify/le2i_tcn.json`
- `artifacts/online_ops_fit_20260315_verify_le2i_bypass/tcn.json`
- `artifacts/ops_reverify_20260315/le2i_tcn.json`

Interpretation docs:

- `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`
- `artifacts/reports/tuning/le2i_opt33_r1_summary.md`
- `artifacts/reports/tuning/le2i_opt33_r2_summary.md`
- `artifacts/reports/tuning/le2i_tcn_op2_quickgrid_summary.md`

### `le2i + GCN`

Config:

- `configs/ops/gcn_le2i.yaml`

Result artifacts:

- `artifacts/online_ops_fit_20260315/le2i_gcn.json`
- `artifacts/online_ops_fit_20260315_verify/le2i_gcn.json`
- `artifacts/online_ops_fit_20260315_verify_le2i_bypass/gcn.json`
- `artifacts/ops_reverify_20260315/le2i_gcn.json`

Interpretation docs:

- `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`
- `artifacts/reports/tuning/le2i_opt33_r4_summary.md`
- `artifacts/reports/tuning/le2i_opt33_r6_summary.md`
- `artifacts/reports/tuning/le2i_opt33_r8_summary.md`

## B. 24-Video Custom Delivery Track

Primary config:

- `configs/delivery/tcn_caucafall_r2_train_hneg_four_video.yaml`

Supporting runtime config:

- `configs/ops/tcn_caucafall.yaml`

Result artifacts:

- `artifacts/fall_test_eval_20260315/delivery_tcn_r2_train_hneg_op2.csv`
- `artifacts/fall_test_eval_20260315/delivery_tcn_r2_train_hneg_op2.json`
- `artifacts/fall_test_eval_20260315/delivery_tcn_r2_train_hneg_op2_metrics.json`
- `artifacts/fall_test_eval_20260315_online_reverify_20260315/tcn_op2_after_targeted_fix.json`
- `artifacts/fall_test_eval_20260315_online_reverify_20260315/tcn_op2_pose_raw_frontend_emulation_final_k2_v2.json`

Interpretation docs:

- `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md`
- `docs/project_targets/FINAL_DEMO_WALKTHROUGH.md`
- `docs/project_targets/DEPLOYMENT_LOCK.md`

## C. Online Replay Repair Sequence

This is the chain used to explain the debugging and online alignment story.

Baseline replay:

- `artifacts/ops_reverify_20260315/online_replay_summary.json`

After direct-window gate fix:

- `artifacts/ops_reverify_20260315_after_gatefix/online_replay_summary.json`

After motion-support fix:

- `artifacts/ops_reverify_20260315_after_motionfix/online_replay_summary.json`

After online OP refit:

- `artifacts/online_ops_fit_20260315/`
- `artifacts/online_ops_fit_20260315_verify/summary.json`
- `artifacts/online_ops_fit_20260315_verify/final_summary.json`
- `artifacts/online_ops_fit_20260315_verify/final_summary_v2.json`
- `artifacts/online_ops_fit_20260315_verify/final_summary_v3.json`

Interpretation docs:

- `artifacts/reports/tuning/caucafall_targeted_fix_comparison.md`
- `artifacts/reports/tuning/caucafall_targeted_retrain_results.md`
- `docs/project_targets/REPLAY_LIVE_ACCEPTANCE_LOCK.md`

## D. GCN Recovery / Overtake Track

Config families:

- `configs/ops/gcn_caucafall_r1_augreg.yaml`
- `configs/ops/gcn_caucafall_r1_ctrl.yaml`
- `configs/ops/gcn_caucafall_r1_recovery.yaml`
- `configs/ops/gcn_caucafall_r2_recallpush_a.yaml`
- `configs/ops/gcn_caucafall_r2_recallpush_b.yaml`

Result artifacts:

- `artifacts/reports/gcn_overtake/`
- `artifacts/reports/gcn_aug/`

Interpretation docs:

- `artifacts/reports/gcn_overtake/RESULTS.md`
- `docs/project_targets/archive/experiments/GCN_ROUND1_RECOVERY_PLAN.md`
- `docs/project_targets/FINAL_CANDIDATES.md`

## E. Stability / Significance / Reliability Track

Configs:

- active runtime configs under `configs/ops/`

Result artifacts:

- `artifacts/figures/stability/`
- `artifacts/reports/stability_summary.csv`
- `artifacts/reports/stability_summary.json`
- `artifacts/reports/op123_stability_summary.csv`
- `artifacts/reports/op123_stability_summary.json`
- `artifacts/reports/significance_summary.json`

Interpretation docs:

- `docs/project_targets/STABILITY_REPORT.md`
- `docs/project_targets/SIGNIFICANCE_REPORT.md`
- `docs/project_targets/ROBUSTNESS_REPORT.md`

## F. Cross-Dataset / Transfer Track

Configs:

- related historical `configs/ops/cross_*`

Result artifacts:

- `artifacts/figures/cross_dataset/`
- `artifacts/reports/cross_dataset_error_taxonomy.md`
- `artifacts/reports/cross_dataset_manifest.json`
- `artifacts/reports/cross_dataset_summary.csv`

Interpretation docs:

- `docs/project_targets/CROSS_DATASET_REPORT.md`
- `docs/project_targets/CLAIM_TABLE.md`
- `docs/project_targets/THESIS_EVIDENCE_MAP.md`

## G. MUVIM Track

Configs:

- `configs/ops/tcn_muvim*.yaml`
- `configs/ops/gcn_muvim*.yaml`

Result artifacts:

- `artifacts/reports/tuning/muvim_r1_plan.md`
- `artifacts/reports/tuning/muvim_r2_summary.md`
- `artifacts/reports/tuning/muvim_r3_summary.md`
- `artifacts/reports/tuning/muvim_r3b_summary.md`
- `artifacts/reports/tuning/muvim_metric_contract_fix.md`

Interpretation docs:

- `docs/project_targets/LOCKED_PARAMS_RUNBOOK.md`
- `docs/project_targets/PLOT_SELECTION_FOR_REPORT.md`

## Suggested Use

When making a claim in the report or thesis:

1. identify the config in this map
2. cite the artifact bundle or result file
3. cite the narrative document that interprets it

That keeps the evidence chain explicit and auditable.
