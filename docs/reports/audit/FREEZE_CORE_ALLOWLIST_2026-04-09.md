# Freeze Core Allowlist

Date: 2026-04-09  
Scope: recommended allowlist for a markable frozen snapshot

## Rule

Only include files or directories from this allowlist in the final frozen snapshot unless there is a clear, documented reason to expand scope.

This allowlist is intentionally narrower than the full branch state.

## 1. Code and Runtime Surface

Include:

- `README.md`
- `docker-compose.yml`
- `src/fall_detection/`
- `server/`
- `apps/`
- `scripts/`
- `tests/`

## 2. Required Config Surface

Include:

- `configs/ops/README.md`
- active runtime ops:
  - `configs/ops/tcn_caucafall.yaml`
  - `configs/ops/gcn_caucafall.yaml`
  - `configs/ops/tcn_le2i.yaml`
  - `configs/ops/gcn_le2i.yaml`
- frozen evidence ops:
  - `configs/ops/tcn_caucafall_r2_train_hneg.yaml`
  - `configs/ops/gcn_caucafall_r2_recallpush_b.yaml`
  - `configs/ops/tcn_le2i_opt33_r2.yaml`
  - `configs/ops/gcn_le2i_opt33_r2.yaml`

Do not include by default:

- `configs/ops/archive/`
- `configs/ops/diagnostic/`
- historical tuning families not directly cited by current report/evidence docs

## 3. Active Project-Target Docs

Include:

- `docs/project_targets/README.md`
- `docs/project_targets/CLAIM_TABLE.md`
- `docs/project_targets/THESIS_EVIDENCE_MAP.md`
- `docs/project_targets/FINAL_CANDIDATES.md`
- `docs/project_targets/STABILITY_REPORT.md`
- `docs/project_targets/SIGNIFICANCE_REPORT.md`
- `docs/project_targets/CROSS_DATASET_REPORT.md`
- `docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md`
- `docs/project_targets/PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md`
- `docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md`
- `docs/project_targets/PAPER_SUBMISSION_READINESS_CHECKLIST.md`
- `docs/project_targets/PROJECT_FINAL_YEAR_EXECUTION_PLAN.md`
- `docs/project_targets/DEPLOYMENT_LOCK.md`
- `docs/project_targets/DEPLOYMENT_DEFAULT_PROFILE.md`
- `docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md`
- `docs/project_targets/FINAL_DEMO_WALKTHROUGH.md`
- `docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md`
- `docs/project_targets/CLEAN_DRY_RUN_MINIMUM_PATH.md`
- `docs/project_targets/SUBMISSION_PACK_INDEX.md`
- `docs/project_targets/FIELD_VALIDATION_RUNBOOK.md`
- `docs/project_targets/FIELD_VALIDATION_MINIMUM_PACK.md`
- `docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md`
- `docs/project_targets/REPLAY_LIVE_ACCEPTANCE_LOCK.md`
- `docs/project_targets/LOCKED_PARAMS_RUNBOOK.md`
- `docs/project_targets/PLOT_EVIDENCE_CHECKLIST.md`
- `docs/project_targets/PROJECT_DELIVERY_EXCELLENCE_STANDARD.md`

Do not include by default:

- `docs/project_targets/archive/`
- `docs/project_targets/supporting/`

## 4. Required Evidence Artifacts

Include:

- `artifacts/reports/stability_summary.csv`
- `artifacts/reports/stability_summary.json`
- `artifacts/reports/significance_summary.json`
- `artifacts/reports/cross_dataset_manifest.json`
- `artifacts/reports/cross_dataset_summary.csv`
- `artifacts/reports/cross_dataset_error_taxonomy.md`
- `artifacts/reports/online_mc_replay_matrix_20260402.csv`
- `artifacts/reports/online_mc_replay_matrix_20260402.json`
- `artifacts/reports/deployment_lock_validation.md`
- `artifacts/reports/deployment_field_eval.json`
- `artifacts/reports/deployment_field_failures.json`
- `artifacts/reports/deployment_field_validation_summary.md`
- `artifacts/reports/deployment_field_observations.csv`
- `artifacts/reports/clean_dry_run_report.md`
- `artifacts/reports/replay_live_acceptance.md`
- `artifacts/reports/release_snapshot.md`
- `artifacts/reports/release_bundle_status.json`

Do not include by default:

- `artifacts/reports/archive/`
- `artifacts/reports/diagnostic/`
- `artifacts/reports/tuning/`
- `artifacts/reports/gcn_aug/`
- `artifacts/reports/gcn_overtake/`
- `artifacts/reports/hneg_cycle/`

## 5. Required Report Figures

Include:

- `artifacts/figures/report/system_architecture_diagram.svg`
- `artifacts/figures/report/alert_policy_flow.svg`
- `artifacts/figures/report/offline_stability_comparison.png`
- `artifacts/figures/report/cross_dataset_transfer_summary.png`
- `artifacts/figures/report/cross_dataset_f1_comparison.png`
- `artifacts/figures/report/online_replay_accuracy_heatmap.png`
- `artifacts/figures/report/online_mc_dropout_delta.png`
- `artifacts/figures/report/stability_f1_errorbars.png`

Do not include by default:

- `artifacts/figures/report/diagnostic/`

## 6. Exclusions by Default

Exclude unless explicitly justified:

- branch-only `research_ops/`
- archived replay history
- diagnostic replay materials
- MUVIM exploratory archived ops
- broad tuning and exploratory engineering traces
- historical planning docs

## 7. Intended Use

Use this allowlist when:

- preparing a markable repository snapshot
- defining a release boundary
- deciding what should move with a frozen export

If a file outside this allowlist is needed, record the reason in the release note or freeze log before adding it.
