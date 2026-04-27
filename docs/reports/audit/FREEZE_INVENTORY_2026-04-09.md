# Freeze Inventory

Date: 2026-04-09  
Scope: current branch-local freeze classification for the thesis/report and defended software package

## Purpose

This document defines the current repository inventory in freeze terms:

- `freeze_core`: should be treated as current source-of-truth for a final reviewed snapshot
- `supporting_branch_only`: useful on this branch for thesis/reporting or engineering traceability, but not part of the minimum frozen release surface
- `archive_or_diagnostic`: intentionally retained history, diagnosis, or intermediate material that must not be treated as current authoritative evidence

This is an inventory document, not a deletion plan.

## 1. Freeze Core

### 1.1 Runtime and application code

Keep as freeze core:

- `src/fall_detection/`
- `server/`
- `apps/`
- `scripts/`
- `tests/`
- `docker-compose.yml`
- `README.md`

Reason:
- these files define the actual runnable system, runtime behavior, evaluation scripts, and current user-facing entrypoints

### 1.2 Active deploy and report configs

Keep as freeze core:

- `configs/ops/tcn_caucafall.yaml`
- `configs/ops/gcn_caucafall.yaml`
- `configs/ops/tcn_le2i.yaml`
- `configs/ops/gcn_le2i.yaml`
- frozen candidate ops still needed for defended evidence:
  - `configs/ops/tcn_caucafall_r2_train_hneg.yaml`
  - `configs/ops/gcn_caucafall_r2_recallpush_b.yaml`
  - `configs/ops/tcn_le2i_opt33_r2.yaml`
  - `configs/ops/gcn_le2i_opt33_r2.yaml`

Reason:
- these drive the live runtime or the frozen rerun evidence now referenced by the report and evidence map

### 1.3 Active project-target control docs

Keep as freeze core:

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

Reason:
- these are the current source-of-truth control, claim, evidence, and delivery documents after the latest alignment pass

### 1.4 Current authoritative evidence artifacts

Keep as freeze core:

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

### 1.5 Current report-facing figures

Keep as freeze core:

- `artifacts/figures/report/system_architecture_diagram.svg`
- `artifacts/figures/report/alert_policy_flow.svg`
- `artifacts/figures/report/offline_stability_comparison.png`
- `artifacts/figures/report/cross_dataset_transfer_summary.png`
- `artifacts/figures/report/cross_dataset_f1_comparison.png`
- `artifacts/figures/report/online_replay_accuracy_heatmap.png`
- `artifacts/figures/report/online_mc_dropout_delta.png`
- `artifacts/figures/report/stability_f1_errorbars.png`

Reason:
- these are the current main report figure pack after the diagnostic figure was split out

## 2. Supporting Branch Only

These files are useful on this branch and should be retained, but they are not the minimum frozen release surface.

### 2.1 Branch-only research operating system

Keep branch-only:

- `research_ops/`

Reason:
- useful for thesis control and paper-evidence discipline
- intentionally not part of the stripped code-only `main` strategy

### 2.2 Supporting project-target docs

Keep branch-only:

- `docs/project_targets/supporting/`

Includes:

- `ABLATION_MATRIX.md`
- `EXPERIMENT_RECORDING_PROTOCOL.md`
- `LATENCY_REPORT.md`
- `OBJECTIVES_EVIDENCE_OUTCOMES.md`
- `OPS_POLICY_REPORT.md`
- `PARAM_PROMOTION_WORKFLOW.md`
- `ROBUSTNESS_REPORT.md`

Reason:
- these support thesis discussion, examiner-facing traceability, or process understanding
- they are not the top-layer freeze-control surface

### 2.3 Supplemental report notes and audits

Keep branch-only unless a final submission pack explicitly needs them:

- `docs/reports/notes/`
- `docs/reports/audit/`
- `docs/reports/readiness/`

Reason:
- they are valuable for internal control, supervision, and final review
- they are not all needed in a minimal external release snapshot

### 2.4 Supplemental evidence and engineering traces

Keep branch-only:

- `artifacts/reports/tuning/`
- `artifacts/reports/gcn_aug/`
- `artifacts/reports/gcn_overtake/`
- `artifacts/reports/hneg_cycle/`
- `artifacts/reports/fault_inject_summary.json`
- `artifacts/reports/infer_profile_cpu_local_tcn_le2i.json`
- `artifacts/reports/op123_per_seed.csv`
- `artifacts/reports/op123_stability_summary.csv`
- `artifacts/reports/op123_stability_summary.json`

Reason:
- these retain engineering depth and thesis traceability
- they are not required for the narrowest frozen evidence claim set

## 3. Archive or Diagnostic

These should remain available but must not be treated as current authoritative evidence.

### 3.1 Archived replay history

Archive/diagnostic only:

- `artifacts/reports/archive/replay_matrix_legacy_20260402/`
- `artifacts/reports/archive/replay_runtime_iterations_20260402/`

Reason:
- retained for audit history and mismatch diagnosis
- superseded by `online_mc_replay_matrix_20260402.*` and the current aligned report text

### 3.2 Diagnostic-only replay artifacts

Archive/diagnostic only:

- `artifacts/reports/diagnostic/online_replay_le2i_perclip_20260402.json`
- `artifacts/figures/report/diagnostic/le2i_per_clip_outcome_heatmap.png`

Reason:
- valid pre-fix diagnosis
- not valid as final runtime-result evidence

### 3.3 Archived planning docs

Archive/diagnostic only:

- `docs/project_targets/archive/planning/`

Includes:

- `PAPER_SECTION_HEADINGS.md`
- `PAPER_SUBMISSION_WEEK_PLAN.md`
- `PLOT_SELECTION_FOR_REPORT.md`

Reason:
- still useful for writing history
- no longer source-of-truth for final freeze state

### 3.4 Archived MUVIM exploratory ops

Archive/diagnostic only:

- `configs/ops/archive/muvim/`

Reason:
- retains quick-search and later exploratory MUVIM branches
- removed from the active runtime surface to reduce accidental reuse

## 4. Current Freeze Assessment

Current state after this cleanup pass:

- `artifacts`: partially freeze-structured
  - main evidence separated from archive and diagnostic material
- `configs/ops`: partially freeze-structured
  - MUVIM exploratory ops archived
  - caucafall/le2i historical tuning families still mixed at root
- `docs/project_targets`: materially improved
  - root now closer to live control docs
  - planning and supporting material separated

## 5. Remaining Gaps Before a True Freeze Snapshot

Still unresolved:

1. `configs/ops/` root still contains many historical caucafall/le2i tuning families that are not yet split into `active` vs `archive`.
2. The overall git worktree is still globally dirty outside the surfaces cleaned here.
3. Final submission/release intent is still not encoded as one explicit allowlist of files to include in a markable snapshot.

## 6. Recommended Next Move

Do not keep moving files blindly.

The next correct step is:

1. define a freeze allowlist for the final markable snapshot
2. define a branch-only allowlist for retained thesis/supporting material
3. leave the remaining historical material in place unless it is causing concrete confusion

That will convert the current cleanup work into a defensible final freeze boundary.
