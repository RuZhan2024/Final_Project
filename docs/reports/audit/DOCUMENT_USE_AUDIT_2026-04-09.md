# Document Use Audit

Date: 2026-04-09

## Purpose

Identify which files under `docs/` are:
- actively useful and should remain visible
- supporting/branch-only and worth keeping but not treating as current source of truth
- archive-only / likely unused in the current project state

This audit does **not** delete anything. It marks files for future keep/archive/remove decisions.

Update:
- the first archival batch from this audit has already been applied
- retired report plans/notes were moved under `docs/archive/reports/`
- older training docs were moved under `docs/archive/training/`

## Audit Method

The review used three signals:
- current directory role declared in `docs/README.md`, `docs/project_targets/README.md`, and `docs/archive/README.md`
- active reference counts from the current repository graph
- whether a document still serves the frozen report/evidence/handoff path

Reference counts were checked against:
- `README.md`
- `docs/`
- `research_ops/`
- `scripts/`
- `server/`
- `apps/`
- `src/`
- `Makefile`

Interpretation rule:
- `0` references does not automatically mean safe to delete
- but `0` references plus archive/historical role is a strong signal that the file is not currently useful in active project flow

## Top-Level Summary

Current `docs/` file count:
- total files: `106`
- `docs/project_targets`: `47`
- `docs/reports`: `45`
- `docs/archive`: `9`
- `docs/training`: `3`
- root-level standalone docs: `2`

## Keep As Active Source Of Truth

These are still actively useful and should remain in the current visible docs surface:

- `docs/project_targets/THESIS_EVIDENCE_MAP.md`
- `docs/project_targets/CLAIM_TABLE.md`
- `docs/project_targets/FINAL_CANDIDATES.md`
- `docs/project_targets/STABILITY_REPORT.md`
- `docs/project_targets/SIGNIFICANCE_REPORT.md`
- `docs/project_targets/CROSS_DATASET_REPORT.md`
- `docs/project_targets/DEPLOYMENT_LOCK.md`
- `docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md`
- `docs/project_targets/FINAL_DEMO_WALKTHROUGH.md`
- `docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md`
- `docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md`
- `docs/project_targets/PROJECT_FINAL_YEAR_EXECUTION_PLAN.md`
- `docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md`
- `docs/project_targets/REPLAY_LIVE_ACCEPTANCE_LOCK.md`
- `docs/project_targets/SUBMISSION_PACK_INDEX.md`
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`
- `docs/reports/readiness/READINESS_REPORT.md`
- `docs/reports/runbooks/USER_GUIDE.md`
- `docs/reports/runbooks/DEMO_RUNBOOK.md`
- `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`
- `docs/reports/audit/FULL_STACK_PROJECT_AUDIT_2026-04-09.md`
- `docs/reports/audit/FULL_CODE_REVIEW_2026-04-09.md`
- `docs/reports/audit/FREEZE_STATUS_2026-04-09.md`
- `docs/reports/audit/SUPERVISOR_HANDOFF_SUMMARY_2026-04-09.md`

## Keep As Supporting / Branch-Only Material

These still have some value, but they are not the main current source of truth:

- `docs/project_targets/supporting/OBJECTIVES_EVIDENCE_OUTCOMES.md`
- `docs/project_targets/supporting/OPS_POLICY_REPORT.md`
- `docs/project_targets/supporting/LATENCY_REPORT.md`
- `docs/project_targets/supporting/ROBUSTNESS_REPORT.md`
- `docs/project_targets/supporting/ABLATION_MATRIX.md`
- `docs/project_targets/supporting/EXPERIMENT_RECORDING_PROTOCOL.md`
- `docs/project_targets/supporting/PARAM_PROMOTION_WORKFLOW.md`
- `docs/project_targets/archive/planning/PAPER_SECTION_HEADINGS.md`
- `docs/project_targets/archive/planning/PLOT_SELECTION_FOR_REPORT.md`
- `docs/reports/notes/HIGH_STANDARD_FINAL_REPORT_EVIDENCE_INVENTORY_2026-03-29.md`
- `docs/reports/notes/HIGH_STANDARD_FINAL_REPORT_TASKS_2026-03-29.md`
- `docs/reports/notes/Compute_Threshold.md`
- `docs/reports/notes/REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md`
- `docs/reports/runbooks/CONFIG_RESULT_EVIDENCE_MAP.md`
- `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md`
- `docs/reports/runbooks/ONLINE_FRONTEND_SMOKE_CHECKLIST.md`
- `docs/reports/runbooks/SUPERVISOR_DELIVERY_MODES.md`
- `docs/reports/checklists/DEPLOYMENT_READINESS_CHECKLIST.md`
- `docs/reports/checklists/REPRODUCIBILITY_CHECKLIST.md`

Recommended treatment:
- keep on this branch
- do not treat as first-stop docs for supervisor handoff
- move further out of the active surface only if you want a stricter minimal-docs submission branch

## Archive-Only / Not Actively Useful

These files are not currently useful in the active delivery path. Most are historical, superseded, or have zero active references.

### Already archive-scoped and safe to keep as history only

- `docs/archive/README.md`
- all files under `docs/archive/audit_completed/`
- all files under `docs/project_targets/archive/experiments/`
- all files under `docs/project_targets/archive/implementation/`
- `docs/project_targets/archive/planning/PAPER_SUBMISSION_WEEK_PLAN.md`
- `docs/project_targets/archive/planning/README.md`

These should remain archive-only and should not be surfaced as current guidance.

### Likely unused in current repo state

These have no active repo references and do not appear to support the current frozen handoff path:

- `docs/reports/audit/ARTIFACT_PORTABILITY_REPORT.md`
- `docs/reports/audit/AUDIT_EXECUTION_TASKS.md`
- `docs/reports/audit/CAUCAFALL_OUTCOMES_AUDIT_2026-03-08.md`
- `docs/reports/audit/CODE_CLEANLINESS_REPORT.md`
- `docs/reports/audit/CONFIG_CONTRACT_MATRIX.md`
- `docs/reports/audit/FULL_PROJECT_AUDIT_2026-04-09.md`
- `docs/reports/audit/FYP_EVALUATION_AGAINST_LECTURE.md`
- `docs/reports/audit/GLOBAL_AUDIT_REPORT.md`
- `docs/reports/audit/INTEGRATION_STATUS_REPORT.md`
- `docs/reports/audit/SUBMISSION_PACK_GAP_AUDIT.md`
- `docs/reports/checklists/DEPLOYMENT_READINESS_CHECKLIST.md`
- `docs/reports/checklists/REPRODUCIBILITY_CHECKLIST.md`
- `docs/reports/notes/HIGH_STANDARD_FINAL_REPORT_FIGURE_TABLE_PLAN_2026-03-29.md`
- `docs/reports/notes/report_note.md`
- `docs/reports/plans/PATCH_PLAN.md`
- `docs/reports/plans/SAFE_GUARD_ALERT_SYSTEM_IMPLEMENTATION_PLAN.md`
- `docs/reports/plans/THIS_WEEK_TASKLIST.md`
- `docs/reports/runbooks/CLEANUP_SCOPE_AUDIT.md`
- `docs/reports/runbooks/RELEASE_RUNBOOK.md`
- `docs/training/TRAINING_STABILITY.md`
- `docs/training/TRAINING_UPGRADES_TASKS.md`
- `docs/training/training_upgrades.md`

Recommended treatment:
- do not delete blindly
- mark as archive-only or move under a stricter `archive/` surface if you want to shrink `docs/reports/` and `docs/training/`

## Special Cases

### `docs/Literature Review.docx`

Status:
- low active repo usefulness
- likely personal background/reference material

Recommendation:
- keep only if you still want the original literature-review artifact in the repo
- otherwise move out of the active `docs/` surface or archive externally

### `docs/reports/README.md`

Status:
- currently referenced as an index role, but its content is outdated relative to the newer freeze/code-review documents

Recommendation:
- keep, but refresh it if you want `docs/reports/` to act as a reliable landing page

## Marked Candidates

### Best candidates to hide from active surface first

Applied in the first archival batch:
- `docs/reports/plans/PATCH_PLAN.md` -> `docs/archive/reports/plans/PATCH_PLAN.md`
- `docs/reports/plans/SAFE_GUARD_ALERT_SYSTEM_IMPLEMENTATION_PLAN.md` -> `docs/archive/reports/plans/SAFE_GUARD_ALERT_SYSTEM_IMPLEMENTATION_PLAN.md`
- `docs/reports/plans/THIS_WEEK_TASKLIST.md` -> `docs/archive/reports/plans/THIS_WEEK_TASKLIST.md`
- `docs/reports/notes/report_note.md` -> `docs/archive/reports/notes/report_note.md`
- `docs/reports/notes/HIGH_STANDARD_FINAL_REPORT_FIGURE_TABLE_PLAN_2026-03-29.md` -> `docs/archive/reports/notes/HIGH_STANDARD_FINAL_REPORT_FIGURE_TABLE_PLAN_2026-03-29.md`
- `docs/training/TRAINING_STABILITY.md` -> `docs/archive/training/TRAINING_STABILITY.md`
- `docs/training/TRAINING_UPGRADES_TASKS.md` -> `docs/archive/training/TRAINING_UPGRADES_TASKS.md`
- `docs/training/training_upgrades.md` -> `docs/archive/training/training_upgrades.md`

- `docs/reports/plans/PATCH_PLAN.md`
- `docs/reports/plans/SAFE_GUARD_ALERT_SYSTEM_IMPLEMENTATION_PLAN.md`
- `docs/reports/plans/THIS_WEEK_TASKLIST.md`
- `docs/reports/notes/report_note.md`
- `docs/training/TRAINING_STABILITY.md`
- `docs/training/TRAINING_UPGRADES_TASKS.md`
- `docs/training/training_upgrades.md`

### Best candidates to move deeper into archive if desired

- `docs/reports/audit/ARTIFACT_PORTABILITY_REPORT.md`
- `docs/reports/audit/AUDIT_EXECUTION_TASKS.md`
- `docs/reports/audit/CODE_CLEANLINESS_REPORT.md`
- `docs/reports/audit/CONFIG_CONTRACT_MATRIX.md`
- `docs/reports/audit/GLOBAL_AUDIT_REPORT.md`
- `docs/reports/audit/INTEGRATION_STATUS_REPORT.md`
- `docs/reports/audit/SUBMISSION_PACK_GAP_AUDIT.md`
- `docs/reports/runbooks/CLEANUP_SCOPE_AUDIT.md`
- `docs/reports/notes/HIGH_STANDARD_FINAL_REPORT_FIGURE_TABLE_PLAN_2026-03-29.md`

## Recommendation

Do not mass-delete `docs/`.

Instead:
1. keep the active handoff/evidence/report docs visible
2. demote stale plans/notes/training docs from the active surface
3. treat zero-reference audit and plan docs as archive-only unless you still need them for personal traceability

This gives you a smaller and cleaner docs surface without breaking the current evidence chain.
