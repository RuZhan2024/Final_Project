# Freeze Status

Date: 2026-04-09  
Source: `./scripts/freeze_manifest.sh`

## Current Verdict

Freeze-core path existence: **pass**  
Freeze-core git cleanliness: **fail**

This means the current frozen boundary is now well defined and machine-checkable, but the repository still does not present a clean freeze snapshot.

## What Is Already Closed

- freeze-core allowlist exists
- branch-support allowlist exists
- report figure root is unified under `artifacts/figures/report`
- diagnostic replay artifacts are split under `artifacts/reports/diagnostic` and `artifacts/figures/report/diagnostic`
- planning/supporting docs are no longer mixed directly into the root `docs/project_targets` layer
- report build has a stronger scripted entrypoint
- canonical test entrypoint now exists

## Freeze-Core Dirty Surface

The latest `freeze_manifest` run shows these still-dirty freeze-core areas:

### Runtime / code layer

- `server/`
- `src/fall_detection/evaluation/fit_ops.py`
- `tests/server/*`
- `apps/src/pages/monitor/components/ModelInfoCard.js`
- `apps/src/pages/settings/SettingsPage.js`
- `docker-compose.yml`

Interpretation:

- the live application/runtime layer is still changing
- this is the highest-priority blocker for a true freeze snapshot

### Config / evidence layer

- `configs/ops/README.md`
- `artifacts/reports/cross_dataset_manifest.json`
- `artifacts/reports/cross_dataset_summary.csv`
- `artifacts/figures/report/cross_dataset_transfer_summary.png`
- untracked main report figures:
  - `cross_dataset_f1_comparison.png`
  - `online_mc_dropout_delta.png`
  - `online_replay_accuracy_heatmap.png`
  - `stability_f1_errorbars.png`
- untracked replay matrix artifacts:
  - `artifacts/reports/online_mc_replay_matrix_20260402.csv`
  - `artifacts/reports/online_mc_replay_matrix_20260402.json`

Interpretation:

- the active evidence layer exists and is internally more coherent than before
- but several authoritative artifacts are still untracked or locally modified

### Control / submission-doc layer

- `docs/project_targets/README.md`
- `docs/project_targets/CLAIM_TABLE.md`
- `docs/project_targets/CROSS_DATASET_REPORT.md`
- `docs/project_targets/THESIS_EVIDENCE_MAP.md`
- `docs/project_targets/FINAL_CANDIDATES.md`
- `docs/project_targets/SIGNIFICANCE_REPORT.md`
- `docs/project_targets/FIELD_VALIDATION_RUNBOOK.md`
- `docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md`
- `docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md`
- `docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md`
- `docs/project_targets/LOCKED_PARAMS_RUNBOOK.md`
- `docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md`
- `docs/project_targets/PLOT_EVIDENCE_CHECKLIST.md`
- `docs/project_targets/PROJECT_FINAL_YEAR_EXECUTION_PLAN.md`
- untracked active control docs:
  - `docs/project_targets/PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md`
  - `docs/project_targets/PAPER_SUBMISSION_READINESS_CHECKLIST.md`
  - `docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md`

Interpretation:

- the active thesis-control layer is still being rewritten
- this is acceptable for branch work, but not for a frozen external snapshot

## Main Remaining Blockers

1. Live runtime code and server paths are still modified.
2. Several freeze-core evidence artifacts remain untracked.
3. Active thesis-control docs remain modified and partially untracked.
4. Freeze-core still overlaps with a broader dirty worktree.

## Recommended Next Sequence

1. Stabilize the live runtime/code layer.
2. Stage and freeze the authoritative evidence artifacts.
3. Stage and freeze the active `docs/project_targets` control set.
4. Re-run:

```bash
./scripts/freeze_manifest.sh
```

5. Do not call the repo freeze-ready until `[freeze-core-git-status]` is empty or intentionally explained.
