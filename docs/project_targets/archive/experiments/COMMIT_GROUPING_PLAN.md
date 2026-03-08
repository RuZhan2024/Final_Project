# Commit Grouping Plan

This plan groups the current work into clean, reviewable commits.

## Commit 1 — Experiment Evidence & Decisions

Scope:
- Experiment tracking + summaries + decision reports.

Include files:
- `tools/track_experiment.py`
- `artifacts/registry/overfit_experiment_registry.csv`
- `artifacts/reports/tuning/overfit_round1_commands.sh`
- `artifacts/reports/tuning/overfit_round1_summary.csv`
- `artifacts/reports/tuning/overfit_round1_summary.md`
- `artifacts/reports/tuning/tcn_round2_commands.sh`
- `artifacts/reports/tuning/tcn_round2_results.md`
- `docs/project_targets/archive/experiments/OVERFIT_ROUND1_PLAN.md`
- `docs/project_targets/archive/experiments/TCN_ROUND2_PLAN.md`
- `docs/project_targets/archive/experiments/GCN_POLICY_ROUND2_RESULTS.md`
- `configs/ops/*r1_*`
- `configs/ops/*r2_*`
- `outputs/metrics/*r1_*` *(if you intend to version metrics outputs)*
- `outputs/metrics/*r2_*` *(if you intend to version metrics outputs)*

Message suggestion:
- `feat(exp): add round1/round2 experiment tracking, summaries, and promotion decisions`

## Commit 2 — Deployment Lock & Release Artifacts

Scope:
- Lock profile + validation + release snapshot + bundle checks.

Include files:
- `configs/ops/tcn_caucafall.yaml`
- `docs/project_targets/DEPLOYMENT_LOCK.md`
- `artifacts/reports/deployment_lock_validation.md`
- `artifacts/reports/release_snapshot.md`
- `tools/run_deployment_lock_validation.sh`
- `tools/check_release_bundle.py`
- `artifacts/reports/release_bundle_status.json`
- `docs/project_targets/FINAL_DEMO_WALKTHROUGH.md`
- `docs/project_targets/THESIS_EVIDENCE_MAP.md`

Message suggestion:
- `chore(release): lock caucafall tcn op2 profile and add deployment validation bundle checks`

## Commit 3 — Runtime Defaults Alignment (Optional Separate Commit)

Scope:
- Backend/frontend/db defaults aligned to locked profile.

Include files:
- `server/core.py`
- `server/routes/settings.py`
- `server/create_db.sql`
- `apps/src/pages/settings/SettingsPage.js`

Message suggestion:
- `fix(defaults): align settings/db/ui fall-threshold and default model to locked tcn profile`

## Notes

- If repo policy avoids committing generated outputs, exclude `outputs/metrics/*` and keep only reports + registry + ops.
- Keep existing unrelated modified files out of these commits unless they are part of the same review intent.
