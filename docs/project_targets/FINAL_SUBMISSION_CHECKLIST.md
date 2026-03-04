# Final Submission Checklist

Use this before final handoff/demo.

## A) Deployment Lock

- [ ] `docs/project_targets/DEPLOYMENT_LOCK.md` exists and matches runtime profile.
- [ ] `configs/ops/tcn_caucafall.yaml` points to intended checkpoint.
- [ ] `artifacts/reports/deployment_lock_validation.md` shows:
  - [ ] API health = yes
  - [ ] predict endpoint exists = yes
  - [ ] active model = TCN
  - [ ] active dataset = caucafall
  - [ ] active OP = OP-2
  - [ ] manual replay non-fall PASS
  - [ ] manual replay fall PASS
  - [ ] verdict PASS checked

## B) Evidence Integrity

- [ ] `docs/project_targets/THESIS_EVIDENCE_MAP.md` includes latest rows:
  - [ ] Tab-Overfit-Round1-13
  - [ ] Tab-GCN-Policy-R2-14
  - [ ] Tab-TCN-Train-R2-15
  - [ ] Tab-Deploy-Lock-16
- [ ] `artifacts/reports/release_snapshot.md` exists.
- [ ] `artifacts/reports/release_bundle_status.json` exists and `"ok": true`.
- [ ] `python tools/check_release_bundle.py` passes.

## C) Experiment Record Completeness

- [ ] `artifacts/registry/overfit_experiment_registry.csv` includes planned/running/done/rejected transitions.
- [ ] Round-1 summary exists:
  - [ ] `artifacts/reports/tuning/overfit_round1_summary.md`
- [ ] GCN policy Round-2 report exists:
  - [ ] `docs/project_targets/GCN_POLICY_ROUND2_RESULTS.md`
- [ ] TCN Round-2 report exists:
  - [ ] `artifacts/reports/tuning/tcn_round2_results.md`

## D) Demo Readiness

- [ ] `docs/project_targets/FINAL_DEMO_WALKTHROUGH.md` reviewed and followed once end-to-end.
- [ ] Settings page can set `caucafall + TCN + OP-2` and persist.
- [ ] Monitor replay can show one non-fall and one fall expected behavior.
- [ ] Events/Dashboard load without API failures during demo path.

## E) Packaging & Commit

- [ ] `docs/project_targets/COMMIT_GROUPING_PLAN.md` reviewed.
- [ ] Commits grouped by topic (experiments / lock / defaults).
- [ ] No unintended files included.
- [ ] Final tag/message prepared.

## F) Optional Final Gate

- [ ] Run a final quick check:
  - `bash tools/run_deployment_lock_validation.sh`
  - `python tools/check_release_bundle.py`
- [ ] Re-open `artifacts/reports/deployment_lock_validation.md` and confirm PASS remains checked.
