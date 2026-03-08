# Final Submission Checklist

Use this checklist as the final release gate before hand-in.

Reference standards:
- `docs/project_targets/PROJECT_DELIVERY_EXCELLENCE_STANDARD.md`
- `docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md`
- `docs/project_targets/SUBMISSION_PACK_INDEX.md`

## A) Code Snapshot (Mandatory)

- [ ] Final commit is pushed to remote.
- [ ] Release tag is created for markable snapshot.
- [ ] No unintended local-only files are included in release.
- [ ] Root `README.md` quickstart has been re-tested.

## B) User Guide & Runbooks (Mandatory)

- [ ] User guide exists and is current:
  - `docs/reports/runbooks/USER_GUIDE.md`
- [ ] Demo runbook exists and is current:
  - `docs/reports/runbooks/DEMO_RUNBOOK.md`
- [ ] Final demo walkthrough exists and is current:
  - `docs/project_targets/FINAL_DEMO_WALKTHROUGH.md`
- [ ] One clean-machine dry run has been completed (or explicitly documented as pending).

## C) Working Artefact Evidence (Mandatory)

- [ ] Deployment lock document is current:
  - `docs/project_targets/DEPLOYMENT_LOCK.md`
- [ ] Deployment lock validation report is PASS:
  - `artifacts/reports/deployment_lock_validation.md`
- [ ] Evidence map has no stale/unmapped claim rows:
  - `docs/project_targets/THESIS_EVIDENCE_MAP.md`
- [ ] Release bundle checker passes:
  - `python tools/check_release_bundle.py`

## D) Demo Recording (Mandatory)

- [ ] Demo video recorded (about 5 min).
- [ ] Video shows:
  - [ ] startup
  - [ ] at least one non-fall case
  - [ ] at least one fall case
  - [ ] end-to-end output/result visibility
- [ ] Video file/link is listed in `SUBMISSION_PACK_INDEX.md`.

## E) Research Integrity (R1-R5 Gate)

- [ ] `R1` Reproducibility: commands regenerate reported artifacts.
- [ ] `R2` Integrity: no test leakage in policy fitting/selection.
- [ ] `R3` Operability: backend + frontend + inference flow runs from docs.
- [ ] `R4` Explainability: policy/decision behavior can be explained from outputs.
- [ ] `R5` Limitations: known limitations are explicitly documented.

## F) Parameter Promotion Gate (Mandatory)

- [ ] Any “improved” metric result has passed both tracks:
  - [ ] paper-comparison track improved
  - [ ] deployment track not regressed (especially FA24h / stability)
- [ ] Accepted parameter set is promoted to reproducible command targets (Makefile or equivalent).
- [ ] Promotion is recorded in:
  - [ ] `docs/project_targets/PARAM_PROMOTION_WORKFLOW.md`
  - [ ] `docs/project_targets/FINAL_CANDIDATES.md`
  - [ ] `docs/project_targets/THESIS_EVIDENCE_MAP.md`
- [ ] Root `README.md` reflects the promoted one-command reproduction path.
- [ ] No report/dissertation number depends on non-promoted ad-hoc command lines.

## G) Final Verification Commands

- [ ] `bash tools/run_deployment_lock_validation.sh`
- [ ] `python tools/check_release_bundle.py`
- [ ] Re-open generated reports and confirm PASS status remains valid.
