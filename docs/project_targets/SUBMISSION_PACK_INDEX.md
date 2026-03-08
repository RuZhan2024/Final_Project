# Submission Pack Index

Use this as the single index for final hand-in materials.

## 1) Code Submission

- Repository root:
  - `README.md`
- Final release snapshot:
  - `git tag <FINAL_TAG>` (to be filled at release time)
- Optional zipped source (if required by submission portal):
  - `artifacts/release/<project_name>_<tag>.zip` (to be generated)

## 2) User Guide

- Primary guide:
  - `docs/reports/runbooks/USER_GUIDE.md`
- Demo-oriented walkthrough:
  - `docs/project_targets/FINAL_DEMO_WALKTHROUGH.md`
- Deployment runbook:
  - `docs/reports/runbooks/DEMO_RUNBOOK.md`

## 3) Working Artefact Evidence

- Deployment lock + validation:
  - `docs/project_targets/DEPLOYMENT_LOCK.md`
  - `artifacts/reports/deployment_lock_validation.md`
- Core performance/evidence mapping:
  - `docs/project_targets/THESIS_EVIDENCE_MAP.md`
- Final alignment status:
  - `docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md`

## 4) Demo Recording

- Recommended location:
  - `artifacts/demo/final_demo_recording.mp4` (or submission platform link)
- Minimum content:
  - system startup
  - one non-fall replay
  - one fall replay
  - output/event behavior visible end-to-end

## 5) Mandatory Pre-Submit Checks

Run and confirm:
- `bash tools/run_deployment_lock_validation.sh`
- `python tools/check_release_bundle.py`

Then verify:
- `docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md`
- `docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md`
