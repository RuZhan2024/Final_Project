# Deployment Field Validation Task Sheet

## Scope
Validate deployment behavior on a small real-world dataset recorded in your own environment.

## Objective
Provide practical evidence that the deployed system is robust outside benchmark datasets.

## Data Collection Plan
- Target size: 20-40 short clips.
- ADL categories:
  - walking
  - sitting/standing
  - bending/picking object
  - intentional lying down
- Event-like categories (safe simulation only):
  - fast sit / controlled collapse-like motion
- Variation factors:
  - lighting (bright/dim)
  - camera angle (front/side)
  - distance (near/far)
  - occlusion (partial)
  - subject count (single/multi-person)

## Protocol Rules
- Keep this set separate from benchmark train/val/test splits.
- No model retraining with this set.
- Use this set only for deployment validation and error analysis.
- If using unlabeled benchmark subsets first (e.g., LE2i unlabeled), report results as stress-test alert-rate proxies only unless clips are verified fall-free.

## Evaluation Tasks
1. Run end-to-end deployment path on each clip.
2. Log for each clip:
- predicted events
- false alerts
- missed events
- detection delay
- runtime errors/failures
3. Export artifacts:
- `artifacts/reports/deployment_field_eval.json`
- `artifacts/reports/deployment_field_failures.json`
- `artifacts/reports/deployment_field_validation_summary.md`

## Metrics to Report
- Event recall proxy (on labeled field events)
- False alerts per hour/day (field estimate)
- Delay p50/p95
- Failure-mode counts by type
- For unlabeled stress runs, clearly mark FA/day as `estimated alert rate` rather than strict false-alarm rate.

## Deliverables
- Summary report row in evidence map.
- Short qualitative error taxonomy with top recurring failure modes.
- Reproduce command(s) used for evaluation.

## Closure Levels

### Minimum paper-safe path
- does not require DB event history
- requires:
  - `deployment_field_manifest.csv`
  - `deployment_field_labels.csv`
  - `deployment_field_observations.csv`
  - `deployment_field_eval.json`
  - `deployment_field_failures.json`
  - `deployment_field_validation_summary.md`

### Enhanced DB-backed path
- optional strengthening evidence only
- adds:
  - `deployment_dual_policy_events.json`
- use only if DB access is available and recent runtime events exist

## Acceptance Criteria
- 20-40 clips processed.
- Reports generated with required metric fields.
- Evidence map updated with artifact paths + command.
- Clear statement of limitations (small sample size, scenario coverage).

## Current Status
- In progress

## Manual TODO (User)
- [ ] Record 20-40 real field clips and place under `data/field/raw/`.
- [ ] Copy templates to working report files:
  - `artifacts/reports/deployment_field_manifest.csv`
  - `artifacts/reports/deployment_field_labels.csv`
  - `artifacts/reports/deployment_field_observations.csv`
- [ ] Fill `artifacts/reports/deployment_field_observations.csv` (from template).
- [ ] Run the minimum paper-safe summarizer:
  - `python tools/summarize_field_validation.py --obs_csv artifacts/reports/deployment_field_observations.csv --hours 1.0 --out_eval_json artifacts/reports/deployment_field_eval.json --out_failures_json artifacts/reports/deployment_field_failures.json --out_markdown artifacts/reports/deployment_field_validation_summary.md`
- [ ] (Optional) If DB-backed runtime evidence is available, also run:
  - `python tools/summarize_dual_policy_events.py --resident_id 1 --hours 24 --out_json artifacts/reports/deployment_dual_policy_events.json`
  - `python tools/summarize_field_validation.py --obs_csv artifacts/reports/deployment_field_observations.csv --hours 1.0 --dual_policy_json artifacts/reports/deployment_dual_policy_events.json --out_eval_json artifacts/reports/deployment_field_eval.json --out_failures_json artifacts/reports/deployment_field_failures.json --out_markdown artifacts/reports/deployment_field_validation_summary.md`
- [ ] (Optional) Start API once for live robustness extension:
  - `source .venv/bin/activate && PYTHONPATH="$(pwd)/src:$(pwd)" uvicorn server.app:app --host 0.0.0.0 --port 8000`

## Execution Pack
- Detailed runbook: `docs/project_targets/FIELD_VALIDATION_RUNBOOK.md`
- Minimum closure guide: `docs/project_targets/FIELD_VALIDATION_MINIMUM_PACK.md`
- Templates:
  - `artifacts/templates/deployment_field_manifest_template.csv`
  - `artifacts/templates/deployment_field_labels_template.csv`
  - `artifacts/templates/deployment_field_observations_template.csv`
- Summarizer script:
  - `tools/summarize_field_validation.py`
  - `tools/summarize_dual_policy_events.py`
