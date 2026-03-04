# Field Validation Runbook

## Purpose
Generate deployment-grade field evidence for the locked strategy:
- Primary deployment target: CAUCAFall
- Comparative benchmark: LE2i
- Real-world reliability evidence: field validation set

## Inputs
Use templates:
- `artifacts/templates/deployment_field_manifest_template.csv`
- `artifacts/templates/deployment_field_labels_template.csv`
- `artifacts/templates/deployment_field_observations_template.csv`

## Step 1: Collect Clips
1. Record 20-40 short clips in real target-like environment.
2. Cover ADL and controlled event-like motions.
3. Save clips under `data/field/raw/`.
4. Fill `deployment_field_manifest.csv` from template.

## Step 2: Label Event Clips
1. Fill `deployment_field_labels.csv`:
- `has_event=1` only for controlled event-like clips.
- Provide `event_start_s` and `event_end_s` for event clips.

## Step 3: Run Deployment Stack
Run API:
```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Optional latency baseline (monitor endpoint):
```bash
python scripts/benchmark_monitor_e2e.py \
  --api_base http://127.0.0.1:8000 \
  --endpoint /api/v1/monitor/predict_window \
  --dataset_code caucafall \
  --mode tcn \
  --op_code OP-2 \
  --n_windows 80 \
  --out_json artifacts/reports/monitor_e2e_benchmark_field.json
```

## Step 4: Fill Observations CSV
Use `deployment_field_observations.csv` with one row per clip:
- `detected_event`
- `false_alert_count`
- `first_detect_s`
- `delay_s`
- `status`
- `failure_type`

## Step 5: Build Field Reports
```bash
python tools/summarize_dual_policy_events.py \
  --resident_id 1 \
  --hours 24 \
  --out_json artifacts/reports/deployment_dual_policy_events.json

python tools/summarize_field_validation.py \
  --obs_csv artifacts/reports/deployment_field_observations.csv \
  --hours 1.0 \
  --dual_policy_json artifacts/reports/deployment_dual_policy_events.json \
  --out_eval_json artifacts/reports/deployment_field_eval.json \
  --out_failures_json artifacts/reports/deployment_field_failures.json \
  --out_markdown artifacts/reports/deployment_field_validation_summary.md
```

## Step 6: Acceptance Gates
- Clip coverage: >= 20 clips.
- Required outputs exist:
  - `artifacts/reports/deployment_field_eval.json`
  - `artifacts/reports/deployment_field_failures.json`
  - `artifacts/reports/deployment_field_validation_summary.md`
- Include in report:
  - `event_recall_proxy`
  - `fa24h_estimate`
  - `delay_p50_s`, `delay_p95_s`
  - `failure_type_counts`

## Step 7: Evidence Map Update
Add a new row in `docs/project_targets/THESIS_EVIDENCE_MAP.md` pointing to:
- command used
- manifest/labels/observations files
- two output JSON reports

## Notes
- Do not retrain with field clips.
- Do not tune thresholds on field test results.
- Keep this set isolated as deployment validation evidence only.
