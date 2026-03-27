# Field Validation Minimum Pack

Date: 2026-03-22

Purpose:
Define the smallest field-validation bundle that is strong enough to support a bounded paper claim without blocking on optional DB-backed runtime summaries.

## Key Decision

There are two valid closure levels:

1. Minimum paper-safe path
- required for closing the current paper claim
- does not require database event history

2. Enhanced DB-backed path
- optional strengthening evidence
- requires `events` table access and a working DB configuration

The paper should not wait on the enhanced path if the minimum path is complete.

## Minimum Paper-Safe Path

Required inputs:
- `data/field/raw/` with the recorded clips
- `artifacts/reports/deployment_field_manifest.csv`
- `artifacts/reports/deployment_field_labels.csv`
- `artifacts/reports/deployment_field_observations.csv`

Required outputs:
- `artifacts/reports/deployment_field_eval.json`
- `artifacts/reports/deployment_field_failures.json`
- `artifacts/reports/deployment_field_validation_summary.md`

Required command:

```bash
python tools/summarize_field_validation.py \
  --obs_csv artifacts/reports/deployment_field_observations.csv \
  --hours 1.0 \
  --out_eval_json artifacts/reports/deployment_field_eval.json \
  --out_failures_json artifacts/reports/deployment_field_failures.json \
  --out_markdown artifacts/reports/deployment_field_validation_summary.md
```

Minimum acceptance gate:
- at least `20` clips
- manifest, labels, and observations files filled
- summary markdown produced
- evidence map updated
- paper wording remains bounded:
  - small field set
  - controlled environment
  - deployment-oriented supporting evidence only

## Enhanced DB-Backed Path

Additional optional output:
- `artifacts/reports/deployment_dual_policy_events.json`

Optional command:

```bash
python tools/summarize_dual_policy_events.py \
  --resident_id 1 \
  --hours 24 \
  --out_json artifacts/reports/deployment_dual_policy_events.json
```

Use this only when:
- the DB is available
- recent events exist in `events`
- the runtime actually logs the dual-policy fields you want to summarize

What it adds:
- counts of safe vs recall alerts
- disagreement ratios
- stronger runtime-policy support in the field-validation summary

What it does not change:
- it is not required to close the minimum paper-safe field-validation claim

## Recommended File Set To Commit

- `artifacts/reports/deployment_field_manifest.csv`
- `artifacts/reports/deployment_field_labels.csv`
- `artifacts/reports/deployment_field_observations.csv`
- `artifacts/reports/deployment_field_eval.json`
- `artifacts/reports/deployment_field_failures.json`
- `artifacts/reports/deployment_field_validation_summary.md`

Optional:
- `artifacts/reports/deployment_dual_policy_events.json`

## Paper-Safe Wording

Use wording like:
- `A small field-validation set was used to assess deployment-oriented behavior under controlled real-world conditions.`
- `The field evidence is intended as bounded supporting validation rather than large-scale real-world proof.`

Do not use wording like:
- `The system is validated for real-world deployment at scale.`
- `The field study proves robust deployment performance across environments.`
