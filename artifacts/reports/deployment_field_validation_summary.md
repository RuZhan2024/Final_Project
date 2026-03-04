# Deployment Field Validation Summary

## Core Metrics

- `event_recall_proxy`: 1.0000
- `event_precision_proxy`: 0.5000
- `fa24h_estimate`: 24.0000
- `delay_p50_s`: 0.8000
- `delay_p95_s`: 0.8000

## Failure Summary

- `status_counts`: {"unknown": 1, "fail": 1, "ok": 1}
- `failure_type_counts`: {"false_alert": 1}

## Dual Policy Runtime Summary

- source: `artifacts/reports/deployment_dual_policy_events.json`
- `dual_policy_counts`: {}
- `ratios`: {"disagreement_ratio_on_dual_rows": 0.0, "dual_rows_ratio": 0.0, "recall_only_ratio_on_disagreements": 0.0}
