# Robustness Report

## Scope
Executable failure-mode validation for deployment path (camera/API/model/policy).

## Fault Matrix
| Fault ID | Injection | Expected Behavior | Required Evidence | Status |
|---|---|---|---|---|
| R1 | Empty skeleton frame | Skip frame, no crash, no false alert spike | `fault_inject_summary.json` + pass flag | In progress (scaffold ready) |
| R2 | Dropped frame bursts | Continue stream with cooldown logic | `fault_inject_summary.json` + recovery indicator | In progress (scaffold ready) |
| R3 | Low-confidence joints | Graceful degradation, avoid NaN propagation | metrics + warning log | Done (confirm NaN fix) |
| R4 | Camera stream end | Clean shutdown, final status emitted | `fault_inject_summary.json` + API/server log | In progress (scaffold ready) |
| R5 | Missing input file | explicit error response (no silent pass) | `fault_inject_summary.json` + HTTP/API log | In progress (scaffold ready) |
| R6 | DB/API write failure | fallback behavior + warning log | `fault_inject_summary.json` + log | In progress (scaffold ready) |

## Planned Tooling
- `tools/fault_inject.py` for repeatable injections.
- Output artifacts:
  - `artifacts/reports/fault_inject_summary.json`
  - `artifacts/reports/fault_inject.log`

## Metrics to Report
- `failure_recovery_rate`
- `false_alert_spike_after_fault`
- `mean_recovery_time_s`

## Current Scaffold Result
- `artifacts/reports/fault_inject_summary.json` generated.
- Smoke status: `6/6` scenarios passed in scaffold mode.
- Next upgrade: bind scenarios to live API/runtime path (session state + DB write path) and append server logs.

## Validation Command Skeleton
```bash
python tools/fault_inject.py --scenario all --out_json artifacts/reports/fault_inject_summary.json
```
