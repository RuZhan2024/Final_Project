# Deployment Lock Validation

- Generated at (UTC): 2026-03-04T23:11:21Z
- API base: `http://127.0.0.1:8000`
- Resident ID: `1`

## Auto checks

| Check | Result | Notes |
|---|---|---|
| API health reachable | yes | GET `http://127.0.0.1:8000/api/health` |
| Predict endpoint exists | yes | `/api/monitor/predict_window` in OpenAPI |
| Active model | TCN | expected: `TCN` |
| Active dataset | caucafall | expected: `caucafall` |
| Active OP | OP-2 | expected: `OP-2` |
| Fall threshold | 0.7099999785423279 | expected around `0.71` |

## Manual replay checks (fill after test)

| Clip type | File | Expected | Observed | Pass/Fail | Notes |
|---|---|---|---|---|---|
| Non-fall | LE2i Office video 27 (replay) | No alert / no repeated false events | No fall alert triggered | PASS | Verified in prior replay run |
| Fall | CAUCAFall fall replay clips (2 clips) | Single clear fall event | Fall event detected on both clips | PASS | Verified in prior replay run |

## Verdict

- [x] PASS: lock profile usable for demo
- [ ] FAIL: requires further tuning/fixes
