# Replay + Live Acceptance Report

- Generated at (UTC): 2026-03-06T01:18:44Z
- API base: `http://127.0.0.1:8000`
- Resident ID: `1`

## Auto checks

| Check | Result | Notes |
|---|---|---|
| API health reachable | no | GET `http://127.0.0.1:8000/api/health` |
| Deploy specs available | 0 | GET `http://127.0.0.1:8000/api/spec` |
| Active model | N/A | expected test profile: `TCN` or `GCN` |
| Active dataset | N/A | expected test profile: `caucafall` |
| Active OP | N/A | expected test profile: `OP-2` |
| MC enabled | False | usually `True` for monitor |
| OP live_guard present (TCN/GCN OP1/2/3) | yes | checks `configs/ops/*_caucafall.yaml` |
| Monitor page summary polling enabled | no | expected: `no` |

## Manual replay acceptance (fixed clips)

| Case | Mode/Profile | Expected | Observed | Pass/Fail | Notes |
|---|---|---|---|---|---|
| Non-fall replay | caucafall + TCN + OP-2 | no fall event |  |  |  |
| Fall replay | caucafall + TCN + OP-2 | clear fall event |  |  |  |
| Non-fall replay | caucafall + GCN + OP-2 | no fall event |  |  |  |
| Fall replay | caucafall + GCN + OP-2 | clear fall event |  |  |  |

## Manual live acceptance (high-performance machine)

| Case | Profile | Expected | Observed | Pass/Fail | Notes |
|---|---|---|---|---|---|
| Standing / ADL | caucafall + TCN + OP-2 | no repeated false fall |  |  |  |
| Controlled fall | caucafall + TCN + OP-2 | fall detected |  |  |  |
| Standing / ADL | caucafall + GCN + OP-2 | no repeated false fall |  |  |  |
| Controlled fall | caucafall + GCN + OP-2 | fall detected |  |  |  |

## Acceptance gate

- [ ] Replay baseline stable and repeatable
- [ ] Live baseline acceptable on target hardware
- [ ] Ready for demo lock
