# Deployment Default Profile (TCN-first)

## Objective
- Keep production/demo behavior stable and low-risk.
- Use `TCN` as default auto-alert model.
- Keep `HYBRID` available for feature demonstration only.

## Default Policy
- `active_dataset_code`: `caucafall`
- `active_model_code`: `TCN`
- `active_op_code`: `OP-2`
- `mc_enabled`: `true`
- `mc_M`: `10`
- `mc_M_confirm`: `25`

Reason:
- Current GCN checkpoint tradeoff is unstable (`high recall -> high FA`, `low FA -> low recall`).
- TCN path is currently the most deployable.

## One-command API Lock (when backend is running)
```bash
curl -s -X PUT "http://127.0.0.1:8000/api/settings?resident_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "active_dataset_code": "caucafall",
    "active_model_code": "TCN",
    "active_op_code": "OP-2",
    "mc_enabled": true,
    "mc_M": 10,
    "mc_M_confirm": 25
  }'
```

Verify:
```bash
curl -s "http://127.0.0.1:8000/api/settings?resident_id=1"
```

Expected key fields:
- `system.active_model_code = "TCN"`
- `system.active_dataset_code = "caucafall"`
- `system.active_op_code = "OP-2"`

## Hybrid Demo Switch (manual, reversible)
```bash
curl -s -X PUT "http://127.0.0.1:8000/api/settings?resident_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "active_model_code": "HYBRID",
    "active_dataset_code": "caucafall",
    "active_op_code": "OP-2"
  }'
```

Revert to default:
```bash
curl -s -X PUT "http://127.0.0.1:8000/api/settings?resident_id=1" \
  -H "Content-Type: application/json" \
  -d '{"active_model_code":"TCN","active_dataset_code":"caucafall","active_op_code":"OP-2"}'
```

## Runtime Acceptance Check
Send a monitor payload and verify backend mode fields:
```bash
curl -s -X POST "http://127.0.0.1:8000/api/monitor/predict_window" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "tcn",
    "dataset_code": "caucafall",
    "op_code": "OP-2",
    "target_T": 48,
    "target_fps": 23,
    "raw_t_ms": [0, 43, 86, 129],
    "raw_xy": [[[0,0]],[[0,0]],[[0,0]],[[0,0]]],
    "raw_conf": [[1],[1],[1],[1]]
  }'
```

Expected:
- `effective_mode: "tcn"` for default profile.
- `triage_state` exists.
- no 5xx error.
