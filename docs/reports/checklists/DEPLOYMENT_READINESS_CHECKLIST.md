# Deployment Readiness Checklist

Date: 2026-03-02

## API + runtime boot
- [x] FastAPI app imports cleanly (`server.app`, `server.deploy_runtime`).
- [x] `/api/health` endpoint exists and returns `{ok, ts}`.
- [x] `/api/spec` endpoint exists.
- [x] `/api/spec` returns non-empty specs/models/datasets for promoted profiles.

Validation:
- `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
- `PYTHONPATH="$(pwd)/src:$(pwd)" python3 - <<'PY'\nfrom fastapi.testclient import TestClient\nfrom server.app import app\nc=TestClient(app)\nprint(c.get('/api/spec').json())\nPY`

## Inference endpoint readiness
- [x] `POST /api/monitor/predict_window` is implemented.
- [x] Live inference returns 200 for standard dataset payload when specs are present.
- [x] Deploy configs expected at `configs/ops/*.yaml` exist at top level in this workspace.

Validation:
- `PYTHONPATH="$(pwd)/src:$(pwd)" python3 - <<'PY'\nfrom fastapi.testclient import TestClient\nfrom server.app import app\nc=TestClient(app)\nxy=[[[0.0,0.0] for _ in range(17)] for _ in range(48)]\nconf=[[1.0 for _ in range(17)] for _ in range(48)]\np={'session_id':'check','mode':'tcn','dataset_code':'le2i','op_code':'OP-2','target_T':48,'xy':xy,'conf':conf}\nr=c.post('/api/monitor/predict_window',json=p)\nprint(r.status_code, r.json().get('triage_state'))\nPY`

## CORS / frontend integration
- [x] CORS middleware configured with localhost defaults.
- [x] Override supported via `CORS_ALLOWED_ORIGINS`.
- [x] Frontend API base configurable via `REACT_APP_API_BASE`.

Validation:
- Inspect `server/main.py` and `apps/src/lib/config.js`.

## Error handling + observability
- [x] Monitor route now logs non-fatal DB failures with structured warnings.
- [x] Rolling latency logging for monitor inference endpoint (p50/p95 windowed logs) is implemented.
- [x] Client has basic monitor throttling (`lastSentRef`, FPS processing cap).

Validation:
- `rg -n "except .*pass|except \(.*\):\s*pass" server src`

## Profiling and performance evidence
- [x] `scripts/profile_infer.py` exists and can profile I/O or model path.
- [x] Profile reports are present for promoted model profiling.
- [x] Checked-in measured p50/p95 profile artifact exists for promoted model.

Validation:
- `python3 scripts/profile_infer.py --win_dir <win_dir> --ckpt <ckpt> --io_only 0 --runs 100 --out_json artifacts/reports/infer_profile_cpu_local_tcn_le2i.json`

## Demo stability requirements (must-pass)
- [x] Non-empty deploy specs for at least one dataset+model.
- [x] `/api/monitor/predict_window` returns 200 with real checkpoint.
- [x] Frontend README replaced with project-specific instructions.
- [x] DB failure surfaced in logs (not silently ignored).
- [x] One reproducible end-to-end demo command set documented.
