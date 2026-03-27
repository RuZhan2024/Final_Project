# Examiner Demo Runbook (Laptop)

Date: 2026-03-02

## 0) Preconditions
- Python 3.10+ available.
- Node.js + npm available.
- Repo cloned locally.
- At least one deploy spec exists in `configs/ops/*.yaml` with reachable checkpoint.

If `/api/spec` is empty, live inference demo cannot run yet.

## 1) Python setup
```bash
python3 -m venv .venv
source .venv/bin/activate
# For examiner reproducibility, prefer lockfile when available:
pip install -r requirements.lock.txt
# Fallback:
# pip install -r requirements.txt
pip install -r requirements_server.txt
pip install -e . --no-build-isolation
```

## 2) Quick backend sanity checks
```bash
PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py
```
Expected: `[ok] api contract smoke passed`

## 3) Start backend
```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Optional replay clip override:
```bash
REPLAY_CLIPS_DIR=/absolute/path/to/replay_clips \
PYTHONPATH="$(pwd)/src:$(pwd)" uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Expected replay clip location if no override is set:
- `data/replay_clips`

## 4) Health and spec checks (curl)
### Health
```bash
curl -sS http://localhost:8000/api/health
```
Expected JSON keys: `ok`, `ts`

### Deploy spec availability
```bash
curl -sS http://localhost:8000/api/spec
```
Expected: non-empty `specs` and `models` for live inference demo.

### CAUCAFall GCN guardrail check
For CAUCAFall GCN deployments, verify OP fitting used `min_tau_high >= 0.40`.
This avoids low-threshold high-false-alert operating points.

Recalibrate (ckpt-only, no training):
```bash
make fit-ops-gcn-caucafall-force-ckpt ADAPTER_USE=1
```

Validate metrics:
```bash
python scripts/eval_metrics.py \
  --win_dir data/processed/caucafall/windows_eval_W48_S12/test \
  --ckpt outputs/caucafall_gcn_W48S12/best.pt \
  --ops_yaml configs/ops/gcn_caucafall.yaml \
  --out_json outputs/metrics/gcn_caucafall.json \
  --fps_default 23
```

## 5) Minimal inference request (JSON payload)
Route signature uses JSON body (`MonitorPredictPayload`), not multipart upload.

```bash
PYTHONPATH="$(pwd)/src:$(pwd)" python3 - <<'PY'
import requests
xy=[[[0.0,0.0] for _ in range(33)] for _ in range(48)]
conf=[[1.0 for _ in range(33)] for _ in range(48)]
payload={
  "session_id":"examiner-1",
  "mode":"tcn",
  "dataset_code":"le2i",
  "op_code":"OP-2",
  "target_T":48,
  "xy":xy,
  "conf":conf,
  "persist":False
}
r=requests.post("http://localhost:8000/api/monitor/predict_window", json=payload, timeout=30)
print(r.status_code)
print(r.json())
PY
```

Expected:
- HTTP 200
- keys include `triage_state`, `models`, `latency_ms`, `effective_mode`

## 6) Frontend startup
```bash
cd apps
npm install
REACT_APP_API_BASE=http://localhost:8000 npm start
```
Open: `http://localhost:3000`

## 7) Live monitor walk-through
1. Open Monitor page.
2. Allow camera access.
3. Confirm API summary shows online status.
4. Select dataset/model/op if UI exposes controls.
5. Observe `p_fall`, triage state, timeline events.

## 8) Replay monitor walk-through
1. Open Monitor page.
2. Switch to Replay mode.
3. Confirm the clip dropdown is populated from `/api/replay/clips`.
4. Select the recommended non-fall clip and run replay.
5. Select the recommended fall clip and run replay.
6. Confirm timeline/event behavior matches the locked demo notes.

## 9) Optional profiling evidence (for viva/demo appendix)
```bash
cd /path/to/repo
source .venv/bin/activate
python3 scripts/profile_infer.py \
  --profile cpu_local \
  --arch tcn \
  --win_dir data/processed/le2i/windows_eval_W48_S12/test \
  --ckpt outputs/le2i_tcn_W48S12/best.pt \
  --io_only 0 \
  --runs 100 \
  --out_json artifacts/reports/infer_profile_cpu_local_tcn_le2i.json
```

## 10) Troubleshooting quick map
- `ModuleNotFoundError: server` when running scripts:
  - use `PYTHONPATH="$(pwd)/src:$(pwd)"` prefix.
- `/api/spec` empty:
  - missing `configs/ops/*.yaml` and/or missing checkpoint files.
- Replay dropdown empty:
  - verify `data/replay_clips` exists or set `REPLAY_CLIPS_DIR`.
  - check `curl -sS http://localhost:8000/api/replay/clips`.
- Monitor predict 404 for dataset/model:
  - deploy spec key not discovered.
- DB-related warnings/errors:
  - verify DB env vars or run demo with non-persistent mode (`persist=false`).
