# RELEASE_RUNBOOK

## 1) Clean Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_server.txt
pip install -e . --no-build-isolation
```

Frontend:
```bash
cd apps
npm install
cd ..
```

## 2) Required Environment Variables

Backend (optional overrides):
```bash
export CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

Frontend:
```bash
export REACT_APP_API_BASE=http://localhost:8000
```

## 3) Minimal No-Training Pipeline Check

Use existing artifacts only (no retraining):
```bash
python -m compileall .
python -c "import fall_detection; import server.app"
make -Bn fit-ops-gcn-le2i
make -Bn eval-gcn-le2i
```

## 4) Start Backend

```bash
source .venv/bin/activate
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Expected startup:
- FastAPI app starts without import errors.
- Health endpoints respond.

## 5) API Smoke Checks

Health:
```bash
curl -s http://localhost:8000/api/health
curl -s http://localhost:8000/api/v1/health
```

Predict window (JSON):
```bash
curl -s -X POST http://localhost:8000/api/monitor/predict_window \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"demo",
    "dataset_code":"le2i",
    "mode":"tcn",
    "target_T":48,
    "xy":[[[0.0,0.0],[0.1,0.1]]],
    "conf":[[1.0,1.0]],
    "persist":false
  }'
```

## 6) Start Frontend

```bash
cd apps
REACT_APP_API_BASE=http://localhost:8000 npm start
```

Open `http://localhost:3000`.

## 7) Troubleshooting
- If frontend can’t reach API: verify `REACT_APP_API_BASE` and backend CORS env var.
- If inference fails for GCN two-stream models with rich feature sets: apply P0 runtime split patch from `PATCH_PLAN.md`.
- If `pytest -q` aborts: this environment currently shows a torch import abort during collection; treat as environment/ABI issue and run targeted smoke checks.
