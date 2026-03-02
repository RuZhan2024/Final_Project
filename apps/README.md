# Frontend (React) Demo App

This app is the examiner-facing UI for the fall-detection system.

## Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000` (or another URL)

## Environment

Create `apps/.env`:

```bash
REACT_APP_API_BASE=http://localhost:8000
```

`REACT_APP_API_BASE` is read in `apps/src/lib/config.js`.

## Run (Local Demo)

From repo root, run backend:

```bash
source .venv/bin/activate
pip install -r requirements_server.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal, run frontend:

```bash
cd apps
npm install
npm start
```

Open `http://localhost:3000`.

## Quick API Checks

Health:

```bash
curl -s http://localhost:8000/health
```

Deploy specs:

```bash
curl -s http://localhost:8000/api/spec
```

Live monitor inference endpoint used by UI:

```bash
curl -s -X POST http://localhost:8000/api/monitor/predict_window \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","dataset_code":"le2i","mode":"tcn","target_T":48,"xy":[[[0,0]]],"conf":[[1.0]]}'
```

## Notes

- Default ports:
  - frontend: `3000`
  - backend: `8000`
- If browser CORS issues appear, verify backend CORS settings in `server/app.py`.
- The UI supports `/api/*` and `/api/v1/*` compatibility routes exposed by the backend.
