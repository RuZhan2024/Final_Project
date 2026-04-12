# Frontend (React) Demo App

This app is the examiner-facing UI for the fall-detection system.

## Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000` (or another URL)

## Environment

Create `applications/frontend/.env`:

```bash
REACT_APP_API_BASE=http://localhost:8000
```

`REACT_APP_API_BASE` is read in `applications/frontend/src/lib/config.js`.

## Run (Local Demo)

From repo root, run backend:

```bash
source .venv/bin/activate
pip install -r requirements_server.txt
uvicorn applications.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal, run frontend:

```bash
cd applications/frontend
npm install
npm start
```

Open `http://localhost:3000`.

## Replay Clips

The Monitor replay dropdown is populated by the backend from:

- default: `data/replay_clips`
- override: `REPLAY_CLIPS_DIR=/absolute/path/to/replay_clips`

Supported replay extensions:

- `.mp4`
- `.mov`
- `.webm`
- `.m4v`

Quick check:

```bash
curl -s http://localhost:8000/api/replay/clips
```

If this returns an empty `clips` list, the replay selector in the UI will also be empty.

## Quick API Checks

Health:

```bash
curl -s http://localhost:8000/api/health
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
- If browser CORS issues appear, verify backend CORS settings in `applications/backend/app.py`.
- The UI supports `/api/*` and `/api/v1/*` compatibility routes exposed by the backend.
