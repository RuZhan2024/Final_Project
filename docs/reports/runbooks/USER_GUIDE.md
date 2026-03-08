# User Guide

## What This Software Does

This system detects fall risk from human pose sequences and serves decisions through a backend API + frontend monitor UI.
It supports dataset/model/policy switching for replay and live monitoring workflows.

## Core Features

- Pose-window inference with TCN/GCN models
- Policy-based alerting (`OP-1/OP-2/OP-3`)
- Replay mode for deterministic demo/testing
- Live mode for webcam monitoring (performance-dependent)
- Event timeline + backend event APIs

## Setup (Fresh Environment)

1. Clone repo and create virtualenv.
2. Install Python dependencies:
   - `pip install -r requirements.txt` (or project equivalent)
3. Install frontend dependencies:
   - `cd apps && npm install`
4. Return to repo root.

## Run (Quickstart)

### A) ML artifacts path
Use the locked workflow described in root `README.md` to ensure checkpoints/ops/metrics are available.

### B) Start backend

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)"
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### C) Start frontend

```bash
cd apps
npm start
```

Open: `http://localhost:3000`

## How to Use

### Replay Mode (recommended for demo)

1. Open Monitor page.
2. Select dataset/model/op profile in Settings (recommended: locked deployment profile).
3. Switch to Replay mode.
4. Load a prepared video clip and play.
5. Observe prediction state, `P(fall)`, and event behavior.

### Live Mode (optional)

1. Switch to Live mode.
2. Allow camera access.
3. Monitor capture FPS and prediction stability.

## API Health/Smoke Checks

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/summary
```

## Configuration and Environment

- Backend base URL for frontend:
  - `REACT_APP_API_BASE` (default `http://localhost:8000`)
- CORS origins:
  - controlled by backend config/env
- Runtime profile lock:
  - see `docs/project_targets/DEPLOYMENT_LOCK.md`

## Known Limitations

- Live mode quality is sensitive to client performance (capture FPS, occlusion, camera angle).
- Replay mode is the authoritative demonstration mode for consistent examiner behavior.
- Cross-dataset behavior may vary; deployment claims should follow locked profile docs and evidence map.

## Troubleshooting

- `Failed to fetch` in frontend:
  - verify backend is running and `REACT_APP_API_BASE` is correct
- Unexpected monitor behavior:
  - re-apply locked settings profile
  - confirm correct checkpoint + ops yaml are active
- Inference route mismatch:
  - check `/openapi.json` and frontend request path alignment
