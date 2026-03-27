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
4. Select a prepared replay clip from the dropdown and play.
5. Observe prediction state, `P(fall)`, and event behavior.

Replay clip source:

- Backend discovers replay videos from `data/replay_clips` by default.
- Override with `REPLAY_CLIPS_DIR=/absolute/path/to/replay_clips` before starting `uvicorn`.
- Supported extensions: `.mp4`, `.mov`, `.webm`, `.m4v`.
- The Monitor UI loads these clips via `/api/replay/clips`.

### Exact Recommended Replay Inputs

Use these exact cases for examiner-facing replay checks:

- Non-fall replay:
  - LE2i identifier: `Office__video__27_`
  - Stable internal artifact path: `data/interim/le2i/pose_npz/Office__video__27_.npz`
  - Expected behavior: no fall alert / no repeated false events
- Fall replay 1:
  - CAUCAFall identifier: `Subject.1__Fall_forward__80e1655b`
  - Stable internal artifact path: `data/interim/caucafall/pose_npz/Subject.1__Fall_forward__80e1655b.npz`
  - Expected behavior: one clear fall event
- Fall replay 2:
  - CAUCAFall identifier: `Subject.4__Fall_backwards__2ea12ecd`
  - Stable internal artifact path: `data/interim/caucafall/pose_npz/Subject.4__Fall_backwards__2ea12ecd.npz`
  - Expected behavior: one clear fall event

The locked demo evidence in `artifacts/reports/deployment_lock_validation.md` uses `LE2i Office video 27` as the non-fall case and two CAUCAFall fall clips as the positive cases. Name the replay video files with the same identifiers so they are easy to select from the Monitor dropdown.

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
