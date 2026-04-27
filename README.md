# Safe Guard Fall Detection

This repository is a pose-based fall
detection system. It provides a runnable frontend and backend that demonstrate
the core project idea: detecting fall-related behaviour from pose-window input
and presenting the result through a monitoring interface.

## Quick Links

- GitHub repository: project source repository for record-keeping and marking
  [Open GitHub repository](https://github.com/RuZhan2024/Final_Project)
- Final tagged snapshot: `assignment3-final`
- Live project: deployed web app for interactive evaluation
  [Open live project](https://fall-detection-frontend.onrender.com/)
- Recorded demo: short walkthrough video of the deployed system
  [Open recorded demo](https://goldsmithscollege-my.sharepoint.com/:v:/g/personal/gru001_campus_goldsmiths_ac_uk/IQBrDgxxknESS6ZryqXdWqG_AVKusWzPt3W8FpREDEV13y4?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=QO0ipZ)
- Replay clips: clips used in `Monitor` -> `Replay Mode`
  [Open replay clips](https://goldsmithscollege-my.sharepoint.com/:f:/g/personal/gru001_campus_goldsmiths_ac_uk/IgBguPeopCB5SKbuR_vhvGBBAYZKJjXrEhQrDEm9ZfMWehY?e=yuLoZd)
- Realtime recordings: self-recorded realtime test videos; `.mp4` trimmed,
  `.mov` full-length
  [Open realtime recordings](https://goldsmithscollege-my.sharepoint.com/:f:/g/personal/gru001_campus_goldsmiths_ac_uk/IgAqe_2PLr2tQrwpqtW7skyGAaq1mAmS2MeDGhBJZ6rAFqw?e=CX4g7S)

## Recommended Assessment Path

Use the project in this order for the clearest evaluation path:

1. Open the live project to review the deployed interface and available pages.
2. Watch the recorded demo for a guided walkthrough of the end-to-end system.
3. Review the replay clips used in `Monitor` -> `Replay Mode`.
4. Review the self-recorded realtime videos as supporting evidence for the
   realtime monitoring path.
5. Use [Quick Start](#quick-start) only if a local run is needed.

## What This Software Does

Safe Guard monitors human pose windows and classifies whether the observed
behaviour is consistent with a fall, an uncertain fall-like event, or normal
activity. The system exposes that behaviour through a web UI with live and
replay monitoring modes.

## Core Features Implemented

- live monitoring page with browser camera input
- replay monitoring page for end-to-end demonstration with promoted clips
- FastAPI backend for runtime inference and monitor responses
- React frontend for dashboard, monitor, events, and settings workflows
- prediction timeline and runtime model information display
- settings flow for dataset, model, operating point, and notification toggles
- event history and skeleton replay viewing for stored events
- optional persistent workflows through SQLite or Docker/MySQL mode

## System Requirements

### Recommended local demo path

- Python `3.10+`
- Python `3.10.x` recommended for dependency compatibility on a fresh machine
- `pip`, `setuptools`, and `wheel` are upgraded automatically inside `.venv`
- Node.js `22.x` with npm
- Node `22.22.0` recommended for frontend parity
- npm `10.x` recommended

### Full persistent-system path

- Docker Desktop or Docker Engine with `docker compose`

## Quick Start

Use this path for the fastest local run.

1. From the repository root, run:

   ```bash
   make bootstrap-dev
   ```

   The bootstrap script automatically looks for a compatible Python `3.10+`
   interpreter, including common Homebrew and `pyenv` installations.

   If your machine has multiple Python versions installed, prefer:

   ```bash
   PY_BIN=python3.10 make bootstrap-dev
   ```

   If you use `pyenv`, make sure `python3.10` resolves to an installed Python
   `3.10.x` interpreter before running the command.

2. Open:
   - frontend: `http://localhost:3000`
   - backend health check: `http://127.0.0.1:8000/api/health`
3. In the frontend, go to `Monitor`.
4. Select `Replay Mode`.
5. Choose a replay clip.
6. Click `Play Replay`.
7. Observe:
   - current prediction output
   - prediction timeline updates
   - runtime model information

Recommended runtime preset:

- dataset: `caucafall`
- model: `TCN`
- operating point: `OP-2`

This path does not require MySQL, external credentials, or raw dataset
downloads.

This is the recommended local setup path for assessment and handover.

To stop the local run:

```bash
make stop-dev
```

## Setup And Run

Run all commands from the repository root.

### Option A: Recommended local demo mode

See [Quick Start](#quick-start).

This option is the recommended local demo path and uses:

- `make bootstrap-dev` for first-time setup and startup
- `make dev` if `.venv` and `applications/frontend/node_modules` already exist

Notes:

- this mode does not require MySQL
- the intended local demo database backend is SQLite via `DB_BACKEND=sqlite`
- no login or test credentials are required
- no raw dataset download is required for the default demo path

### Option B: Full persistent-system mode

Use this when database-backed persistence needs to be demonstrated.

```bash
docker compose up
```

Open:

- frontend: `http://localhost:3000`
- backend health check: `http://127.0.0.1:8000/api/health`

This starts:

- `frontend`
- `backend`
- `mysql`

If port `3306` is already in use:

```bash
MYSQL_PORT=3307 docker compose up
```

## How To Use The System

### Monitor page

Use this page to demonstrate the main project feature.

- `Realtime Mode`
  - starts browser-camera monitoring
  - shows the current prediction, pose overlay, and runtime status
- `Replay Mode`
  - loads a replay clip and runs the end-to-end detection path
  - is the recommended mode for a stable end-to-end demo

Expected result:

- the prediction state updates as the clip or live input progresses
- the timeline bar records recent prediction windows
- the model information card shows runtime thresholds and monitor parameters

### Dashboard page

Use this page to show:

- system summary
- whether monitoring is enabled
- latest backend latency
- whether the frontend can reach the backend

### Events page

Use this page to show:

- stored events
- filterable event history
- review status updates
- skeleton replay for stored event clips when available

### Settings page

Use this page to show:

- dataset selection
- model selection
- operating point selection
- notification toggle behaviour
- caregiver/Telegram configuration

## Validation

### Basic runtime check

Backend health endpoint:

```text
http://127.0.0.1:8000/api/health
```

### Project checks

From the repository root:

```bash
git diff --check
python3 -m compileall applications/backend ml/src/fall_detection
cd applications/frontend && npm run typecheck
cd ../..
bash ops/scripts/run_canonical_tests.sh all
```

These checks are the recommended validation path before a release or handover.

## Dependencies

Primary project dependencies include:

- FastAPI backend
- React + TypeScript frontend
- NumPy / PyTorch / MediaPipe for the pose and inference pipeline
- SQLite for the lightweight demo path
- MySQL for the full persistent Docker path

Dependency files:

- `requirements.txt`
- `requirements-dev.txt`
- `requirements_server.txt`
- `applications/frontend/package.json`

Recommended toolchain versions for assessment:

- Python `3.10.x`
- Node.js `22.22.0`
- npm `10.x`

## Demo Assets

If the live project is unavailable, use the local setup path above.

### Evidence Notes

The replay clips are the prepared demonstration clips used by the frontend
`Replay Mode` workflow.

The realtime recordings are the self-recorded realtime videos created for this
project.

Suggested demo clips:

- `realtime/realtime_adl_submission.mp4`
- `realtime/realtime_fall_submission.mp4`

Recording formats in the realtime recordings folder:

- `.mp4` files are trimmed versions for quick review
- `.mov` files are full-length recordings

The replay clips can be shown through the running frontend in `Monitor` ->
`Replay Mode`, while the realtime recordings provide separate evidence for the
realtime monitoring path.

## Repository Scope

This repository contains the working software artefact and the supporting code
needed to run and verify it.

Included:

- runnable frontend
- runnable backend
- runtime configuration files
- promoted runtime/demo assets
- validation scripts and tests
- training/evaluation code retained for project traceability

Not required for the default demonstration path:

- full retraining from raw datasets
- raw dataset redistribution
- external production notification credentials

## Repository Layout

```text
applications/
  backend/      FastAPI backend, runtime services, routes, and repositories
  frontend/     React frontend, pages, hooks, and monitoring UI
ml/             model, calibration, evaluation, and training code
ops/            validation scripts and operational utilities
configs/        runtime configuration and operating-point presets
artifacts/      promoted demo assets used by the runnable system
qa/             integration and contract-style test coverage
```

This layout separates the runnable application from the retained ML and ops
code so the main demo path stays simple while the project remains traceable.

## Datasets

Supported dataset codes:

- `le2i`
- `urfd`
- `caucafall`
- `muvim`

Dataset availability summary:

- `CAUCAFall`
  - primary benchmark and deployment-target dataset in this project
  - not redistributed in this repository
- `LE2i`
  - comparative dataset
  - not redistributed in this repository
- `URFD`
  - supported by the codebase
  - not redistributed in this repository
- `MUVIM`
  - restricted-access exploratory dataset
  - not redistributed in this repository

## Known Limitations

- this system is a focused prototype, not a fully productized care platform
- the default quick-start path demonstrates the runnable monitoring system, not
  full model retraining from raw datasets
- raw datasets are not redistributed, so full reproduction requires separately
  acquired data
- database-backed workflows are implemented, but require SQLite or the
  Docker/MySQL runtime mode to be configured
- Telegram notifications and generated summaries require external credentials
  and deployment configuration
- SMS and phone-call escalation are not implemented in this prototype and are
  treated as future work
- replay mode is intended for demonstration and interface review, not as a
  claim of clinically validated deployment
- the system demonstrates end-to-end feasibility and core project behaviour,
  but it is not presented as a clinically validated or deployment-ready safety
  product

## Deployment Notes

The backend supports:

- SQLite on a local machine
- SQLite on a persistent Render disk
- MySQL in Docker Compose mode

Relevant configuration files:

- `docker-compose.yml`
- `Dockerfile.backend`
- `render.yaml`

For frontend Render parity checks:

```bash
make frontend-render-check
```
