# Safe Guard Fall Detection

This repository contains the working software artefact for a pose-based fall
detection system. It includes a runnable frontend and backend that demonstrate
the core project idea: detecting fall-related behaviour from pose-window input
and presenting the result through a monitoring interface.

## Overview

The system detects fall-related behaviour from pose-window input and shows the
result through a web interface. Users can run the system locally, play a
promoted replay clip, and observe prediction output, timeline updates, and
runtime model information.

## Demo Access and Evidence

### Deployed frontend

- live monitor page: `https://fall-detection-frontend.onrender.com/monitor`

If the deployed frontend is unavailable, use the local setup path below.

### Shared demo assets

- SharePoint folder: `https://goldsmithscollege-my.sharepoint.com/:f:/g/personal/gru001_campus_goldsmiths_ac_uk/IgCUcyLaZ3TwTp0WaWv_0KKAAT1UlGpgH_1nLZhxLxTbLMw?e=a0E9Zd`

Recommended evidence files:

- `realtime/realtime_adl_submission.mp4`
- `realtime/realtime_fall_submission.mp4`

These are the trimmed realtime evidence clips and are the recommended versions
for quick review. Full-length recordings are also provided in the same folder:

- `realtime/realtime_adl.mov`
- `realtime/realtime_fall.mov`

The `replay/` folder contains the replay clips used as supporting evidence for
the authenticity of the demo inputs. Replay behaviour is best viewed through
the running frontend on the `Monitor` page in `Replay Mode`.

## Quick Start

### Fastest way to run

From the repository root:

```bash
make bootstrap-dev
```

Then open:

- frontend: `http://localhost:3000`
- backend health check: `http://127.0.0.1:8000/api/health`

### Fastest way to verify the core feature

1. Open `http://localhost:3000`
2. Go to `Monitor`
3. Select `Replay Mode`
4. Choose a replay clip
5. Click `Play Replay`
6. Observe:
   - current prediction output
   - prediction timeline updates
   - runtime model information

### Notes

- no login or test credentials are required
- no raw dataset download is required for the default demo path
- no MySQL setup is required for the default demo path
- the recommended runtime demo preset is:
  - dataset: `caucafall`
  - model: `TCN`
  - operating point: `OP2`

## Release Snapshot

This repository should be associated with the final tagged release snapshot.
Update this section with the final tag once it has been created.

## Core Features Implemented

- pose-based fall detection demonstrated through a working software system
- React frontend for monitoring predictions
- FastAPI backend for online inference
- replay monitoring for end-to-end demonstration using promoted replay clips
- realtime monitoring using browser camera input
- runtime model information panel
- prediction timeline updates in the frontend
- optional event history workflow when database-backed mode is enabled
- configurable runtime profiles for dataset, model, and operating point selection

## Included in This Repository

This repository is focused on the working software artefact and its supporting
runtime assets, configuration, and validation utilities.

Included:

- runnable frontend
- runnable backend
- configuration files
- promoted runtime assets for demo and review
- test scripts and release checks
- source code for the software system

Retained for research traceability, but not required for the default demo path:

- dataset-dependent extraction scripts
- preprocessing scripts
- training scripts
- evaluation scripts

## Runtime Modes

This repository supports two main runtime modes.

### 1. Lightweight demo mode

Recommended for local demonstration and recording.

Includes:

- frontend
- backend
- promoted runtime assets
- no MySQL required

Run with:

```bash
make bootstrap-dev
```

### 2. Full persistent-system mode

Use this only if database persistence also needs to be demonstrated.

Includes:

- frontend
- backend
- MySQL

Run with:

```bash
docker compose up
```

## Requirements

### Local demo mode

- Python `3.10+`
- Node.js / npm
- Node `22.22.0` recommended for frontend parity

### Full persistent mode

- Docker Desktop or Docker Engine with `docker compose`

## Setup and Run

Run all commands from the repository root.

### Option A: Recommended local demo

```bash
make bootstrap-dev
```

This command:

- creates `.venv` if missing
- installs Python dependencies if missing
- installs frontend dependencies if missing
- starts the backend on `127.0.0.1:8000`
- starts the frontend on `127.0.0.1:3000`

Open:

- frontend: `http://localhost:3000`
- backend health: `http://127.0.0.1:8000/api/health`

To stop:

```bash
make stop-dev
```

Or, if the current run is attached to your terminal:

```bash
Ctrl-C
```

Notes:

- use `make dev` if `.venv` and `applications/frontend/node_modules` already exist
- this mode does not require MySQL
- DB-backed features fall back gracefully when DB is unavailable
- for cloud deployment, the backend also supports `DB_BACKEND=sqlite`

### Option B: Full system with MySQL persistence

```bash
docker compose up
```

Open:

- frontend: `http://localhost:3000`
- backend health: `http://127.0.0.1:8000/api/health`

This starts:

- `frontend`
- `backend`
- `mysql`

If port `3306` is already in use:

```bash
MYSQL_PORT=3307 docker compose up
```

Makefile wrappers are also available:

```bash
make compose-up
make compose-down
```

## How to Use the System

### Recommended path: Replay Mode

This is the primary end-to-end workflow in the submitted software artefact.

1. Start the system with `make bootstrap-dev`
2. Open `http://localhost:3000`
3. Go to `Monitor`
4. Switch to `Replay Mode`
5. Select a replay clip from the dropdown
6. Click `Play Replay`
7. Observe:
   - predicted fall-related output
   - timeline updates
   - runtime model panel

### Optional path: Realtime Mode

1. Stay on the `Monitor` page
2. Switch to `Realtime Mode`
3. Allow camera access in the browser
4. Click `Start Realtime`
5. Observe the live prediction output and timeline updates

### Optional path: Event History

1. Start the full database-backed system with `docker compose up`
2. Open the `Event History` page
3. Review stored events and related status data

## Sample Inputs and Access

- no login is required
- no test account is required
- the recommended sample inputs are the promoted replay clips available through `Replay Mode`
- the default review path does not require raw training datasets

## Environment Variables and Configuration

The repository includes:

- `.env.example` as the documented variable template
- `applications/backend/config.py` as the backend config parser

For the default local demo path, the provided safe defaults are intended to
minimise setup friction.

Important configuration groups:

### 1. App defaults

- `APP_BASE_URL`
- `APP_TIMEZONE`
- `CORS_ALLOWED_ORIGINS`
- `SESSION_TTL_S`
- `SESSION_MAX_STATES`

### 2. Runtime paths

- `SQLITE_PATH`
- `SAFE_GUARD_SQLITE_PATH`
- `EVENT_CLIPS_DIR`

### 3. Database mode

- `DB_BACKEND`
- `DB_HOST`
- `DB_PORT`
- `DB_USER`
- `DB_PASS`
- `DB_NAME`

### 4. Optional integrations

- `TELEGRAM_BOT_TOKEN`
- `TWILIO_*`
- `RESEND_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

Configuration rules:

- keep real secrets out of git
- use private local env files or deployment secret stores for real credentials
- treat `.env.example` as documentation, not as a real secret store
- the default local demo does not require third-party notification credentials

## Repository Layout

```text
.
├── applications/
│   ├── backend/                 # FastAPI backend
│   └── frontend/                # React frontend
├── ml/
│   └── src/fall_detection/      # Core ML package
├── ops/
│   ├── configs/                 # labels, splits, ops, delivery profiles
│   ├── scripts/                 # utility and orchestration scripts
│   └── deploy_assets/           # promoted checkpoints and replay clips
├── qa/
│   └── tests/                   # maintained test suite
├── data/                        # raw/interim/processed datasets
├── outputs/                     # local and experimental outputs
└── Makefile                     # main project entrypoint
```

## Common Commands

List available targets:

```bash
make help
```

Start and stop the software:

```bash
make bootstrap-dev
make dev
make stop-dev
make compose-up
make compose-down
```

## Validation

Run the release checks:

```bash
make release-check
```

This currently verifies:

- git status visibility
- release boundary script availability
- Python compileability for maintained backend, ML, and ops code
- frontend production build

Recommended test subsets:

```bash
./ops/scripts/run_canonical_tests.sh torch-free
./ops/scripts/run_canonical_tests.sh contract
./ops/scripts/run_canonical_tests.sh monitor
```

## Demo Recording

For a short software demonstration, show the following:

1. start the system
2. open the frontend
3. go to `Monitor`
4. select `Replay Mode`
5. choose a replay clip
6. click `Play Replay`
7. show the input selection
8. show the prediction output
9. show the timeline updates
10. show the runtime model panel

This demonstrates one meaningful feature working end-to-end.

## Scope Notes

The default usage path is the runnable software and demo path, not full ML
reproduction. In practice:

- `make bootstrap-dev` and `make dev` do not require raw training datasets
- `docker compose up` is only needed if database persistence must also be shown
- raw datasets, `data/interim`, and `data/processed` are not distributed with this submission
- full data extraction, preprocessing, training, and evaluation require separately acquired datasets
- full ML reproduction is optional and requires separately arranged dataset access

## Runtime Assets

This submission distinguishes between experimental outputs and promoted runtime
assets.

### Promoted runtime assets

- `ops/deploy_assets/manifest.json`
  - source of truth for shipped runtime assets
- `ops/deploy_assets/checkpoints/`
  - promoted checkpoints approved for runtime loading
- `ops/deploy_assets/replay_clips/`
  - promoted replay clips approved for reviewer and demo use
- `ops/configs/ops/*.yaml`
  - canonical runtime operating-point profiles

### Experimental outputs

- `outputs/`
  - local or experimental training and evaluation outputs
  - not treated as shipped runtime assets unless explicitly promoted

## Optional Research Workflows

These workflows are not part of the default demo path. They are retained for
research traceability.

### Prepare data

```bash
make pipeline-data-caucafall
make pipeline-data-le2i
```

No-extract variants:

```bash
make pipeline-caucafall-noextract
make pipeline-le2i-noextract
```

### Train models

```bash
make train-tcn-caucafall
make train-gcn-caucafall
```

### Fit operating points and evaluate

```bash
make fit-ops-caucafall
make fit-ops-gcn-caucafall
make eval-caucafall
make eval-gcn-caucafall
```

### Run auto pipelines

```bash
make pipeline-auto-tcn-caucafall ADAPTER_USE=1
make pipeline-auto-gcn-caucafall ADAPTER_USE=1
```

If extraction should be included:

```bash
AUTO_DO_EXTRACT=1 make pipeline-auto-tcn-caucafall ADAPTER_USE=1
```

## Deployment Notes

Recommended cloud deployment shape:

- frontend on Render static hosting
- backend on Render web service
- SQLite on a persistent disk

Suggested backend environment variables:

```bash
DB_BACKEND=sqlite
SQLITE_PATH=/var/data/cloud_demo.sqlite3
SAFE_GUARD_ENABLED=1

TELEGRAM_BOT_TOKEN=your_telegram_bot_token
CAREGIVER_TELEGRAM_CHAT_ID=your_telegram_chat_id

AI_REPORTS_ENABLED=1
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
OPENAI_TIMEOUT_S=12

APP_BASE_URL=https://your-frontend-domain.onrender.com
CORS_ALLOWED_ORIGINS=https://your-frontend-domain.onrender.com
```

Frontend parity check before deployment:

```bash
make frontend-render-check
```

This runs:

- `npm ci`
- `npm run build`

## Datasets

Supported dataset codes:

- `le2i`
- `urfd`
- `caucafall`
- `muvim`

Expected default raw roots:

- `data/raw/LE2i`
- `data/raw/UR_Fall_clips`
- `data/raw/CAUCAFall`
- `data/raw/MUVIM`

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

- this submission is a focused prototype, not a fully productized care platform
- the default reviewer path demonstrates the runnable monitoring system, not full model retraining from raw datasets
- raw datasets are not redistributed, so full reproduction requires separately acquired data
- some features are environment-dependent, including database persistence, Telegram notifications, and generated summaries
- replay mode is intended for demonstration and interface review, not as a claim of clinically validated deployment
- the system demonstrates end-to-end feasibility and core project behaviour, but it is not presented as a clinically validated or deployment-ready safety product

## Project Scope

This repository is intended to provide:

- a functioning software prototype
- a clear demonstration of the project's core feature
- at least one meaningful end-to-end workflow
- runnable code with setup instructions
- a user guide explaining how to run and use the system
- clear acknowledgement of limitations
