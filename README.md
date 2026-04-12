# Fall Detection v2

An end-to-end fall detection project with:

- pose-based preprocessing and window generation
- TCN and GCN model training/evaluation
- a FastAPI backend for online inference
- a React frontend for live and replay monitoring

The canonical Python package code lives in `ml/src/fall_detection`.
The repository now uses a normalized top-level structure:

- `applications/frontend` for the React client
- `applications/backend` for the FastAPI service
- `ml/src/fall_detection` for the ML package
- `ops/` for operational configs, scripts, and deploy assets
- `qa/tests` for the maintained test suite

## Submission Overview

This repository is prepared to be reviewed in two modes:

1. lightweight demo mode
   - frontend + backend
   - no MySQL required
   - lowest setup friction
2. full persistent-system mode
   - frontend + backend + MySQL
   - Docker-backed database persistence
   - preferred when the database layer also needs to be demonstrated

For the current monitor demo, the recommended review path is:

- dataset: `caucafall`
- model: `TCN`
- operating point: `OP2`

That is the main delivery profile used in the project's online validation and four-folder custom video verification.
That is the preferred live demonstration profile because it is the strongest bounded online replay row in the current fixed 24-clip matrix. It should be treated as the runtime demo preset, not as a replacement for the broader evidence pack.

## Reviewer Note

The default reviewer path for this repository is the runnable system/demo path, not the full ML reproduction path.

- `make bootstrap-dev`, `make dev`, and `docker compose up` do not require raw training datasets
- raw datasets, `data/interim`, and `data/processed` are not distributed with this submission
- dataset-dependent extraction, preprocessing, training, and evaluation commands are retained for research traceability, but they require separately acquired datasets
- as a result, reviewers are not expected to rerun the full ML pipeline unless dataset access has been arranged separately

## Quick Start

Run all commands from the repository root.

### Option A: One-command local demo

```bash
make bootstrap-dev
```

What it does:

- creates `.venv` if missing
- installs Python dependencies if missing
- installs frontend dependencies if missing
- starts backend on `127.0.0.1:8000`
- starts frontend on `127.0.0.1:3000`

Open:

- frontend: `http://localhost:3000`
- backend health: `http://127.0.0.1:8000/api/health`

Notes:

- use `make dev` if `.venv` and `applications/frontend/node_modules` already exist
- this mode does not require MySQL
- DB-backed features fall back gracefully when DB is unavailable
- for cloud deployment, the backend now also supports `DB_BACKEND=sqlite`

Stop the local demo:

```bash
make stop-dev
```

```bash
# if `make bootstrap-dev` or `make dev` is running in the current terminal
Ctrl-C
```

### Option B: One-command full system with persistent MySQL

```bash
docker compose up
```

Open:

- frontend: `http://localhost:3000`
- backend health: `http://127.0.0.1:8000/api/health`

What it starts:

- `frontend`
- `backend`
- `mysql`

Notes:

- MySQL data is persisted in the `mysql_data` Docker volume
- the database is initialized from `applications/backend/create_db.sql`
- if host port `3306` is already occupied, move only the exposed MySQL port:

```bash
MYSQL_PORT=3307 docker compose up
```

Makefile wrappers are also available:

```bash
make compose-up
make compose-down
```

## Requirements

### Local dev mode

- Python 3.10+
- Node.js / npm

### Full Docker mode

- Docker Desktop or Docker Engine with `docker compose`

## Repository Layout

```text
.
├── applications/
│   ├── backend/                 # FastAPI backend
│   └── frontend/                # React frontend
├── ml/
│   └── src/fall_detection/      # Core package: data, training, eval, deploy runtime
├── ops/
│   ├── configs/                 # labels, splits, ops, delivery profiles
│   ├── scripts/                 # utility and orchestration scripts
│   └── deploy_assets/           # promoted runtime-approved checkpoints and replay clips
├── qa/
│   └── tests/                   # smoke and contract tests
├── data/                        # raw/interim/processed datasets
├── outputs/                     # experimental/local training outputs (not promoted runtime assets)
└── Makefile                     # main project entrypoint
```

## Common Workflows

List available targets:

```bash
make help
```

### Start the system

```bash
make bootstrap-dev
make dev
make stop-dev
make compose-up
make compose-down
```

### Stop the local dev servers

```bash
make stop-dev
```

```bash
# if the current local run is attached to your terminal
Ctrl-C
```

### Prepare data

These commands are optional research-reproduction workflows. They require separately acquired datasets and are not part of the default reviewer/demo path.

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

If raw extraction should be included:

```bash
AUTO_DO_EXTRACT=1 make pipeline-auto-tcn-caucafall ADAPTER_USE=1
```

## Validation Before Submission

Run the static release checks:

```bash
make release-check
```

This currently verifies:

- git status visibility
- release boundary script availability
- Python compileability for `ml/src/fall_detection`, `applications/backend`, and `ops/scripts`
- frontend production build

Recommended contract/smoke test subset:

```bash
./ops/scripts/run_canonical_tests.sh torch-free
```

Server app / contract subset:

```bash
./ops/scripts/run_canonical_tests.sh contract
```

Torch-dependent monitor subset:

```bash
./ops/scripts/run_canonical_tests.sh monitor
```

## Runtime Profiles and Delivery Notes

Current online deployment profiles are loaded from:

- `ops/configs/ops/tcn_caucafall.yaml`
- `ops/configs/ops/gcn_caucafall.yaml`
- `ops/configs/ops/tcn_le2i.yaml`
- `ops/configs/ops/gcn_le2i.yaml`

Important delivery note:

## Render Deployment Notes

Recommended cloud deployment shape:

- frontend on Render static hosting
- backend on Render web service
- app data on SQLite with a persistent disk
- Telegram caregiver notification
- optional generated event summary via Gemini API

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

Operational note:

- caregiver `telegram_chat_id` can come from the app database or the env fallback
- Telegram and summary-provider credentials should be stored only in Render environment variables
- SMS / phone / email escalation are future-work channels, not the current implemented delivery path

## Configuration and Secrets

Backend runtime configuration is centered on:

- `.env.example` as the documented variable inventory
- `applications/backend/config.py` as the canonical config parsing layer
- deployment environment variables as the source for real secrets

Configuration should be read in four groups:

1. safe local defaults
   - `APP_BASE_URL`
   - `APP_TIMEZONE`
   - `CORS_ALLOWED_ORIGINS`
   - `SESSION_TTL_S`
   - `SESSION_MAX_STATES`
2. runtime paths
   - `SQLITE_PATH`
   - `SAFE_GUARD_SQLITE_PATH`
   - `EVENT_CLIPS_DIR`
3. backend mode selection
   - `DB_BACKEND`
   - `DB_HOST`
   - `DB_PORT`
   - `DB_USER`
   - `DB_PASS`
   - `DB_NAME`
4. secret-bearing integrations
   - `TELEGRAM_BOT_TOKEN`
   - `TWILIO_*`
   - `RESEND_API_KEY`
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`

Operational rules:

- keep real secrets out of git and store them only in local private env files or deployment secret stores
- prefer SQLite for local demo and supervisor review
- treat `.env.example` as the documented contract, not as a source of real credentials
- if a deployment requires third-party delivery or AI summaries, configure those values explicitly rather than relying on undeclared defaults

Suggested Render blueprint:

- [render.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/render.yaml)
- backend uses a persistent disk mounted at `/var/data`
- frontend should set `REACT_APP_API_BASE` to the backend Render URL

- the main reviewed online path is `caucafall + TCN + OP2`
- demo and replay delivery profiles live under `ops/configs/delivery/`

## Datasets and Data Modes

Supported dataset codes:

- `le2i`
- `urfd`
- `caucafall`
- `muvim`

Expected raw roots by default:

- `data/raw/LE2i`
- `data/raw/UR_Fall_clips`
- `data/raw/CAUCAFall`
- `data/raw/MUVIM`

Dataset availability:

- `CAUCAFall`
  - primary benchmark and deployment-target dataset in this project
  - publicly available for research use from its original dataset source
  - source: `https://data.mendeley.com/datasets/7w7fccy7ky/4`
  - this repository does not redistribute the raw dataset
- `LE2i`
  - comparative and transfer-boundary dataset in this project
  - publicly available for research use from its original dataset source
  - source: `http://le2i.cnrs.fr/Fall-detection-Dataset?lang=fr`
  - this repository does not redistribute the raw dataset
- `URFD`
  - supported by the codebase as an additional dataset option
  - publicly available from its original dataset source
  - source: `http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html`
  - this repository does not redistribute the raw dataset
- `MUVIM`
  - secondary exploratory dataset in this project
  - obtained from the Intelligent Assistive Technology and Systems Lab, University of Toronto, under a signed research data agreement
  - not redistributed in this repository
  - access is restricted and must be arranged directly with the data owner, subject to the provider's terms

Required acknowledgment for `MUVIM`-based research use:

- citation: S. Denkovski, S. S. Khan, B. Malamis, S. Y. Moon, B. Ye and A. Mihailidis, "Multi Visual Modality Fall Detection Dataset," *IEEE Access*, 2022, doi: `10.1109/ACCESS.2022.3211939`
- acknowledgment: `The author acknowledges the support of the Intelligent Assistive Technology and Systems Lab (IATSL) at the University of Toronto through the sharing of data related to this research.`

Two common usage modes:

1. raw mode
   - extract poses
   - preprocess poses
   - generate labels and splits
   - build train/eval windows
2. raw-free mode
   - start from existing `data/interim/...` or `data/processed/...`
   - useful when processed data already exists

## Notes for Reviewers

- if you only want to see the system working, use `make bootstrap-dev`
- if you want DB persistence included, use `docker compose up`
- if local `3306` is already occupied, use `MYSQL_PORT=3307 docker compose up`
- this branch is intentionally trimmed to runnable code, configuration, tests, and deployment assets
- runtime-generated artifacts such as datasets, checkpoints produced during training, and local event clips are not part of the tracked submission surface

## Runtime Asset Promotion

The cleaned submission branch distinguishes between experimental ML outputs and
runtime-approved deploy assets.

- `ops/deploy_assets/manifest.json`
  - tracked source of truth for promoted runtime assets
  - lists the shipped checkpoints, canonical runtime operating-point YAMLs, and
    replay clips intended for reviewer/demo use
- `ops/deploy_assets/checkpoints/`
  - promoted checkpoints approved for backend runtime loading on this branch
- `ops/deploy_assets/replay_clips/`
  - promoted replay clips approved for reviewer/demo playback
- `ops/configs/ops/*.yaml`
  - canonical runtime operating-point profiles; the promoted subset is listed in
    `ops/deploy_assets/manifest.json`
- `outputs/`
  - experimental or locally produced training/evaluation outputs
  - not treated as shipped runtime assets unless explicitly promoted into
    `ops/deploy_assets/` and recorded in the manifest

This branch does not rely on private research notes to identify deployable ML
assets. Reviewers can use `ops/deploy_assets/manifest.json` as the explicit
runtime promotion contract.


// uvicorn applications.backend.app:app --host 0.0.0.0 --port 8000 --reload
