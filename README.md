# Fall Detection v2

An end-to-end fall detection project with:

- pose-based preprocessing and window generation
- TCN and GCN model training/evaluation
- a FastAPI backend for online inference
- a React frontend for live and replay monitoring

The canonical Python package code lives in `src/fall_detection`.

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

- use `make dev` if `.venv` and `apps/node_modules` already exist
- this mode does not require MySQL
- DB-backed features fall back gracefully when DB is unavailable
- for cloud deployment, the backend now also supports `DB_BACKEND=sqlite`

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
- the database is initialized from `server/create_db.sql`
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
├── src/fall_detection/          # Core package: data, training, eval, deploy runtime
├── server/                      # FastAPI backend
├── apps/                        # React frontend
├── scripts/                     # Utility and orchestration scripts
├── configs/                     # labels, splits, ops, delivery profiles
├── data/                        # raw/interim/processed datasets
├── outputs/                     # checkpoints and training outputs
├── artifacts/                   # evaluation outputs, figures, evidence bundles
├── tests/                       # smoke and contract tests
├── docs/                        # active docs, runbooks, submission guidance
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
make compose-up
make compose-down
```

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
- Python compileability for `src/fall_detection`, `server`, and `scripts`
- frontend production build

Recommended contract/smoke test subset:

```bash
./scripts/run_canonical_tests.sh torch-free
```

Server app / contract subset:

```bash
./scripts/run_canonical_tests.sh contract
```

Torch-dependent monitor subset:

```bash
./scripts/run_canonical_tests.sh monitor
```

## Runtime Profiles and Delivery Notes

Current online deployment profiles are loaded from:

- `configs/ops/tcn_caucafall.yaml`
- `configs/ops/gcn_caucafall.yaml`
- `configs/ops/tcn_le2i.yaml`
- `configs/ops/gcn_le2i.yaml`

Important delivery note:

## Render Deployment Notes

Recommended cloud deployment shape:

- frontend on Render static hosting
- backend on Render web service
- app data on SQLite with a persistent disk
- Telegram caregiver notification
- AI message analysis via Gemini API

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
- Telegram and Gemini credentials should be stored only in Render environment variables
- SMS / phone / email escalation are future-work channels, not the current implemented delivery path

Suggested Render blueprint:

- [render.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/render.yaml)
- backend uses a persistent disk mounted at `/var/data`
- frontend should set `REACT_APP_API_BASE` to the backend Render URL

- the main reviewed online path is `caucafall + TCN + OP2`
- the four-folder custom delivery evaluation lives under `configs/delivery/` and `artifacts/fall_test_eval_20260315*/`

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

Two common usage modes:

1. raw mode
   - extract poses
   - preprocess poses
   - generate labels and splits
   - build train/eval windows
2. raw-free mode
   - start from existing `data/interim/...` or `data/processed/...`
   - useful when processed data already exists

## Documentation Guide

Use these documents as the main entrypoints:

- documentation index: `docs/README.md`
- submission pack index: `docs/project_targets/SUBMISSION_PACK_INDEX.md`
- supervisor delivery modes: `docs/reports/runbooks/SUPERVISOR_DELIVERY_MODES.md`
- user guide: `docs/reports/runbooks/USER_GUIDE.md`
- demo runbook: `docs/reports/runbooks/DEMO_RUNBOOK.md`
- final demo walkthrough: `docs/project_targets/FINAL_DEMO_WALKTHROUGH.md`
- delivery alignment status: `docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md`
- experiment evidence index: `docs/reports/runbooks/EXPERIMENT_EVIDENCE_INDEX.md`
- config-to-result evidence map: `docs/reports/runbooks/CONFIG_RESULT_EVIDENCE_MAP.md`

## Evidence and Reporting

For report and thesis writing, the key evidence locations are:

- runtime and delivery evidence: `configs/ops/`, `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`
- custom 24-video delivery evidence: `configs/delivery/`, `artifacts/fall_test_eval_20260315/`, `artifacts/fall_test_eval_20260315_online_reverify_20260315/`
- online repair and refit evidence: `artifacts/online_ops_fit_20260315*/`, `artifacts/ops_reverify_20260315*/`
- figures and summaries: `artifacts/figures/`, `artifacts/reports/`

Use `docs/reports/runbooks/EXPERIMENT_EVIDENCE_INDEX.md` as the shortest evidence navigation page.

## Notes for Reviewers

- if you only want to see the system working, use `make bootstrap-dev`
- if you want DB persistence included, use `docker compose up`
- if local `3306` is already occupied, use `MYSQL_PORT=3307 docker compose up`
- archived teaching/tutorial material has been moved under `docs/archive/tutorial_materials/`
- active project/report/thesis material is under `docs/project_targets/`, `docs/reports/`, and `artifacts/`


// uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
