# Fall Detection v2

This repository contains a full fall-detection stack:
- ML pipeline (`src/fall_detection`, `scripts`, `Makefile`)
- Backend API (`server`)
- Frontend app (`apps`)

The project has been refactored to a package-first layout. Canonical code now lives under `src/fall_detection`.

## Project Structure

```text
.
├── src/fall_detection/        # Canonical ML/data/deploy code
├── scripts/                   # Thin CLI entrypoints
├── server/                    # FastAPI backend
├── apps/                      # React frontend
├── configs/                   # labels/splits/ops and config artifacts
├── data/                      # raw/interim/processed datasets
├── Makefile                   # End-to-end pipeline orchestration
├── pyproject.toml             # Editable package config
└── Integrated-Refactor-Master-Plan.md
```

## Setup

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install package in editable mode:

```bash
pip install -e . --no-build-isolation
```

Note: `--no-build-isolation` is recommended in restricted/offline environments.

### 2) Verify CLI entrypoints

```bash
fd-make-windows --help
fd-fit-ops --help
```

## Datasets

Supported datasets:
- `le2i`
- `urfd` (URFall alias in adapter layer)
- `caucafall`
- `muvim`

Raw data roots expected by default:
- `data/raw/LE2i`
- `data/raw/UR_Fall_clips`
- `data/raw/CAUCAFall`
- `data/raw/MUVIM`

## ML Workflow (Makefile)

List available targets:

```bash
make help
```

### Data pipeline (single dataset)

From raw data:

```bash
make pipeline-data-le2i
```

No extraction (expects `data/interim/<ds>/pose_npz_raw/*.npz`):

```bash
make pipeline-le2i-noextract
```

### Train

```bash
make train-tcn-le2i
make train-gcn-le2i
```

### Fit operating points, evaluate, plot

```bash
make fit-ops-le2i
make eval-le2i
make plot-le2i

make fit-ops-gcn-le2i
make eval-gcn-le2i
make plot-gcn-le2i
```

### Full pipelines

```bash
make pipeline-le2i
make pipeline-gcn-le2i
```

## Adapter Mode for Windowing

Window builders can be routed through the dataset adapter layer:

```bash
make windows-le2i ADAPTER_USE=1
make windows-eval-urfd ADAPTER_USE=1
make windows-unlabeled-caucafall ADAPTER_USE=1
```

Adapter knobs:
- `ADAPTER_DATASET_<ds>`
- `ADAPTER_URFALL_TARGET_FPS` (default `25.0`)

## Backend (FastAPI)

Install server deps (if separate):

```bash
pip install -r requirements_server.txt
```

Run backend:

```bash
uvicorn server.app:app --reload --port 8000
```

Health endpoint:
- `GET http://localhost:8000/api/health`

## Frontend (React)

```bash
cd apps
npm install
npm start
```

Optional API base URL:

```bash
export REACT_APP_API_BASE=http://localhost:8000
```

## Troubleshooting

### MediaPipe extraction fails with OpenGL / NSOpenGLPixelFormat

If extraction crashes with errors like:
- `Could not create an NSOpenGLPixelFormat`
- `kGpuService ... cannot be created`

this is an environment OpenGL/MediaPipe runtime issue, not a Makefile wiring issue.

Workarounds:
1. Run on a machine/session with OpenGL-capable graphics context.
2. Generate `pose_npz_raw` elsewhere and run `pipeline-<ds>-noextract`.

### OMP shared-memory error in restricted environments

Errors like `OMP: Error #179 ... Can't open SHM2` are environment restrictions. Use a local environment with shared-memory support for full train/eval runs.

## Refactor Plan

The active migration plan is tracked in:
- `Integrated-Refactor-Master-Plan.md`
