# Cross-Platform One-Click Portability Tasks - 2026-05-07

Branch: `feat/ctr-gcn-upgrade`

This task note defines the work needed to restore and harden the old MacBook-style
one-command project experience across macOS, Linux, and Windows.

The target is a portable project that can be copied or cloned onto another
machine and then installed, started, validated, and used for offline ML workflows
with minimal manual setup.

## Product Goal

The project should support these one-command workflows:

- First-time install and full-stack start:
  - macOS/Linux: `make bootstrap-dev` or `./ops/scripts/dev.sh bootstrap`
  - Windows PowerShell: `.\ops\scripts\dev.ps1 bootstrap`
  - universal fallback: `python ops/scripts/dev.py bootstrap`
- Start already-installed local app:
  - macOS/Linux: `make dev`
  - Windows PowerShell: `.\ops\scripts\dev.ps1 start`
  - universal fallback: `python ops/scripts/dev.py start`
- Stop local app:
  - macOS/Linux: `make stop-dev`
  - Windows PowerShell: `.\ops\scripts\dev.ps1 stop`
  - universal fallback: `python ops/scripts/dev.py stop`
- Validate local app:
  - `python ops/scripts/dev.py smoke`
- Run local full-stack demo:
  - backend: `http://127.0.0.1:8000`
  - frontend: `http://127.0.0.1:3000`
  - default runtime: promoted `caucafall_tcn`, `OP-2`, 24 replay clips
- Run offline ML smoke:
  - `python ops/scripts/dev.py offline-smoke`
- Run reproducible offline pipeline when raw data exists:
  - `python ops/scripts/dev.py offline-pipeline --dataset caucafall --model ctr-gcn`
  - `python ops/scripts/dev.py offline-pipeline --dataset caucafall --model tcn`

## Current State

Already present:

- macOS/Linux shell bootstrap:
  - `ops/scripts/bootstrap_dev.sh`
  - `ops/scripts/start_fullstack.sh`
- Windows PowerShell bootstrap:
  - `ops/scripts/bootstrap_dev.ps1`
  - `ops/scripts/start_fullstack.ps1`
  - `ops/scripts/stop_fullstack.ps1`
- Makefile app entry points:
  - `make bootstrap-dev`
  - `make dev`
  - `make stop-dev`
- Docker Compose persistent stack:
  - `docker compose up`
- Runtime assets:
  - `ops/deploy_assets/manifest.json`
  - `ops/deploy_assets/checkpoints/caucafall_tcn_best.pt`
  - `ops/configs/ops/tcn_caucafall.yaml`
  - 24 replay clips under `ops/deploy_assets/replay_clips/`
- Offline ML entry points:
  - `fd-train-tcn`
  - `fd-train-ctr-gcn`
  - Make targets for labels, splits, windows, training, fitting, evaluation, plots

Known gaps:

- `make` is not available by default on Windows.
- Makefile recipes are Unix-first (`bash`, `lsof`, shell syntax), so Makefile
  cannot be the only portability interface.
- macOS/Linux scripts and Windows scripts duplicate behavior rather than sharing
  one orchestration layer.
- Offline ML workflows are powerful but not packaged as simple portable commands.
- Bootstrap currently installs dependencies but does not run a unified smoke
  verification at the end.
- Raw data availability is not explicitly checked before extraction/training
  workflows.
- README still needs a cross-platform quick-start table that separates:
  - local demo
  - offline ML
  - persistent Docker stack
  - troubleshooting
- There is no single machine-readable environment doctor that reports Python,
  Node, npm, torch, GPU, MediaPipe, ports, runtime assets, and raw-data status.

## Implementation Status

Implemented in this pass:

- `ops/scripts/dev.py`
  - `doctor`
  - `bootstrap`
  - `start`
  - `stop`
  - `smoke`
- `ops/scripts/dev.ps1`
  - Windows PowerShell wrapper around `dev.py`
- `ops/scripts/dev.sh`
  - macOS/Linux shell wrapper around `dev.py`
- Makefile app entry points now call the shared Python orchestration:
  - `make bootstrap-dev`
  - `make dev`
  - `make stop-dev`
- README Quick Start now lists macOS/Linux, Windows PowerShell, and universal
  Python commands.

Verified on Windows:

- `python ops/scripts/dev.py doctor --strict`
- `python ops/scripts/dev.py bootstrap --skip-start`
- `python ops/scripts/dev.py stop`
- `.\ops\scripts\dev.ps1 start --detached`
- `python ops/scripts/dev.py smoke`

Still pending:

- macOS validation
- Linux validation
- Docker smoke wrapper
- offline-smoke command
- offline-pipeline command
- converting legacy bootstrap/start scripts into thin wrappers

## Design Direction

Use a Python standard-library orchestrator as the cross-platform source of truth:

- `ops/scripts/dev.py`

Add thin wrappers around it:

- `ops/scripts/dev.sh`
- `ops/scripts/dev.ps1`

Keep Makefile for macOS/Linux convenience and offline power-user workflows, but
make it call the same shared scripts where possible.

This avoids relying on `make`, `bash`, `lsof`, or PowerShell as the only control
plane. Python is already required by the project and is the most portable local
orchestration layer.

## Task Plan

### Phase 1: Define Unified Command Contract

Create a command contract for:

- `doctor`
  - report OS, Python, pip, Node, npm, Git, Docker availability
  - report CPU/GPU/Torch availability
  - report required runtime assets
  - report raw-data availability per dataset
  - report whether ports 8000/3000 are free or occupied by project processes
- `bootstrap`
  - create or reuse platform-local venv
  - install Python dependencies
  - install editable project package
  - install frontend dependencies
  - sync MediaPipe assets
  - run `doctor`
  - optionally start the app
- `start`
  - start backend and frontend
  - wait for backend health
  - wait for frontend readiness
  - write process state under `.make/dev-state.json`
- `stop`
  - stop only project-owned backend/frontend processes
  - clear state file
- `smoke`
  - call backend health
  - call settings endpoint
  - verify deploy specs exactly include `caucafall_tcn`
  - verify replay clips count is 24
  - verify frontend index returns 200
- `test`
  - run backend QA
  - run frontend typecheck/test/build
- `offline-smoke`
  - import TCN and CTR-GCN training modules
  - construct tiny synthetic window tensors
  - run minimal model forward passes
  - validate fit/eval utilities import cleanly
- `offline-pipeline`
  - require raw data
  - run extraction, labels, splits, windows, train, fit, eval for selected model
  - fail early with clear instructions if raw data is missing

### Phase 2: Normalize Environments

Implement consistent environment layout:

- macOS/Linux venv: `.venv`
- Windows venv: `.venv-win`
- optional override: `FD_VENV_DIR`
- Python override:
  - macOS/Linux: `PY_BIN=python3.10`
  - Windows: `-Python C:\Path\python.exe`
- Node version:
  - `.nvmrc` remains `22.22.0`
  - warn when Node is not 22.x
- Frontend dependency command:
  - prefer `npm ci` when `package-lock.json` exists
  - fall back to `npm install` only when lockfile is missing

Acceptance:

- Bootstrap never writes machine-specific absolute paths into tracked files.
- Bootstrap can be rerun safely.
- Bootstrap emits actionable messages instead of Python stack traces for common
  missing tools.

### Phase 3: App Runtime One-Click

Unify app start behavior:

- backend:
  - `uvicorn applications.backend.app:app --host 127.0.0.1 --port 8000`
  - set `PYTHONPATH=<repo>/ml/src;<repo>`
  - write stdout/stderr logs to temp directory
- frontend:
  - `npm start`
  - set `REACT_APP_API_BASE=http://127.0.0.1:8000`
  - set `BROWSER=none` by default
- state:
  - store real listener PIDs, URLs, and log paths
  - use JSON state readable on all OSes

Acceptance:

- `python ops/scripts/dev.py start --detached` starts both services.
- `python ops/scripts/dev.py stop` stops only project processes.
- `python ops/scripts/dev.py smoke` passes after start.
- Existing Windows `.ps1` and macOS/Linux `.sh` scripts remain supported.

### Phase 4: Offline ML One-Click

Package offline ML workflows into simple commands:

- `offline-smoke`
  - no raw data required
  - runs on CPU
  - validates core model and feature plumbing
- `offline-check-data`
  - reports presence of `data/raw/caucafall`, `data/raw/le2i`, etc.
  - reports expected source type: images vs videos
- `offline-pipeline`
  - model choices: `tcn`, `ctr-gcn`
  - dataset choices: `caucafall`, `le2i`, later others if retained
  - optional flags:
    - `--skip-extract`
    - `--smoke`
    - `--device cpu|cuda|auto`
    - `--seed`
    - `--window W,S`

Acceptance:

- Without raw data, offline pipeline exits with a clear message and no partial
  output mutation.
- With raw data, caucafall TCN and CTR-GCN can be launched through one command.
- Seeds and output directories are printed at the start of each run.
- Generated outputs remain ignored unless explicitly promoted.

### Phase 5: Docker Path

Keep Docker as the persistent-system one-command option:

- `docker compose up`
- `docker compose down`

Tasks:

- verify frontend Dockerfile exists and is documented
- ensure Docker backend uses current promoted runtime assets
- add `docker-smoke` command to validate backend, frontend, and MySQL health
- ensure `.env.example` has safe defaults and no private values

Acceptance:

- Fresh Docker run works from clone on Linux/macOS/Windows Docker Desktop.
- Port overrides work:
  - `BACKEND_PORT`
  - `FRONTEND_PORT`
  - `MYSQL_PORT`

### Phase 6: Documentation

Update documentation with a single portable quick-start table:

| Goal | macOS/Linux | Windows PowerShell | Universal |
| --- | --- | --- | --- |
| install + start | `make bootstrap-dev` | `.\ops\scripts\dev.ps1 bootstrap` | `python ops/scripts/dev.py bootstrap` |
| start app | `make dev` | `.\ops\scripts\dev.ps1 start` | `python ops/scripts/dev.py start` |
| stop app | `make stop-dev` | `.\ops\scripts\dev.ps1 stop` | `python ops/scripts/dev.py stop` |
| smoke test | `python ops/scripts/dev.py smoke` | same | same |
| offline smoke | `python ops/scripts/dev.py offline-smoke` | same | same |
| Docker stack | `docker compose up` | `docker compose up` | same |

Documentation must include:

- prerequisites
- first-run commands
- rerun commands
- stop commands
- logs location
- raw-data requirements
- GPU notes
- common Windows Git/PowerShell execution-policy notes
- common port-conflict fixes

## Cross-Platform Test Matrix

Minimum matrix before declaring done:

- Windows 11 PowerShell
  - `.\ops\scripts\dev.ps1 bootstrap -SkipStart`
  - `.\ops\scripts\dev.ps1 start -Detached`
  - `python ops/scripts/dev.py smoke`
  - `.\ops\scripts\dev.ps1 stop`
- macOS
  - `make bootstrap-dev`
  - `make dev`
  - `python ops/scripts/dev.py smoke`
  - `make stop-dev`
- Linux
  - `python ops/scripts/dev.py bootstrap --skip-start`
  - `python ops/scripts/dev.py start --detached`
  - `python ops/scripts/dev.py smoke`
  - `python ops/scripts/dev.py stop`
- Docker
  - `docker compose up`
  - `python ops/scripts/dev.py docker-smoke`
  - `docker compose down`

## Final Definition Of Done

The project is portable when:

- a fresh clone can install and run locally on Windows without `make`
- a fresh clone can install and run locally on macOS/Linux with `make` or Python
- local demo mode does not require raw datasets, MySQL, or private credentials
- offline ML commands are discoverable and fail clearly if raw data is missing
- runtime smoke passes with one promoted deploy spec: `caucafall_tcn`
- replay clips count is 24
- frontend build passes
- backend QA passes
- README contains a short, accurate, cross-platform quick start
- old MacBook-style one-command workflow is restored without making Windows a
  second-class path

## Suggested Implementation Order

1. Add `ops/scripts/dev.py` with `doctor`, `bootstrap`, `start`, `stop`, `smoke`.
2. Convert `bootstrap_dev.sh`, `start_fullstack.sh`, and PowerShell scripts into
   thin wrappers or make them call the shared Python orchestration.
3. Add `offline-smoke` and `offline-check-data`.
4. Add `offline-pipeline` wrapper around existing Make/Python ML commands.
5. Update README quick start and troubleshooting.
6. Run the full test matrix and record results in a new verification note.
