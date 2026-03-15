# Fall Detection v2

End-to-end fall detection project with:
- ML pipeline (pose windows -> TCN/GCN training -> fit operating points -> evaluation)
- FastAPI backend for live/replay inference
- React frontend monitor UI

Canonical Python package code lives in `src/fall_detection`.

---

## 1) Quick Start

Run from repo root.

### A. One command to run ML pipeline (CAUCAFall, TCN + GCN)

```bash
AUTO_DO_EXTRACT=0 ADAPTER_USE=1 make pipeline-auto-tcn-caucafall pipeline-auto-gcn-caucafall
```

Notes:
- `AUTO_DO_EXTRACT=0` means no raw extraction is triggered.
- This assumes your `data/interim` / `data/processed` inputs already exist.

### B. One command to bootstrap + start backend + frontend

```bash
make bootstrap-dev
```

Open:
- Frontend: `http://localhost:3000`
- Backend health: `http://127.0.0.1:8000/api/health`

Notes:
- `make bootstrap-dev` creates `.venv` and installs missing frontend deps if needed.
- If your environment is already prepared, `make dev` starts faster.
- The command fails fast if port `8000` or `3000` is already in use.
- When the frontend process exits, the script also stops the backend it started.

---

## 2) Project Layout

```text
.
├── src/fall_detection/          # Core package (data, training, eval, deploy runtime)
├── scripts/                     # CLI entrypoints used by Makefile
├── server/                      # FastAPI backend
├── apps/                        # React frontend
├── configs/                     # labels, splits, ops
├── data/                        # raw/interim/processed
├── outputs/                     # checkpoints, metrics, plots
├── artifacts/                   # reports, manifests, audit outputs
├── tests/                       # smoke + contract tests
└── Makefile                     # primary orchestration interface
```

---

## 3) Environment Setup

## 3.1 Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

For reproducible installs:

```bash
pip install -r requirements.lock.txt
```

## 3.2 Frontend

```bash
cd apps
npm install
cd ..
```

## 3.3 Optional: DB-backed endpoints

If you want DB persistence for settings/events/caregivers, configure env vars (`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, `DB_NAME`).

---

## 4) Datasets and Data Modes

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
1. **Raw mode**: extraction + preprocess + labels + splits + windows
2. **Raw-free mode**: start from existing `data/interim/.../pose_npz_raw` or `data/processed/...`

---

## 5) Core Makefile Workflows

Show all available targets:

```bash
make help
```

## 5.1 Data preparation

```bash
make pipeline-data-le2i
make pipeline-data-caucafall
```

No-extract mode:

```bash
make pipeline-le2i-noextract
make pipeline-caucafall-noextract
```

## 5.2 Training

```bash
make train-tcn-caucafall
make train-gcn-caucafall
```

## 5.3 Fit operating points + evaluate + plot

```bash
make fit-ops-caucafall
make fit-ops-gcn-caucafall
make eval-caucafall
make eval-gcn-caucafall
make plot-caucafall
make plot-gcn-caucafall
make plot-confmat-caucafall
make plot-confmat-gcn-caucafall
make plot-failure-caucafall
make plot-failure-gcn-caucafall
make plot-balance-caucafall
```

## 5.4 Full pipelines

```bash
make pipeline-caucafall
make pipeline-gcn-caucafall
```

## 5.5 Auto pipelines (single command families)

```bash
make pipeline-auto-tcn-caucafall ADAPTER_USE=1
make pipeline-auto-gcn-caucafall ADAPTER_USE=1
```

Optional extraction toggle:

```bash
AUTO_DO_EXTRACT=1 make pipeline-auto-tcn-caucafall ADAPTER_USE=1
```

---

## 6) Locked Reproducibility Profiles (CAUCAFall)

Project includes locked TCN/GCN reproducibility paths.

Reproduce locked profiles:

```bash
make repro-best-caucafall ADAPTER_USE=1
make repro-best-muvim ADAPTER_USE=1
```

Promote locked ops to canonical deploy ops:

```bash
make apply-locked-ops-caucafall ADAPTER_USE=1
```

Current canonical deploy files:
- `configs/ops/tcn_caucafall.yaml`
- `configs/ops/gcn_caucafall.yaml`

LE2i paper-comparison diagnostic profile (GCN, scene-scoped start-guard):

```bash
make train-best-gcn-le2i-paper ADAPTER_USE=1
make repro-best-gcn-le2i-paper ADAPTER_USE=1
```

Artifacts:
- `configs/ops/gcn_le2i_paper_profile.yaml`
- `outputs/metrics/gcn_le2i_opt33_r8_dataside_noise_paperops.json`
- Full locked-parameter runbook:
  - `docs/project_targets/LOCKED_PARAMS_RUNBOOK.md`

Locked LE2i GCN training parameters (dataset-specific, does not alter global defaults):
- resume: `outputs/le2i_gcn_W48S12_opt33_r4_recallpush_promoted/best.pt`
- epochs/min_epochs: `45/8`, lr: `3e-4`, batch: `128`
- features: `motion=1, conf=1, bone=1, bonelen=1, two_stream=1, fuse=concat`
- robustness: `x_noise_std=0.01, x_quant_step=0.002`
- regularization: `dropout=0.18, mask_joint_p=0.00, mask_frame_p=0.00`
- training stability: `use_ema=1, ema_decay=0.999, deterministic=1, num_workers=0`

---

## 7) Backend API

Run backend:

```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Useful endpoints:
- `GET /api/health`
- `GET /api/spec`
- `GET /api/settings?resident_id=1`
- `POST /api/monitor/predict_window` (legacy HTTP path, still available)
- `WS /api/monitor/ws` (live monitor primary inference path)

OpenAPI:
- `http://127.0.0.1:8000/openapi.json`

---

## 8) Frontend

Run frontend:

```bash
cd apps
npm start
```

Default URL:
- `http://localhost:3000`

API base URL is configured in frontend config/env (default localhost backend).

---

## 9) Replay + Live Acceptance Lock

Use this to verify:
1. replay mode is stable/repeatable
2. live mode is acceptable on target hardware

One command:

```bash
bash tools/run_replay_live_acceptance.sh
```

Outputs:
- `artifacts/reports/replay_live_acceptance.md`

Detailed lock/runbook:
- `docs/project_targets/REPLAY_LIVE_ACCEPTANCE_LOCK.md`

Recommended profile for acceptance:
- dataset: `caucafall`
- model: `TCN` (primary), `GCN` (secondary)
- op: `OP-2`
- transport: `WebSocket` (`/api/monitor/ws`)

---

## 10) Quality and Audit Commands

Quick integrity checks:

```bash
python -m compileall src/fall_detection server scripts
python -c "import fall_detection; import server.app"
```

Audit gates:

```bash
make audit-smoke
make audit-ci
```

API contract smoke:

```bash
PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py
```

---

## 11) Common Issues

## 11.1 `pipeline-auto-*` fails at extraction

You likely do not have raw files matching expected glob patterns. Use raw-free mode:

```bash
AUTO_DO_EXTRACT=0 ADAPTER_USE=1 make pipeline-auto-tcn-caucafall
```

## 10.2 `fit_ops` feature mismatch (e.g. 17 vs 33 joints)

Windows and checkpoint were built with different feature contracts. Rebuild windows and retrain consistently (same adapter/joint layout + feature flags).

## 10.3 OpenMP shared-memory error in restricted environments

Use:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

before running eval/fit commands.

## 10.4 Frontend “Failed to fetch”

Check:
- backend is running on the expected host/port
- frontend API base URL points to backend
- CORS settings permit frontend origin

---

## 11) Recommended Demo Flow (Examiner/Supervisor)

1. Run ML command (Section 1A)
2. Run app command (Section 1B)
3. In UI Settings:
   - dataset: `caucafall`
   - model: `TCN` or `GCN`
   - OP: `OP-2`
4. Use Replay mode with prepared demo clips

---

## 12) Additional Docs

- Documentation index: `docs/README.md`
- Submission pack index: `docs/project_targets/SUBMISSION_PACK_INDEX.md`
- Plot selection guide for report/paper: `docs/project_targets/PLOT_SELECTION_FOR_REPORT.md`
- Backend details: `server/README.md`
- Reports/checklists: `docs/reports/`
- Project targets and execution plan: `docs/project_targets/`
