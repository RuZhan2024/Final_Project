# GLOBAL_AUDIT_REPORT

## Architecture Overview (End-to-End Integration Chain)
This repository is structured as a package-first ML system with thin script wrappers and Makefile orchestration.

1. Data ingestion and pose extraction starts via Make targets (`extract-%`, `preprocess-%`) in [Makefile](./Makefile) and dispatches to wrappers in `scripts/` (for example `scripts/extract_pose_videos.py`, `scripts/preprocess_pose.py`).
2. Labeling and splitting follow (`labels-%`, `splits-%`) and feed window generation (`windows-%`, `windows-eval-%`, `fa-windows-%`), implemented by `scripts/make_windows.py` -> `src/fall_detection/data/windowing/make_windows_impl.py`.
3. Model training uses `train-tcn-%` and `train-gcn-%` targets in [Makefile](./Makefile:815) and [Makefile](./Makefile:837), mapped to `scripts/train_tcn.py` / `scripts/train_gcn.py`, which delegate into `src/fall_detection/training/train_tcn.py` and `train_gcn.py`.
4. Operating-point fitting and evaluation are wired by `fit-ops-%`, `fit-ops-gcn-%`, `eval-%`, `eval-gcn-%` targets in [Makefile](./Makefile:874), [Makefile](./Makefile:886), [Makefile](./Makefile:911), [Makefile](./Makefile:921), using `src/fall_detection/evaluation/fit_ops.py` and `metrics_eval.py`.
5. Artifacts are emitted to `outputs/*`, `configs/ops/*`, and portable bundle descriptors under `artifacts/`.
6. Backend runtime is FastAPI (`server/app.py` -> `server/main.py`), with route modules in `server/routes/*` and deploy inference path in `server/deploy_runtime.py`.
7. Frontend is React under `apps/`, consuming backend via `apps/src/lib/apiClient.js` + `apps/src/lib/config.js` (`REACT_APP_API_BASE`, default `http://localhost:8000`).

The intended full chain is:
`Data -> Windows -> Train -> Fit Ops -> Eval -> API -> Frontend`.

## Scope and Audit Method
- Static audit plus smoke checks only.
- No expensive training reruns.
- Commands run:
  - `python -m compileall .`
  - `python -c "import fall_detection"`
  - `python -c "import server.app"`
  - `pytest -q`
  - `make -Bn <target>` dry-runs for representative targets
  - grep scans for absolute paths, exception handling patterns, import hacks, CORS and route contracts

## Build Integrity Results
- PASS: `python -m compileall .`
- PASS: `python -c "import fall_detection"`
- PASS: `python -c "import server.app"`
- FAIL: `pytest -q` aborts during torch import from test collection
  - Evidence: torch import abort stack originates from `tests/test_hardneg_split_guard.py` importing `src/fall_detection/training/train_gcn.py`, ending in fatal abort (exit code 134).

## Key Findings

### P0-1: Runtime two-stream GCN feature split is hard-coded and conflicts with canonical feature contract
- Evidence:
  - Canonical contract explicitly forbids hard-coded slicing in [src/fall_detection/core/features.py](./src/fall_detection/core/features.py:10).
  - Canonical two-stream split is implemented in [src/fall_detection/core/features.py](./src/fall_detection/core/features.py:458).
  - Runtime still uses hard-coded `F==2/3/4/5` branches and raises for other dimensions in [server/deploy_runtime.py](./server/deploy_runtime.py:458) and [server/deploy_runtime.py](./server/deploy_runtime.py:476).
  - Current ops configs include bone+bone_len+conf enabled (`use_bone: true`, `use_bone_length: true`) in `configs/ops/*.yaml`, which can produce `F>5` in canonical input.
- Impact:
  - Potential runtime inference failure for valid trained feature configurations.
  - Train/eval/runtime parity risk.

### P0-2: `pipeline-auto-*` does not include fit-ops/eval despite documented DAG comment
- Evidence:
  - Help text claims auto pipeline includes fit-ops/eval ([Makefile](./Makefile:535), [Makefile](./Makefile:536)).
  - Actual implementation runs windows, optional FA, train, then plot only ([Makefile](./Makefile:1111), [Makefile](./Makefile:1119), [Makefile](./Makefile:1125)); no `fit-ops-*` or `eval-*` dependency.
- Impact:
  - Single-command pipeline can skip metrics/ops generation unexpectedly.
  - Reproducibility and release workflow ambiguity.

### P1-1: Portability leak in reproducibility manifest (absolute path)
- Evidence:
  - Absolute checkpoint path embedded in [artifacts/repro/RESULTS_20260302_204401/manifest.json](./artifacts/repro/RESULTS_20260302_204401/manifest.json:4).
  - Main bundle uses relative paths in [artifacts/artifact_bundle.json](./artifacts/artifact_bundle.json).
- Impact:
  - Cross-machine replay issues for that repro artifact.

### P1-2: Documentation drift in frontend README health endpoint
- Evidence:
  - `apps/README.md` instructs `curl http://localhost:8000/health` ([apps/README.md](./apps/README.md:45)).
  - Actual routes are `/api/health` and `/api/v1/health` in [server/routes/health.py](./server/routes/health.py:12).
- Impact:
  - Demo friction and false negatives in quick checks.

### P2-1: Residual import bootstrap hacks in core scripts
- Evidence:
  - `sys.path.insert` bootstrap in training/evaluation scripts (for example [src/fall_detection/training/train_tcn.py](./src/fall_detection/training/train_tcn.py:24), [src/fall_detection/training/train_gcn.py](./src/fall_detection/training/train_gcn.py:23), `src/fall_detection/evaluation/plot_*`).
- Impact:
  - Architecture cleanliness and packaging discipline gap; not immediate runtime breakage.

## Positives (Evidence-backed)
- Makefile DAG is comprehensive and dataset-scoped with collision guards (`windows-eval-%`, `eval-gcn-%`, etc.).
- Fit/eval tooling has explicit policy and sweep controls with strong contracts in `fit_ops.py` and `metrics_eval.py`.
- API route version parity (`/api/*` and `/api/v1/*`) is consistently implemented.
- CORS defaults include common local frontend origins and env override (`CORS_ALLOWED_ORIGINS`) in [server/main.py](./server/main.py:24).
- Checkpoint/ops outputs are present for both TCN and GCN on LE2i + CAUCAFall in `outputs/metrics` and `configs/ops`.

## Readiness Verdict
- Pipeline integrity: **Yellow** (core chain works, but auto-pipeline drift and runtime GCN split contract risk are significant).
- Reproducibility: **Yellow** (good tooling, but pytest instability in current environment and one non-portable manifest).
- Deployment readiness: **Yellow/Red boundary** for GCN two-stream deployments with extended features until P0-1 is patched.

## Recommended Immediate Order
1. Patch runtime two-stream split to call canonical `split_gcn_two_stream` logic (P0).
2. Fix auto-pipeline target DAG or help text mismatch (P0).
3. Normalize repro manifest paths to relative (P1).
4. Update frontend README health command (P1).
