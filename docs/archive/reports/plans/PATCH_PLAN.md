# PATCH_PLAN

## P0 (Pipeline-breaking / Contract-breaking)

### P0-1: Fix runtime two-stream GCN feature split parity
- Files:
  - `server/deploy_runtime.py`
- Evidence:
  - Hard-coded feature slicing and `RuntimeError` for `F` outside {2,3,4,5} at lines 458-476.
  - Canonical contract requires layout-based slicing (`src/fall_detection/core/features.py:10`, `:458`).
- Minimal patch strategy:
  - Replace hard-coded `F` branch with `split_gcn_two_stream` using `feat_cfg` from spec.
  - Keep fallback behavior for missing motion only via canonical helper.
- Validation command:
  - `python -m compileall server/deploy_runtime.py src/fall_detection/core/features.py`
  - `python -c "import server.app"`
  - `python -c "from server.deploy_runtime import predict_spec; print('import-ok')"`

### P0-2: Align `pipeline-auto-*` behavior with documented chain
- Files:
  - `Makefile`
- Evidence:
  - Help text says auto-pipelines include fit-ops/eval (lines 535-536).
  - Target implementations at lines 1111-1125 omit fit/eval.
- Minimal patch strategy (choose one):
  - Preferred: add `fit-ops-*` + `eval-*` invocation before `plot-*` in auto targets.
  - Alternative: adjust help text to explicitly state train+plot only.
- Validation command:
  - `make -Bn pipeline-auto-tcn-le2i`
  - `make -Bn pipeline-auto-gcn-le2i`

## P1 (Robustness / Reproducibility)

### P1-1: Remove absolute path leakage in repro manifests
- Files:
  - `scripts/reproduce_claim.py`
  - existing generated `artifacts/repro/*/manifest.json` (regenerate)
- Evidence:
  - absolute `/Users/...` path in `artifacts/repro/RESULTS_20260302_204401/manifest.json:4`.
- Minimal patch strategy:
  - write checkpoint path relative to manifest dir or repo root.
  - add optional `--emit_absolute_paths` if needed for debugging.
- Validation command:
  - `python scripts/reproduce_claim.py --help`
  - `rg -n '/Users/' artifacts/repro/**/manifest.json`

### P1-2: Fix frontend README health endpoint command
- Files:
  - `apps/README.md`
- Evidence:
  - docs use `/health` (line 45), route is `/api/health` (`server/routes/health.py:12`).
- Minimal patch strategy:
  - update command to `/api/health` and optionally include `/api/v1/health`.
- Validation command:
  - `rg -n 'localhost:8000/(health|api/health)' apps/README.md`

## P2 (Cleanliness / Engineering maturity)

### P2-1: Reduce path-bootstrap import hacks
- Files:
  - `src/fall_detection/training/train_tcn.py`
  - `src/fall_detection/training/train_gcn.py`
  - selected eval plot scripts
- Evidence:
  - `sys.path.insert` bootstrap blocks at script top.
- Minimal patch strategy:
  - rely on package install (`pip install -e .`) and wrappers; remove bootstrap where safe.
- Validation command:
  - `python -m compileall src/fall_detection/training src/fall_detection/evaluation`

### P2-2: Tighten broad exception handling in critical paths
- Files:
  - `src/fall_detection/evaluation/metrics_eval.py`
  - `src/fall_detection/core/features.py`
- Evidence:
  - multiple broad `except Exception` blocks.
- Minimal patch strategy:
  - narrow exception types where practical and add structured warning logs.
- Validation command:
  - `python -m compileall src/fall_detection/evaluation/metrics_eval.py src/fall_detection/core/features.py`

### P2-3: Add type-check gate (optional but recommended)
- Files:
  - `pyproject.toml` (or new `mypy.ini`)
- Evidence:
  - no mypy/pyright config detected.
- Minimal patch strategy:
  - configure a minimal mypy scope for `src/fall_detection/core` and `server`.
- Validation command:
  - `mypy src/fall_detection/core server`
