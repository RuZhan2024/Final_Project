# AUDIT_EXECUTION_TASKS

## Mission Plan (Step-by-Step)

### Phase 0 — Repo Map + Entrypoints
- [x] Enumerate core Makefile DAG targets (extract/preprocess/labels/splits/windows/windows-eval/fa-windows/train/fit-ops/eval/plot).
- [x] Map training/eval entrypoints (`train_tcn.py`, `train_gcn.py`, `fit_ops.py`, `metrics_eval.py`, plot scripts).
- [x] Map backend/frontend entrypoints (`server/app.py`, `server/deploy_runtime.py`, React API client/env).
- [x] Identify instruction docs (`AGENTS.md`, `README.md`, `docs/*`) and note drift.
- Output: architecture overview written in `GLOBAL_AUDIT_REPORT.md`.

### Phase 1 — Fast Build Integrity Checks
- [x] Run `python -m compileall .`.
- [x] Run import smoke checks:
  - [x] `python -c "import fall_detection"`
  - [x] `python -c "import server.app"`
- [x] Run `pytest -q` and capture failure evidence if any.
- Output: PASS/FAIL recorded in `GLOBAL_AUDIT_REPORT.md` and P0 entries in `PATCH_PLAN.md`.

### Phase 2 — Parameter & Config Contract Matching
- [x] Extract Makefile commands (dry-run evidence) and map CLI flags to argparse definitions.
- [x] Validate `configs/ops/*.yaml` schema compatibility with eval/fit_ops parsers.
- [x] Verify training metric/scheduler/imbalance settings coherence and deterministic validation behavior.
- Output: `CONFIG_CONTRACT_MATRIX.md` with PASS/WARN/FAIL and file-line evidence.

### Phase 3 — Artifacts & Portability
- [x] Enumerate bundle/report artifacts.
- [x] Validate referenced files and relative-vs-absolute paths.
- [x] Check server expectations against produced artifacts.
- Output: `ARTIFACT_PORTABILITY_REPORT.md` with checklist and minimal patch actions.

### Phase 4 — End-to-End Integration Audit (Server ↔ Apps ↔ ML)
- [x] Audit backend import hygiene and route inventory.
- [x] Compare frontend API calls against backend routes/methods.
- [x] Check CORS and frontend API base env handling.
- [x] Verify runtime feature/preprocessing parity with training pipeline and flag drift.
- Output: `INTEGRATION_STATUS_REPORT.md` including curl commands + demo startup commands.

### Phase 5 — Code Cleanliness & Industry Hygiene
- [x] Scan for TODO/FIXME, broad exception swallowing, `allow_pickle=True`, absolute paths, tracked junk files.
- [x] Assess typing/config maturity (mypy gate presence/absence).
- [x] Rank top cleanup items with evidence.
- Output: `CODE_CLEANLINESS_REPORT.md`.

### Phase 6 — Release Runbook
- [x] Write reproducible examiner runbook: install, env vars, no-training checks, backend/frontend launch, curl smoke tests.
- Output: `RELEASE_RUNBOOK.md`.

### Patch Planning
- [x] Classify findings as P0/P1/P2.
- [x] Provide file list, minimal patch strategy, validation command per item.
- Output: `PATCH_PLAN.md`.

## Deliverables Status
- [x] `GLOBAL_AUDIT_REPORT.md`
- [x] `INTEGRATION_STATUS_REPORT.md`
- [x] `CONFIG_CONTRACT_MATRIX.md`
- [x] `ARTIFACT_PORTABILITY_REPORT.md`
- [x] `CODE_CLEANLINESS_REPORT.md`
- [x] `RELEASE_RUNBOOK.md`
- [x] `PATCH_PLAN.md`

## Execution Notes
- No expensive training runs were performed.
- Only static/smoke/dry-run checks were used.
- P0 blockers were documented, not broadly refactored.
