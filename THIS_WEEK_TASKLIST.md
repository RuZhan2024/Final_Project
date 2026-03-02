# This Week Tasklist (Thesis + Demo Critical Path)
Updated: 2026-03-02
Overall status: 9/9 completed

## 1. Regenerate deploy specs for one demo profile (LE2i TCN or CAUCAFall GCN)
- Effort: M
- Risk: High
- Status: Completed (2026-03-02)
- Why: `/api/spec` is empty and monitor inference returns 404 without `configs/ops/*.yaml`.
- Acceptance criteria:
  - `configs/ops/<arch>_<dataset>.yaml` exists
  - referenced checkpoint exists
  - `/api/spec` returns non-empty `specs`
- Validate:
  - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 - <<'PY'\nfrom fastapi.testclient import TestClient\nfrom server.app import app\nprint(TestClient(app).get('/api/spec').json())\nPY`
- Evidence:
  - `configs/ops/tcn_le2i.yaml` and `configs/ops/gcn_caucafall.yaml` restored and loaded by runtime.
  - `/api/spec` returned non-empty `specs` during smoke checks.

## 2. Fix artifact bundle integrity
- Effort: S
- Risk: Medium
- Status: Completed (2026-03-02)
- Why: bundle currently references missing reports, fails audit.
- Acceptance criteria:
  - `scripts/audit_artifact_bundle.py` exits 0
  - all paths in `artifacts/artifact_bundle.json` resolve
- Validate:
  - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/audit_artifact_bundle.py --bundle_json artifacts/artifact_bundle.json`
- Evidence:
  - Audit output: `[ok] artifact bundle valid: artifacts/artifact_bundle.json`.

## 3. Capture full run config for CAUCAFall GCN checkpoint
- Effort: S
- Risk: High
- Status: Completed (2026-03-02)
- Why: headline AP trace exists, but exact training config artifact is missing.
- Acceptance criteria:
  - `outputs/caucafall_gcn_W48S12/train_config.json` exists (or equivalent manifest)
  - includes seed, feature flags, training hyperparameters
- Validate:
  - `test -f outputs/caucafall_gcn_W48S12/train_config.json`
- Evidence:
  - Created `outputs/caucafall_gcn_W48S12/train_config.json` from checkpoint `train_cfg`.
  - Contains seed/features/hyperparameters (`seed=33724876`, `lr=0.001`, feature flags).

## 4. Remove hard-negative split leakage risk
- Effort: M
- Risk: High
- Status: Completed (2026-03-02)
- Why: hard-negative lists are loaded without split-origin checks.
- Acceptance criteria:
  - training scripts reject `hard_neg_list` entries outside approved split roots
  - unit test added for rejection behavior
- Validate:
  - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 -m pytest -q tests/test_train_tcn_hardneg_prefix.py`
  - add equivalent GCN test
- Evidence:
  - Added split guard + override flag to training entrypoints:
  - `src/fall_detection/training/train_tcn.py`
  - `src/fall_detection/training/train_gcn.py`
  - Added coverage in `tests/test_hardneg_split_guard.py`.

## 5. Regenerate non-degenerate ops sweeps
- Effort: M
- Risk: High
- Status: Completed (2026-03-02)
- Why: current `configs/ops/*.sweep.json` fail ops sanity.
- Acceptance criteria:
  - ops sanity audit passes
  - OP1/OP2/OP3 contain finite recall/F1 and alert counts
- Validate:
  - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/audit_ops_sanity.py --ops_dir configs/ops`
- Evidence:
  - Regenerated `configs/ops/tcn_le2i.sweep.json` and `configs/ops/gcn_caucafall.sweep.json`.
  - Audit output: `[ok] ops sanity passed`.

## 6. Produce one thesis claim manifest (command + ckpt hash + metrics)
- Effort: S
- Risk: Medium
- Status: Completed (2026-03-02)
- Why: examiner must be able to verify headline claim provenance.
- Acceptance criteria:
  - `artifacts/repro/RESULTS_*/manifest.json` exists
  - includes commit hash, checkpoint hash, metrics rows
- Validate:
  - `python3 scripts/reproduce_claim.py --dataset caucafall --model gcn --run 0`
- Evidence:
  - Created `artifacts/repro/RESULTS_20260302_204401/manifest.json`.
  - Includes git metadata, checkpoint hash, and metrics entries.

## 7. Add server-side non-fatal error logging for DB failures in monitor route
- Effort: S
- Risk: Medium
- Status: Completed (2026-03-02)
- Why: current monitor route silently swallows DB failures.
- Acceptance criteria:
  - DB errors produce structured warning logs with context (`resident_id`, `session_id`, action)
  - inference response still returns success where possible
- Validate:
  - simulate DB-off and call monitor endpoint; check log output
- Evidence:
  - Added warnings in `server/routes/monitor.py` for DB default-read and event-persist failures.
  - Added test `tests/test_monitor_fault_injection.py::test_predict_window_logs_db_default_read_failure`.
  - Validation result: status `200` plus warning log with `resident_id/session_id`.

## 8. Capture latency profile artifact for promoted demo model
- Effort: S
- Risk: Medium
- Status: Completed (2026-03-02)
- Why: deployment claims need measured p50/p95 evidence.
- Acceptance criteria:
  - `artifacts/reports/infer_profile_*.json` present with mean/median/p95
- Validate:
  - `python3 scripts/profile_infer.py --win_dir <win_dir> --ckpt <ckpt> --io_only 0 --runs 100 --out_json artifacts/reports/infer_profile_cpu_local_tcn_le2i.json`
- Evidence:
  - Created `artifacts/reports/infer_profile_cpu_local_tcn_le2i.json`.
  - Measured latency: `mean=4.158ms`, `median=4.119ms`, `p95=4.754ms`.

## 9. Replace frontend README with project-specific run + env instructions
- Effort: S
- Risk: Medium
- Status: Completed (2026-03-02)
- Why: current frontend README is boilerplate CRA and not examiner-friendly.
- Acceptance criteria:
  - includes `REACT_APP_API_BASE`, backend startup, expected routes
- Validate:
  - manual doc review + fresh run from README steps
- Evidence:
  - Replaced `apps/README.md` with project-specific instructions:
  - `REACT_APP_API_BASE`, backend startup, frontend startup, `/health`, `/api/spec`, and monitor curl example.
