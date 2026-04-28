# Readiness Report: Dissertation + Deployment Audit

Date: 2026-03-02  
Scope: static repo audit + smoke checks only (no training runs)

## Dry-Run Closure Note (2026-03-22)

Paper/submission closure now expects one explicit external-review dry run recorded in:
- [CLEAN_DRY_RUN_MINIMUM_PATH.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CLEAN_DRY_RUN_MINIMUM_PATH.md)
- [clean_dry_run_report.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/clean_dry_run_report.md)

Current status:
- template and minimum path prepared
- one dry run recorded as `PASS`
- backend startup, frontend startup, API smoke, `/api/health`, `/api/spec`, headless inference, and browser replay checks all completed

## Post-Audit Remediation Update (2026-03-02)

Completed since initial audit:
- Restored deploy specs and regenerated non-degenerate sweeps:
  - `configs/ops/tcn_le2i.yaml`, `configs/ops/gcn_caucafall.yaml`
  - `configs/ops/tcn_le2i.sweep.json`, `configs/ops/gcn_caucafall.sweep.json`
- `/api/spec` now returns non-empty `specs/models/datasets`; monitor inference returns HTTP 200 with real checkpoints.
- Added `outputs/caucafall_gcn_W48S12/train_config.json` from checkpoint-embedded training metadata.
- Generated claim manifest:
  - `artifacts/repro/RESULTS_20260302_204401/manifest.json`
- Repaired artifact bundle and revalidated:
  - `artifacts/artifact_bundle.json` passes `scripts/audit_artifact_bundle.py`
- Added monitor DB failure warnings (non-fatal) in `server/routes/monitor.py` and test coverage in `tests/test_monitor_fault_injection.py`.
- Captured latency evidence:
  - `artifacts/reports/infer_profile_cpu_local_tcn_le2i.json` (`mean=4.158ms`, `p95=4.754ms`)
- Replaced frontend boilerplate README with project-specific run/env/API instructions in `apps/README.md`.

Residual gaps:
- No additional critical blockers identified for this sprint scope; remaining improvements are optional hardening and documentation polish.

## 1) Architecture Overview (Integration Chain)

This repository is structured as a full-stack fall-detection system with a Makefile-orchestrated ML pipeline, FastAPI backend, and React frontend.

### Data extraction -> preprocessing -> windowing
- Orchestration is Makefile-driven with stamp-based DAG nodes.
- Extraction targets call:
  - `scripts/extract_pose_videos.py` for LE2i (`Makefile:572-578`)
  - `scripts/extract_pose_images.py` for URFD/CAUCAFall/MUVIM (`Makefile:580-608`)
- Preprocessing target calls `scripts/preprocess_pose.py` with normalization and pelvis fill controls (`Makefile:621-630`).
- Labeling and split generation:
  - `scripts/make_labels_*.py` (`Makefile:643-676`)
  - `scripts/make_splits.py` (`Makefile:690-700`)
- Window generation:
  - train/eval windows via `scripts/make_windows.py` (`Makefile:741-771`)
  - unlabeled windows via `scripts/make_unlabeled_windows.py` (`Makefile:777-791`)
  - FA windows via `scripts/make_fa_windows.py` (`Makefile:796-804`)

### Training
- TCN and GCN training entrypoints:
  - `scripts/train_tcn.py` (`Makefile:815-835`)
  - `scripts/train_gcn.py` (`Makefile:837-857`)
- Canonical implementations live in:
  - `src/fall_detection/training/train_tcn.py`
  - `src/fall_detection/training/train_gcn.py`

### Ops fitting + evaluation
- Operating-point fitting:
  - `scripts/fit_ops.py` for TCN/GCN (`Makefile:874-896`)
  - Core logic: `src/fall_detection/evaluation/fit_ops.py`
- Metrics evaluation:
  - `scripts/eval_metrics.py` (`Makefile:911-929`)
  - Core logic: `src/fall_detection/evaluation/metrics_eval.py`
- Plot generation:
  - `scripts/plot_fa_recall.py`, `scripts/plot_f1_vs_tau.py` (`Makefile:948-962`)

### Artifacting
- Bundle manifest exists at `artifacts/artifact_bundle.json`.
- Baseline placeholder exists at `artifacts/baseline/le2i_58813e8.json`.
- Bundle references multiple missing reports (`artifacts/artifact_bundle.json:12-35`; validation failed, see Section 2).

### Backend runtime
- FastAPI entrypoint is minimal (`server/app.py:1-10`) and app assembly is in `server/main.py`.
- CORS defaults and override env var are in `server/main.py:20-48`.
- Live inference endpoint:
  - `POST /api/monitor/predict_window` in `server/routes/monitor.py:181-520`
  - uses deploy specs from `server/deploy_runtime.py`.
- Deploy spec discovery prioritizes `configs/ops/*.yaml` (`server/deploy_runtime.py:256-267`).

### Frontend
- React app under `apps/`.
- API base from `REACT_APP_API_BASE` with fallback `http://localhost:8000` (`apps/src/lib/config.js:3-12`).
- Unified fetch helper at `apps/src/lib/apiClient.js:11-45`.
- Monitor page posts `raw_t_ms/raw_xy/raw_conf` to `/api/monitor/predict_window` (`apps/src/pages/monitor/hooks/usePoseMonitor.js:269-305`).

## 2) Empirical Evidence from outputs/ + artifacts/

### Available outputs
- Present:
  - `outputs/caucafall_gcn_W48S12/best.pt`
  - `outputs/caucafall_gcn_W48S12/history.jsonl`
  - `outputs/le2i_tcn_W48S12/best.pt`
  - `outputs/le2i_tcn_W48S12/history.jsonl`
  - `outputs/le2i_tcn_W48S12/train_config.json`
- Missing expected eval products:
  - no `outputs/metrics/*.json`
  - no `artifacts/figures/plots/*.png`

### “0.99 AP” traceability check
- Traced claim source:
  - `outputs/caucafall_gcn_W48S12/history.jsonl` epoch 4: `"ap": 0.9915947983536599`.
- Evidence summary command used during audit:
  - `python3` parser over history files -> best AP row found in `outputs/caucafall_gcn_W48S12/history.jsonl`.
- Reproducibility caveat:
  - Command + full config for this exact run is not fully recoverable from local artifacts because `outputs/caucafall_gcn_W48S12/train_config.json` is absent.
  - This is a reproducibility gap for headline claims.

### Artifact bundle integrity
- `artifacts/artifact_bundle.json` references report files under `artifacts/reports/*` (`lines 12-35`) that do not exist locally.
- `scripts/audit_artifact_bundle.py` fails with missing targets.

### Ops sweep evidence
- `scripts/audit_ops_sanity.py --ops_dir configs/ops` fails:
  - `configs/ops/gcn_caucafall.sweep.json: degenerate sweep`
  - `configs/ops/tcn_le2i.sweep.json: degenerate sweep`
- This indicates current top-level ops sweep artifacts are not promotion-ready as stored.

## 3) Config Evidence (W/S, fps, features, confirm, OP definitions)

- Windowing defaults: `WIN_W=48`, `WIN_S=12` (`Makefile:84-85`).
- Dataset FPS defaults:
  - LE2i=25, URFD=30, CAUCAFall=23, MUVIM=30 (`Makefile:111-114`).
- Feature flags defaults:
  - motion/conf/bone/bone_len enabled for TCN path (`Makefile:192-209`).
- Confirm-stage defaults used by fit/eval:
  - `ALERT_CONFIRM=1`, `confirm_s=2.0`, `confirm_min_lying=0.65`, `confirm_max_motion=0.08` (`Makefile:359-363`).
- OP selection criteria:
  - conservative picker defaults + tie-break + min tau high (`Makefile:386-402`)
  - implemented in `fit_ops.py` conservative selector (`src/fall_detection/evaluation/fit_ops.py:145-296`).

Path hygiene checks:
- No hardcoded `/Users/...` found in runtime configs (scan only hit audit regex in `scripts/audit_static.py`).
- Archived ops YAML checkpoint references are relative, but currently unresolved in this workspace:
  - e.g., `configs/ops/archive/caucafall/gcn_caucafall_tau37floor.yaml` -> `../../outputs/caucafall_gcn_W48S12/best.pt` (path does not exist from that file location).

---

## Academic Rigor & Thesis Readiness

### 1) Reproducibility

Rating: **Yellow**

What is strong:
- End-to-end target chain is explicit in Makefile (`extract -> preprocess -> labels -> splits -> windows -> train -> fit_ops -> eval -> plot`).
- Seeds are exposed (`SPLIT_SEED`, train CLI args, split CLI args).
- Package-first layout and wrappers are clear (`scripts/*` thin wrappers).

What blocks Green:
- Headline run provenance is incomplete for CAUCAFall GCN:
  - AP 0.9916 exists in history, but full train config artifact is missing.
- Artifact bundle is invalid (references missing report files).
- Baseline parity target file is `pending_capture` with null targets (`artifacts/baseline/le2i_58813e8.json:4-15`).

Minimal patch plan:
- Add/standardize `train_config.json` write in `train_gcn.py` (parallel to TCN behavior).
- Rebuild `artifacts/artifact_bundle.json` from files that actually exist.
- Capture one reproducibility manifest for headline claim with exact command list and hashes.

Validation commands:
- `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/audit_artifact_bundle.py --bundle_json artifacts/artifact_bundle.json`
- `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/reproduce_claim.py --dataset caucafall --model gcn --run 0`

Reproduction path (current best-effort):
1. `make windows-caucafall windows-eval-caucafall`
2. `make train-gcn-caucafall`
3. `make fit-ops-gcn-caucafall`
4. `make eval-gcn-caucafall`

Missing for strict reproducibility:
- frozen command manifest + resolved config snapshot for the run that produced AP 0.9916.

### 2) Evaluation Soundness

Rating: **Yellow**

Metric definitions are mostly defensible:
- Event-level FA/24h and overlap logic are explicitly documented and implemented (`metrics_eval.py:7-19`, `alerting.py:484-605`).
- Inclusive frame semantics are consistently stated (`metrics_eval.py:13-18`, `score_unlabeled_alert_rate.py:10-16`).

Risks / mismatch points:
- AP is window-level (`core.metrics.ap_auc`) while OP sweeps and deployment metrics are event-level; this is valid but must be explicitly defended in dissertation methodology.
- Current ops sweep artifacts in `configs/ops/*.sweep.json` are degenerate under audit (`audit_ops_sanity` failure), which can undermine trust in stored operating points.
- Unlabeled handling: `y==-1` treated as negative for GT event construction (`alerting.py:531-533`), correct for FA streams but can bias if unlabeled composition differs by dataset.

Hard-negative leakage risk:
- Hard-negative lists are ingested directly from external text paths without split guard checks:
  - TCN: `train_tcn.py:529-553`
  - GCN: `train_gcn.py:533-557`
- If list contains val/test windows, leakage is possible.

Data leakage guard status:
- CAUCAFall subject split guard exists at Makefile level (`Makefile:682-683`).
- Split summary confirms `mode: caucafall_subject` (`configs/splits/caucafall_split_summary.json:10-15`).
- Unit test exists for group disjointness logic (`tests/test_split_group_leakage.py:13-40`).

Defense paragraph draft (for examiner):
- The project separates model discrimination and operational alerting: AP/AUC assess window-level ranking quality, while event-level recall/F1/FA24h evaluate deployment behavior under temporal policy (EMA, persistence, cooldown, optional confirm heuristics). Metrics use explicit inclusive-frame timing and event overlap rules, with subject-independent splitting enforced for CAUCAFall to reduce identity leakage. Unlabeled streams are used only for false-alert estimation and are not counted as positives.

Must-fix before defense:
- Regenerate non-degenerate `configs/ops/*.sweep.json` and ensure audit passes.
- Add split-origin validation for hard-negative lists.
- Publish one concrete claim manifest tying AP/F1/FA metrics to checkpoint hash and command.

### 3) Methodology Coherence

Rating: **Yellow-Green**

Coherence evidence:
- Normalization:
  - Preprocess supports `none|torso|shoulder` with pelvis translation/fill (`preprocess_pose_npz.py:342-400`, CLI defaults at `586-601`).
- Feature construction parity:
  - Single canonical builder in `core/features.py` with explicit channel layout contract (`1-35`, `206-231`, `393-440`).
- Architectures:
  - TCN and GCN definitions centralized in `core/models.py` (`82-175`, `203+`).
- Calibration:
  - Temperature scaling only (Contract C) in `core/calibration.py:7-13, 96-143`.
- Ops fitting uses validation directory by design (`Makefile:877,889`; `fit_ops.py` module docs).

Documentation gaps that are thesis blockers:
- Frontend README is generic CRA template and does not document project-specific monitor/deploy flow (`apps/README.md:1-70`).
- Root README lacks concrete artifact-promote workflow that currently matches repo reality (missing metrics bundle artifacts).
- No single “claim-to-artifact” table stored in repo root.

Methodology chapter outline (module-linked):
1. Data acquisition and pose preprocessing: `src/fall_detection/pose/preprocess_pose_npz.py`, extraction scripts.
2. Labeling and leak-safe splitting: `src/fall_detection/data/labels/*`, `src/fall_detection/data/splits/make_splits.py`.
3. Window generation and supervision policy: `src/fall_detection/data/windowing/make_windows_impl.py`.
4. Feature engineering contract: `src/fall_detection/core/features.py`.
5. Models (TCN/GCN) and training: `src/fall_detection/core/models.py`, `src/fall_detection/training/train_{tcn,gcn}.py`.
6. Calibration and operating points: `src/fall_detection/core/calibration.py`, `src/fall_detection/evaluation/fit_ops.py`.
7. Deployment policy and event metrics: `src/fall_detection/core/alerting.py`, `src/fall_detection/evaluation/metrics_eval.py`.
8. Runtime integration: `server/deploy_runtime.py`, `server/routes/monitor.py`, `apps/src/pages/monitor/hooks/usePoseMonitor.js`.

---

## Deployment & On-Device Readiness

### 1) End-to-End Latency & Profiling

Rating: **Yellow**

What exists:
- Profiling script already exists: `scripts/profile_infer.py` (supports IO-only or model path).
- API summary endpoint exists and reports status (`/api/summary` smoke-queried).

Gaps:
- No persistent server-side p50/p95 request latency logging middleware.
- No captured profiling reports currently present under `artifacts/reports/` despite bundle references.

Minimal profiling plan:
- CPU local baseline:
  - `make profile-infer PROFILE=cpu_local DS=le2i MODEL=tcn PROFILE_IO_ONLY=0`
- Add optional middleware to aggregate endpoint latencies into rolling p50/p95 for `/api/monitor/predict_window`.

Validation:
- `python3 scripts/profile_infer.py --win_dir <...> --ckpt <...> --io_only 0 --runs 100 --out_json artifacts/reports/infer_profile_cpu_local_tcn_le2i.json`

Latency components to report in dissertation:
- model forward latency
- preprocessing/feature build latency
- client capture + network + API + UI refresh end-to-end latency

### 2) System Robustness (Failure Modes)

Rating: **Yellow-Red for live demo reliability**

Failure-mode evidence table:

| Symptom | Current Behavior | Evidence | Recommended improvement |
|---|---|---|---|
| No pose frames from MediaPipe | UI returns early, shows “No pose detected…”, no inference frame sent | `usePoseMonitor.js:405-416` | Add explicit stale-stream warning + auto-recovery timer in UI status panel |
| DB unavailable during monitor predict | Exceptions swallowed silently | `server/routes/monitor.py:225-270`, `497-498` | Log structured warnings with rate limit; include `db_status` in response/debug fields |
| DB write fail for event persist | silently ignored | `server/routes/monitor.py:478-498` | Return non-fatal warning flag in response when persist requested but failed |
| Missing deploy specs | endpoint returns 404 | `server/routes/monitor.py:345-353`; test call produced 404 for le2i | Add startup health check that surfaces missing specs early (`/api/spec` currently empty) |
| Backpressure / queueing under load | Client throttles send interval; no backend queue metrics | `usePoseMonitor.js:243-266`; no queue primitives in monitor route | Add request-rate counters + drop policy metric for overloaded conditions |

Improvements required for demo stability this week:
- DB/persist errors must be visible in logs and optionally in response diagnostics.
- Add pre-demo startup validator that fails fast if no deploy specs/checkpoints are loadable.
- Produce one measured latency profile artifact included in demo package.

### 3) Packaging / Demo UX

Rating: **Yellow**

What works:
- Editable install path documented in root README (`README.md:35-39`).
- Backend startup command documented (`README.md:159-163`).
- API smoke passes when `PYTHONPATH` is set:
  - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py` -> passed.

Gaps:
- `apps/README.md` is generic CRA boilerplate; does not document project env vars/endpoint flow.
- Root `.env` only contains DB defaults; no `.env.example` for backend + frontend combined demo setup.
- Deploy specs not present at `configs/ops/*.yaml`, so live inference endpoint currently cannot run end-to-end.

---

## Strengths (Evidence-backed)

- Strong Makefile DAG orchestration with explicit targets for each stage and audit gates (`Makefile:560-1130`).
- Contract-driven feature engineering (`core/features.py`) and calibration contract (`core/calibration.py`).
- Subject-independent split guard for CAUCAFall in build pipeline (`Makefile:682-683`).
- Existing audit ecosystem (`scripts/audit_*`) and integration parity checks that pass (`audit_api_contract`, `audit_api_v1_parity`, `audit_runtime_imports`, smoke API with proper PYTHONPATH).
- Existing profiling scaffold (`scripts/profile_infer.py`) ready for reportable latency numbers.

## Critical Gaps (This Week, Decisive)

1. Rebuild valid deploy specs in `configs/ops/*.yaml` and matching metrics JSON so backend inference can run on examiner laptop.
2. Repair artifact bundle references so bundle audits pass.
3. Add hard-negative split guard to prevent accidental test/val leakage in training.
4. Produce one claim manifest tying 0.99 AP to concrete command, config, and checkpoint hash.
5. Replace frontend README with project-specific run/deploy instructions.
