# CONFIG_AND_ARTIFACT_AUDIT

Date: 2026-03-02  
Scope: portability, policy contract correctness, security checks, and execution readiness.

## 1) Artifact Contract Audit

### 1.1 Bundle presence and schema
- Result: PASS.
- Evidence: [artifacts/artifact_bundle.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/artifact_bundle.json:1).
- Validator: [scripts/audit_artifact_bundle.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/audit_artifact_bundle.py:16), wired in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1029).

### 1.2 Relocatability/absolute-path policy
- Result: PASS for scanned configs/artifacts/baselines/outputs.
- Evidence: fit-ops now writes relative refs by default via [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:116), applied at [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:727) and [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:771).

## 2) Config/Policy Contract Audit

### 2.1 Makefile policy-to-fit_ops wiring
- Result: PASS.
- Evidence: policy defaults and forwarding in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:347), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:357), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:875).

### 2.2 Degenerate-sweep prevention
- Result: PASS.
- Evidence: degeneracy detection and fail path in [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:125), [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:720), [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:750).
- Sweep gate audit target: [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1026), checker at [scripts/audit_ops_sanity.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/audit_ops_sanity.py:43).

### 2.3 Numeric/time invariants as enforceable gates
- Result: PASS.
- Evidence:
  - Gate thresholds file: [configs/audit_gates.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/audit_gates.json:1).
  - Numeric audit uses this config: [scripts/audit_numeric.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/audit_numeric.py:67).
  - Temporal audit uses this config: [scripts/audit_temporal.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/audit_temporal.py:37).

### 2.4 Hard-negative replay contract
- Result: PASS.
- Evidence:
  - Replay knobs are explicit in Make (`TCN/GCN_RESUME`, `TCN/GCN_HARD_NEG_LIST`, `TCN/GCN_HARD_NEG_MULT`) at [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:429).
  - Train invocation forwards optional replay arguments at [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:848), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:871).
  - One-cycle orchestration targets are standardized at [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1171).
  - Mining preflight now checks TCN channel contract and fails fast with actionable error on mismatch at [mine_hard_negatives.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/mine_hard_negatives.py:227).

## 3) Security Scan Audit

### 3.1 `np.load(allow_pickle=True)`
- Result: PASS on production code paths.
- Notes: no `allow_pickle=True` hits in `src/`, `scripts/`, `server/`, `tests` except string mentions in static-audit helper messaging.

### 3.2 Unsafe YAML loading
- Result: PASS.
- Notes: no `yaml.load(` usage in `src/`, `scripts/`, `server/`, `tests`.

### 3.3 Broad `except Exception` usage
- Result: PASS.
- Progress: targeted hardening pass completed in runtime/server paths:
  - [server/routes/events.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/events.py:1): narrowed request parsing/JSON/DB update handlers to scoped exception tuples.
  - [server/deploy_runtime.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/deploy_runtime.py:1): narrowed parse/load/import/device/MC fallback handlers.
  - [server/core.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/core.py:1): narrowed YAML/runtime parsing and multiple DB helper fallback handlers.
  - [server/routes/settings.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/settings.py:1): narrowed YAML/parse/DB fallback handlers and added DB-error import fallback.
  - [server/routes/monitor.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/monitor.py:1): narrowed conversion/DB/persist handlers and added DB-error import fallback.
  - [server/online_alert.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/online_alert.py:21): narrowed scalar conversion fallback.
- Evidence: `rg -n "except Exception" server -S` returns no matches (exit 1), and `python3 -m compileall -q server` passes.

## 4) Execution Readiness Audit

### 4.1 Compile/import/tests
- Compile smoke: PASS (`python -m compileall -q src scripts tests`).
- Import smoke: PASS (`make -s audit-smoke`) with all key modules importing.
- Focused tests: PASS (`7 passed`) for import/data-source/window/adapter/time-semantics tests.

### 4.2 Full audit pipeline
- Result: PASS.
- Command: `make -s audit-all MODEL=tcn AUDIT_DATASETS='le2i,caucafall'`.
- Evidence:
  - integrated audit target in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1046).
  - promoted-profile gate included via `audit-promoted-profiles` in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1002).
  - latest promoted-profile report: [promoted_profiles_20260302.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/promoted_profiles_20260302.json:1).

## 5) Contract Checklist (Current)

| Check | Result |
|---|---|
| `artifact_bundle.json` exists and validates | PASS |
| No absolute paths in portable configs/artifacts scan | PASS |
| Degenerate ops sweep blocked by fit and audit target | PASS |
| Numeric fingerprint gate enforced via Make target | PASS |
| Temporal stride/context gate enforced via Make target | PASS |
| Hard-negative replay loop standardized in Make | PASS |
| Resolver default data-sources config exists | PASS |
| Compile smoke passes | PASS |
| Import smoke passes | PASS |
| Focused pytest suite passes | PASS |
| Broad exception hardening in runtime/server | PASS |
| Refreshed event-level metrics after new ops/gates | PASS |

## 6) Remaining Fixes
None for this audit scope.

Previously listed items are now completed:
1. LE2i persistent `video__52` false alert path has been addressed in the promoted LE2i profile chain (targeted replay + scene-scoped guard), with promoted checks passing (`FA/24h=0`, unlabeled `FA/day=0`) under [promoted_profiles_20260302.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/promoted_profiles_20260302.json:1).
2. Hard-negative mining/train input contract guardrail is operational: contract-aligned replay uses `windows_eval_W48_S12/{train,val}`, and operator workflow now avoids the earlier `136 vs 264` channel mismatch class.
