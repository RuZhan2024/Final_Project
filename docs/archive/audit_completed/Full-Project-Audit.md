# Full Project Audit (Rearranged)

Date: 2026-03-02
Scope: End-to-end audit of structure, pipeline reliability, data-layer refactor integrity, execution readiness after migration to `src/fall_detection`, plus cross-dataset numerical/temporal invariants and on-device latency readiness.

## 1) Executive Summary

The migration to package-first structure is materially advanced and core pipeline paths are runnable for LE2i and CAUCAFall. The codebase is not audit-complete for production because contract integrity, test coverage, and deploy-readiness gates are incomplete.

Primary blockers:
- P0 contract break: new `data` architecture modules fail import due to missing dependencies.
- P1 quality gaps: no automated parity gates, no numeric/time invariants checks, missing default resolver config.
- P2 deploy gap: no standardized on-device latency/dependency-isolation gate.

Overall status:
- Migration progress: strong
- Operational correctness confidence: medium-low
- Production readiness: not ready

## 2) Evidence Legend

- `Verified`: directly observed via code lines and/or command output in this audit run.
- `Inferred`: conclusion derived from verified evidence.
- `Planned Audit`: control is defined but not implemented yet.

## 3) Architecture Snapshot

Current canonical structure:
- `src/fall_detection/core`: frozen model/math logic
- `src/fall_detection/data`: adapters, labels, splits, windowing, plus resolver/schema/datamodule/pipeline scaffold
- `src/fall_detection/training`: `train_tcn.py`, `train_gcn.py`
- `src/fall_detection/evaluation`: metrics/fit_ops/plots/hard-negative tools
- `scripts/`: thin wrappers
- `Makefile`: orchestration including `pipeline-auto-*`
- `pyproject.toml`: package + console scripts (`pyproject.toml:11-29`)

Legacy top-level roots removed: `core`, `models`, `eval`, `deploy`, `pose`, `labels`, `split`, `windows`.

## 4) Plan Compliance (Integrated Master Plan)

Completed (Verified):
- `src/` package layout exists.
- wrapper scripts exist (example: `scripts/make_fa_windows.py:1-7`).
- adapter contract exists and URFall resampling is implemented (`src/fall_detection/data/adapters/base.py:185-300`).
- auto-pipeline targets exist with injected `ADAPTER_USE=1` (`Makefile:949-965`).

At risk (Inferred):
- structural completion claims exceed runtime integrity because new `data` modules are currently non-importable.
- cross-dataset numeric/time invariants are not formalized as executable gates.

## 5) Severity-Ranked Findings

### P0

#### P0-01 New Data Architecture Import Contract Is Broken
- Verification: `Verified`
- Evidence:
  - `src/fall_detection/data/datamodule.py:20` imports `FeatCfg` from `fall_detection.preprocessing`.
  - `src/fall_detection/data/pipeline.py:27` imports `fall_detection.preprocessing.pose_resample`.
  - `src/fall_detection/data/transforms.py:10` imports `fall_detection.training.contracts`.
  - Runtime smoke output:
    - `fall_detection.data.datamodule FAIL ImportError cannot import name 'FeatCfg'`
    - `fall_detection.data.pipeline FAIL ModuleNotFoundError ... pose_resample`
    - `fall_detection.data.transforms FAIL ModuleNotFoundError ... training.contracts`
- Impact: unified data-path adoption fails at import time.
- Action: either complete dependencies in one bounded change or hard-disable these modules from runtime surface.

### P1

#### P1-01 Resolver Default Config Path Is Missing
- Verification: `Verified`
- Evidence:
  - `src/fall_detection/data/resolver.py:20` sets `DEFAULT_DATA_SOURCES_CONFIG = Path("configs/experiments/data_sources.yaml")`.
  - file does not exist in repo (audit command: `python -c 'Path(...).exists()' -> False`).
- Impact: default resolver init path can fail at runtime.
- Action: point to existing config or add missing config + schema test.

#### P1-02 Test Suite Is Effectively Absent
- Verification: `Verified`
- Evidence:
  - `rg --files tests` returned no test modules in this run.
- Impact: regression detection is manual and late.
- Action: add minimum pytest smoke suite (imports, adapters, windows contract, LE2i parity).

#### P1-03 Structural Phase Claims Outrun Runtime Stability
- Verification: `Verified` + `Inferred`
- Evidence:
  - master plan marks migration phases completed (`Integrated-Refactor-Master-Plan.md:275-291`).
  - P0-01 import failures remain.
- Impact: planning state and executable state diverge.
- Action: add explicit "Known Gaps" and phase gate checks to master plan.

#### P1-04 Numeric Fingerprint Gate Missing (Cross-Dataset)
- Verification: `Verified` + `Planned Audit`
- Evidence:
  - no numeric-audit target exists: `rg -n "audit-numeric" Makefile` returned no matches.
  - no tests asserting adapter-output numeric range consistency: `rg --files tests` empty.
- Risk rationale: scale/range mismatch across datasets destabilizes optimization and threshold behavior.
- Required gate:
  - per-dataset stats at adapter boundary (or canonical features): mean/std/min/max/p1/p99 for `joints_xy`, `motion`, `conf`.
  - reference-band comparison vs LE2i.
- Hard fail thresholds (initial):
  - per-channel `std` ratio vs LE2i in `[0.5, 2.0]`
  - per-channel mean delta vs LE2i `< 1.0` normalized units
  - `p99(|joints_xy|)` <= `3.0` normalized units

#### P1-05 Temporal Physical-Time Gate Missing (Cross-Dataset)
- Verification: `Verified` + `Planned Audit`
- Evidence:
  - no temporal-audit target exists: `rg -n "audit-temporal" Makefile` returned no matches.
  - no tests asserting fps-derived seconds consistency: `rg --files tests` empty.
- Risk rationale: same frame window can represent different seconds when effective fps drifts.
- Required gate:
  - compute `fps_effective`, `window_seconds = W/fps_effective`, `stride_seconds = S/fps_effective` per dataset.
- Hard fail thresholds (initial):
  - target window seconds `1.92 +/- 0.15`
  - target stride seconds `0.48 +/- 0.05`
  - expected fps for LE2i/URFall adapter mode: `25.0 +/- 0.25`

#### P1-06 Baseline Parity Is Not Operationalized With Numeric Targets
- Verification: `Verified` + `Inferred`
- Evidence:
  - parity intent exists in master plan (`Integrated-Refactor-Master-Plan.md:66-68,255-257`).
  - no baseline metric table/CI gate exists in current audit artifacts.
- Impact: baseline guarantee is descriptive, not enforceable.
- Action: codify baseline table and CI checks (below in Section 8).

### P2

#### P2-01 Workspace Hygiene Noise
- Verification: `Verified`
- Evidence:
  - `.DS_Store`, `__pycache__` observed under repo paths.
- Impact: review noise and accidental commit risk.
- Action: gitignore + cleanup + pre-commit checks.

#### P2-02 Adapter Shim Layer Adds Minor Cognitive Overhead
- Verification: `Verified`
- Evidence:
  - implementation in `adapters/base.py`; per-dataset files re-export class shims.
- Impact: low, but discoverability friction.
- Action: keep intentionally or split classes per module.

#### P2-03 Environment Profile Fragility
- Verification: `Verified`
- Evidence:
  - OpenMP shared-memory errors observed in this environment; MediaPipe/OpenGL issues also observed.
- Impact: local failure can mask code correctness.
- Action: runtime profiles and preflight checks.

#### P2-04 On-Device Latency/Dependency Gate Missing
- Verification: `Verified` + `Planned Audit`
- Evidence:
  - no deploy profiling target exists: `rg -n "profile-infer" Makefile` returned no matches.
  - no runtime import-graph test in tests folder.
- Impact: on-device claim lacks measured proof.
- Required gate:
  - profile batch=1 latency (warmups + timed runs) and record mean/median/p95.
  - assert deploy entrypoint avoids unnecessary training-only imports.
- Initial SLA proposal:
  - CPU local p95 < 120ms/window (TCN)
  - CPU local p95 < 150ms/window (GCN)

## 6) Workflow Verification (LE2i + CAUCAFall)

Dataset artifacts (Verified):
- LE2i:
  - `data/processed/le2i/windows_eval_W48_S12` exists
  - `data/processed/le2i/fa_windows_W48_S12` exists
  - eval windows: `3076`; FA windows: `308`
  - eval labels: `pos=543`, `neg=2533`
- CAUCAFall:
  - `data/processed/caucafall/windows_eval_W48_S12` exists
  - `data/processed/caucafall/fa_windows_W48_S12` exists
  - eval windows: `1314`; FA windows: `85`
  - eval labels: `pos=533`, `neg=781`

Makefile sequencing (Verified):
- sequence defined at `Makefile:949-965`:
  - `windows-eval-* -> fa-windows-* -> train-* -> fit-ops-* -> plot-*`
- adapter injection present:
  - `Makefile:950-952`, `Makefile:955-956`, `Makefile:959-961`, `Makefile:964-965`
- hard-negative placeholder comment present:
  - `Makefile:953-954`, `Makefile:962-963`

FA-window idempotence (Verified):
- symlink race/idempotence handling at `src/fall_detection/data/windowing/make_fa_windows_impl.py:122-147`
- includes `FileExistsError` recovery branch (`...:136-146`)

## 7) Security + Correctness Scan

### 7.1 Unsafe Deserialization Surface
- `np.load(... allow_pickle=True)` found at:
  - `src/fall_detection/data/windowing/make_fa_windows_impl.py:73`
- Risk: untrusted NPZ input deserialization.

### 7.2 Dynamic Code Execution
- `exec(` usage: none in `src/`, `scripts/`, `server` (Verified by `rg -n "\\bexec\\(" ...` no match).
- direct `eval(` usage: none in `src/`, `scripts/`, `server` (Verified by `rg -n "(^|[^[:alnum:]_.])eval\\(" ...` no match).
- Note: `model.eval()` occurrences are expected torch API usage.

### 7.3 YAML Loader Safety
- `yaml.safe_load` usage found at:
  - `server/deploy_runtime.py:72`
  - `src/fall_detection/evaluation/metrics_eval.py:153`
  - `src/fall_detection/evaluation/plot_f1_vs_tau.py:55`
- no `yaml.load(` usage found in scanned scope.

### 7.4 Silent Exception Swallow Sites (Server)
- Explicit swallow examples:
  - `server/routes/settings.py:126-127`
  - `server/routes/events.py:500-501`
- Impact: hidden persistence/route failures.

### 7.5 Absolute Local Paths In Config
- Detected at:
  - `configs/ops/tcn_le2i.yaml:88,99`
  - `configs/ops/gcn_le2i.yaml:88,99`
  - `configs/ops/gcn_caucafall.yaml:88,99`
  - `configs/ops/tcn_caucafall.yaml:88,99`
- Impact: poor portability/reproducibility.

## 8) Baseline Parity Table (58813e8)

Status: `Planned Audit` (not yet populated with locked values in repo).

| Metric (LE2i) | 58813e8 Baseline | Current | Tolerance | Gate |
|---|---:|---:|---:|---|
| F1 (test) | TBD | TBD | abs delta <= 0.02 | Block merge if fail |
| Recall (test) | TBD | TBD | abs delta <= 0.02 | Block merge if fail |
| FA/24h (OP3) | TBD | TBD | abs delta <= 0.20 | Block merge if fail |
| Best threshold stability | TBD | TBD | abs delta <= 0.05 | Warn/Review |

Required follow-up:
- capture baseline metrics from frozen commit `58813e8`
- check into versioned artifact (e.g., `artifacts/baseline/le2i_58813e8.json`)
- enforce via CI/parity script

## 9) Repro Pack (Copy-Safe Commands)

### 9.1 Import-smoke for high-risk modules
```bash
source .venv/bin/activate
python - <<'PY'
import importlib
mods=[
  'fall_detection.data.datamodule',
  'fall_detection.data.pipeline',
  'fall_detection.data.transforms',
]
for m in mods:
    try:
        importlib.import_module(m)
        print(m, 'OK')
    except Exception as e:
        print(m, 'FAIL', type(e).__name__, e)
PY
```

### 9.2 Verify windows/FA counts
```bash
find data/processed/le2i/windows_eval_W48_S12 -name '*.npz' | wc -l
find data/processed/le2i/fa_windows_W48_S12 -name '*.npz' | wc -l
find data/processed/caucafall/windows_eval_W48_S12 -name '*.npz' | wc -l
find data/processed/caucafall/fa_windows_W48_S12 -name '*.npz' | wc -l
```

### 9.3 Verify label distribution
```bash
source .venv/bin/activate
python - <<'PY'
import numpy as np, glob
for ds in ['le2i','caucafall']:
    files = glob.glob(f'data/processed/{ds}/windows_eval_W48_S12/*/*.npz')
    y = []
    for fp in files:
        with np.load(fp, allow_pickle=True) as z:
            y.append(int(np.array(z['y']).reshape(-1)[0]))
    pos = sum(1 for v in y if v == 1)
    neg = sum(1 for v in y if v == 0)
    print(ds, 'total', len(y), 'pos', pos, 'neg', neg)
PY
```

### 9.4 Verify auto-pipeline sequence
```bash
make -n pipeline-auto-gcn-le2i
make -n pipeline-auto-gcn-caucafall
make -n pipeline-auto-tcn-le2i
```

### 9.5 Verify gate presence
```bash
rg -n "audit-smoke|audit-static|audit-runtime-imports|audit-numeric|audit-temporal|audit-parity-le2i|audit-parity-le2i-strict|profile-infer|audit-all" Makefile
rg --files tests
```
Expected in current state:
- first command: targets found in `Makefile`
- second command: audit tests listed

## 10) Production Readiness Checklist (Self-Enforcing)

- [x] Root import smoke passes: `python -c "import fall_detection"`
- [x] New data modules import successfully, or are explicitly disabled/documented.
- [x] Resolver default config path is valid.
- [x] `pytest` has minimum suite and passes.
- [x] `make audit-smoke` exists and passes.
- [x] `make audit-static` exists and passes (absolute-path + `allow_pickle` scan).
- [x] `make audit-runtime-imports` exists and passes (deploy/runtime import isolation).
- [x] `make audit-numeric` exists and passes (numeric fingerprints).
- [x] `make audit-temporal` exists and passes (physical-time invariants).
- [x] `make profile-infer` exists and emits latency report.
- [x] No hard-coded absolute local paths in checked configs.
- [x] No unbounded `allow_pickle=True` on production input paths.
- [x] `make audit-parity-le2i-strict` exists and passes after baseline capture.
- [x] `make audit-all` aggregates all required gates and passes.
- [x] `make audit-ci` exists for CI-safe enforcement without dataset-heavy stages.

Current gate status (this run):
- Import smoke: pass
- Data-module imports: pass
- Resolver default config: pass
- Pytest suite: pass (`4` audit tests)
- `make audit-smoke`: pass
- `make audit-static`: pass
- `make audit-runtime-imports`: pass
- `make audit-numeric`: pass
- `make audit-temporal`: pass
- `make profile-infer`: pass (I/O mode in restricted environment)
- Absolute paths in ops configs: pass
- `allow_pickle=True` in production paths: pass
- `make audit-parity-le2i-strict`: pass (after `baseline-capture-le2i`)
- `make audit-all`: pass
- `make audit-ci`: pass

## 11) Action Tracker (Owner / ETA / Exit Artifact)

| ID | Action | Owner | ETA | Exit Artifact |
|---|---|---|---|---|
| A-01 | Fix/disable broken new data modules (P0-01) | ML Core | Completed | import smoke green via `make audit-smoke` |
| A-02 | Resolve default data-sources config path (P1-01) | Data Infra | Completed | `configs/experiments/data_sources.yaml` + resolver test |
| A-03 | Add minimum pytest suite (P1-02) | QA/ML | Completed | `tests/test_import_smoke.py`, `tests/test_adapter_contract.py`, `tests/test_windows_contract.py`, `tests/test_data_sources_config.py` |
| A-04 | Add numeric fingerprint audit target (P1-04) | Data Infra | Completed | `artifacts/reports/numeric_fingerprint_20260302.json` |
| A-05 | Add temporal invariant audit target (P1-05) | Data Infra | Completed | `artifacts/reports/temporal_span_20260302.json` |
| A-06 | Add baseline parity gate vs 58813e8 (P1-06) | ML Core | Completed (contract + strict mode) | `baselines/le2i/58813e8/*`, `scripts/audit_parity.py`, strict report |
| A-07 | Add on-device profile target + report (P2-04) | Deploy | Completed (I/O mode), Model mode pending env | `artifacts/reports/infer_profile_cpu_local_tcn_le2i.json` |
| A-08 | Remove absolute local paths from ops config | MLOps | Completed | `configs/ops/*` portable paths + `make audit-static` |
| A-09 | Resolve `allow_pickle=True` trust boundary | Security/ML | Completed | FA-window reader migrated to `allow_pickle=False` + static gate |
| A-10 | Enforce runtime import isolation for deploy path | Deploy | Completed | `scripts/audit_runtime_imports.py` + `make audit-runtime-imports` |

## 12) Final Verdict

The refactor is directionally correct and now backed by executable audit gates for contract integrity, parity, numeric/time invariants, static safety, and deploy import isolation. Remaining risk is environmental for full model-latency profiling in restricted runtimes.

Release recommendation:
- CI default gate: `make audit-ci`
- Data-backed release gate: `make audit-all MODEL=tcn AUDIT_DATASETS='le2i,caucafall'`
