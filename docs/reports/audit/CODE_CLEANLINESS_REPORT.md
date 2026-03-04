# CODE_CLEANLINESS_REPORT

## Summary
Codebase is generally clean and functional, but there are contract and hygiene gaps that reduce industrial readiness.

## Top Findings (Ranked)

1. **Runtime two-stream GCN hard-coded feature slicing (critical contract debt)**
- Evidence: `server/deploy_runtime.py:458-476`.
- Why it matters: violates canonical feature contract and can crash on valid feature sets.

2. **Auto-pipeline documentation/implementation drift**
- Evidence: `Makefile:535-536` vs `Makefile:1111-1125`.
- Why it matters: silently skips fit/eval in single-command flow.

3. **Absolute path leakage in reproducibility manifest**
- Evidence: `artifacts/repro/RESULTS_20260302_204401/manifest.json:4`.
- Why it matters: non-portable artifacts.

4. **Frontend quick-check docs use non-existent health route**
- Evidence: `apps/README.md:45` uses `/health`; actual is `server/routes/health.py:12` `/api/health`.
- Why it matters: avoidable demo failures.

5. **Residual `sys.path` bootstrap hacks in script modules**
- Evidence: `src/fall_detection/training/train_tcn.py:24-32`, `train_gcn.py:23-31`, `evaluation/plot_*`.
- Why it matters: package hygiene and import determinism.

6. **Broad exception usage in evaluation path (error observability reduction)**
- Evidence: `src/fall_detection/evaluation/metrics_eval.py` has multiple broad `except Exception` blocks (for example lines around `70`, `156`, `686`).
- Why it matters: may hide malformed YAML/metrics conditions.

7. **Broad exception usage in feature/meta utilities**
- Evidence: `src/fall_detection/core/features.py:108,117,126` and `src/fall_detection/core/models.py:45,50`.
- Why it matters: debugging degraded when malformed inputs appear.

8. **Audit scripts include many broad exceptions with silent fallback behavior**
- Evidence: `scripts/audit_profile_budget.py:55`, `scripts/audit_runtime_imports.py:15`, etc.
- Why it matters: false-green audit risk.

9. **No static typing gate configured (mypy/pyright absent)**
- Evidence: no `mypy`/`pyright` config detected in root files.
- Why it matters: contract regressions reach runtime.

10. **pytest unstable in current env due to torch import abort**
- Evidence: `pytest -q` fatal abort at torch import during collection.
- Why it matters: CI confidence gap.

## Additional Scan Results
- `allow_pickle=True`: no insecure uses found in src/scripts/server/tests scans.
- `except Exception: pass` exact pattern: not found in scanned app paths.
- Tracked `.DS_Store` / `__pycache__`: none via `git ls-files`.

## Quick-Fix Plan
- P0: fix runtime feature split + auto-pipeline drift.
- P1: improve portability docs/manifests and frontend README endpoint.
- P2: reduce broad exception swallowing and remove path-bootstrap helpers where possible.

## Suggested Validation Commands
```bash
python -m compileall .
python -c "import fall_detection; import server.app"
make -Bn pipeline-auto-gcn-le2i
pytest -q
```
