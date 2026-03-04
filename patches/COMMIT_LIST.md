# Patch Commit List

Branch: `fix/p0-runtime-parity-autopipeline`

## Applied Fixes

1. `58ef76f` — `fix(runtime): use canonical two-stream GCN feature split`
- File: `server/deploy_runtime.py`
- Validation:
  - `python -m compileall server/deploy_runtime.py src/fall_detection/core/features.py`
  - `python -c "import server.app"`
  - `python -c "from server.deploy_runtime import predict_spec; print('ok')"`

2. `0097126` — `fix(make): run fit-ops and eval explicitly in auto pipelines`
- File: `Makefile`
- Validation:
  - `make -Bn pipeline-auto-tcn-le2i | rg "fit-ops-|eval-|plot-|train-tcn"`
  - `make -Bn pipeline-auto-gcn-le2i | rg "fit-ops-gcn-|eval-gcn-|plot-gcn-|train-gcn"`

3. `91e002e` — `fix(repro): emit repo-relative paths in manifest`
- File: `scripts/reproduce_claim.py`
- Validation:
  - `python -m compileall scripts/reproduce_claim.py`
  - `python scripts/reproduce_claim.py --dataset caucafall --model gcn --run 0 --out_dir /tmp/repro_portability_check`
  - `rg -n '/Users/' /tmp/repro_portability_check/manifest.json`

4. `ea1cf08` — `docs(apps): fix health check endpoint to /api/health`
- File: `apps/README.md`
- Validation:
  - `rg -n "localhost:8000/(health|api/health)" apps/README.md`
