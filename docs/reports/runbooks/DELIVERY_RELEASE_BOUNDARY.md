# Delivery Release Boundary

This document defines the release boundary for the current delivery-grade full-stack setup.

The repository currently contains a large amount of unrelated research churn:

- archived operating-point sweeps
- thesis/report edits
- figures
- course/tutorial material
- experimental configs

Those files are not part of the minimum releasable product for:

- one-command startup
- online monitor runtime
- frontend monitor behavior
- backend API contract stability

## Release Core

These files make up the current delivery-grade release boundary.

### Startup and Developer Entry Points

- `Makefile`
- `README.md`
- `scripts/start_fullstack.sh`
- `scripts/bootstrap_dev.sh`
- `scripts/release_doctor.sh`

### Backend Runtime and API Contract

- `server/db.py`
- `server/deploy_runtime.py`
- `server/routes/monitor.py`
- `server/routes/events.py`

### Frontend Runtime

- `apps/src/pages/monitor/hooks/usePoseMonitor.js`
- `apps/src/App.js`

### Release Hygiene

- `.gitignore`
- `tests/conftest.py`

## Release Docs

These docs are safe to ship with the release because they describe the active online behavior:

- `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`
- `docs/reports/runbooks/ONLINE_FRONTEND_SMOKE_CHECKLIST.md`
- `docs/reports/runbooks/DELIVERY_RELEASE_BOUNDARY.md`

## Explicitly Out Of Scope

Do not treat these as required release payload unless there is a separate documentation or research deliverable:

- `artifacts/**`
- `artifacts/figures/**`
- `configs/ops/archive/**`
- `configs/ops/grid_*`
- `configs/ops/cross_*`
- `configs/ops/*muvim*`
- `docs/project_targets/**`
- `docs/course/**`
- `docs/tutorial/**`
- `teaching/**`
- ad-hoc plotting scripts
- experiment-only YAMLs and sweep JSONs

## Current Release Commands

Print the current release candidate subset:

```bash
bash scripts/release_manifest.sh
```

Run the static release checks:

```bash
make release-check
```

## Recommended Commit Strategy

For a clean delivery commit, stage only the release-core and release-doc files.

Example:

```bash
git add \
  .gitignore \
  Makefile \
  README.md \
  apps/src/App.js \
  apps/src/pages/monitor/hooks/usePoseMonitor.js \
  server/db.py \
  server/deploy_runtime.py \
  server/routes/events.py \
  server/routes/monitor.py \
  scripts/bootstrap_dev.sh \
  scripts/release_doctor.sh \
  scripts/start_fullstack.sh \
  tests/conftest.py \
  docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md \
  docs/reports/runbooks/ONLINE_FRONTEND_SMOKE_CHECKLIST.md \
  docs/reports/runbooks/DELIVERY_RELEASE_BOUNDARY.md
```

This avoids accidentally bundling unrelated research changes into the software delivery branch.
