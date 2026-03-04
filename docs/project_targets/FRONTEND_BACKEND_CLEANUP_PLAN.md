# Frontend Backend Cleanup Plan

## Scope
- Clean and align frontend/backend contracts for live monitoring.
- Limit runtime product scope to:
  - Datasets: `caucafall`, `le2i`
  - Models: `TCN`, `GCN`
- Remove legacy hybrid/unused dataset paths from user-facing logic.

## Objectives
- Single, stable API contract between UI and server for monitor/settings/events.
- No frontend calls to unsupported modes/datasets.
- Backend normalization and defaults must be strict and predictable.
- Fix observed UX issues:
  - Start button should work on first click.
  - Idle user should not be flooded by false fall events due to unsafe default alert gating.

## Current Issues (Audit Summary)
- Backend route contracts still expose legacy values:
  - `server/routes/operating_points.py` still allows `HYBRID` and defaults `dataset_code` to `muvim`.
  - `server/core.py` `_derive_ops_params_from_yaml` still has hybrid merge branch.
- Frontend still contains hybrid mappings:
  - `apps/src/lib/modelCodes.js`
  - `apps/src/pages/monitor/utils.js`
  - `apps/src/pages/monitor/hooks/useOperatingPointParams.js`
  - `apps/src/pages/Events.js`
- Some backend defaults/comments still reference hybrid:
  - `server/routes/dashboard.py`
  - `server/routes/events.py`
- Monitor endpoint supports legacy mode parsing and dual branches; needs strict model scope while keeping safety gating behavior.

## Phases

### Phase 1: Contract Lock (P0)
- Restrict API model/dataset surface:
  - `operating_points`: only `TCN|GCN`, dataset normalized to `caucafall|le2i`.
  - settings payload descriptions updated accordingly.
- Acceptance:
  - `/openapi.json` shows no `HYBRID` in these route descriptions.
  - Invalid dataset/model inputs normalize to supported defaults instead of drifting.

### Phase 2: Backend Logic Cleanup (P0)
- Normalize monitor defaults and mode parsing:
  - unknown/legacy mode -> `tcn`
  - supported modes remain `tcn|gcn`
- Keep deployment-safe behavior:
  - TCN monitor response uses safe-policy state when available.
  - Event persistence in TCN mode uses safe-policy alert gate if configured.
- Acceptance:
  - No 404 mismatch for monitor prediction endpoint.
  - Returned payload always includes deterministic `effective_mode` in `tcn|gcn`.

### Phase 3: Frontend Logic Cleanup (P0)
- Remove hybrid options and mappings from settings/monitor/events flows.
- Ensure operating point fetch only requests `TCN|GCN`.
- Keep first-click start behavior (pending-start latch) intact.
- Acceptance:
  - UI only shows `TCN`, `GCN`, `CAUCAFall`, `LE2I`.
  - Start works on first click after controller init.

### Phase 4: Consistency & Hygiene (P1)
- Replace stale defaults/comments in server modules (`events`, `dashboard`).
- Keep DB compatibility (no schema-breaking migration in this cleanup).
- Acceptance:
  - No user-facing hybrid wording in active UI/API paths.

## Validation Commands
- Python syntax:
  - `python -m py_compile server/core.py server/deploy_runtime.py server/routes/monitor.py server/routes/settings.py server/routes/operating_points.py server/routes/events.py server/routes/dashboard.py`
- Frontend syntax:
  - `node --check apps/src/lib/modelCodes.js`
  - `node --check apps/src/pages/monitor/utils.js`
  - `node --check apps/src/pages/monitor/hooks/useOperatingPointParams.js`
  - `node --check apps/src/pages/settings/SettingsPage.js`
  - `node --check apps/src/pages/Events.js`
  - `node --check apps/src/monitoring/MonitoringContext.js`
- Runtime contract check:
  - `curl -s http://127.0.0.1:8000/openapi.json | grep -E \"operating_points|monitor/predict_window|dataset_code|model_code\"`
  - `curl -i -X POST http://127.0.0.1:8000/api/monitor/predict_window -H \"Content-Type: application/json\" -d '{...}'`

## Deliverables
- This plan document.
- Minimal code patches for backend/frontend contract alignment.
- Validation result summary (what passed/what remains).
