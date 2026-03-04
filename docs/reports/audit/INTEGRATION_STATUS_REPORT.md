# INTEGRATION_STATUS_REPORT

## Backend Import Hygiene
Status: **Mostly PASS with cleanup debt**

- Backend entrypoint is thin and clean: `server/app.py` imports `app` from `server.main`.
- Route assembly in `server/main.py` is explicit and module-local.
- No backend-side legacy `core.*` import drift detected for runtime paths.
- Cleanup debt: path-bootstrap hacks remain in some ML scripts (not backend runtime critical).

## Backend Routes (Observed)
- Health:
  - `GET /api/health`
  - `GET /api/v1/health`
- Monitor:
  - `POST /api/monitor/predict_window`
  - `POST /api/v1/monitor/predict_window`
  - `POST /api/monitor/reset_session`
  - `POST /api/v1/monitor/reset_session`
- Settings:
  - `GET/PUT /api/settings`
  - `GET/PUT /api/v1/settings`
- Events:
  - `GET /api/events`
  - `GET /api/events/summary`
  - `PUT /api/events/{event_id}/status`
  - `POST /api/events/test_fall`
  - `POST /api/events/{event_id}/skeleton_clip`
- Plus dashboard/specs/operating_points/notifications/caregivers mirrors under `/api` and `/api/v1`.

## Frontend API Calls vs Backend Contract
Status: **PASS**

Frontend calls are `/api/*` and align with backend routes.
Examples in `apps/src`:
- `/api/settings`
- `/api/events`, `/api/events/summary`, `/api/events/{id}/status`
- `/api/monitor/predict_window`
- `/api/monitor/reset_session`
- `/api/spec`, `/api/summary`, `/api/operating_points`

No `/api/v1/*` usage in frontend is required because backend exposes both.

## CORS and Env Config
Status: **PASS**

- Default allowed origins include localhost 3000 and 5173 in `server/main.py`.
- Override supported via `CORS_ALLOWED_ORIGINS` env var.
- Frontend base URL: `REACT_APP_API_BASE` in `apps/src/lib/config.js` (default `http://localhost:8000`).

## Critical Integration Drift

### 1) Runtime preprocessing parity risk for two-stream GCN (P0)
- Training/eval canonical split uses `split_gcn_two_stream(X, feat_cfg)`.
- Runtime (`server/deploy_runtime.py`) hard-codes slices for `F in {2,3,4,5}` and fails otherwise.
- This can break inference when feature config includes bone/bone_len channels.

### 2) Auto-pipeline target chain drift (P0)
- `pipeline-auto-tcn-*` and `pipeline-auto-gcn-*` currently run windows/train/plot only.
- They do not invoke fit-ops/eval despite the documented text in `make help`.

## API Smoke Commands

### Health
```bash
curl -s http://localhost:8000/api/health
curl -s http://localhost:8000/api/v1/health
```

### Inference (JSON payload)
Route expects `MonitorPredictPayload` (`server/core.py`), accepts `xy/conf` or preferred `raw_*` fields.

```bash
curl -s -X POST http://localhost:8000/api/monitor/predict_window \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo",
    "dataset_code": "le2i",
    "mode": "tcn",
    "target_T": 48,
    "xy": [[[0.0,0.0],[0.0,0.0]]],
    "conf": [[1.0,1.0]],
    "persist": false
  }'
```

## Minimal Demo Start Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_server.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

```bash
cd apps
npm install
REACT_APP_API_BASE=http://localhost:8000 npm start
```

## Integration Verdict
- API ↔ frontend contract: **Green**
- ML runtime parity: **Yellow/Red** until GCN two-stream split parity fix lands
- End-to-end demo reliability: **Yellow**
