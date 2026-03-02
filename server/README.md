# Server (FastAPI)

This folder contains the **server-side** component that bridges:

1) the front-end (captures RGB, extracts skeletons), and
2) the trained fall-detection models (TCN / GCN / TCN+GCN).

The server accepts **skeleton windows** and returns:
- per-model probability `p_fall` (and optional MC-dropout uncertainty),
- triage state: `not_fall | uncertain | fall`,
- alert level: `none | possible_fall | fall_detected`.

It also keeps an **in-memory per-session** state machine for temporal rules.

## Run

From the repo root:

```bash
source .venv/bin/activate
pip install -r requirements_server.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Front-end -> server payload

POST ` /api/monitor/predict_window `

- `mode`: `"tcn" | "gcn" | "dual"`
- `model_tcn`, `model_gcn`: optional model IDs (strings) when mode is `dual`
- preferred live payload:
  - `raw_t_ms`: capture timestamps in milliseconds, shape `[N]`
  - `raw_xy`: pose coordinates, shape `[N,J,2]`
  - `raw_conf`: optional confidence, shape `[N,J]`
  - `window_end_t_ms`: optional explicit window end timestamp
- runtime windowing params:
  - `target_T`: fixed window length used for inference (default `48`)
  - `dataset_code`: dataset profile (`le2i | urfd | caucafall | muvim`) to resolve expected FPS
- compatibility fallback:
  - `xy` and `conf` are still accepted when `raw_*` is unavailable

Response includes:
- `triage_state`: `not_fall | uncertain | fall`
- `models`: per-model output (`mu`, `sigma`, triage details)
- `event_id`: persisted event id when `persist=true` and a new fall starts

## Additional UI helper endpoints

- `PUT /api/events/{event_id}/status`
- `POST /api/notifications/test`

Versioned compatibility aliases are also available under `/api/v1/*` for
health, monitor, settings, events, specs, dashboard, operating points,
caregivers, and notifications endpoints.

## Runtime config source of truth

- FastAPI live inference uses deploy specs discovered from `configs/ops/*.yaml`
  (checkpoint path + `ops` + `alert_cfg`).
- `configs/deploy_modes.yaml` is used by offline deploy scripts (for example
  `src/fall_detection/deploy/run_modes.py`) and is not the primary config source
  for `server/routes/monitor.py`.

## Integration audit commands

From repo root:

```bash
make audit-api-contract
make audit-api-smoke
make audit-integration-contract
```

The server is **stateless per request**, but keeps a **session state** keyed by
`session_id` to implement possible/confirmed fall logic.

## DB (optional)

If you want the DB-backed endpoints (dashboard/events), set:

```bash
export DB_HOST=...
export DB_PORT=3306
export DB_USER=...
export DB_PASS=...
export DB_NAME=...
```

and install PyMySQL:

```bash
pip install pymysql
```
