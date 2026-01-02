# Server (FastAPI)

This folder contains the **server-side** component that bridges:

1) the front-end (captures RGB, extracts skeletons), and
2) the trained fall-detection models (TCN / GCN / TCN+GCN).

The server accepts **skeleton windows** (`xy`, `conf`) and returns:
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
- `xy`: shape `[T,33,2]`
- `conf`: shape `[T,33]`
- `fps`: optional (defaults to model's `fps_default`)
- `timestamp_ms`: optional (recommended)

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
