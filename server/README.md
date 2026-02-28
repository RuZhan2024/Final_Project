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
- Preferred compact path for live use:
  - `raw_t_ms`: `[N]`
  - `raw_xy_flat`: `[N*J*2]`
  - `raw_conf_flat`: `[N*J]`
  - `raw_joints`: `J` (typically `33`)
- `fps`: optional (defaults to model's `fps_default`)
- `timestamp_ms`: optional (recommended)
- `mc_sigma_tol`: optional positive float for adaptive MC early-stop
- `mc_se_tol`: optional positive float for adaptive MC standard-error early-stop

Response includes:

- `models.<name>.mc_n_used`: MC samples used for that model
- top-level `mc_n_used`: compact per-model map

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

## Session Memory Guardrails

The monitor session cache is in-memory and now has built-in cleanup.

- `SESSION_TTL_S` (default `1800`): remove sessions inactive for this many seconds.
- `SESSION_MAX_STATES` (default `1000`): hard cap on retained sessions.

Example:

```bash
export SESSION_TTL_S=900
export SESSION_MAX_STATES=300
```

## Backend tests

From repo root:

```bash
make install-dev
make test-server
make test-server-cov
```

Coverage threshold defaults to `70` and can be overridden:

```bash
make test-server-cov COVERAGE_MIN=75
```

`make test-server-cov` also writes `coverage.xml` for CI artifacts/reporting.

CI coverage policy:
- Pull requests and non-`main` branches require at least `75%`.
- Pushes to `main` require at least `81%`.
