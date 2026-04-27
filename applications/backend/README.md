# Server (FastAPI)

This folder contains the **server-side** component that bridges:

1) the front-end (captures RGB, extracts skeletons), and
2) the trained fall-detection models (TCN / GCN / TCN+GCN).

The server accepts **skeleton windows** and returns:
- per-model probability `p_fall` (and optional MC-dropout uncertainty),
- triage state: `not_fall | uncertain | fall`,
- alert level: `none | possible_fall | fall_detected`.

It also keeps an **in-memory per-session** state machine for temporal rules.

## Backend structure

The backend is organized around explicit assembly and configuration boundaries:

- `applications/backend/application.py`
  - FastAPI application factory and route registration
- `applications/backend/config.py`
  - environment-backed runtime configuration and path resolution
- `applications/backend/schemas.py`
  - shared API payload models
- `applications/backend/routes/`
  - HTTP/WebSocket transport layer
- `applications/backend/services/`
  - runtime and response orchestration
- `applications/backend/repositories/`
  - persistence-facing read/write logic
- `applications/backend/core.py`
  - shared backend helpers and in-memory fallback state

## Run

From the repo root:

```bash
source .venv/bin/activate
pip install -r requirements_server.txt
uvicorn applications.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

## Front-end -> server payload

POST ` /api/monitor/predict_window `

- `mode`: `"tcn" | "gcn" | "hybrid"`
- `model_tcn`, `model_gcn`: optional model IDs (strings) when mode is `hybrid`
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
- `POST /twilio/webhook`

Versioned compatibility aliases are also available under `/api/v1/*` for
health, monitor, settings, events, specs, dashboard, operating points,
caregivers, and notifications endpoints.

## Runtime config source of truth

- FastAPI live inference uses deploy specs discovered from `ops/configs/ops/*.yaml`
  (checkpoint path + `ops` + `alert_cfg`).
- `ops/configs/deploy_modes.yaml` is used by offline deploy scripts (for example
  `ml/src/fall_detection/deploy/run_modes.py`) and is not the primary config source
  for `applications/backend/routes/monitor.py`.

## Integration audit commands

From repo root:

```bash
make audit-api-contract
make audit-api-smoke
make audit-integration-contract
```

The server is **stateless per request**, but keeps a **session state** keyed by
`session_id` to implement possible/confirmed fall logic.

## Safe Guard notifications

The repository now includes a threshold-aware "Safe Guard" notification layer
under `applications/backend/notifications/`.

Current implemented delivery path:

- Telegram caregiver alert
- optional generated text summary inside the Telegram message
- local SQLite audit trail for notification attempts and feedback

Reserved / future-work channels:

- SMS
- phone-call escalation
- email escalation

Design goals:

- classify persisted alert events into Tier 1 / Tier 2 / Tier 3
- keep outbound notifications off the inference request path
- audit delivery attempts in local SQLite
- keep MySQL `events` as the canonical event source of truth

### Enable it

From repo root:

```bash
cp .env.example .env
source .venv/bin/activate
uvicorn applications.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Important environment variables:

```bash
DB_BACKEND=sqlite
SQLITE_PATH=applications/backend/cloud_demo.sqlite3
SAFE_GUARD_ENABLED=1
SAFE_GUARD_SQLITE_PATH=applications/backend/safe_guard_notifications.sqlite3
HIGH_CONF_MARGIN=0.08
LOW_UNCERTAINTY_THRESHOLD=0.05
HIGH_UNCERTAINTY_THRESHOLD=0.15
ALERT_COOLDOWN_SECONDS=60
TELEGRAM_BOT_TOKEN=
CAREGIVER_TELEGRAM_CHAT_ID=
AI_REPORTS_ENABLED=1
AI_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
OPENAI_TIMEOUT_S=12
APP_BASE_URL=http://127.0.0.1:3000
CORS_ALLOWED_ORIGINS=http://127.0.0.1:3000
```

If Telegram credentials are not configured, Safe Guard still runs in
audit mode and records skipped delivery attempts in SQLite.

For Render-style cloud deployment, prefer:

- `DB_BACKEND=sqlite`
- `SQLITE_PATH` on a persistent disk mount
- caregiver Telegram Chat ID stored in the app database
- Telegram bot token and summary-provider keys stored as Render environment variables
- Render persistent disks require a paid web-service plan

Telegram behavior:

- a caregiver Telegram alert is attempted for each fall-like event
- the message includes a generated event analysis section when the selected summary provider key is configured
- if summary generation fails or is disabled, the message falls back to a deterministic summary
- SMS, phone, and email escalation remain future-work channels rather than the current implemented path

### Runtime integration

Safe Guard is triggered after a new fall event has already been persisted to
the main `events` table inside `applications/backend/routes/monitor.py`.

The monitor payload can now include:

- `location`: optional event location label used in caregiver notifications

The current front-end integration sends:

- `camera_live` for live camera monitoring
- replay video filename for replay mode

### Local audit database

Safe Guard writes local audit records to SQLite. By default:

```text
applications/backend/safe_guard_notifications.sqlite3
```

This local SQLite database stores:

- notification event audit rows
- per-channel delivery attempts
- caregiver feedback records

It does not replace MySQL `events`.

### Demo script

Run the minimal demo:

```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/ml/src:$(pwd)" python3 ops/scripts/demo_safe_guard_notifications.py
```

The demo simulates:

- one Tier 1 event
- one Tier 2 event
- one Tier 3 event

and writes results to:

```text
applications/backend/safe_guard_demo.sqlite3
```

### Legacy webhook surface

`POST /twilio/webhook` still exists as legacy scaffolding, but it is not part
of the current defended Telegram-first delivery path.

## Safe Guard acceptance checklist

Use this checklist before treating the feature as release-ready.

### 1. Basic startup

- copy `.env.example` to `.env`
- set `SAFE_GUARD_ENABLED=1`
- start the backend:

```bash
source .venv/bin/activate
uvicorn applications.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Expected:

- `/api/health` responds successfully
- backend starts without import errors
- `applications/backend/safe_guard_notifications.sqlite3` is created when Safe Guard handles events

### 2. Demo script

Run:

```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/ml/src:$(pwd)" python3 ops/scripts/demo_safe_guard_notifications.py
```

Expected:

- one Tier 1 event printed
- one Tier 2 event printed
- one Tier 3 event printed
- `applications/backend/safe_guard_demo.sqlite3` is created

### 3. SQLite audit verification

Check the local SQLite database for:

- `notification_events`
- `notification_attempts`
- `caregiver_feedback`

Verify key fields exist and are populated:

- `event_id`
- `probability`
- `threshold`
- `margin`
- `uncertainty`
- `alert_tier`
- `telegram_status`

If Telegram is not configured, delivery status should show skipped or failed audit entries without crashing the server.

### 4. Runtime monitor integration

Trigger a real fall event through the monitor UI.

Expected:

- MySQL `events` receives the new event
- SQLite `notification_events` receives the same event reference
- event metadata includes:
  - `threshold`
  - `margin`
  - `uncertainty`
  - `location`

Current front-end location behavior:

- live camera -> `camera_live`
- replay mode -> replay filename

### 5. Non-blocking behavior

While a fall event is processed:

- `/api/monitor/predict_window` must still return quickly
- notification failures must be logged only
- notification failures must not return 500 from the inference path

### 6. Deduplication

Trigger repeated alerts for the same incident.

Expected:

- duplicate Telegram sends are suppressed
- the same `event_id` should not be re-delivered on the same channel
- cooldown behavior should suppress repeated alert delivery inside the configured window

### 7. Tier logic

Validate three cases:

- Tier 1: high margin and low uncertainty
- Tier 2: alert-worthy but not Tier 1
- Tier 3: non-alert-worthy

Expected:

- Tier 1 -> Telegram alert
- Tier 2 -> Telegram alert if currently configured by policy
- Tier 3 -> no caregiver notification, audit only

### 8. Real channel integration

After configuring:

- `TELEGRAM_BOT_TOKEN`
- caregiver `telegram_chat_id`

Verify:

- Telegram message is short, readable, and includes event context
<<<<<<< HEAD
- AI summary appears when configured
=======
- generated summary appears when configured
>>>>>>> feature/monitor-architecture-refactor
- failure in the Telegram channel does not block inference or event persistence

### 9. Legacy webhook feedback

The Twilio webhook remains present only as legacy surface and should not be
treated as part of the current release path.

### 10. Regression check

Re-check these existing features:

- Event History still loads correctly
- Event History status update still works
- monitor inference still works in live mode
- replay mode still persists events when configured to persist
- setting `SAFE_GUARD_ENABLED=0` does not break the old path

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
