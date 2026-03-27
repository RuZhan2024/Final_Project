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

## Safe Guard notifications

The repository now includes a threshold-aware "Safe Guard" notification layer
under `server/notifications/`.

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
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Important environment variables:

```bash
DB_BACKEND=sqlite
SQLITE_PATH=server/cloud_demo.sqlite3
SAFE_GUARD_ENABLED=1
SAFE_GUARD_SQLITE_PATH=server/safe_guard_notifications.sqlite3
HIGH_CONF_MARGIN=0.08
LOW_UNCERTAINTY_THRESHOLD=0.05
HIGH_UNCERTAINTY_THRESHOLD=0.15
ALERT_COOLDOWN_SECONDS=60
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_PHONE=
CAREGIVER_PHONE=
RESEND_API_KEY=
EMAIL_FROM=
CAREGIVER_EMAIL=
AI_REPORTS_ENABLED=1
AI_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
OPENAI_TIMEOUT_S=12
APP_BASE_URL=http://127.0.0.1:3000
CORS_ALLOWED_ORIGINS=http://127.0.0.1:3000
```

If Twilio / Resend credentials are not configured, Safe Guard still runs in
audit mode and records skipped delivery attempts in SQLite.

For Render-style cloud deployment, prefer:

- `DB_BACKEND=sqlite`
- `SQLITE_PATH` on a persistent disk mount
- caregiver contact details stored in the app database
- Resend and AI provider keys stored as Render environment variables
- Render persistent disks require a paid web-service plan

Email behavior:

- a detailed caregiver email is attempted for each fall-like event
- the email includes an AI-generated event analysis section when the selected AI provider key is configured
- if AI generation fails or is disabled, the email falls back to a deterministic summary
- SMS and phone-call escalation remain optional and are controlled by settings

### Runtime integration

Safe Guard is triggered after a new fall event has already been persisted to
the main `events` table inside `server/routes/monitor.py`.

The monitor payload can now include:

- `location`: optional event location label used in caregiver notifications

The current front-end integration sends:

- `camera_live` for live camera monitoring
- replay video filename for replay mode

### Local audit database

Safe Guard writes local audit records to SQLite. By default:

```text
server/safe_guard_notifications.sqlite3
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
PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/demo_safe_guard_notifications.py
```

The demo simulates:

- one Tier 1 event
- one Tier 2 event
- one Tier 3 event

and writes results to:

```text
server/safe_guard_demo.sqlite3
```

### Twilio feedback webhook

Safe Guard exposes:

```text
POST /twilio/webhook
```

Supported reply codes:

- `1` -> mark event as false alarm / false positive
- `2` -> mark event as confirmed fall
- `3` -> mark event as resolved / assistance provided

Correlation behavior:

- prefer explicit `event_id`
- otherwise try to parse `Ref:<event_id>` from SMS body
- otherwise fall back to the most recent unresolved local Safe Guard audit event

## Safe Guard acceptance checklist

Use this checklist before treating the feature as release-ready.

### 1. Basic startup

- copy `.env.example` to `.env`
- set `SAFE_GUARD_ENABLED=1`
- start the backend:

```bash
source .venv/bin/activate
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Expected:

- `/api/health` responds successfully
- backend starts without import errors
- `server/safe_guard_notifications.sqlite3` is created when Safe Guard handles events

### 2. Demo script

Run:

```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/demo_safe_guard_notifications.py
```

Expected:

- one Tier 1 event printed
- one Tier 2 event printed
- one Tier 3 event printed
- `server/safe_guard_demo.sqlite3` is created

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
- `phone_status`
- `sms_status`
- `email_status`

If Twilio / Resend are not configured, delivery status should show skipped or failed audit entries without crashing the server.

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

- duplicate phone / SMS / email sends are suppressed
- the same `event_id` should not be re-delivered on the same channel
- cooldown behavior should suppress repeated alert delivery inside the configured window

### 7. Tier logic

Validate three cases:

- Tier 1: high margin and low uncertainty
- Tier 2: alert-worthy but not Tier 1
- Tier 3: non-alert-worthy

Expected:

- Tier 1 -> email always, phone if enabled, SMS if enabled
- Tier 2 -> email always, no phone, SMS if enabled
- Tier 3 -> no caregiver notification, audit only

### 8. Real channel integration

After configuring:

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM_PHONE`
- `CAREGIVER_PHONE`
- `RESEND_API_KEY`
- `EMAIL_FROM`
- `CAREGIVER_EMAIL`

Verify:

- phone call content is short and non-technical
- SMS is brief and includes event context
- email includes the full event detail payload
- failure in one channel does not block the others

### 9. Webhook feedback

Test:

```text
POST /twilio/webhook
```

Reply mapping:

- `1` -> false alarm
- `2` -> confirmed fall
- `3` -> resolved / assistance provided

Expected:

- canonical MySQL event status updates
- SQLite feedback row is recorded
- resolved events are marked no longer unresolved in SQLite

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
