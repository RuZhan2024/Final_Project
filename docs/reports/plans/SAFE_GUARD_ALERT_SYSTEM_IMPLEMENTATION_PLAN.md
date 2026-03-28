# SAFE_GUARD_ALERT_SYSTEM_IMPLEMENTATION_PLAN

## Purpose

This document translates the proposed "Safe Guard Intelligent Alert System" into an implementation plan that fits the current architecture of this repository.

It is not a generic design note. It is a project-specific implementation contract for this codebase.

## Current Project Reality

The current runtime pipeline already has these properties:

- live inference runs through `server/routes/monitor.py`
- event persistence already exists in the MySQL-backed `events` flow
- event review/status updates already exist through `/api/events/{event_id}/status`
- a minimal DB-backed notification logging path already exists in `server/notifications_service.py`
- event deduplication and cooldown already exist in the monitor path before event persistence

The proposed Safe Guard system must therefore extend the existing design rather than replace it.

## Core Decision

The Safe Guard system will be implemented as a notification and audit layer on top of the existing event creation flow.

It will not:

- replace the current event persistence contract
- introduce SQLite as the primary event source of truth
- classify raw model windows independently of the existing runtime alert policy

It will:

- consume threshold-aware runtime event outputs after the current monitor policy has decided an alert-worthy event exists
- assign a notification tier for that event
- route notifications asynchronously
- store notification audit records locally in SQLite
- expose caregiver feedback webhook handling that can update the canonical event review state

## Why The Original Spec Needs Adjustment

### 1. `is_fall` alone is not sufficient in this project

The current runtime path does not rely on a single boolean fall decision. It uses:

- `triage_state`
- `safe_alert`
- `recall_alert`
- per-model outputs
- operating-point thresholds
- quality gates
- motion gates
- cooldown / dedup logic

Therefore the Safe Guard classifier must sit after the existing event-generation logic, not before it.

### 2. Tier classification must be threshold-aware and policy-aware

The original requirement correctly says severity must not be hardcoded using absolute probability bands.

For this repository, that rule must be strengthened:

- classification must be relative to the active operating threshold
- classification must also respect the current runtime policy output
- a high-uncertainty non-event must not become a notification candidate purely because uncertainty is high

### 3. SQLite must not become a second primary event database

This project already uses MySQL `events` for:

- Event History
- review state
- event metadata
- clip attachment references

If SQLite also stores the main event record independently, the system will split into two truth sources.

That is not acceptable for this codebase.

So SQLite will be used only for:

- notification audit
- dispatch attempt tracking
- webhook feedback caching / correlation support
- optional daily summary support

MySQL `events` remains the canonical event record.

## Recommended Architecture

### Canonical event flow

1. `server/routes/monitor.py` decides a new event should be persisted.
2. A row is inserted into MySQL `events`.
3. A new Safe Guard notification manager is called with the persisted event context.
4. The notification manager classifies the event into Tier 1 / Tier 2 / Tier 3 using threshold-aware logic.
5. Notification work is pushed to a background queue.
6. Phone / SMS / email sending happens outside the inference request path.
7. Delivery attempts and outcomes are recorded in SQLite.
8. Caregiver feedback webhook updates MySQL event status and records feedback in SQLite.

### Responsibility split

- MySQL:
  - canonical event record
  - event review state
  - event metadata used by UI

- SQLite:
  - notification audit log
  - channel attempt state
  - dedup cache support
  - caregiver feedback notes / mirrored dispatch state

- In-memory queue + worker:
  - non-blocking outbound dispatch

## Revised Input Contract

The Safe Guard manager should not classify from only:

- `is_fall`
- `probability`
- `uncertainty`
- `threshold`

Instead it should accept a project-aligned event payload with at least:

- `event_id`
- `resident_id`
- `timestamp`
- `location`
- `model_code`
- `dataset_code`
- `op_code`
- `triage_state`
- `safe_alert`
- `recall_alert`
- `probability`
- `uncertainty`
- `threshold`
- `margin`

Recommended runtime source:

- `probability`:
  - use the persisted event score or the primary model display probability
- `threshold`:
  - use the active runtime threshold for the operating point
- `margin`:
  - `probability - threshold`
- `triage_state` and alert flags:
  - use current monitor output

## Revised Tier Rules For This Project

### Event eligibility

Safe Guard tiering should only run for events that have already entered the project event flow.

That means one of these must be true:

- a new MySQL event row has been persisted
- runtime policy has explicitly classified the incident as an alert-worthy event

Tiering must not run on every raw inference window.

### Tier 1: High Confidence Fall

Condition:

- canonical runtime event exists
- final runtime state is alert-worthy
- `margin >= HIGH_CONF_MARGIN`
- `uncertainty < LOW_UNCERTAINTY_THRESHOLD`

Action:

- always send email
- send phone if `notify_phone` is enabled
- send SMS if `notify_sms` is enabled

### Tier 2: Borderline / Ambiguous Alert Event

Condition:

- canonical runtime event exists
- alert-worthy event is present
- Tier 1 is not satisfied

Typical reasons:

- margin is above threshold but not strong enough
- uncertainty is elevated
- event should be reviewed but does not justify phone escalation

Action:

- always send email
- never send phone
- send SMS if `notify_sms` is enabled

### Tier 3: Silent / Log Only

Condition:

- no canonical alert-worthy event exists

Action:

- no phone
- no SMS
- no email
- optional local audit logging only

Important:

In this repository, Tier 3 should not create a new primary MySQL event row. It may create only a local audit entry if needed for operational analysis.

## Notification Policy Mapping To Existing Settings

Current settings already expose:

- `notify_sms`
- `notify_phone`
- `notify_on_every_fall`

Safe Guard should interpret them like this:

- email:
  - always enabled for Tier 1 and Tier 2
- phone:
  - only if `notify_phone` is true and tier is Tier 1
- SMS:
  - only if `notify_sms` is true and tier is Tier 1 or Tier 2
- global gate:
  - if `notify_on_every_fall` is false, suppress all outbound caregiver notifications

## Threshold Source Of Truth

The threshold must not be hardcoded.

It should come from the runtime operating point selected for the active deploy profile:

- use the threshold already resolved by `server/routes/monitor.py`
- store it in event metadata when the event is created
- pass it into Safe Guard classification

Recommended persisted metadata additions:

- `threshold`
- `margin`
- `uncertainty`
- `notification_tier`
- `notification_policy_version`

## Deduplication Strategy

This project already has event-level deduplication in the monitor route.

Safe Guard dedup must be layered, not duplicated blindly.

Recommended strategy:

1. Primary dedup:
   - rely on the existing event creation cooldown in `monitor.py`

2. Notification dedup:
   - deduplicate by `event_id`
   - if the same `event_id` has already had a channel delivered successfully, do not resend that channel

3. Fallback cooldown:
   - if `event_id` is unavailable in any future call site, apply `ALERT_COOLDOWN_SECONDS` on a per-resident basis

This prevents repeated phone/SMS/email dispatch for the same unresolved incident.

## SQLite Scope

SQLite should be introduced as a local notification audit database only.

### Required tables

Recommended minimal table set:

- `notification_events`
  - mirrors dispatch-relevant event context
- `notification_attempts`
  - one row per channel attempt
- `caregiver_feedback`
  - webhook replies and operator notes

### Required stored fields

At minimum store:

- `event_id`
- `timestamp`
- `resident_id`
- `location`
- `is_fall_like_event`
- `probability`
- `threshold`
- `margin`
- `uncertainty`
- `alert_tier`
- `phone_attempted`
- `sms_attempted`
- `email_attempted`
- `phone_status`
- `sms_status`
- `email_status`
- `caregiver_feedback`
- `notes`

Important:

SQLite is an audit store. It does not replace MySQL `events`.

## Webhook Design

### Required endpoint

- `POST /twilio/webhook`

### Original spec issue

The original idea of "mark the most recent unresolved alert event" is unsafe when multiple alerts happen close together.

### Repository-specific rule

Webhook handling must prefer explicit event correlation.

Recommended message design:

- outgoing SMS includes a short incident reference
- outgoing email includes canonical `event_id`
- when possible, SMS reply parsing should extract an `event_id` token

### Fallback behavior

If explicit correlation is unavailable:

- fall back to most recent unresolved alert for that resident only
- record that fallback matching was used

### Reply mapping

- `"1"` -> false positive
- `"2"` -> confirmed fall
- `"3"` -> resolved / assistance provided

### Canonical update target

Webhook feedback should update:

- MySQL `events.status` when available
- MySQL `events.meta` for legacy fallback
- SQLite audit entry for local traceability

## Async Delivery Design

Use a lightweight in-process queue and background worker thread.

Do not introduce Celery at this stage.

### Why

- current deployment is single FastAPI service
- real-time inference path must remain low-latency
- queue/thread is enough for the current scale

### Required manager interface

Recommended interface:

- `NotificationManager.handle_event(...) -> None`

Behavior:

- validate event payload
- classify tier
- enqueue dispatch work
- return immediately

External API failures must never break the inference request path.

## Channel Content Rules

These parts of the original spec are correct and should be kept.

### Phone call

- short emergency-style message
- mention fall detected
- mention location
- do not mention probability, threshold, margin, or uncertainty

### SMS

- short summary
- include event type, time, and location
- Tier 2 explicitly says the event is ambiguous and should be checked
- keep text concise

### Email

- detailed event report
- include:
  - event_id
  - timestamp
  - location
  - probability
  - threshold
  - margin
  - uncertainty
  - alert tier
  - actions taken
  - recommendation

## Recommended Code Layout

Add a focused notification package under `server/`:

```text
server/
  notifications/
    __init__.py
    models.py
    classifier.py
    manager.py
    queue_worker.py
    sqlite_store.py
    twilio_client.py
    email_client.py
    templates.py
    config.py
  routes/
    twilio_webhook.py
```

### Module responsibilities

- `models.py`
  - Pydantic or dataclass models for event payloads and preferences
- `classifier.py`
  - threshold-aware tier assignment
- `manager.py`
  - main orchestration interface
- `queue_worker.py`
  - background worker and safe dispatch execution
- `sqlite_store.py`
  - schema creation and audit writes
- `twilio_client.py`
  - phone/SMS wrapper with retries and timeouts
- `email_client.py`
  - SMTP wrapper with retries and timeouts
- `templates.py`
  - per-channel message builders
- `config.py`
  - environment variable loading and defaults
- `routes/twilio_webhook.py`
  - caregiver feedback entrypoint

## Integration Points In This Repository

### Primary integration point

Inside `server/routes/monitor.py`, immediately after MySQL event persistence succeeds.

Current flow:

- event row inserted
- `dispatch_fall_notifications(...)` called

Recommended replacement:

- keep event insert unchanged
- replace the current DB-only notification helper with `NotificationManager.handle_event(...)`

### Transitional strategy

To reduce risk:

1. keep `dispatch_fall_notifications(...)` temporarily as compatibility path
2. add new `NotificationManager`
3. gate usage behind a config flag
4. migrate channel dispatch to new manager
5. later deprecate old helper

## Environment Variables

The original env list is mostly appropriate.

Recommended additions to align with current app:

- `SAFE_GUARD_ENABLED`
- `SAFE_GUARD_SQLITE_PATH`
- `SAFE_GUARD_WORKER_QUEUE_SIZE`
- `SAFE_GUARD_RETRY_COUNT`
- `SAFE_GUARD_HTTP_TIMEOUT_S`

Keep these from the original spec:

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM_PHONE`
- `CAREGIVER_PHONE`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `EMAIL_FROM`
- `CAREGIVER_EMAIL`
- `APP_BASE_URL`
- `HIGH_CONF_MARGIN`
- `LOW_UNCERTAINTY_THRESHOLD`
- `HIGH_UNCERTAINTY_THRESHOLD`
- `ALERT_COOLDOWN_SECONDS`

## Recommended Rollout Phases

### Phase 1: Internal Safe Guard core

- add domain models
- add tier classifier
- add SQLite audit store
- add queue worker
- add dry-run channel clients

Success criterion:

- classification and queueing work without external providers

### Phase 2: Real channel adapters

- implement Twilio SMS / call wrapper
- implement SMTP email wrapper
- add retry and timeout handling

Success criterion:

- dispatch works asynchronously and failures are isolated

### Phase 3: Runtime integration

- wire manager into `monitor.py`
- persist threshold / margin / uncertainty in event metadata
- attach dispatch summary to event response for diagnostics

Success criterion:

- runtime inference latency does not materially regress

### Phase 4: Feedback loop

- add `/twilio/webhook`
- map replies to canonical event status updates
- write feedback to SQLite audit tables

Success criterion:

- caregiver replies are visible in Event History state

## Explicit Non-Goals For First Implementation

Do not include these in v1:

- Celery / Redis
- distributed worker infrastructure
- automatic model retraining
- active learning data selection
- full incident management UI

## Final Recommendation

The Safe Guard concept is valid for this repository, but it must be implemented as:

- threshold-aware
- policy-aware
- event-driven
- MySQL-primary
- SQLite-audit-only
- asynchronous and non-blocking

That is the correct project-specific interpretation of the original specification.

## Next Implementation Step

The next coding task should implement:

1. notification domain models
2. threshold-aware tier classifier
3. SQLite audit store
4. background queue worker
5. `NotificationManager.handle_event(...)`
6. a new webhook route
7. integration into `server/routes/monitor.py` behind a feature flag
