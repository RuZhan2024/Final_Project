# Full Code Review Tasks

Date: 2026-04-09  
Purpose: track the supervisor-prep full code review in two phases

## Baseline

- Branch: current working branch at review time
- Review date: 2026-04-09
- Primary environment: local macOS development environment
- Workspace state at task-doc upgrade time: mixed worktree allowed during active review, but findings must always reference the code state actually inspected
- Primary evidence/result document:
  - `docs/reports/audit/FULL_CODE_REVIEW_2026-04-09.md`

## Scope

This review is intentionally split into two layers:

1. Line-by-line code review
   - inspect file logic directly
   - check defaults, control flow, error handling, state management, persistence behavior, and hidden assumptions
   - record findings per file or per module batch

2. Workflow mismatch review
   - verify that reviewed modules still compose correctly as a system
   - check for frontend/backend, runtime/persistence, replay/realtime, and docs/code drift

## Review Policy

- Do not mark a module as reviewed unless its files were actually read.
- Prefer freeze-core code first.
- Record real findings only; do not pad the audit with cosmetic comments.
- A finding must carry severity and blocker status.
- A fix must be justified before it is applied.
- Do not refactor unrelated code during review.
- After a fix, use `fixed_pending_verify` until verification is complete.
- A file can be marked:
  - `not_started`
  - `in_review`
  - `reviewed_no_blocker`
  - `reviewed_with_findings`
  - `fixed_pending_verify`
  - `closed`

## Finding Template

Each recorded finding should include:

- `ID`
- `Severity`: `critical | high | medium | low`
- `Status`
- `Release blocker`: `yes | no`
- `File(s)`
- `What was checked`
- `Finding`
- `Impact`
- `Fix`
- `Verification`

## Batch Plan

### Batch 1: ML Pipeline

Status: `fixed_pending_verify`

Files/modules:
- `src/fall_detection/preprocessing/*`
- `src/fall_detection/pose/*`
- `src/fall_detection/core/*`
- `src/fall_detection/evaluation/*`
- `src/fall_detection/training/*`
- `src/fall_detection/deploy/*`

Goals:
- review extract/preprocess/features/contracts
- review train/eval/fit_ops logic
- review deployment-side shared feature path

Deliverables:
- update `FULL_CODE_REVIEW_2026-04-09.md`
- record findings and fixes

Exit criteria:
- all listed ML pipeline files needed for the main path were read
- core data flow was traced from raw pose to deploy-time features
- findings were written with file refs
- justified fixes were applied where needed
- post-fix verification was recorded

### Batch 2: Server

Status: `fixed_pending_verify`

Files/modules:
- `server/core.py`
- `server/db.py`
- `server/deploy_runtime.py`
- `server/notifications_service.py`
- `server/services/*`
- `server/repositories/*`
- `server/notifications/*`
- `server/routes/*`
- `server/app.py`
- `server/main.py`

Goals:
- review runtime defaults and fallback logic
- review persistence and schema assumptions
- review notification and caregiver paths
- review monitor route and event flow

Deliverables:
- update `FULL_CODE_REVIEW_2026-04-09.md`
- list findings with file refs
- apply justified fixes

Exit criteria:
- all listed server areas were read directly
- runtime defaults and persistence assumptions were checked
- route-to-service-to-repository glue was traced for major paths
- findings were written with file refs
- justified fixes were applied and verified

### Batch 3: Frontend

Status: `fixed_pending_verify`

Files/modules:
- `apps/src/monitoring/*`
- `apps/src/pages/*`
- `apps/src/features/*`
- `apps/src/lib/*`
- `apps/src/App.js`

Goals:
- review page state flow
- review API contract assumptions
- review monitor, replay, settings, dashboard, events paths

Deliverables:
- update `FULL_CODE_REVIEW_2026-04-09.md`
- record findings and fixes

Exit criteria:
- all listed frontend modules needed for live monitor, replay, settings, dashboard, and events were read
- API contract assumptions were checked against current backend behavior
- findings were written with file refs
- justified fixes were applied and verified

### Batch 4: Scripts and Tests

Status: `fixed_pending_verify`

Files/modules:
- tooling/build/freeze:
  - `scripts/*`
- verification/coverage:
  - `tests/*`

Goals:
- review canonical tooling
- review build/freeze/test scripts
- review whether tests still match current contracts

Deliverables:
- update `FULL_CODE_REVIEW_2026-04-09.md`
- record stale or missing test coverage

Exit criteria:
- build/freeze/test entrypoints were read
- stale test assumptions were identified where present
- scripts and tests were reviewed as separate concerns
- findings were written with file refs
- justified fixes were applied and verified

## Workflow Review

Status: `fixed_pending_verify`

This phase starts only after the four code batches have been reviewed.

Rules:
- run workflow review only on the post-fix code state
- recheck previously found mismatches; do not assume they stayed fixed

Checks:
- startup with missing DB or fallback settings
- settings -> monitor -> event -> notification path
- realtime vs replay persistence semantics
- fallback behavior with and without DB
- frontend/backend preset consistency
- evidence/demo path consistency with runtime defaults
- runtime behavior when notification delivery is unavailable

Deliverables:
- final update to `FULL_CODE_REVIEW_2026-04-09.md`
- closure summary:
  - what was reviewed
  - what was fixed
  - what remains acceptable risk

## Fix Discipline

- Keep fixes minimal and evidence-based.
- Do not broaden scope during review unless a discovered defect requires it.
- Move a finding to `fixed_pending_verify` immediately after the patch lands.
- Only move a finding to `closed` after an explicit verification step.

## Accepted Risk Table

- `risk`: torch-backed tests remain unverified on this local machine
  - `why accepted`: local `import torch` aborts before pytest collection; torch-free and frontend regression slices are passing
  - `release blocker`: no for code-review completion, yes for full environment verification elsewhere
  - `revisit later`: yes

- `risk`: two narrow frontend state-sync fixes still rely mainly on static review rather than dedicated unit tests
  - `why accepted`: low blast radius, surrounding API contracts are covered, and no new runtime mismatch was found in workflow review
  - `release blocker`: no
  - `revisit later`: yes

## Current Known Findings Already Entered

The initial pass already identified and partly fixed:
- fallback preset drift
- monitoring toggle persistence divergence
- caregiver fallback save ambiguity
- quoted env value parsing weakness
- replay persistence ambiguity

These do not remove the need for the full batched review below.

## Execution Log

### 2026-04-09

- task document created
- initial findings document already exists:
  - `docs/reports/audit/FULL_CODE_REVIEW_2026-04-09.md`
- next execution target:
  - `Batch 1: ML Pipeline`
- ML pipeline first-pass review started
- first ML findings already fixed and verified:
  - non-strict datamodule split fallback no longer leaks all windows into train/val/test
  - training checkpoints no longer write fake default `pose_preprocess` metadata
  - direct trainer bootstrap path now points at `src`, not the wrong package subdirectory
  - evaluation window discovery now matches recursive training/deploy discovery
  - unified pipeline windows now write frame-index metadata instead of millisecond timestamps into `w_start/w_end`
  - deploy-side MC dropout now keeps BatchNorm in eval mode and enables only dropout layers
  - CAUCAFall raw-label discovery now uses the project nominal `23 FPS` instead of `25 FPS`
- verification completed:
  - `python3 -m py_compile src/fall_detection/data/datamodule.py src/fall_detection/training/train_tcn.py src/fall_detection/training/train_gcn.py`
  - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_datamodule_split_contract.py tests/test_data_sources_config.py tests/test_pose_preprocess_config.py`
  - `python3 -m py_compile src/fall_detection/evaluation/metrics_eval.py src/fall_detection/evaluation/score_unlabeled_alert_rate.py src/fall_detection/deploy/run_alert_policy.py`
  - `python3 -m py_compile src/fall_detection/data/pipeline.py src/fall_detection/deploy/common.py tests/test_deploy_common_mc_contract.py`
  - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_data_pipeline_window_metadata.py tests/test_datamodule_split_contract.py tests/test_data_sources_config.py tests/test_pose_preprocess_config.py`
- environment limitation recorded:
  - the new torch-backed deploy MC regression test could not be executed here because importing `torch` aborts during pytest collection in this local environment
- Batch 1 main-path review is functionally complete; only the torch-backed deploy MC regression remains blocked by the local environment
- next execution target:
  - `Batch 2: Server`
- server findings started:
  - active-op normalization is now unified across settings load/save, in-memory fallback, and monitor request context
  - monitor/events notification responses now expose real Safe Guard enqueue state instead of legacy queue-log summaries
  - `/api/notifications` now reads the Safe Guard SQLite audit store instead of the legacy MySQL notification queue log
  - v2 synthetic test-fall events now write `pending_review`, matching current UI and summary contracts
  - runtime-created monitor events now also write `pending_review` when the active schema supports status tracking
  - dashboard today counts now support the current `events.type` schema instead of only the old `event_type` path
  - server entrypoint, deploy-spec listing, operating-points listing, health route, and Twilio feedback webhook were read and did not show a new blocking mismatch in this pass
- frontend findings started:
  - legacy monitor operating-point fallback now includes `dataset_code`, so fallback thresholds cannot silently drift back to CAUCAFall when LE2I is active
  - `Test Fall` no longer draws a success-looking fall marker when the backend request failed
  - caregiver form fields now re-sync when the loaded caregiver payload changes under the same record id
  - app shell, dashboard/events polling hooks, and shared frontend utils were read and did not show a new blocking mismatch in this pass
- verification completed:
  - `python3 -m py_compile server/services/monitor_context_service.py server/services/settings_service.py server/core.py server/repositories/settings_repository.py tests/server/test_op_code_normalization.py`
  - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/server/test_op_code_normalization.py`
  - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/server/test_runtime_core.py -k "active_op_code or derive_ops_params_from_yaml_modes"`
  - `python3 -m py_compile server/notifications/models.py server/notifications/manager.py server/notifications/sqlite_store.py server/routes/notifications.py server/routes/events.py server/services/monitor_runtime_service.py tests/server/test_notification_manager.py tests/server/test_notifications_routes.py`
  - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/server/test_notification_manager.py tests/server/test_notifications_routes.py tests/server/test_op_code_normalization.py`
  - `python3 -m py_compile server/routes/events.py tests/server/test_events_test_fall_status_contract.py`
  - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/server/test_notification_manager.py tests/server/test_notifications_routes.py tests/server/test_op_code_normalization.py tests/server/test_events_test_fall_status_contract.py`
  - `python3 -m py_compile server/repositories/monitor_repository.py tests/server/test_monitor_repository_event_status.py`
  - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/server/test_monitor_repository_event_status.py tests/server/test_events_test_fall_status_contract.py tests/server/test_notification_manager.py tests/server/test_notifications_routes.py tests/server/test_op_code_normalization.py`
  - `python3 -m py_compile server/repositories/dashboard_repository.py tests/server/test_dashboard_repository_counts.py`
  - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/server/test_dashboard_repository_counts.py tests/server/test_monitor_repository_event_status.py tests/server/test_events_test_fall_status_contract.py tests/server/test_notification_manager.py tests/server/test_notifications_routes.py tests/server/test_op_code_normalization.py`
  - `cd apps && CI=1 npm test -- --runInBand --watchAll=false --watchman=false src/features/monitor/api.test.js`
- environment limitation recorded:
  - server tests importing `server.main` or `server.routes.monitor` still hit the local `torch` abort during collection, so verification is currently limited to torch-free server slices
- Batch 4 scripts/tests review started
- scripts/tests findings fixed:
  - canonical test coverage now includes the new ML and server regression tests added during this review
  - a dedicated `frontend` mode now runs the monitor API regression test
- tooling reviewed with no new blocker in this pass:
  - `scripts/build_report.sh`
  - `scripts/freeze_manifest.sh`
  - `scripts/generate_report_figures.py`
  - `scripts/build_cross_dataset_summary.py`
  - `scripts/plot_cross_dataset_transfer.py`
- verification completed:
  - `bash -n scripts/run_canonical_tests.sh`
  - `./scripts/run_canonical_tests.sh torch-free`
  - `./scripts/run_canonical_tests.sh frontend`
- workflow review completed on the post-fix code state:
  - `./scripts/freeze_manifest.sh`
  - `./scripts/run_canonical_tests.sh contract` now fails fast with a clear torch-environment prerequisite message instead of a pytest abort trace
  - no new code-path mismatch was identified in the defended `settings -> monitor -> event -> notification` or `realtime vs replay` paths
- next execution target:
  - `final verify / commit`
