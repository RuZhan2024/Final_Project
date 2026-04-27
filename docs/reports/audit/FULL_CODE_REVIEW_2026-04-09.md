# Full Code Review

Date: 2026-04-09  
Scope: end-to-end code review of the current frozen project state before supervisor handoff  
Baseline: current branch worktree at review time, after `freeze_manifest.sh` passed and worktree was clean

## Verdict

The codebase is structurally strong and the main ML, backend, frontend, and notification paths are all present and coherent. The main remaining risk is not missing implementation, but **workflow mismatch across fallback paths, UI assumptions, and persistence behavior**.

I found **4 actionable code-level mismatches** in the initial pass. A first remediation pass has now closed the two P0 issues and partially closed one P1 issue. The replay-persistence mismatch remains open.

The first ML-pipeline batch has now also been reviewed. That batch has added **7 real findings** so far, all fixed in the current worktree.

The server batch is now underway and has added **5 real findings** so far, all fixed in the current worktree. The frontend batch has also started and has already added **2 real findings** plus one smaller state-sync issue, all fixed in the current worktree. The scripts/tests batch has now added **1 real finding**, also fixed in the current worktree.

## Findings

### 13. Medium: server-side active-op handling accepted multiple spellings but did not normalize them consistently across load, save, and runtime context assembly

Evidence:
- Monitor request context previously used raw `upper().strip()` instead of the shared op normalizer in [monitor_context_service.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/services/monitor_context_service.py#L75).
- Settings persistence/load paths also used plain uppercase handling in [settings_repository.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/repositories/settings_repository.py).
- In-memory fallback updates did the same in [core.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/core.py#L847).

Impact:
- Inputs such as `op2`, `OP2`, or `2` could be accepted by some paths but then stored or propagated under different string forms.
- That creates avoidable drift between request payloads, DB state, in-memory fallback state, and event metadata.

Fix:
- Server-side settings load/save, in-memory fallback updates, and monitor request-context assembly now all use the shared `_norm_op_code()` contract.

Verification:
- [test_op_code_normalization.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_op_code_normalization.py)

Status: closed in current worktree  
Priority: P1  
Final release blocker: no

### 14. High: server notification APIs and monitor/event responses were exposing legacy queue-log state instead of the real Safe Guard delivery path

Evidence:
- `persist_monitor_event()` previously called the legacy [dispatch_fall_notifications()](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications_service.py#L29) helper before calling the real `NotificationManager`, and then returned that legacy summary as `notification_dispatch`.
- `/api/events/test_fall` also wrote legacy queue rows and returned that summary instead of the real Safe Guard enqueue result in [events.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/events.py#L299).
- `/api/notifications` read MySQL `notifications_log`, while actual Telegram delivery attempts and statuses were recorded in the Safe Guard SQLite store in [sqlite_store.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/sqlite_store.py#L14).

Impact:
- API responses could say a notification was "attempted" or show queued rows even when the real Telegram path had not yet enqueued, had been dedup-suppressed, or later failed.
- The server exposed two different notification truths: MySQL queue logs and SQLite delivery audit.
- That is a direct workflow mismatch for review/demo validation because operator-visible state could diverge from actual caregiver delivery state.

Fix:
- `NotificationManager.handle_event()` now returns an explicit dispatch-acceptance result.
- `monitor` persistence and `events.test_fall` now report the real Safe Guard enqueue state instead of the legacy MySQL queue summary.
- `/api/notifications` now reads the Safe Guard SQLite audit store, which is the actual source of delivery status.

Verification:
- [test_notification_manager.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_notification_manager.py)
- [test_notifications_routes.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_notifications_routes.py)

Status: closed in current worktree  
Priority: P0  
Final release blocker: resolved in current worktree

### 15. Medium: synthetic v2 test-fall events still wrote the legacy `unreviewed` status while the rest of the stack uses `pending_review`

Evidence:
- The v2 insert path in [events.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/events.py#L295) still populated `status="unreviewed"`.
- Frontend labels, filters, and local today-summary logic only treat `pending_review` as the active review-pending state:
  - [eventLabels.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/lib/eventLabels.js#L13)
  - [Events.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/Events.js#L41)
  - [useEventsData.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/events/hooks/useEventsData.js#L67)

Impact:
- Test fall events created through the supported API could land in a status spelling that the UI no longer treats as the canonical pending-review value.
- That makes event-history review behavior and summary counts depend on legacy spelling tolerance instead of one explicit contract.

Fix:
- The v2 test-fall path now writes `pending_review`, matching the rest of the stack's current event-review contract.

Verification:
- [test_events_test_fall_status_contract.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_events_test_fall_status_contract.py)

Status: closed in current worktree  
Priority: P1  
Final release blocker: no

### 16. Medium: monitor-created events could land with empty review status even though the UI and summaries assume `pending_review`

Evidence:
- [insert_monitor_event()](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/repositories/monitor_repository.py#L61) previously inserted only `resident_id, type, severity, model_code, operating_point_id, score, meta`.
- In the current SQLite schema, `events.status` exists in [db.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/db.py#L193).
- Event history, review flow, and pending summary logic all assume new reviewable events default to `pending_review`:
  - [events_service.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/services/events_service.py#L244)
  - [Events.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/Events.js#L62)

Impact:
- Real monitor-created events could appear in the event table with empty/null status while synthetic test-fall events and UI assumptions used `pending_review`.
- That makes pending-review counts and review-state UX depend on insertion path instead of one shared event contract.

Fix:
- Monitor event insertion now inspects the live `events` schema and writes `status='pending_review'` whenever that column exists, while remaining compatible with the older MySQL schema that lacks the field.

Verification:
- [test_monitor_repository_event_status.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_monitor_repository_event_status.py)

Status: closed in current worktree  
Priority: P1  
Final release blocker: no

### 17. High: dashboard today counts were still keyed to legacy `event_type` schema and could ignore the current `events.type` table entirely

Evidence:
- [load_today_counts()](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/repositories/dashboard_repository.py#L55) previously counted only when `events.event_type` existed, otherwise falling back to `fall_events`.
- The current repo-native `events` schema uses `type` together with `event_time` / `ts` in:
  - [create_db.sql](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/create_db.sql#L73)
  - [db.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/db.py#L193)

Impact:
- On the active schema, Dashboard could report `falls_detected=0` and `false_alarms=0` even when the events table already contained today's fall and false-alarm events.
- That is a high-severity UI/runtime mismatch because it makes the top-level system summary unreliable.

Fix:
- Dashboard counting now supports the current `events.type` schema first, selects the best available time column (`event_time`, `ts`, then `created_at`), and still retains backward compatibility with older schemas.

Verification:
- [test_dashboard_repository_counts.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_dashboard_repository_counts.py)

Status: closed in current worktree  
Priority: P0  
Final release blocker: resolved in current worktree

### 18. High: frontend legacy operating-point fallback silently requested the wrong dataset thresholds

Evidence:
- [fetchOperatingPoints()](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/features/monitor/api.js#L60) previously queried `/api/operating_points` with only `model_code`.
- The backend endpoint defaults `dataset_code` to `caucafall`, so the fallback path in [useOperatingPointParams.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/monitor/hooks/useOperatingPointParams.js#L78) could fetch CAUCAFall thresholds even when the active dataset was LE2I.

Impact:
- When the UI falls back from YAML-derived deploy params to the legacy operating-points endpoint, the page could silently show and use thresholds for the wrong dataset.
- That is a real frontend/backend contract mismatch because the active dataset selector no longer guaranteed dataset-matched OP thresholds in the fallback path.

Fix:
- Frontend operating-point requests now always include `dataset_code`, and the fallback hook passes the currently active dataset through to the backend.

Verification:
- [api.test.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/features/monitor/api.test.js)

Status: closed in current worktree  
Priority: P0  
Final release blocker: resolved in current worktree

### 19. Medium: `Test Fall` could paint a fall marker in the monitor timeline even when the backend request failed

Evidence:
- The `testFall()` handler in [usePoseMonitor.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/monitor/hooks/usePoseMonitor.js#L1218) previously added a forced fall marker even when the request threw or when the response was not an `ok` result.

Impact:
- Operators could see a visual fall marker and reasonably infer that the backend accepted the synthetic fall event even when the request actually failed.
- That creates false-success feedback during monitor demos and UI validation.

Fix:
- The timeline marker is now added only after a confirmed `ok` backend response.
- Failure paths now surface an error through the existing `predictError` channel instead of drawing a fake successful fall.

Verification:
- static code review of [usePoseMonitor.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/monitor/hooks/usePoseMonitor.js#L1218)

Status: closed in current worktree  
Priority: P1  
Final release blocker: no

### 20. Low: caregiver form state only re-synced when the caregiver ID changed, not when the loaded caregiver fields changed under the same ID

Evidence:
- [SettingsPage.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/settings/SettingsPage.js#L35) previously reloaded local caregiver form fields only on `primary?.id`.

Impact:
- If the backend reloaded or normalized caregiver data for the same record ID, the form could keep showing stale name/chat-id values until a full remount.

Fix:
- The caregiver form now re-synchronizes when `id`, `name`, or `telegram_chat_id` changes.

Verification:
- static code review of [SettingsPage.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/settings/SettingsPage.js#L35)

Status: closed in current worktree  
Priority: P2  
Final release blocker: no

### 21. Medium: the canonical test entrypoint had drifted behind the current freeze-core regression set

Evidence:
- [run_canonical_tests.sh](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/run_canonical_tests.sh) previously covered only the older smoke/API/notification subset under `torch-free`.
- Newly added freeze-core regression tests from this review were not included, including:
  - [test_datamodule_split_contract.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/test_datamodule_split_contract.py)
  - [test_pose_preprocess_config.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/test_pose_preprocess_config.py)
  - [test_data_pipeline_window_metadata.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/test_data_pipeline_window_metadata.py)
  - [test_data_pipeline_caucafall_label_fps.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/test_data_pipeline_caucafall_label_fps.py)
  - [test_op_code_normalization.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_op_code_normalization.py)
  - [test_events_test_fall_status_contract.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_events_test_fall_status_contract.py)
  - [test_monitor_repository_event_status.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_monitor_repository_event_status.py)
  - [test_dashboard_repository_counts.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_dashboard_repository_counts.py)
- The frontend regression test [api.test.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/features/monitor/api.test.js) also had no canonical entrypoint.

Impact:
- A passing “canonical” test run no longer implied that the current freeze-core regression suite had actually been exercised.
- That weakens handoff confidence because regression protection exists in the repo but was not wired into the documented test entrypoint.

Fix:
- Expanded `torch-free` coverage in `run_canonical_tests.sh` to include the new ML and server regression tests.
- Added a dedicated `frontend` mode for the monitor API regression test.
- Updated `all` mode and usage text so the canonical entrypoint now reflects the current freeze-core regression set.
- Added a graceful torch preflight for `contract` and `monitor` modes so known environment failure now surfaces as an explicit prerequisite error instead of a pytest abort trace.

Verification:
- `bash -n scripts/run_canonical_tests.sh`
- `./scripts/run_canonical_tests.sh torch-free`
- `./scripts/run_canonical_tests.sh frontend`

Status: closed in current worktree  
Priority: P1  
Final release blocker: no

### 1. High: documented live preset, backend fallback preset, and frontend fallback preset still diverge

Evidence:
- Backend in-memory defaults still boot into `LE2I + TCN + OP-2` with `mc_enabled=false` in [server/core.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/core.py#L740).
- Settings UI fallback still assumes `active_dataset_code="le2i"` and `mc_enabled=true` in [SettingsPage.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/settings/SettingsPage.js#L53).
- Monitor page fallback also assumes `active_dataset_code="le2i"` and `mc_enabled=true` in [Monitor.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/Monitor.js#L47).

Impact:
- In DB-unavailable or partially initialized mode, backend and frontend can boot with different assumptions.
- This directly conflicts with the current documented demo/review preset `CAUCAFall + TCN + OP-2`.
- The result is a real handoff risk: the UI can present one default while the runtime and evidence pack assume another.

Required action:
- Unify fallback defaults across backend and frontend to the defended live preset.
- Treat the documented demo preset as the single source of truth for no-DB operation.

Status: closed in current worktree  
Priority: P0  
Final release blocker: resolved in current worktree

### 2. High: monitoring toggle can leave runtime state and persisted state diverged when settings persistence fails

Evidence:
- Monitoring is optimistically switched before persistence in [MonitoringContext.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/monitoring/MonitoringContext.js#L116).
- On failure, the catch block only records the error and returns `false`; it does not restore runtime state or force a refresh in [MonitoringContext.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/monitoring/MonitoringContext.js#L148).

Impact:
- If `persistSettings()` fails after `safeStart()` succeeds, the camera/runtime loop can remain active while the backend still believes monitoring is disabled.
- If disable persistence fails, the frontend can stop local monitoring while the persisted backend state still says monitoring is enabled.
- This is a real workflow mismatch, not just a UI nuisance, because Monitor and Settings no longer describe the same system state.

Required action:
- Make toggle behavior transactional.
- On persistence failure, revert local runtime state or force an immediate refresh/reconciliation before returning success/failure.

Status: closed in current worktree  
Priority: P0  
Final release blocker: resolved in current worktree

### 3. Medium: replay detections were silently non-persistent unless monitoring was on or clip storage was enabled

Evidence:
- Replay/live payloads only set `persist` when `monitoringOnRef.current || storeEventClips` in [usePoseMonitor.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/monitor/hooks/usePoseMonitor.js#L632).

Impact:
- Users can replay a clip, see a fall classification in the UI, and reasonably expect an event-history row or notification.
- In practice, replay falls remain transient unless monitoring is actively on or event-clip storage is enabled.
- This already caused operator confusion during validation because replay “worked” visually but did not generate an event.

Required action:
- Make replay persistence explicit in the UI, or introduce a dedicated replay-persist switch.
- Do not leave this as an implicit hidden rule.

Status: closed in current worktree  
Priority: P1  
Final release blocker: resolved in current worktree

### 4. Medium: caregiver save path can silently degrade to in-memory fallback while the frontend still treats it as a normal save

Evidence:
- Backend caregiver upsert falls back to in-memory and still returns a success-shaped payload with `db_available=false` in [caregivers.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/caregivers.py#L60).
- Frontend caregiver hook ignores `db_available` and only reloads caregivers after the request in [useCaregivers.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/settings/hooks/useCaregivers.js#L34).

Impact:
- The Settings page can present caregiver data as “saved” even when persistence actually degraded to in-memory fallback.
- That is especially risky for Telegram delivery, because the user may believe the contact is durable while the next restart loses it.

Required action:
- Surface fallback persistence explicitly in the caregiver UX.
- Treat `db_available=false` as a warning state, not a normal successful save.

Status: partially_closed in current worktree  
Priority: P1  
Final release blocker: no, but should be fixed or clearly signalled

### 5. Low: local env loader does not normalize quoted values

Evidence:
- The local env loader copies `value.strip()` directly into `os.environ` in [config.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/config.py#L12).

Impact:
- If a token is stored as `"TOKEN"` or `'TOKEN'`, the quotes are preserved and downstream auth fails with misleading errors.
- This is a robustness issue that already matters in notification setup flows.

Required action:
- Strip balanced single or double quotes when loading local env files.

Status: closed in current worktree  
Priority: P2  
Final release blocker: no

### 6. High: non-strict `UnifiedWindowDataModule` could silently leak the full window set into train/val/test

Evidence:
- In non-strict mode, `_resolve_split()` previously returned `tuple(windows)` whenever split IDs were missing in [datamodule.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/data/datamodule.py#L150).
- That behavior applied not only to `predict`, but also to `train`, `val`, and `test`.

Impact:
- A missing or broken split manifest could silently turn into train/val/test leakage instead of a visible empty split or a hard failure.
- This is a real ML workflow defect because it contaminates evaluation semantics while still looking superficially “working”.

Fix:
- The fallback now returns all windows only for `predict`.
- Missing non-predict splits in non-strict mode now resolve to empty tuples instead of leaking the entire dataset.

Verification:
- [test_datamodule_split_contract.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/test_datamodule_split_contract.py)

Status: closed in current worktree  
Priority: P0  
Final release blocker: resolved in current worktree

### 7. Medium: training checkpoints were embedding default `pose_preprocess` metadata even when the true training preprocess was unknown

Evidence:
- `build_data_cfg_dict()` in both trainers wrote `normalize_pose_preprocess_cfg(None)` into checkpoint metadata:
  - [train_tcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_tcn.py#L141)
  - [train_gcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_gcn.py#L562)

Impact:
- The checkpoint looked like it carried explicit preprocessing provenance, but it was really only carrying repo defaults.
- That is misleading for deploy parity, because downstream runtime code may trust checkpoint metadata more than it should.

Fix:
- Trainers now store only `fps_default` unless real pose-preprocess metadata is explicitly available.
- This removes false certainty instead of recording fabricated provenance.

Verification:
- `python3 -m py_compile src/fall_detection/training/train_tcn.py src/fall_detection/training/train_gcn.py`

Status: closed in current worktree  
Priority: P1  
Final release blocker: no

### 8. Medium: direct execution bootstrap in both trainer modules pointed at the wrong path

Evidence:
- Both trainer modules claimed to bootstrap the "repo root" using `here.parents[1]`, but for
  `src/fall_detection/training/*.py` that path resolves to `src/fall_detection`, not the importable source root:
  - [train_tcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_tcn.py#L23)
  - [train_gcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_gcn.py#L22)

Impact:
- Direct module execution could still fail to import `fall_detection.*` even though the file comments implied it was supported.
- This is a tooling correctness issue for manual training/debug flows.

Fix:
- Both bootstraps now add `here.parents[2]`, which is the actual `src` root containing the `fall_detection` package.

Verification:
- `python3 -m py_compile src/fall_detection/training/train_tcn.py src/fall_detection/training/train_gcn.py`

Status: closed in current worktree  
Priority: P2  
Final release blocker: no

### 9. Medium: evaluation window discovery was less permissive than training/deploy discovery

Evidence:
- Training, replay, and deploy-time runners discover windows recursively.
- `metrics_eval.py` and `score_unlabeled_alert_rate.py` previously only scanned top-level `*.npz` files:
  - [metrics_eval.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/metrics_eval.py#L214)
  - [score_unlabeled_alert_rate.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/score_unlabeled_alert_rate.py#L50)

Impact:
- Nested window directories could evaluate as partial or empty datasets even though training and deploy-time scripts would still see the full set.
- This is an experiment-parity mismatch, not just a convenience issue.

Fix:
- Both evaluators now use recursive discovery to match the rest of the pipeline.

Verification:
- `python3 -m py_compile src/fall_detection/evaluation/metrics_eval.py src/fall_detection/evaluation/score_unlabeled_alert_rate.py`

Status: closed in current worktree  
Priority: P1  
Final release blocker: no

### 10. High: unified pipeline windows stored millisecond timestamps in fields that downstream code interprets as frame indices

Evidence:
- The canonical exporter wrote rounded `start_ms` / `end_ms` directly into `w_start` / `w_end` in [pipeline.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/data/pipeline.py#L594).
- Downstream evaluation, alerting, and offline deploy runners consistently treat those same fields as inclusive frame indices:
  - [metrics_eval.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/metrics_eval.py#L14)
  - [run_alert_policy.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/deploy/run_alert_policy.py#L41)
  - [alerting.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/alerting.py#L203)

Impact:
- Windows exported by the unified data pipeline could carry timing metadata in a different unit than the rest of the stack expects.
- That can distort event timing, false-alert grouping, FA/24h estimates, and offline alert-policy replay even when the model probabilities themselves are correct.
- This is a true ML-to-runtime contract mismatch.

Fix:
- `export_windows()` now writes `w_start` / `w_end` as inclusive frame indices on the target-FPS timeline, matching downstream consumers.

Verification:
- [test_data_pipeline_window_metadata.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/test_data_pipeline_window_metadata.py)

Status: closed in current worktree  
Priority: P0  
Final release blocker: resolved in current worktree

### 11. Medium: deploy-side MC dropout path violated the repo's own uncertainty contract by switching the whole model to train mode

Evidence:
- [deploy/common.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/deploy/common.py#L237) previously implemented MC sampling with `model.train()` around repeated forwards.
- [core/uncertainty.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/uncertainty.py#L4) explicitly defines the deployment contract as “keep BatchNorm in eval mode; enable only dropout layers”.

Impact:
- Deploy-time uncertainty estimates could drift from the intended contract because BatchNorm statistics were being updated/read in training mode during MC sampling.
- That is a runtime-parity defect, not just an implementation style issue.

Fix:
- `predict_mu_sigma()` now delegates to `core.uncertainty.mc_predict_mu_sigma()`, which preserves eval mode globally and enables only dropout layers during sampling.

Verification:
- [test_deploy_common_mc_contract.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/test_deploy_common_mc_contract.py)

Status: closed in current worktree  
Priority: P1  
Final release blocker: no

### 12. Medium: CAUCAFall raw-label reconstruction used the wrong nominal FPS during pre-extraction discovery

Evidence:
- [pipeline.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/data/pipeline.py#L1177) previously passed `fps=25.0` into `_infer_caucafall_spans_from_frame_annotations(...)`.
- Elsewhere in the project, CAUCAFall is consistently treated as `23 FPS`.

Impact:
- Rebuilding labels and splits directly from raw CAUCAFall videos could shift inferred fall spans before any pose extraction occurred.
- That creates a dataset-preparation mismatch between raw-label reconstruction and the rest of the training/evaluation stack.

Fix:
- Raw CAUCAFall label discovery now uses the project’s nominal `23 FPS`.

Verification:
- [test_data_pipeline_caucafall_label_fps.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/test_data_pipeline_caucafall_label_fps.py)

Status: closed in current worktree  
Priority: P1  
Final release blocker: no

## Areas Reviewed With No Blocking Mismatch Found

The following areas were reviewed and did not show a blocking code-level mismatch in this pass:

- ML data pipeline structure in [pipeline.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/data/pipeline.py)
- OP fitting / evaluation path in [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py)
- Telegram-first notification dispatch in [manager.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/manager.py)
- settings snapshot / response assembly in [settings.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/settings.py)
- FastAPI app assembly in [main.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/main.py) and [app.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/app.py)
- deploy spec discovery/listing in [deploy_runtime.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/deploy_runtime.py) and [specs.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/specs.py)
- operating-point listing and YAML fallback in [operating_points.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/operating_points.py)
- caregiver CRUD fallback path in [caregivers.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/caregivers.py)
- lightweight health endpoint in [health.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/health.py)
- Twilio reply-to-feedback bridge in [twilio_webhook.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/twilio_webhook.py)
- top-level app shell and route mounting in [App.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/App.js)
- monitoring context assembly and retry logic in [MonitoringContext.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/monitoring/MonitoringContext.js)
- dashboard polling hook in [useDashboardSummary.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/dashboard/hooks/useDashboardSummary.js)
- events polling/review hook in [useEventsData.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/events/hooks/useEventsData.js)
- shared frontend API helpers in [apiClient.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/lib/apiClient.js), [config.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/lib/config.js), and [booleans.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/lib/booleans.js)
- monitor UI helper logic in [utils.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/monitor/utils.js), [constants.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/monitor/constants.js), [ControlsCard.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/monitor/components/ControlsCard.js), and [LiveMonitorCard.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/monitor/components/LiveMonitorCard.js)
- pose preprocessing path in [preprocess_pose_npz.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/pose/preprocess_pose_npz.py)
- canonical feature builder in [features.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/features.py)
- deploy feature parity path in [common.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/deploy/common.py)
- checkpoint bundle helpers in [ckpt.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/ckpt.py)
- temperature-calibration helpers in [calibration.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/calibration.py)
- model builder and input-dimension inference in [models.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/models.py)
- FA/24h sweep helpers in [metrics.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/metrics.py)
- MC-dropout utilities in [uncertainty.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/uncertainty.py)
- alert policy and operating-point selection in [alerting.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/alerting.py)
- offline triage runner glue in [run_modes.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/deploy/run_modes.py)
- training protocol definitions in [contracts.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/contracts.py)
- canonical test orchestration in [run_canonical_tests.sh](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/run_canonical_tests.sh)
- freeze allowlist check in [freeze_manifest.sh](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/freeze_manifest.sh)
- report build entrypoint in [build_report.sh](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/build_report.sh)
- report figure generation in [generate_report_figures.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/generate_report_figures.py)
- cross-dataset summary and plotting tooling in [build_cross_dataset_summary.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/build_cross_dataset_summary.py) and [plot_cross_dataset_transfer.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/plot_cross_dataset_transfer.py)

That is not a claim that these files are perfect; it means this review pass did not find a release-blocking workflow mismatch there.

## Workflow Review Summary

Status: fixed_pending_verify in current worktree

The post-fix workflow review did not expose a new code-path mismatch across the main defended system paths:

- `settings -> monitor -> event -> notification`
  - fallback preset alignment, monitoring rollback, pending-review status contracts, and Safe Guard delivery reporting are now internally consistent across frontend and backend
- `realtime vs replay`
  - replay persistence is now explicit in the UI instead of being hidden behind monitoring state or clip-storage side effects
- `freeze/build/test tooling`
  - `freeze_manifest.sh` still resolves the defended allowlist correctly
  - `run_canonical_tests.sh torch-free` and `run_canonical_tests.sh frontend` both pass on this machine
  - `contract` and `monitor` modes now fail fast with a clear torch-environment prerequisite message instead of an opaque pytest abort

The remaining workflow limitation is environmental rather than a newly found code mismatch: torch-backed verification is still blocked on this local machine, so final closure of the torch-backed monitor/contract path requires rerun in a clean torch environment.

## Accepted Risks

- Risk: torch-backed tests remain unverified on this local machine
  - Why accepted: the failure reproduces as an environment-level `import torch` abort before pytest collection, and the affected code paths already have torch-free regression coverage where feasible
  - Release blocker: no for code review completion, yes for full environment verification elsewhere
  - Revisit later: yes

- Risk: two small frontend state fixes remain protected mainly by static review rather than dedicated unit tests
  - Why accepted: the changes are narrow UI-state guards with low blast radius, and the surrounding end-to-end contracts are already covered
  - Release blocker: no
  - Revisit later: yes

## Recommended Fix Order

1. Add a targeted regression test for monitoring toggle rollback on settings persistence failure.
2. Add a targeted regression test for caregiver fallback-warning behavior.
3. Decide whether the replay-persist toggle should default to off permanently or remember the previous operator choice.
4. Re-run the new torch-backed deploy MC test in an environment where `import torch` does not abort during collection.
5. Re-run the torch-backed deploy MC, contract, and monitor test slices in an environment where `import torch` is stable.

## Review Notes

- This review is code-focused. It does not replace the broader repository/evidence audit in [FULL_STACK_PROJECT_AUDIT_2026-04-09.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/audit/FULL_STACK_PROJECT_AUDIT_2026-04-09.md).
- The new deploy MC regression test is syntactically valid, but full execution is currently blocked in this local environment because importing `torch` aborts during pytest collection.
