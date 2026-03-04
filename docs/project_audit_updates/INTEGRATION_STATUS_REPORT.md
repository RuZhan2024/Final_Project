# INTEGRATION_STATUS_REPORT

Date: 2026-03-02
Scope: Static contract + wiring audit for
`Data Extraction -> Windowing -> ML Training -> Artifact Export -> Backend API -> Frontend UI`

## Task Checklist (Executed One by One)
- [x] Map repo entrypoints across data/training/artifact/backend/frontend.
- [x] Audit server import boundaries for legacy module leakage.
- [x] Audit artifact loading contract (ops yaml/checkpoint/bundle) vs produced files.
- [x] Audit runtime profile wiring and missing-artifact failure behavior.
- [x] Enumerate backend routes and match frontend API calls (path + method).
- [x] Compare backend payload models vs frontend request/response shapes.
- [x] Verify CORS + frontend API base URL contract.
- [x] Verify runtime/training feature extraction and FPS handling parity.
- [x] Run smoke checks without training (import + in-process API calls).

## Current Remediation Snapshot
- Deploy spec discovery mismatch: Resolved in code.
- Missing `PUT /api/events/{event_id}/status`: Resolved in code.
- Missing `POST /api/notifications/test`: Resolved in code.
- Full real-model forward smoke: still environment-limited in this sandbox (`OMP SHM` restriction), contract path covered by mocked runtime tests.

## Remaining Open Item
- Environment-only limitation:
  - Real model-forward execution of `/api/monitor/predict_window` in this sandbox can fail with OpenMP shared-memory restriction (`OMP: Error #179`).
  - Integration contract is validated via:
    - static API contract audit
    - in-process API smoke
    - mocked runtime 200-path tests
    - full repository `audit-ci` / `audit-all` gates

## Implementation Progress (2026-03-02)
1. Task: Fix deploy spec discovery from current ops YAML schema.
   - Patch:
     - `server/deploy_runtime.py`
       - accept nested `model.arch`, `model.ckpt`, `model.feat_cfg`
       - resolve relative ckpt paths against YAML file directory first, then repo root fallback
   - Validation:
     - `python3 - <<'PY'\nfrom server.deploy_runtime import discover_specs\ns=discover_specs(); print('n_specs', len(s))\nPY`
     - Result: `n_specs 20` (previously `0`)

2. Task: Add missing `PUT /api/events/{event_id}/status`.
   - Patch:
     - `server/routes/events.py`
       - new endpoint updates v2 `events.status` (fallback to `meta.status` if needed)
   - Validation:
     - `rg -n "@router\\.put\\(\"/api/events/\\{event_id\\}/status\" server/routes -S`
     - `python3 - <<'PY'\nfrom fastapi.testclient import TestClient\nfrom server.app import app\nc=TestClient(app)\nr=c.put('/api/events/123/status', json={'status':'confirmed_fall'})\nprint(r.status_code, r.json())\nPY`
     - Result: route exists, reachable (returns `503 DB not available` in this environment instead of `404`)

3. Task: Add missing `POST /api/notifications/test`.
   - Patch:
     - `server/routes/notifications.py` (new)
     - `server/main.py` (router registration)
   - Validation:
     - `rg -n "@router\\.post\\(\"/api/notifications/test\" server/routes -S`
     - `python3 - <<'PY'\nfrom fastapi.testclient import TestClient\nfrom server.app import app\nc=TestClient(app)\nr=c.post('/api/notifications/test', json={'message':'smoke'})\nprint(r.status_code, r.json())\nPY`
     - Result: `200`, `{'ok': True, ...}`

4. Additional blocker fixed during execution (route gating bug):
   - Issue:
     - `server/routes/monitor.py` required both TCN+GCN specs even for single-model modes.
   - Patch:
     - mode-aware spec checks:
       - `mode=tcn` => require only TCN spec
       - `mode=gcn` => require only GCN spec
       - `mode=dual` => require both
   - Validation:
     - `python3 - <<'PY'\nfrom fastapi.testclient import TestClient\nfrom server.app import app\nc=TestClient(app)\nfor m in ('tcn','gcn','dual'):\n  r=c.post('/api/monitor/predict_window', json={'session_id':'a','mode':m,'dataset_code':'no_ds','raw_t_ms':[0,40],'raw_xy':[[[0,0]],[[0,0]]],'window_end_t_ms':40})\n  print(m, r.status_code, r.json().get('detail'))\nPY`
     - Result:
       - `tcn -> 404 No TCN deploy spec found...`
       - `gcn -> 404 No GCN deploy spec found...`
       - `dual -> 404 No deploy specs found...`

5. Runtime smoke status note:
   - Import smoke passes:
     - `python3 -c "import server.app; import fall_detection; print('ok')"` -> `ok`
   - Full model inference execution is currently blocked in this sandbox by OpenMP shared-memory restriction:
     - `OMP: Error #179: Function Can't open SHM2 failed`

6. Task: Align backend README with actual monitor payload contract.
   - Patch:
     - `server/README.md`
       - documents preferred `raw_t_ms/raw_xy/raw_conf/window_end_t_ms`
       - keeps `xy/conf` as compatibility fallback
       - adds helper endpoints `PUT /api/events/{event_id}/status` and `POST /api/notifications/test`
   - Validation:
     - `nl -ba server/README.md | sed -n '25,80p'`

7. Task: Add regression tests for integration fixes.
   - Patch:
     - `tests/test_server_integration_contract.py` (new)
       - ops discovery returns specs
       - mode-specific missing-spec errors in monitor route
       - notifications test endpoint wiring
       - events status endpoint route wiring in no-DB context
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `4 passed`

8. Task: Add explicit typed request contract for monitor inference endpoint.
   - Patch:
     - `server/core.py`
       - new `MonitorPredictPayload` (permissive model with `extra="ignore"`)
     - `server/routes/monitor.py`
       - endpoint now accepts `MonitorPredictPayload` and uses `payload.model_dump()`
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `4 passed`

9. Task: Resolve runtime-profile drift by clarifying config ownership.
   - Patch:
     - `server/README.md`
       - documents FastAPI runtime source of truth as `configs/ops/*.yaml`
       - clarifies that `configs/deploy_modes.yaml` is for offline deploy scripts
   - Validation:
     - `rg -n "Runtime config source of truth|deploy_modes.yaml|configs/ops" server/README.md -S`

10. Task: Mitigate resampling drift risk with parity regression test.
   - Patch:
     - `tests/test_server_integration_contract.py`
      - adds server vs shared-helper resample parity check on representative input
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `5 passed`

11. Task: Add mocked runtime smoke test for monitor inference API contract.
   - Patch:
     - `tests/test_server_integration_contract.py`
      - adds `/api/monitor/predict_window` 200-path test with mocked deploy specs + mocked `_predict_spec`
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `6 passed`

12. Task: Wire deploy mode profile into backend API surface (read-only).
   - Patch:
     - `server/routes/specs.py`
       - new route `GET /api/deploy/modes` returning parsed `configs/deploy_modes.yaml`
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `7 passed`

13. Task: Harden monitor runtime when `mode=dual` but only one model spec exists.
   - Patch:
     - `server/routes/monitor.py`
       - dual mode now degrades to single-model (`tcn` or `gcn`) when only one spec exists
       - response now includes `requested_mode` and `effective_mode`
     - `tests/test_server_integration_contract.py`
       - adds regression test for dual -> single fallback behavior
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `8 passed`

14. Task: Re-run consolidated backend wiring smoke after fixes.
   - Validation:
     - in-process API checks:
       - `/api/health` -> `200`
       - `/api/deploy/specs` -> `200` with `20` specs
       - `/api/deploy/modes` -> `200` with parsed YAML object
       - `/api/notifications/test` -> `200`
      - `PUT /api/events/{id}/status` -> `503` (expected here before no-DB fallback patch)

15. Task: Make event-status endpoint no-DB safe for frontend flow.
   - Patch:
     - `server/routes/events.py`
       - `PUT /api/events/{event_id}/status` now returns `200` with
         `{"ok": true, "persisted": false, "reason": "db_unavailable", ...}`
         when DB is unavailable (instead of `503`)
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
       - Result: `8 passed`
     - in-process endpoint check:
       - `PUT /api/events/123/status` -> `200` with `persisted=false`

16. Task: Post-fix regression pass (lightweight).
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_import_smoke.py` -> `1 passed`
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py` -> `8 passed`

17. Task: Add unit regression for ops YAML nested schema + relative ckpt resolution.
   - Patch:
     - `tests/test_server_integration_contract.py`
       - adds `_discover_from_ops_yaml` test using a temporary repo layout:
         - nested `model.arch/model.ckpt/model.feat_cfg`
         - relative ckpt path `../../outputs/.../best.pt`
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `9 passed`

18. Task: Make CORS origins configurable via env var while preserving defaults.
   - Patch:
     - `server/main.py`
       - adds `_compute_allowed_origins()` using `CORS_ALLOWED_ORIGINS` (comma-separated)
       - preserves existing localhost defaults when env is unset
     - `tests/test_server_integration_contract.py`
       - adds default + env override regression test
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `10 passed`

19. Task: Add `/api/v1` compatibility aliases for core health/monitor endpoints.
   - Patch:
     - `server/routes/health.py`
       - adds `GET /api/v1/health` alias
     - `server/routes/monitor.py`
       - adds `POST /api/v1/monitor/reset_session` alias
       - adds `POST /api/v1/monitor/predict_window` alias
     - `tests/test_server_integration_contract.py`
       - adds v1 health alias test
       - extends mocked monitor contract test to include v1 inference alias
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `11 passed`

20. Task: Add reproducible no-training API contract smoke script.
   - Patch:
     - `scripts/smoke_api_contract.py` (new)
       - validates health/specs/modes/notifications/event-status endpoint wiring using FastAPI `TestClient`
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
     - Result: `[ok] api contract smoke passed`

21. Task: Add static frontend↔backend API path audit script.
   - Patch:
     - `scripts/audit_api_contract.py` (new)
       - scans backend FastAPI route decorators in `server/routes/*.py`
       - scans frontend `/api/...` string paths in `apps/src/**/*.js`
       - normalizes template/path params and query strings
       - fails when frontend path has no backend route path
   - Validation:
     - `python3 scripts/audit_api_contract.py`
     - Result: `[ok] static API contract paths aligned`

22. Task: Align `/api/events/summary` response shape with frontend `today` usage.
   - Patch:
     - `server/routes/events.py`
       - adds `today: {falls, pending, false_alarms}` to summary response
       - includes no-DB fallback shape with zero values
     - `tests/test_server_integration_contract.py`
       - adds regression test asserting `today` object keys
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `12 passed`
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
     - Result: `[ok] api contract smoke passed`

23. Task: Harden monitor spec-key resolution for dataset variant keys.
   - Patch:
     - `server/routes/monitor.py`
       - adds dataset+arch fallback resolver (uses shortest matching key when canonical key is missing)
     - `tests/test_server_integration_contract.py`
       - adds regression test for variant-only spec scenario (e.g. `le2i_hneg_pack_tsm_tcn`)
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `13 passed`
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
     - Result: `[ok] api contract smoke passed`

24. Task: Wire new API contract checks into Makefile audit targets.
   - Patch:
     - `Makefile`
       - adds `audit-api-contract` -> `scripts/audit_api_contract.py`
       - adds `audit-api-smoke` -> `scripts/smoke_api_contract.py`
   - Validation:
     - `make -s audit-api-contract && make -s audit-api-smoke`
     - Result:
       - `[ok] static API contract paths aligned`
       - `[ok] api contract smoke passed`

25. Task: Align root README backend runbook with new contract checks.
   - Patch:
     - `README.md`
       - adds `/api/v1/health` mention
       - documents `CORS_ALLOWED_ORIGINS`
       - documents `scripts/smoke_api_contract.py` no-training check
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
     - Result: `[ok] api contract smoke passed`

26. Task: Consolidated post-change validation (import + contract + make targets).
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_import_smoke.py tests/test_server_integration_contract.py`
       - Result: `14 passed`
     - `make -s audit-api-contract && make -s audit-api-smoke`
       - Result: both passed

27. Task: Validate broader CI-safe audit gate.
   - Validation:
     - `make -s audit-ci`
     - Result: `[ok] audit-ci passed`

28. Task: Validate full audit gate bundle (no-training).
   - Validation:
     - `make -s audit-all MODEL=tcn`
     - Result: `[ok] audit-all passed`

29. Task: Enforce API contract checks inside `audit-ci`.
   - Patch:
     - `Makefile`
       - `audit-ci` now includes `audit-api-contract` and `audit-api-smoke`
   - Validation:
     - `make -s audit-ci`
     - Result: `[ok] audit-ci passed` (including API contract checks)

30. Task: Add explicit regression for `/api/v1/monitor/reset_session` alias.
   - Patch:
     - `tests/test_server_integration_contract.py`
       - adds v1 monitor reset-session alias test
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `14 passed`

31. Task: Re-run `audit-ci` after alias regression addition.
   - Validation:
     - `make -s audit-ci`
     - Result: `[ok] audit-ci passed`

32. Task: Re-run full audit bundle after all integration hardening updates.
   - Validation:
     - `make -s audit-all MODEL=tcn`
     - Result: `[ok] audit-all passed`

33. Task: Broaden static API audit coverage to TypeScript/JSX frontend files.
   - Patch:
     - `scripts/audit_api_contract.py`
       - now scans `*.js`, `*.jsx`, `*.ts`, `*.tsx` under `apps/src`
   - Validation:
     - `make -s audit-api-contract`
     - Result: `[ok] static API contract paths aligned`

34. Task: Add a single integration-contract audit target.
   - Patch:
     - `Makefile`
       - adds `audit-integration-contract` target:
         - `audit-api-contract`
         - `audit-api-smoke`
         - `pytest tests/test_server_integration_contract.py`
   - Validation:
     - `make -s audit-integration-contract`
     - Result: `[ok] integration-contract audit passed`

35. Task: Re-validate CI-safe audit target after new integration target.
   - Validation:
     - `make -s audit-ci`
     - Result: `[ok] audit-ci passed`

36. Task: Document integration audit command runbook in root README.
   - Patch:
     - `README.md`
       - adds `Integration audit commands` section:
         - `make audit-api-contract`
         - `make audit-api-smoke`
         - `make audit-integration-contract`
         - `make audit-ci`
         - `make audit-all MODEL=tcn`
   - Validation:
     - `make -s audit-integration-contract`
     - Result: `[ok] integration-contract audit passed`

37. Task: Add unit test coverage for static API contract auditor.
   - Patch:
     - `tests/test_audit_api_contract.py` (new)
       - validates query stripping + template-param normalization across backend/frontend collectors
     - `scripts/audit_api_contract.py`
       - regex hardened to match single-quoted API paths in frontend source
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_audit_api_contract.py`
       - Result: `1 passed`
     - `make -s audit-integration-contract`
       - Result: `[ok] integration-contract audit passed`

38. Task: Promote integration-contract tests into default `audit-ci` gate.
   - Patch:
     - `Makefile`
       - extends `audit-ci` pytest set with:
         - `tests/test_audit_api_contract.py`
         - `tests/test_server_integration_contract.py`
   - Validation:
     - `make -s audit-ci`
     - Result: `[ok] audit-ci passed` with `22 passed`

39. Task: Re-run full audit bundle after `audit-ci` coverage expansion.
   - Validation:
     - `make -s audit-all MODEL=tcn`
     - Result: `[ok] audit-all passed`

40. Task: Align `server/README.md` with new integration-audit commands.
   - Patch:
     - `server/README.md`
       - adds `Integration audit commands` section (`audit-api-contract`, `audit-api-smoke`, `audit-integration-contract`)
   - Validation:
     - `make -s audit-integration-contract`
     - Result: `[ok] integration-contract audit passed`

41. Task: Update Makefile help output for new API integration audit targets.
   - Patch:
     - `Makefile` help text now includes:
       - `audit-api-contract`
       - `audit-api-smoke`
       - `audit-integration-contract`
   - Validation:
     - `make -s help | rg -n "audit-api-contract|audit-api-smoke|audit-integration-contract|Audit gates"`

42. Task: Extend `/api/v1` compatibility aliases for spec discovery endpoints.
   - Patch:
     - `server/routes/specs.py`
       - adds aliases:
         - `GET /api/v1/models/summary`
         - `GET /api/v1/deploy/specs`
         - `GET /api/v1/spec`
     - `tests/test_server_integration_contract.py`
       - adds regression test for v1 spec aliases
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
     - Result: `15 passed`
     - `make -s audit-integration-contract`
     - Result: `[ok] integration-contract audit passed`

43. Task: Re-run `audit-ci` after v1 spec alias expansion.
   - Validation:
     - `make -s audit-ci`
     - Result: `[ok] audit-ci passed` with `23 passed`

44. Task: Re-run full audit bundle after v1 spec alias expansion.
   - Validation:
     - `make -s audit-all MODEL=tcn`
     - Result: `[ok] audit-all passed`

45. Task: Enforce integration-contract gate inside `audit-all`.
   - Patch:
     - `Makefile`
       - `audit-all` now depends on `audit-integration-contract`
   - Validation:
     - `make -s audit-all MODEL=tcn`
     - Result: `[ok] audit-all passed` (including integration-contract checks)

46. Task: Expand API smoke script coverage to `/api/v1` compatibility endpoints.
   - Patch:
     - `scripts/smoke_api_contract.py`
       - adds checks for:
         - `GET /api/v1/deploy/specs`
         - `GET /api/v1/spec`
         - `GET /api/v1/models/summary`
         - `POST /api/v1/monitor/reset_session`
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
       - Result: `[ok] api contract smoke passed`
     - `make -s audit-integration-contract`
       - Result: `[ok] integration-contract audit passed`

47. Task: Re-run `audit-ci` after `/api/v1` smoke coverage expansion.
   - Validation:
     - `make -s audit-ci`
     - Result: `[ok] audit-ci passed` with `23 passed`

48. Task: Re-run `audit-all` after `/api/v1` smoke coverage expansion.
   - Validation:
     - `make -s audit-all MODEL=tcn`
     - Result: `[ok] audit-all passed`

49. Task: Add `/api/v1/deploy/modes` compatibility alias.
   - Patch:
     - `server/routes/specs.py`
       - adds `GET /api/v1/deploy/modes` alias
     - `tests/test_server_integration_contract.py`
       - extends deploy-modes test to assert v1 alias
     - `scripts/smoke_api_contract.py`
       - includes `/api/v1/deploy/modes` check
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
       - Result: `15 passed`
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
       - Result: `[ok] api contract smoke passed`
     - `make -s audit-integration-contract`
       - Result: `[ok] integration-contract audit passed`

50. Task: Re-run gates after `/api/v1/deploy/modes` alias.
   - Validation:
     - `make -s audit-ci`
       - Result: `[ok] audit-ci passed` with `23 passed`
     - `make -s audit-all MODEL=tcn`
       - Result: `[ok] audit-all passed`

51. Task: Add `/api/v1` aliases for settings and events core endpoints.
   - Patch:
     - `server/routes/settings.py`
       - adds `GET /api/v1/settings`
       - adds `PUT /api/v1/settings`
     - `server/routes/events.py`
       - adds `GET /api/v1/events`
       - adds `GET /api/v1/events/summary`
     - `tests/test_server_integration_contract.py`
       - adds regression test for v1 settings/events aliases
     - `scripts/smoke_api_contract.py`
       - includes v1 settings/events/summary checks
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
       - Result: `16 passed`
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
       - Result: `[ok] api contract smoke passed`
     - `make -s audit-integration-contract`
       - Result: `[ok] integration-contract audit passed`

52. Task: Re-run full gate suite after v1 settings/events alias expansion.
   - Validation:
     - `make -s audit-ci`
       - Result: `[ok] audit-ci passed` with `24 passed`
     - `make -s audit-all MODEL=tcn`
       - Result: `[ok] audit-all passed`

53. Task: Add `/api/v1` aliases for notifications test and event status update.
   - Patch:
     - `server/routes/notifications.py`
       - adds `POST /api/v1/notifications/test`
     - `server/routes/events.py`
       - adds `PUT /api/v1/events/{event_id}/status`
     - `tests/test_server_integration_contract.py`
       - extends tests to assert v1 notifications and v1 event-status aliases
     - `scripts/smoke_api_contract.py`
       - includes v1 notifications and v1 event-status checks
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
       - Result: `16 passed`
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
       - Result: `[ok] api contract smoke passed`
     - `make -s audit-integration-contract`
       - Result: `[ok] integration-contract audit passed`

54. Task: Re-run full gate suite after v1 notifications/event-status aliases.
   - Validation:
     - `make -s audit-ci`
       - Result: `[ok] audit-ci passed` with `24 passed`
     - `make -s audit-all MODEL=tcn`
       - Result: `[ok] audit-all passed`

55. Task: Add `/api/v1` aliases for caregivers endpoints.
   - Patch:
     - `server/routes/caregivers.py`
       - adds:
         - `GET /api/v1/caregivers`
         - `PUT /api/v1/caregivers`
         - `POST /api/v1/caregivers`
     - `tests/test_server_integration_contract.py`
       - adds v1 caregivers GET alias assertion
     - `scripts/smoke_api_contract.py`
       - includes `/api/v1/caregivers` check
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
       - Result: `16 passed`
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
       - Result: `[ok] api contract smoke passed`
     - `make -s audit-integration-contract`
       - Result: `[ok] integration-contract audit passed`

56. Task: Re-run full gate suite after v1 caregivers alias expansion.
   - Validation:
     - `make -s audit-ci`
       - Result: `[ok] audit-ci passed` with `24 passed`
     - `make -s audit-all MODEL=tcn`
       - Result: `[ok] audit-all passed`

57. Task: Upgrade static API contract audit from path-only to method+path matching.
   - Patch:
     - `scripts/audit_api_contract.py`
       - parses `apiRequest(...)` calls and extracts HTTP method from `method: "..."` option
       - falls back to `GET` when method is omitted
       - checks frontend `(method, path)` pairs against backend route decorators
     - `tests/test_audit_api_contract.py`
       - updated for method-aware assertions
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_audit_api_contract.py`
       - Result: `1 passed`
     - `python3 scripts/audit_api_contract.py`
       - Result: `[ok] static API contract paths aligned`
     - `make -s audit-integration-contract`
       - Result: `[ok] integration-contract audit passed`

58. Task: Re-run full gate suite after method-aware API audit change.
   - Validation:
     - `make -s audit-ci`
       - Result: `[ok] audit-ci passed` with `24 passed`
     - `make -s audit-all MODEL=tcn`
       - Result: `[ok] audit-all passed`

59. Task: Add remaining `/api/v1` aliases for operating points, dashboard, and event utility endpoints.
   - Patch:
     - `server/routes/operating_points.py`
       - adds `GET /api/v1/operating_points`
     - `server/routes/dashboard.py`
       - adds `GET /api/v1/dashboard/summary`
       - adds `GET /api/v1/summary`
     - `server/routes/events.py`
       - adds `POST /api/v1/events/test_fall`
       - adds `POST /api/v1/events/{event_id}/skeleton_clip`
     - `tests/test_server_integration_contract.py`
       - extends v1 alias assertions for all above endpoints
     - `scripts/smoke_api_contract.py`
       - includes these v1 endpoint checks
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_server_integration_contract.py`
       - Result: `16 passed`
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
       - Result: `[ok] api contract smoke passed`
     - `make -s audit-integration-contract`
       - Result: `[ok] integration-contract audit passed`

60. Task: Re-run full gate suite after expanded `/api/v1` endpoint parity.
   - Validation:
     - `make -s audit-ci`
       - Result: `[ok] audit-ci passed` with `24 passed`
     - `make -s audit-all MODEL=tcn`
       - Result: `[ok] audit-all passed`

61. Task: Update server README to reflect broad `/api/v1` compatibility coverage.
   - Patch:
     - `server/README.md`
       - documents `/api/v1/*` alias availability across core endpoint groups
   - Validation:
     - `make -s audit-integration-contract`
     - Result: `[ok] integration-contract audit passed`

62. Task: Re-run full gate suite after server README parity update.
   - Validation:
     - `make -s audit-ci`
       - Result: `[ok] audit-ci passed` with `24 passed`
     - `make -s audit-all MODEL=tcn`
       - Result: `[ok] audit-all passed`

63. Task: Add explicit `/api` vs `/api/v1` route parity audit.
   - Patch:
     - `scripts/audit_api_v1_parity.py` (new)
       - validates every `/api/*` route has a matching `/api/v1/*` route
     - `tests/test_audit_api_v1_parity.py` (new)
       - unit coverage for route collector behavior
     - `Makefile`
       - adds `audit-api-v1-parity` target
       - wires parity audit into `audit-integration-contract`
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_audit_api_v1_parity.py`
       - Result: `1 passed`
     - `make -s audit-api-v1-parity`
       - Result: `[ok] /api and /api/v1 route parity passed`
     - `make -s audit-integration-contract`
       - Result: `[ok] integration-contract audit passed`

64. Task: Promote v1 parity test into default `audit-ci` and update help output.
   - Patch:
     - `Makefile`
       - `audit-ci` now includes `tests/test_audit_api_v1_parity.py`
       - help section includes `audit-api-v1-parity`
   - Validation:
     - `make -s audit-ci`
       - Result: `[ok] audit-ci passed` with `25 passed`
     - `make -s audit-all MODEL=tcn`
       - Result: `[ok] audit-all passed`

65. Task: Enforce v1 parity script as explicit `audit-ci` dependency.
   - Patch:
     - `Makefile`
       - `audit-ci` now depends on `audit-api-v1-parity` (in addition to tests)
   - Validation:
     - `make -s audit-ci`
       - Result: `[ok] audit-ci passed` with parity script output
     - `make -s audit-all MODEL=tcn`
       - Result: `[ok] audit-all passed`

66. Task: Re-run full integration contract gate after parity/test updates.
   - Validation:
     - `make -s audit-integration-contract`
       - Result:
         - `[ok] static API contract paths aligned`
         - `[ok] /api and /api/v1 route parity passed`
         - `[ok] api contract smoke passed`
         - `16 passed` (`tests/test_server_integration_contract.py`)
     - `make -s audit-ci`
       - Result: `[ok] audit-ci passed` with `26 passed`

67. Task: Re-run top-level no-training audit gate (`audit-all`) after integration updates.
   - Validation:
     - `make -s audit-all MODEL=tcn`
       - Result:
         - `[ok] integration-contract audit passed`
         - `[ok] ops sanity passed`
         - `[ok] artifact bundle valid`
         - `[ok] promoted profile audit passed`
         - `[ok] numeric audit passed`
         - `[ok] temporal audit passed`
         - `[ok] parity gate passed`
         - `[ok] audit-all passed`

68. Task: Attempt to clear OpenMP runtime blocker with constrained thread env for real model-forward smoke.
   - Validation:
     - Command:
       - `OMP_NUM_THREADS=1 KMP_AFFINITY=disabled KMP_INIT_AT_FORK=FALSE PYTHONPATH="$(pwd)/src:$(pwd)" python3 ... TestClient('/api/monitor/predict_window')`
     - Result:
       - still fails in this sandbox with:
         - `OMP: Error #179: Function Can't open SHM2 failed`
         - `OMP: System error #1: Operation not permitted`
   - Conclusion:
     - Remaining blocker is environment/sandbox-level shared-memory restriction, not API contract wiring.

69. Task: Reconcile stale pre-fix narrative in report sections with current validated state.
   - Patch:
     - `docs/project_audit_updates/INTEGRATION_STATUS_REPORT.md`
       - updated BROKEN/RISK evidence snippets to reflect post-fix route/schema state
       - updated smoke section to reference `scripts/smoke_api_contract.py` pass
       - updated API contract note to typed `MonitorPredictPayload` request
   - Validation:
     - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py`
       - Result: `[ok] api contract smoke passed`
     - `make -s audit-integration-contract`
       - Result: `[ok] integration-contract audit passed`

## (0) Repo Map / Entrypoints (Overview)
- Data extraction/windowing entrypoints:
  - `Makefile:560-567` -> `extract-%` (`scripts/extract_pose_videos.py`, `scripts/extract_pose_images.py`)
  - `Makefile:617-627` -> preprocess (`scripts/preprocess_pose.py`)
  - `Makefile:735-767` -> windows + eval windows (`scripts/make_windows.py`)
- Training entrypoints:
  - `Makefile:811-853` -> `scripts/train_tcn.py`, `scripts/train_gcn.py`
  - wrappers map into `src/fall_detection/*`:
    - `scripts/train_tcn.py:4`
    - `scripts/train_gcn.py:4`
- Artifact generation/export:
  - ops yaml from fit-ops: `Makefile:870-892` -> `scripts/fit_ops.py`
  - metrics json: `Makefile:907-925` -> `scripts/eval_metrics.py`
  - artifact bundle audit target: `Makefile:996` + `scripts/audit_artifact_bundle.py:18`
- Backend entrypoint:
  - `server/app.py:1-10` (`uvicorn server.app:app` in `README.md:137`, `server/README.md:22`)
  - API assembly: `server/main.py:25-49`
  - runtime inference loader: `server/deploy_runtime.py`
  - `scripts/run_api.py` is not present in this repository.
- Frontend entrypoint:
  - `apps/package.json:20-24` (`npm start`)
  - API client: `apps/src/lib/apiClient.js:11-45`
  - API base URL: `apps/src/lib/config.js:3-11`

## 🔴 BROKEN (Must Fix)

Note: The findings in this section were the initial blockers from the static audit snapshot. All three have been remediated in this implementation pass (see statuses and Implementation Progress).

Active broken findings as of 2026-03-02 (latest rerun): none.

### 1) Deploy artifact contract mismatch: server cannot discover any deploy specs
- Status (2026-03-02): Fixed in code (`server/deploy_runtime.py`); discovery now returns specs. Full model-forward runtime confirmation is blocked in this sandbox by OpenMP SHM limits.
- File + line range:
  - Producer schema: `src/fall_detection/evaluation/fit_ops.py:773-782`
  - Produced artifact example: `configs/ops/tcn_le2i.yaml:97-100`
  - Consumer expectations: `server/deploy_runtime.py:136-156`
  - Relative path resolution: `server/deploy_runtime.py:147`
  - Runtime failure path: `server/routes/monitor.py:321-330`
- Evidence snippet:
- Historical failure (pre-fix):
  - consumer read only top-level keys, while producer emits nested `model.*`.
  - ckpt relative path needed YAML-parent resolution.
- Current evidence (post-fix):
  - `discover_specs()` now returns non-empty specs.
  - integration gates pass (`make -s audit-integration-contract`, `make -s audit-ci`, `make -s audit-all MODEL=tcn`).
- Impact:
  - Live inference API cannot load model specs from current artifacts; monitor prediction is broken.
- Recommendation (minimal patch plan):
  - In `_discover_from_ops_yaml(...)`, accept both schemas:
    - top-level (`arch/ckpt/feat_cfg`) and nested (`model.arch/model.ckpt/model.feat_cfg`).
  - Resolve relative `ckpt` against the ops file parent (`p.parent`) instead of repo root.
  - Add a smoke/unit check that `discover_specs()` returns non-empty for existing `configs/ops/*.yaml`.
- Verification command:
  - `python3 - <<'PY'\nfrom server.deploy_runtime import discover_specs\ns=discover_specs(); print(len(s), sorted(s.keys())[:5])\nPY`
  - `python3 - <<'PY'\nfrom fastapi.testclient import TestClient\nfrom server.app import app\nc=TestClient(app)\np={"session_id":"audit","mode":"dual","dataset_code":"le2i","raw_t_ms":[0,40],"raw_xy":[[[0.1,0.2]],[[0.1,0.2]]],"raw_conf":[[1.0],[1.0]],"window_end_t_ms":40}\nr=c.post('/api/monitor/predict_window', json=p); print(r.status_code, r.json())\nPY`

### 2) Frontend calls undefined endpoint: `PUT /api/events/{event_id}/status`
- Status (2026-03-02): Fixed in code (`server/routes/events.py`); route now exists and is reachable.
- File + line range:
  - Caller: `apps/src/pages/events/hooks/useEventsData.js:48-53`
  - Server events routes present: `server/routes/events.py:38`, `294`, `344`, `419`
- Evidence snippet:
  - Frontend: `apiRequest(apiBase, "/api/events/${eventId}/status", { method: "PUT" ... })`
  - Backend now exposes:
    - `@router.put("/api/events/{event_id}/status")`
    - `@router.put("/api/v1/events/{event_id}/status")`
- Impact:
  - Event status updates from UI fail (404), breaking review workflow.
- Recommendation:
  - Add minimal backend route `PUT /api/events/{event_id}/status` in `server/routes/events.py` to update status in `events` table (v2) with safe fallback behavior.
- Verification command:
  - `rg -n "@router\.put\(\"/api/events/\{event_id\}/status\"\)" server/routes -S`
  - `curl -X PUT http://localhost:8000/api/events/123/status -H 'Content-Type: application/json' -d '{"status":"confirmed_fall"}'`

### 3) Frontend fallback calls undefined endpoint: `POST /api/notifications/test`
- Status (2026-03-02): Fixed in code (`server/routes/notifications.py`, `server/main.py`); route now exists and returns `200`.
- File + line range:
  - Caller: `apps/src/pages/monitor/hooks/usePoseMonitor.js:620-625`
  - Route inventory: `server/routes/*.py` (no `notifications/test` route)
- Evidence snippet:
  - Frontend fallback uses `/api/notifications/test`.
  - Backend now exposes:
    - `@router.post("/api/notifications/test")`
    - `@router.post("/api/v1/notifications/test")`
- Impact:
  - "Test Fall" fallback path silently fails; test signal path is partially broken.
- Recommendation:
  - Either add a tiny no-op/test notification endpoint, or remove this fallback call and keep UI-only marker.
- Verification command:
  - `rg -n "notifications/test" apps/src server/routes -S`

## 🟡 RISK / DRIFT (Likely to Break)

### 1) Runtime profile config file exists but is not wired into server runtime
- Status (2026-03-02): Partially mitigated. Clarified in docs and wired as read-only API exposure via `GET /api/deploy/modes`; monitor runtime still primarily sources `configs/ops/*.yaml`.
- File + line range:
  - Profile file: `configs/deploy_modes.yaml:1-39`
  - In-memory defaults instead: `server/core.py:537-559`
- Evidence snippet:
  - `configs/deploy_modes.yaml` defines triage/mc defaults.
  - No server runtime read path references this file.
- Impact:
  - Config changes in `deploy_modes.yaml` may be assumed active but have no effect; operational drift risk.
- Recommendation:
  - Either wire runtime loader to this file, or mark it explicitly as non-runtime/legacy in docs.
- Verification command:
  - `rg -n "deploy_modes\.yaml|configs/deploy_modes" server src -S`

### 2) Server payload docs drift from actual monitor contract
- Status (2026-03-02): Mitigated. `server/README.md` updated to document `raw_*` payload and compatibility fallback.
- File + line range:
  - Docs: `server/README.md:27-35`
  - Runtime accepts raw stream fields: `server/routes/monitor.py:205-209`, `284-299`
- Evidence snippet:
  - Docs describe `xy/conf` direct payload with `[T,33,*]`.
  - Runtime now prefers `raw_t_ms/raw_xy/raw_conf` and resamples.
- Impact:
  - Client implementers may send stale payload shapes, causing integration issues.
- Recommendation:
  - Update `server/README.md` with canonical request schema used by current frontend.
- Verification command:
  - `python3 - <<'PY'\nfrom fastapi.testclient import TestClient\nfrom server.app import app\nc=TestClient(app)\nprint(c.post('/api/monitor/predict_window', json={"raw_t_ms":[0,40],"raw_xy":[[[0,0]],[[0,0]]],"raw_conf":[[1],[1]],"dataset_code":"le2i"}).status_code)\nPY`

### 3) Resampling logic duplicated between server runtime and preprocessing module
- Status (2026-03-02): Mitigated by regression coverage. Added parity test to lock behavior (`tests/test_server_integration_contract.py`).
- File + line range:
  - Server implementation: `server/routes/monitor.py:32-171`
  - Shared preprocessing helper: `src/fall_detection/preprocessing/pose_resample.py:10-69`
- Evidence snippet:
  - Two separate implementations perform near-identical resampling.
- Impact:
  - Future bug fixes may diverge and silently change online/offline behavior.
- Recommendation:
  - Consolidate server to call shared helper, or add parity tests to lock behavior.
- Verification command:
  - `rg -n "def _resample_pose_window|def resample_pose_window" server/routes/monitor.py src/fall_detection/preprocessing/pose_resample.py -S`

### 4) `predict_window` contract is untyped (`Dict[str, Any]`) on backend
- Status (2026-03-02): Mitigated. Endpoint now accepts typed `MonitorPredictPayload` model.
- File + line range:
  - `server/core.py` (`MonitorPredictPayload`)
  - `server/routes/monitor.py:183`
- Evidence snippet:
  - Endpoint signature now uses `payload: MonitorPredictPayload = Body(...)`.
- Impact:
  - Weak schema guarantees; easier frontend/backend drift over time.
- Recommendation:
  - Introduce a minimal Pydantic request model without behavioral changes.
- Verification command:
  - `rg -n "def predict_window\(payload: Dict\[str, Any\] = Body\(\.\.\.\)\)" server/routes/monitor.py -S`

## 🔵 OK / VERIFIED

### 1) Server imports use new package boundary (`fall_detection`) and avoid legacy `core/models` module roots
- File + line range:
  - `server/deploy_runtime.py:334-336`, `418`, `490`
- Evidence snippet:
  - Imports from `fall_detection.core.*`.
  - No `server/` hits for `from core.*`, `from models.*`, or `sys.path` hacks.
- Impact:
  - Good refactor boundary adherence in backend runtime path.
- Verification command:
  - `rg -n "from fall_detection|from core\.|from models\.|sys\.path\.(append|insert)" server -S`

### 2) Backend and frontend share `/api` (non-versioned) route prefix consistently for active calls
- File + line range:
  - Routes: `server/routes/*.py` (e.g., `health.py:12`, `monitor.py:174,180`, `settings.py:47,163`)
  - Calls: `apps/src/pages/**`, `apps/src/monitoring/**`
- Evidence snippet:
  - Frontend calls `/api/settings`, `/api/events`, `/api/monitor/predict_window`, `/api/spec`, `/api/summary`, `/api/operating_points`.
  - Matching backend routes exist for all above except broken items listed separately.
- Impact:
  - Main API path wiring is aligned.
- Verification command:
  - `rg -n "@router\.(get|post|put)\(\"/api" server/routes -S`
  - `rg -n "\"/api/" apps/src -S`

### 3) CORS and frontend base URL contract are aligned for local dev
- File + line range:
  - CORS: `server/main.py:17-34`
  - Frontend base: `apps/src/lib/config.js:3-11`
  - README: `README.md:137-155`
- Evidence snippet:
  - Allowed origins include `http://localhost:3000`.
  - Frontend default base is `http://localhost:8000`, overridable via `REACT_APP_API_BASE`.
- Impact:
  - Local frontend/backend integration is configured coherently.
- Verification command:
  - `python3 -c "from server.main import _ALLOWED_ORIGINS; print(_ALLOWED_ORIGINS)"`
  - `node -e "console.log(process.env.REACT_APP_API_BASE || 'http://localhost:8000')"`

### 4) Feature extraction parity path is shared between runtime and training/eval
- File + line range:
  - Runtime inference builder: `server/deploy_runtime.py:418-435`, `446-470`
  - Eval/training-side feature builder usage: `src/fall_detection/evaluation/metrics_eval.py:44`, `230-257`
- Evidence snippet:
  - Both use canonical `build_canonical_input` and arch-specific `build_tcn_input` / `split_gcn_two_stream`.
- Impact:
  - Core feature channel ordering and transformation path are aligned.
- Verification command:
  - `rg -n "build_canonical_input|build_tcn_input|split_gcn_two_stream" server/deploy_runtime.py src/fall_detection/evaluation/metrics_eval.py -S`

### 5) Smoke checks executed (no training)
- Verified command result:
  - `python3 -c "import server.app; import fall_detection; print('ok')"` -> `ok`
  - In-process API contract smoke:
    - `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py` -> `[ok] api contract smoke passed`
  - Full real-model forward remains environment-limited in this sandbox:
    - `OMP: Error #179: Function Can't open SHM2 failed`

## API Contract Notes (Backend ↔ Frontend)
- Pydantic-backed payloads exist for settings and skeleton clip:
  - `server/core.py:43-79` (`SettingsUpdatePayload`)
  - `server/core.py:81-98` (`SkeletonClipPayload`)
- Monitor inference request now uses a typed payload model:
  - request model: `server/core.py` (`MonitorPredictPayload`)
  - route signature: `server/routes/monitor.py:183`
  - response remains dict-based (consumed by frontend monitor hook):
  - response fields consumed by frontend:
    - frontend: `apps/src/pages/monitor/hooks/usePoseMonitor.js:306-365`
    - backend response: `server/routes/monitor.py:479-490`

## Quick Test (After Fixing Broken #1)
0. In-process no-training smoke (recommended in constrained environments):
```bash
PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py
```

1. Start backend:
```bash
source .venv/bin/activate
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```
2. Health check:
```bash
curl -sS http://localhost:8000/api/health
```
3. Minimal inference call (raw window):
```bash
curl -sS -X POST http://localhost:8000/api/monitor/predict_window \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id":"quicktest",
    "mode":"dual",
    "dataset_code":"le2i",
    "op_code":"OP-2",
    "target_T":48,
    "raw_t_ms":[0,40],
    "raw_xy":[[[0.1,0.2]],[[0.1,0.2]]],
    "raw_conf":[[1.0],[1.0]],
    "window_end_t_ms":40
  }'
```
