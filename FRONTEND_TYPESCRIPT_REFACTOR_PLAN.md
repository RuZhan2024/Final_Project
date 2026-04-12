# Frontend TypeScript Refactor Plan

## Purpose

This document defines a practical TypeScript migration plan for the current frontend in `applications/frontend`. The goal is not a cosmetic file extension change. The goal is to make the frontend safer to evolve, easier to review, and less dependent on implicit runtime assumptions.

The migration should preserve current behaviour while introducing typed boundaries around:

- API payloads and responses
- monitor runtime state
- event-history replay data
- settings forms and feature flags
- frontend-side MediaPipe and WebSocket orchestration

## Current Frontend Snapshot

The current frontend is a Create React App application using React 19, React Router 7, CSS modules, and plain JavaScript.

Observed structure:

- `applications/frontend/src/App.js`
- `applications/frontend/src/pages/*`
- `applications/frontend/src/features/*`
- `applications/frontend/src/lib/*`
- `applications/frontend/src/monitoring/MonitoringContext.js`

Observed size and migration pressure:

- `42` JavaScript source files
- `6` CSS files
- `src/pages/monitor/hooks/usePoseMonitor.js`: `1384` lines
- `src/pages/settings/SettingsPage.js`: `483` lines
- `src/pages/Monitor.js`: `308` lines
- `src/pages/Events.js`: `314` lines
- `src/pages/Dashboard.js`: `182` lines

Observed architectural shape:

- the app is already partially feature-oriented
- API access is centralized only lightly through `src/lib/apiClient.js`
- backend payloads are mostly untyped objects
- monitor runtime logic is concentrated in one large hook
- state contracts are inferred from server responses rather than declared

This means the TypeScript migration should be treated as a boundary-hardening project, not as a search-and-replace exercise.

## Why TypeScript Here

The current frontend has several risk patterns that TypeScript can materially improve:

1. API payload drift is easy to introduce.
   `apiRequest()` returns `any`-like data and pages infer shapes ad hoc.

2. Settings and monitoring contracts are implicit.
   `settings?.system || {}` and similar patterns make invalid states easy to normalize silently.

3. Monitor logic is state-heavy and event-driven.
   `usePoseMonitor.js` coordinates MediaPipe, WebSocket, replay, clip upload, timeline state, prediction state, and session state. This is exactly where typed state machines and discriminated unions pay off.

4. Event replay and monitor payloads have domain-specific structures.
   Skeleton clips, replay clip metadata, operating point params, and prediction responses should be typed once and reused.

5. The current app already has enough modularity to support incremental migration.
   It does not need a rewrite or framework switch before adopting TypeScript.

## Non-Goals

This refactor should not:

- replace Create React App with Vite in the same phase
- redesign the UI
- rewrite monitor behaviour or alert semantics
- convert every file in one pass
- introduce a large state library unless current boundaries truly require it

Framework migration can be considered later. It should not be coupled to the initial TypeScript migration.

## Target State

The frontend should end this migration with:

- TypeScript enabled through CRA-compatible configuration
- typed API clients with shared request/response contracts
- typed feature modules under `features/`, `pages/`, `lib/`, and `monitoring/`
- the monitor runtime split into smaller typed units
- no new `.js` feature code added in active paths
- JavaScript retained only temporarily in explicitly deferred legacy areas

## Recommended Target Structure

The existing layout is mostly usable. The main change is stronger typing and clearer contract ownership.

Recommended structure:

```text
applications/frontend/
  src/
    app/
      routes.tsx
      providers.tsx
    features/
      dashboard/
        api.ts
        types.ts
      events/
        api.ts
        types.ts
      monitor/
        api.ts
        media.ts
        prediction.ts
        socketClient.ts
        types.ts
        windowing.ts
      settings/
        api.ts
        types.ts
    lib/
      apiClient.ts
      config.ts
      dates.ts
      ui.ts
    monitoring/
      MonitoringContext.tsx
      types.ts
    pages/
      Dashboard.tsx
      Events.tsx
      Monitor.tsx
      Settings.tsx
      monitor/
        components/
        hooks/
        types.ts
      settings/
        hooks/
```

This keeps the current mental model but adds explicit ownership for types and contracts.

## Migration Rules

The migration should follow these rules:

1. Preserve current behaviour unless a bug is explicitly being fixed.
2. Convert boundary files first, not leaf utilities first.
3. Every phase must leave the frontend buildable and testable.
4. No file should be renamed to `.ts` or `.tsx` unless it also gains meaningful types.
5. Do not move the monitor feature to a new architecture and TypeScript at the same time.
6. Avoid broad `any` casts as a migration shortcut.
7. Temporary compatibility types are allowed, but every such type should have a named follow-up target.
8. No new feature code should be added in plain `.js` once the TypeScript toolchain is enabled.

## Phase Plan

### Phase 1: Enable TypeScript Without Behaviour Change

Scope:

- add `tsconfig.json`
- add TypeScript and React type dependencies
- add CRA-compatible TypeScript entry support
- convert the minimum bootstrap files needed for compiler activation

Recommended first conversions:

- `src/index.js` -> `src/index.tsx`
- `src/App.js` -> `src/App.tsx`
- `src/reportWebVitals.js` -> `src/reportWebVitals.ts`
- `src/setupTests.js` can remain JS initially

Exit checks:

- frontend starts with `npm start`
- frontend builds with `npm run build`
- no TypeScript compiler bootstrap errors
- route mounting still works unchanged

### Phase 2: Type the Shared Contracts

Scope:

- type `lib/apiClient`
- define frontend API response models
- define shared monitor/settings/event types
- remove implicit `object`-shaped payload assumptions

Priority files:

- `src/lib/apiClient.js` -> `src/lib/apiClient.ts`
- `src/lib/config.js` -> `src/lib/config.ts`
- `src/features/dashboard/api.js` -> `src/features/dashboard/api.ts`
- `src/features/events/api.js` -> `src/features/events/api.ts`
- `src/features/settings/api.js` -> `src/features/settings/api.ts`
- `src/features/monitor/api.js` -> `src/features/monitor/api.ts`

Required type groups:

- `SystemSettings`
- `SettingsResponse`
- `DashboardSummary`
- `EventRecord`
- `EventSkeletonClip`
- `ReplayClip`
- `MonitorPredictResponse`
- `OperatingPointProfile`

Exit checks:

- API modules return typed promises
- pages stop treating backend responses as free-form objects
- `apiRequest()` is generic and typed end-to-end
- no feature API module exports untyped response shapes

### Phase 3: Type the Context and Page Boundaries

Scope:

- type the monitoring context
- type top-level page props and page-local state
- reduce `null` and fallback ambiguity

Priority files:

- `src/monitoring/MonitoringContext.js` -> `src/monitoring/MonitoringContext.tsx`
- `src/pages/Dashboard.js` -> `src/pages/Dashboard.tsx`
- `src/pages/Events.js` -> `src/pages/Events.tsx`
- `src/pages/Settings.js` -> `src/pages/Settings.tsx`
- `src/pages/settings/SettingsPage.js` -> `src/pages/settings/SettingsPage.tsx`

Focus areas:

- `settings` shape from context
- event review state
- caregiver form state
- settings update patch types
- dashboard polling and response contracts

Exit checks:

- `useMonitoring()` returns a declared interface
- page props are explicit
- settings and event pages no longer rely on broad `|| {}` fallback patterns for core contracts
- form handlers use typed change events and typed patch objects

### Phase 4: Split and Type the Monitor Feature

Scope:

- type and reduce the monitor runtime path
- split the largest hook into typed subdomains
- give MediaPipe, replay, timeline, and clip-upload flows explicit contracts

This is the highest-value phase.

Priority files:

- `src/pages/Monitor.js` -> `src/pages/Monitor.tsx`
- `src/pages/monitor/hooks/usePoseMonitor.js` -> staged refactor
- `src/pages/monitor/hooks/useReplayClips.js` -> `useReplayClips.ts`
- `src/pages/monitor/hooks/useApiSpec.js` -> `useApiSpec.ts`
- `src/pages/monitor/hooks/useApiSummary.js` -> `useApiSummary.ts`
- `src/pages/monitor/hooks/useOperatingPointParams.js` -> `useOperatingPointParams.ts`
- `src/features/monitor/media.js` -> `media.ts`
- `src/features/monitor/socketClient.js` -> `socketClient.ts`
- `src/features/monitor/prediction.js` -> `prediction.ts`
- `src/features/monitor/windowing.js` -> `windowing.ts`

Recommended internal split for `usePoseMonitor`:

- `useMonitorSessionState`
- `useMonitorMediaPipeRuntime`
- `useMonitorPredictionLoop`
- `useMonitorReplayRuntime`
- `useMonitorClipUpload`
- `useMonitorTimeline`

Required type groups:

- `PoseFrame`
- `WindowFrame`
- `PendingClip`
- `MonitorSessionState`
- `PredictionUIState`
- `ReplayPlaybackState`
- `SocketPredictClient`
- `MonitorController`

Exit checks:

- `usePoseMonitor` is reduced substantially from current size
- monitor helper modules expose typed public contracts
- websocket payloads and responses are typed
- skeleton clip upload path is typed end-to-end
- no monitor module uses implicit `any` data for server responses

### Phase 5: Tighten Tests and Migration Enforcement

Scope:

- update existing tests for TypeScript paths
- add a no-new-JS rule for active frontend code
- add typecheck to CI

Recommended additions:

- `npm run typecheck`
- CI step for `tsc --noEmit`
- contract tests around typed monitor API parsing

Exit checks:

- CI runs build plus typecheck
- new frontend code paths are TypeScript by default
- remaining `.js` files are explicitly tracked as deferred legacy files

## Recommended Compiler Policy

Start conservative, then tighten.

Initial `tsconfig` direction:

- `allowJs: true`
- `checkJs: false`
- `strict: false`
- `noEmit: true`
- `jsx: react-jsx`

After Phases 2 to 4:

- `allowJs: true`
- `strict: true`
- `noImplicitAny: true`
- `noUncheckedIndexedAccess: true`
- `exactOptionalPropertyTypes: true`

Final desired state:

- `allowJs: false` for active source directories
- legacy JS retained only temporarily outside core paths, if any remain

## File Prioritization

The migration should not be done alphabetically. It should be done by leverage.

Highest priority:

1. `src/lib/apiClient.js`
2. `src/monitoring/MonitoringContext.js`
3. `src/features/monitor/api.js`
4. `src/pages/settings/SettingsPage.js`
5. `src/pages/events/components/EventSkeletonClipPanel.js`
6. `src/pages/monitor/hooks/usePoseMonitor.js`

Why these first:

- they carry most backend-contract risk
- they concentrate nullability and state ambiguity
- they define the shapes reused by the rest of the frontend

## Key Type Boundaries To Introduce Early

### Settings

Introduce explicit types for:

- `SystemSettings`
- `SettingsPayload`
- `SettingsPatch`
- `CaregiverRecord`

This will remove the current pattern of deriving many booleans and codes from partially known objects.

### Events

Introduce explicit types for:

- `EventStatus`
- `EventType`
- `EventRecord`
- `EventReviewUpdate`
- `EventSkeletonClipResponse`

This will make review actions and replay availability checks much safer.

### Monitor

Introduce explicit types for:

- monitor mode: `camera | video`
- triage state: `not_fall | uncertain | fall`
- monitor alert state
- timeline entry shape
- predict request/response shape
- replay clip metadata
- clip upload payload

This is where TypeScript will deliver the biggest reduction in accidental regressions.

## Shared Type Sources

The migration should establish a small number of explicit type-source files early, instead of scattering interface declarations across pages and hooks.

Recommended first type files:

```text
applications/frontend/src/features/settings/types.ts
applications/frontend/src/features/events/types.ts
applications/frontend/src/features/monitor/types.ts
applications/frontend/src/monitoring/types.ts
applications/frontend/src/lib/apiTypes.ts
```

Recommended ownership:

- `features/settings/types.ts`
  - `SystemSettings`
  - `SettingsResponse`
  - `SettingsPatch`
  - `CaregiverRecord`
- `features/events/types.ts`
  - `EventRecord`
  - `EventType`
  - `EventStatus`
  - `EventReviewUpdate`
  - `EventSkeletonClip`
- `features/monitor/types.ts`
  - `MonitorPredictRequest`
  - `MonitorPredictResponse`
  - `ReplayClip`
  - `PredictionState`
  - `PendingClip`
  - `MonitorController`
- `monitoring/types.ts`
  - `MonitoringContextValue`
  - `MonitoringSettingsState`
  - `MonitoringCommandResult`
- `lib/apiTypes.ts`
  - shared generic API wrapper types
  - error payload shapes
  - reusable primitive DTOs if they are not feature-specific

Dependency rule:

- pages should import feature types
- context should import feature types
- API modules should define or import feature types
- large hooks should not define ad hoc duplicate interfaces inline unless they are truly private

This gives the migration a type graph that matches the feature graph.

## Large File Decomposition Plan

The current frontend has several files that are too large to convert directly to TypeScript without first reducing responsibility concentration.

### 1. `usePoseMonitor.js`

Current state:

- `1384` lines
- owns MediaPipe runtime, camera/replay media, prediction loop, WebSocket fallback, timeline state, session state, skeleton clip upload, and replay timing

This file should not be converted directly to `.ts`.

Recommended split:

- `useMonitorMediaRuntime.ts`
  - camera stream setup
  - replay video setup
  - media cleanup
  - playback rate synchronization
- `useMonitorPredictionLoop.ts`
  - window slicing
  - predict payload construction
  - socket client orchestration
  - prediction request/response handling
- `useMonitorSessionState.ts`
  - session id
  - prediction UI state
  - triage state
  - timeline accumulation
- `useMonitorClipCapture.ts`
  - pending clip creation
  - timer scheduling
  - upload completion and dedupe
- `useMonitorReplayState.ts`
  - replay current time
  - duration
  - replay progress UI state

Recommended retained orchestration file:

- `usePoseMonitor.ts`
  - composes the smaller hooks
  - exposes the stable public monitor hook contract to the page layer

Exit checks for this decomposition:

- no single monitor hook exceeds roughly `400` to `500` lines
- clip upload no longer shares unrelated local state with replay playback
- prediction transport logic is isolated from media lifecycle logic
- the public `usePoseMonitor` return shape is explicitly typed

### 2. `SettingsPage.js`

Current state:

- `483` lines
- mixes settings rendering, caregiver form state, toast UX, patch creation, backend action handling, and settings normalization

Recommended split:

- `SettingsPage.tsx`
  - page composition only
- `components/CaregiverCard.tsx`
  - caregiver form and edit/save controls
- `components/MonitoringSettingsCard.tsx`
  - monitoring toggles and monitoring-specific controls
- `hooks/useSettingsToasts.ts`
  - status/error toast coordination
- `settings/form.ts`
  - patch-building helpers
  - settings normalization helpers

Recommended type ownership:

- form patch shapes come from `features/settings/types.ts`
- page-local editing state is local to the settings feature

Exit checks:

- the page no longer constructs large mixed patch objects inline
- caregiver form state is isolated from generic settings state
- toast and local error handling are not interwoven with API contract parsing

### 3. `Events.js`

Current state:

- `314` lines
- owns filters, event review state, review modal opening, status mutation, and skeleton replay coordination

Recommended split:

- `Events.tsx`
  - page composition and layout
- `components/EventFiltersBar.tsx`
  - filter UI only
- `components/EventReviewModal.tsx`
  - review modal wiring
- `hooks/useEventFilters.ts`
  - date/type/status/model filter state
- existing `EventSkeletonClipPanel` remains separate and becomes typed

Recommended type ownership:

- filter state becomes `EventFilters`
- review state becomes `EventReviewState`
- event list entries come from `EventRecord`

Exit checks:

- filter state is typed and isolated from review state
- event status update payloads are typed
- replay-ready versus no-replay state is derived from typed event metadata rather than loose object inspection

### 4. `Monitor.js`

Current state:

- `308` lines
- still carries page-level orchestration, prop wiring, and monitor control composition

Recommended split:

- `Monitor.tsx`
  - layout and top-level feature wiring only
- `hooks/useMonitorPageState.ts`
  - page-specific state that does not belong in the runtime hook
- preserve existing card components, but type their props explicitly

Exit checks:

- `Monitor.tsx` becomes a composition layer
- card props are explicit interfaces, not inferred from object spreading
- page concerns are kept separate from runtime concerns

## Risks

1. The monitor feature is large and event-driven.
   Converting it too quickly will create noise and slow review.

2. CRA TypeScript support is workable, but not ideal for large modern codebases.
   The migration should avoid mixing framework replacement with type migration.

3. Backend contracts may still evolve.
   Shared frontend types should be introduced in a way that is easy to update from API truth.

4. The current code uses implicit permissive defaults.
   TypeScript will expose these immediately. That is useful, but it will create short-term migration friction.

## Suggested Execution Order

Recommended practical order:

1. bootstrap TypeScript
2. type shared API contracts
3. type monitoring context
4. type settings and events pages
5. split and type monitor feature
6. enforce typecheck in CI

This order keeps the highest-traffic feature, `Monitor`, from becoming the first migration bottleneck.

## Definition of Done

The frontend TypeScript refactor should be considered complete when:

- the frontend runs and builds with TypeScript enabled
- all shared backend contracts used by the frontend are typed
- `MonitoringContext` is typed
- settings, events, dashboard, and monitor page boundaries are typed
- the monitor feature no longer relies on one large untyped orchestration hook
- CI includes `tsc --noEmit`
- no new active frontend code is added in plain JavaScript

## Recommended Immediate Next Step

Do not start with `usePoseMonitor.js`.

Start with:

1. `tsconfig.json`
2. package TypeScript dependencies
3. `src/index.tsx`
4. `src/App.tsx`
5. `src/lib/apiClient.ts`
6. `src/features/*/api.ts`

That sequence gives the migration a typed boundary model before touching the heaviest runtime hook.
