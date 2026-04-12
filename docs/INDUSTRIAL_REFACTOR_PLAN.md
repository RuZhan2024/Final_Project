# Industrial Refactor Plan

## Purpose

This document defines a production-oriented refactor path for the repository. The goal is not to turn the project into a full enterprise platform in one step, but to move it from a strong research prototype toward a cleaner, more maintainable, and more operationally credible system.

The plan is designed for incremental delivery. Each phase should leave the repository runnable.

## Scope Boundary

This refactor stabilizes the backend boundary first. It does not attempt to fully restructure the wider monorepo into separate packages in a single pass.

That means the immediate focus is:

- backend application assembly
- backend runtime configuration
- backend API contracts
- backend runtime asset handling
- backend route/service/repository boundaries

The training, evaluation, and broader monorepo package structure may be improved later, but they are not the primary surface of the first industrialization pass.

## Current Problems

The repository already contains substantial engineering work, but it still shows several research-first patterns that would be weak in a production codebase:

- backend app assembly, environment loading, and runtime path resolution are too distributed
- API payload models, state helpers, schema probing, and fallback logic are mixed inside `server/core.py`
- runtime-generated assets and long-lived repository assets are not cleanly separated
- backend route handlers still carry too much orchestration responsibility in some paths
- configuration truth sources have historically drifted
- training, evaluation, deployment, and application concerns still live in one monorepo without a sufficiently explicit boundary model

## Refactor Objectives

The industrial refactor should improve the following properties:

1. Clear assembly boundary
   - application factory, route registration, and runtime config loading should be explicit and centralized

2. Clear configuration boundary
   - environment-backed runtime configuration and path resolution should have one source of truth

3. Clear contract boundary
   - API payload schemas should live separately from shared backend helpers

4. Clear runtime asset boundary
   - generated clips, local databases, logs, and caches should be treated as runtime data, not as structural repository content

5. Clear service layering
   - route handlers should increasingly become transport adapters
   - service and repository layers should own orchestration and persistence

6. Verifiable structure
   - key boundaries should be enforced by tests, not only by convention

## Migration Rules

The refactor should follow these rules throughout every phase:

1. No broad rename-only churn unless it removes an identified boundary risk.
2. Each phase must preserve current API behavior unless an explicit compatibility change is documented.
3. Temporary compatibility shims are allowed, but they should be removed by the next phase that makes them unnecessary.
4. No route should gain new orchestration responsibility during the refactor.
5. New business logic should not be added to compatibility surfaces created only to ease migration.
6. The default runnable demo path must remain usable at the end of each phase.

## Runtime Asset Classification

The repository should distinguish clearly between the following classes of assets:

1. Source-controlled artifacts
   - code, configuration, test fixtures, and intentional deploy assets

2. Reproducible generated artifacts
   - reports, plots, summaries, and reproducible experimental outputs that can be rebuilt from tracked commands

3. Runtime-local mutable data
   - event clips, local SQLite databases, local logs, temporary exports, and cache-like state

4. Secret-bearing configuration
   - environment variables, API credentials, local private env files, and any external-service tokens

The industrialization pass should reduce ambiguity between these classes, especially where research evidence, demo assets, and runtime-generated data currently share one monorepo.

## Target Backend Shape

The intended backend shape is:

```text
server/
  app.py                  # stable ASGI entrypoint
  application.py          # app factory and route registration
  config.py               # runtime config and path resolution
  schemas.py              # shared request/response payload models
  core.py                 # shared backend helpers and in-memory fallbacks only
  routes/                 # transport layer
  services/               # orchestration and runtime logic
  repositories/           # persistence logic
  notifications/          # outbound delivery subsystem
```

This does not require a full rewrite. It requires reducing ambiguity about where new code belongs.

## Phased Refactor Plan

### Phase 1: Assembly and Configuration Hardening

Scope:

- introduce a dedicated FastAPI application factory
- centralize runtime configuration and path resolution
- move shared API payload models into a dedicated schema module
- ensure route modules depend on schemas and services rather than on broad utility surfaces
- treat event clip storage and SQLite locations as config-resolved runtime paths

Expected outcome:

- the backend can be reasoned about as an assembled application rather than a set of side-effectful imports

Exit checks:

- `server/app.py` is a stable ASGI entrypoint only
- application assembly flows through one explicit factory path
- environment/path reads for backend runtime state are routed through `server/config.py`
- request payload models no longer live inside `server/core.py`
- at least one config or assembly test fails if this boundary is broken

### Phase 2: Core Decomposition

Scope:

- reduce `server/core.py` further by moving unrelated helper families into narrower modules
- separate:
  - schema probing helpers
  - in-memory fallback state
  - event clip privacy/runtime helpers
  - normalization and coercion helpers

Expected outcome:

- `core.py` becomes a thin compatibility layer or disappears entirely

Core strategy:

- short term: `core.py` remains as a compatibility-only surface
- medium term: no new business logic may be added to `core.py`
- long term: `core.py` is removed once imports and responsibilities have been fully migrated

Exit checks:

- no new schema or route-specific logic remains in `core.py`
- helper families moved out of `core.py` have a narrower destination module
- imports no longer require `core.py` as a catch-all dependency
- at least one test fails if a migrated helper family is pulled back into the old boundary

### Phase 3: Route Slimming

Scope:

- move more orchestration out of route handlers into services
- keep route handlers focused on:
  - request parsing
  - service invocation
  - HTTP mapping
- avoid database and filesystem logic directly inside routes when a service already exists

Expected outcome:

- route handlers become easier to test and safer to evolve

Route rule:

- route handlers should be limited to request parsing, service invocation, and HTTP response mapping
- filesystem, database, and orchestration logic should move behind service or repository boundaries

Exit checks:

- selected high-value routes no longer perform direct orchestration inline
- route modules show reduced direct filesystem or persistence handling
- new logic added during refactor lands in services or repositories rather than routes
- at least one route-contract or service-boundary test would fail if orchestration moved back into the route

### Phase 4: Runtime Asset Boundary

Scope:

- formalize runtime-generated data as non-source assets
- standardize directories for:
  - event clips
  - SQLite files
  - logs
  - temporary exports
- document what belongs in source control and what does not

Expected outcome:

- the repository becomes easier to review and safer to expose externally

Exit checks:

- runtime-local mutable data paths are explicitly documented
- generated event clips and local databases are not treated as structural repository content
- `.gitignore` and runtime path config agree on what is mutable local state
- at least one verification path checks the expected runtime asset locations or ignores

### Phase 5: Release and Verification Hardening

Scope:

- strengthen targeted tests for config, app assembly, route contracts, and service boundaries
- improve CI-oriented command surfaces
- define a minimal release check for:
  - import sanity
  - runtime path sanity
  - config parsing
  - core API contracts

Expected outcome:

- structural refactors become lower risk because the repository has explicit guardrails

Verification strategy:

- app assembly tests
- config parsing and path resolution tests
- route contract tests
- service/repository boundary tests

No phase is complete without at least one test that would fail if the new boundary were violated.

Exit checks:

- each introduced structural boundary has at least one targeted test
- release-oriented checks cover import sanity and config sanity
- demo path viability is rechecked after structural changes
- new structural rules are documented close to the code they govern

## Non-Goals

This refactor does not attempt to deliver all of the following in one pass:

- full enterprise MLOps
- full model registry infrastructure
- cloud-native observability stack
- full microservice decomposition
- production SRE-level logging, tracing, and rollout automation

Those would be larger follow-on efforts. The purpose here is to make the monorepo significantly cleaner and more professional without destabilizing the current project.

## Refactor Risks

The refactor itself carries risks that should be treated explicitly:

1. accidental API breakage
   - route signatures or response structures may drift while code is being reorganized

2. path and configuration regressions
   - centralizing config may break previously implicit runtime assumptions

3. mixed old/new import patterns lingering too long
   - compatibility layers can become permanent if they are not actively retired

4. runtime asset confusion
   - moving clip, SQLite, or output boundaries can create uncertainty about what is source-controlled versus runtime-local

5. verification blind spots
   - if structure changes faster than tests, the repository can become more elegant-looking but less reliable

## Compatibility and Rollback Expectations

The refactor should remain conservative in terms of compatibility:

- the default demo path should stay runnable after every phase
- public API compatibility should be preserved unless a change is explicitly documented
- short-lived compatibility shims are acceptable when they reduce migration risk
- no phase should be merged until its focused verification checks pass

## Acceptance Criteria

The refactor should be considered successful when the following are true:

- backend startup can be explained through one application factory
- runtime config and key paths are resolved through one config module
- API payload schemas no longer live inside a broad helper module
- runtime-generated clips and local databases are clearly treated as runtime data
- route handlers show reduced orchestration responsibility
- focused tests verify the new boundaries

## Immediate Work Items

The first implementation pass on the `refactor/industrial-structure` branch should prioritize:

1. backend application factory extraction
2. centralized runtime config module
3. dedicated `server/schemas.py`
4. event clip/runtime path centralization
5. focused tests for config and structural boundaries

## Reviewer Positioning

This refactor should be described as an industrialization pass, not as a claim that the repository is now a production platform. The honest and defensible claim is:

> the project is being restructured from a research-first monorepo toward a cleaner, more production-oriented application layout with stronger configuration, contract, and runtime boundaries.

That is the right standard for this repository and this stage of the project.
