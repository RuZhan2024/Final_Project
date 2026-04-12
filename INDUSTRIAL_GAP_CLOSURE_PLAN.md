# Industrial Gap Closure Plan

This document defines the next practical steps for moving the cleaned
`refactor/industrial-structure` branch closer to an industry-grade repository.
It is intentionally scoped to high-value improvements that strengthen delivery,
operability, and maintainability without turning the project into a full
enterprise platform rewrite.

## Goal

The goal is not to claim full enterprise maturity in one step. The goal is to
close the highest-value gaps that still separate the current branch from a more
credible industrial codebase:

- CI/CD is partial
- observability is basic
- secrets and configuration governance are not yet strong enough
- MLOps lifecycle handling is only implicit
- compatibility layers and legacy shims still exist

## Scope

This plan targets the following areas:

1. CI/CD
2. Observability
3. Secrets and configuration governance
4. Compatibility and legacy cleanup
5. Lightweight MLOps lifecycle controls

This plan does not attempt to:

- turn the repository into a multi-service enterprise platform
- add full cloud infrastructure automation
- implement a complete model registry platform
- redesign the product or ML methodology

## Execution Rules

- Each task must preserve the current runnable reviewer/demo path.
- No task should reintroduce report-writing or research-process material into
  this clean submission branch.
- Configuration changes must prefer one explicit source of truth.
- No new compatibility shim should be introduced unless a removal plan is
  documented.
- Every completed area must include at least one verification step that would
  fail if the intended boundary were broken.

## Priority Order

1. CI/CD
2. Secrets and configuration governance
3. Observability
4. Compatibility cleanup
5. Lightweight MLOps controls

This order is intentional. It prioritizes improvements that most directly
increase confidence in the repository as deliverable software.

## Workstream 1: CI/CD

### Objective

Make the repository automatically verify its canonical paths on every change.

### Tasks

- Add a GitHub Actions workflow for:
  - Python dependency install
  - frontend dependency install
  - `bash -n` shell checks for operational scripts
  - `ops/scripts/release_doctor.sh`
  - selected `pytest` subsets from `qa/tests`
- Add a second lightweight workflow for pull requests that runs only the fast
  smoke/contract subset.
- Make workflow logs easy to read and fail fast on path/config regressions.

### Exit Checks

- Pull requests trigger automated checks without manual local setup.
- A broken import path, invalid shell script, or failing selected test blocks
  the workflow.
- `release_doctor.sh` passes in CI using the canonical normalized structure.

## Workstream 2: Secrets and Configuration Governance

### Objective

Make runtime configuration explicit, documented, and safer to operate.

### Tasks

- Add or tighten `.env.example` so required environment variables are clear.
- Audit backend config reads and ensure they route through
  `applications/backend/config.py`.
- Classify configuration into:
  - required secrets
  - optional secrets
  - safe local defaults
  - runtime path configuration
- Remove ambiguous or duplicated fallback behaviour where possible.
- Document required deployment variables for local, Docker, and Render-style
  usage.

### Exit Checks

- A new developer can identify required secrets from one documented source.
- Backend runtime config reads are centralized through the config layer.
- Missing required secrets fail clearly rather than degrading silently.

## Workstream 3: Observability

### Objective

Improve runtime visibility so failures and behaviour are easier to understand.

### Tasks

- Add structured request logging for backend API paths.
- Add explicit startup logging for backend configuration mode, DB backend, and
  active runtime profile.
- Separate health and readiness semantics where appropriate.
- Add focused logging around event persistence and notification dispatch.
- Add a documented debug mode for local diagnosis without flooding normal logs.

### Exit Checks

- Backend startup logs state the active runtime mode and critical config.
- Key monitor/event/notification failures are visible without manual print
  debugging.
- Health/readiness behaviour is documented and testable.

## Workstream 4: Compatibility and Legacy Cleanup

### Objective

Reduce the remaining transitional layers now that the industrialized structure
 has been established.

### Tasks

- Continue shrinking `applications/backend/core.py`.
- Remove route-level compatibility aliases that only exist for migration-era
  test shims.
- Update tests to target canonical modules directly where possible.
- Review operational scripts for any remaining old-path assumptions.
- Remove obsolete compatibility comments and stale naming references.

### Exit Checks

- `applications/backend/core.py` is closer to a compatibility shell than an
  implementation home.
- Routes rely on canonical helpers/services rather than migration aliases.
- No active script depends on removed top-level compatibility symlinks.

## Workstream 5: Lightweight MLOps Lifecycle Controls

### Objective

Make model/config/runtime promotion more explicit without building a full
 enterprise MLOps platform.

### Tasks

- Define a minimal promotion contract for:
  - training outputs
  - selected checkpoints
  - operating-point YAMLs
  - runtime deploy assets
- Document what counts as:
  - experimental output
  - promoted runtime asset
  - local-only artifact
- Add a lightweight manifest for shipped checkpoints and replay assets.
- Make `README` and operational scripts refer only to promoted runtime assets,
  not arbitrary experiment outputs.

### Exit Checks

- The repository distinguishes clearly between experimental and deployable ML
  assets.
- A reviewer can identify which checkpoints/configs are intended for runtime
  use.
- Promotion state is documented without relying on private research notes.

## Suggested Delivery Batches

### Batch A: 1 day

- CI smoke workflow
- `.env.example` cleanup
- `config.py` audit pass
- release/check script review

### Batch B: 1 to 2 days

- structured backend logging
- health/readiness hardening
- notification/event logging improvements
- compatibility alias cleanup round

### Batch C: 2 to 3 days

- lightweight ML asset manifest
- promotion contract documentation
- final `core.py` reduction pass

## Merge Criteria For This Plan

This gap-closure plan should be considered effectively executed when:

- CI runs the canonical smoke/release path automatically
- config/secrets handling is centralized and documented
- backend logging is strong enough for operational debugging
- compatibility layers are materially reduced again
- promoted ML/runtime assets are explicitly distinguished from experiment output

At that point, the repository would still not be a full enterprise system, but
it would be substantially closer to a disciplined industry-style project than a
research prototype with cleanup.

## Current Status On This Branch

- Workstream 1: completed first usable pass
- Workstream 2: completed first usable pass
- Workstream 3: completed first usable pass
- Workstream 4: completed first usable pass
- Workstream 5: completed first usable pass once `ops/deploy_assets/manifest.json`
  and its validation test are present
