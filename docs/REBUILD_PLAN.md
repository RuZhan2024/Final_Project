# Safe Guard Clean Rebuild Plan

Branch: `codex/clean-rebuild`

Baseline: `origin/main` at `aad3751`

Purpose: rebuild the project into a clean, robust, readable, industrial-style fall detection system while keeping the original repository history and using `main` as a reference source.

## 1. Rebuild Intent

This rebuild is not a cosmetic refactor. It is a controlled reconstruction of the project boundaries:

- ML pipeline should be reproducible, typed, and artifact-driven.
- Backend should expose stable APIs and isolate transport, business logic, persistence, and model runtime.
- Frontend should be modern, fast, maintainable, and visually polished.
- Runtime claims should match promoted artifacts and documentation.
- Demo safety should be explicit. No public write or notification endpoints should be exposed without a guard.

The current project contains valuable work: pose processing, TCN and GCN experiments, replay clips, runtime profiles, backend routes, frontend monitoring UI, tests, and report evidence. The rebuild should preserve that value, but not preserve the current coupling.

## 2. Guiding Principles

1. Keep the repository, rebuild the structure.
2. `main` is reference material, not the place to continue patching.
3. Start with a minimal vertical slice before migrating every feature.
4. Use explicit contracts between layers.
5. Prefer boring, well-understood architecture over clever abstractions.
6. Keep model runtime independent from research experiments.
7. Make every important artifact traceable to config, code, data, and metrics.
8. Treat safety, privacy, and access control as first-class project concerns.
9. Every migration step must be testable.
10. Delete legacy code only after the replacement path is verified.

## 3. Non-Goals

The first rebuild phase should not:

- Rewrite all ML algorithms at once.
- Move model inference into the frontend.
- Replace FastAPI with Next.js API routes.
- Reproduce every historical experiment before a minimal demo works.
- Keep MySQL and SQLite and Postgres logic all tangled together.
- Keep the old Makefile as the orchestration layer.
- Present historical 23/24 replay results as the current runtime claim.

## 4. Target System Shape

The rebuilt system should have four clear domains:

```text
applications/
  backend/       FastAPI API, runtime inference, persistence, security
  frontend/      Next.js UI

ml/
  src/fall_detection/
    data/        dataset adapters, schema, splits
    pose/        extraction and preprocessing
    features/    windowing and feature transforms
    models/      TCN, GCN, PoseC3D-lite candidates
    training/    train loop and checkpoints
    evaluation/  metrics, replay evaluation, calibration
    runtime/     promoted model loading and prediction
    cli/         typed ML pipeline CLI

ops/
  deploy_assets/ promoted runtime artifacts only
  configs/       production and experiment configs

qa/
  tests/         backend, frontend, ML, contract, replay smoke
```

## 5. Technology Choices

### Backend

Recommended:

- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- Alembic
- SQLite for local demo
- Postgres for production-style deployment
- Uvicorn

Rationale:

- FastAPI is already a good fit for model-backed APIs.
- Pydantic gives clear input and response contracts.
- SQLAlchemy plus Alembic is cleaner than hand-maintaining many DB dialect branches.
- Postgres is a better long-term production target than MySQL for this project.

Decision:

- Keep FastAPI.
- Replace ad hoc DB compatibility over time with repositories backed by SQLAlchemy.
- Keep SQLite local path for simple demo.
- Do not move core backend logic into Next.js API routes.

### Frontend

Recommended:

- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui or a small local component system
- TanStack Query
- Zustand only for small runtime UI state
- Playwright for browser smoke tests

Rationale:

- Next.js gives clean routing, layouts, loading states, and a modern build chain.
- Monitor is browser-heavy, so it should be implemented as client-only components.
- Next.js should own UI and routing, not ML inference.

Important rule:

- Camera, MediaPipe, canvas, video replay, WebSocket, and prediction loops must live behind `"use client"` and dynamic imports where appropriate.

### ML Framework

Recommended first phase:

- Keep PyTorch.
- Hide framework details behind `ModelRuntime` and `ModelRegistry`.
- Do not migrate to Keras in phase 1.

Rationale:

- Existing checkpoints and evidence are PyTorch-based.
- CTR-GCN-style models are natural in PyTorch.
- The current problem is pipeline structure, not the framework.

Future option:

- Keras or TensorFlow can be evaluated later if the model set narrows to TCN-like architectures or if TFLite/mobile/browser deployment becomes a goal.

### ML Orchestration

Recommended:

- Typer CLI
- YAML configs
- Pydantic config validation
- run manifests
- promoted deployment manifests

Makefile should become a thin convenience layer, not the project brain.

## 6. Model Strategy

The rebuilt model system should support multiple candidates behind one interface.

Initial tiers:

```text
Tier 0: Geometry and heuristic sanity baseline
Tier 1: TCN-lite production default
Tier 2: CTR-GCN or ST-GCN research benchmark
Tier 3: PoseC3D-lite robustness candidate
Tier 4: Transformer candidate, later research only
```

Recommended first production default:

- TCN-lite

Why:

- Small.
- Fast.
- Good for real-time sliding windows.
- Easier to deploy and explain.
- Strong enough for binary fall monitoring when the pipeline is clean.

Recommended research benchmark:

- CTR-GCN or simpler ST-GCN.

Recommended new candidate:

- PoseC3D-lite or PoseConv3D-inspired heatmap representation.

Why:

- It may be more robust to pose noise and cross-dataset differences.
- It directly addresses one of this project's biggest issues: runtime pose quality.

Do not make Transformer the first rebuilt model. It is interesting, but likely too complex and data-hungry for the first clean release.

## 7. ML Pipeline Design

The new ML pipeline should be a typed CLI, not a large Makefile.

Target command shape:

```bash
fd data validate --config configs/data/caucafall.yaml
fd pose extract --config configs/pose/mediapipe.yaml
fd windows build --config configs/windows/caucafall.yaml
fd split make --config configs/splits/caucafall.yaml
fd train --config configs/experiments/tcn_caucafall_seed42.yaml
fd eval --run runs/2026-08-14_1530_tcn_caucafall_seed42
fd calibrate --run runs/2026-08-14_1530_tcn_caucafall_seed42
fd promote --run runs/2026-08-14_1530_tcn_caucafall_seed42 --target ops/deploy_assets
fd runtime smoke
```

Pipeline stages:

```text
raw data
  -> dataset adapters
  -> pose extraction
  -> pose preprocessing
  -> window generation
  -> split validation
  -> training
  -> calibration
  -> evaluation
  -> promotion
  -> backend runtime inference
```

Each stage must have:

- input contract
- output contract
- config schema
- deterministic output path
- validation command
- tests for edge cases

## 8. Artifact Rules

Training and evaluation outputs should go to `runs/`, not `outputs/` as an unstructured dumping ground.

Example:

```text
runs/2026-08-14_1530_tcn_caucafall_seed42/
  config.requested.yaml
  config.resolved.yaml
  manifest.json
  checkpoint.pt
  metrics.json
  calibration.json
  evaluation/
    replay_metrics.json
    confusion_matrix.json
  logs/
```

Promotion copies only approved runtime files:

```text
ops/deploy_assets/
  manifest.json
  checkpoints/
  profiles/
  replay_clips/
```

Backend runtime must only read `ops/deploy_assets/manifest.json`.

It must not:

- scan historical config directories
- read arbitrary `outputs/`
- infer production behavior from report artifacts
- silently pick a checkpoint by filename

## 9. Backend Architecture

Target backend structure:

```text
applications/backend/
  app/
    main.py
    application.py
    config.py
    security.py
    errors.py
    logging.py
    contracts/
    routes/
    services/
    repositories/
    runtime/
    db/
```

Layer responsibilities:

- Routes: HTTP and WebSocket transport only.
- Contracts: Pydantic request and response models.
- Services: business logic and use cases.
- Repositories: database reads and writes.
- Runtime: model loading and inference.
- Security: API key, demo guard, CORS, access policy.
- Config: environment parsing and validation.

Monitor prediction should become a use case:

```text
MonitorPredictUseCase
  -> normalize request
  -> validate pose window
  -> evaluate quality
  -> run model runtime
  -> apply decision policy
  -> optionally persist event
  -> return typed response
```

No route file should own the full monitor decision pipeline.

## 10. Frontend Architecture

Target frontend structure:

```text
applications/frontend/
  app/
    layout.tsx
    page.tsx
    monitor/page.tsx
    events/page.tsx
    settings/page.tsx
  features/
    monitor/
      api.ts
      components/
      hooks/
      runtime/
      types.ts
    events/
    settings/
  components/
    ui/
    layout/
  lib/
    api-client.ts
    config.ts
    dates.ts
  contracts/
    api-types.ts
  public/
    mediapipe/
```

Monitor performance rules:

- Store frame buffers in refs, not React state.
- Do not set state every video frame.
- Update display state at controlled intervals.
- Load MediaPipe only on the Monitor page.
- Use dynamic imports for browser-only code.
- Keep WebSocket transport isolated.
- Keep prediction parsing isolated from UI components.

## 11. API Contract Strategy

The backend should be the source of truth for API contracts.

Recommended:

- Pydantic models for requests and responses.
- OpenAPI generation from FastAPI.
- Generate TypeScript types for frontend.
- Contract tests for critical endpoints.

Critical endpoints for phase 1:

```text
GET  /api/health
GET  /api/spec
POST /api/monitor/predict_window
POST /api/monitor/reset_session
WS   /api/monitor/ws
```

Compatibility policy:

- Either support `/api/v1/*` consistently or formally remove it everywhere.
- Do not remove route aliases while tests or frontend still rely on them.

## 12. Security and Privacy

Minimum requirements:

- No real secrets in the repository or working tree intended for sharing.
- `.env.local.private` must not be loaded from the wrong path.
- Public write endpoints require an API key or demo guard.
- Notification test endpoints require an explicit guard.
- Skeleton clip storage must default to off.
- Anonymized skeleton mode must be documented and tested.
- CORS is not authentication.

Recommended modes:

```text
APP_MODE=local
APP_MODE=demo
APP_MODE=production
```

Mode behavior:

- local: convenient, local-only defaults.
- demo: public-safe, write actions guarded or disabled.
- production: strict config, auth required, no silent secret defaults.

## 13. Configuration Rules

Configuration should be explicit and validated.

Backend config:

- Read env from the actual repo root for local development.
- Prefer environment variables in deployment.
- Fail fast for production-required config.
- Use typed config objects.

ML config:

- YAML files validated by Pydantic.
- No hidden defaults for critical dataset or model fields.
- Resolved config saved into every run directory.

Frontend config:

- API base URL explicit.
- No hard-coded production URLs in feature code.
- Browser-only config isolated from server config.

## 14. Makefile Strategy

The new Makefile should be short.

Allowed responsibilities:

- install
- dev
- test
- lint
- backend
- frontend
- ml-smoke
- clean

Not allowed:

- full ML pipeline logic
- long chains of dataset-specific commands
- experiment decision logic
- report evidence generation
- deployment profile selection

The real pipeline belongs in `fd` CLI commands.

## 15. Testing Strategy

Test pyramid:

```text
unit tests
  contracts
  windowing
  config parsing
  decision policy
  metrics

integration tests
  backend API
  runtime smoke with mocked model
  database repository behavior
  frontend API client contract

browser smoke tests
  dashboard loads
  monitor page loads
  replay mode starts
  settings page loads

ML smoke tests
  config validation
  split leakage guard
  tiny training/eval dry run where possible
```

CI phase 1 should run:

- Python compile or Ruff check.
- Pytest contract slice.
- Frontend typecheck.
- Frontend build.
- API contract smoke.

CI phase 2 should add:

- Playwright smoke.
- runtime manifest validation.
- replay smoke with fixture.

## 16. Documentation Strategy

Documentation should be separated by audience:

```text
docs/
  REBUILD_PLAN.md
  ADR/
  USER_GUIDE.md
  DEVELOPER_GUIDE.md
  RUNTIME_CLAIMS.md
  ML_PIPELINE.md
```

Rules:

- Current runtime claim must appear in one canonical document.
- Historical results must be marked as historical or retired.
- 22/24 OP2-only and 23/24 motion-gated result must not be mixed.
- README should explain how to run the current system, not every historical experiment.

## 17. Migration Plan

### Phase 0: Branch and Ground Rules

Status: started.

Tasks:

- Create `codex/clean-rebuild`.
- Keep old worktree as reference.
- Add this rebuild plan.
- Decide initial tech stack.

Done when:

- Clean branch exists.
- Rebuild plan is committed.
- First ADRs exist.

### Phase 1: Minimal Clean Skeleton

Tasks:

- Add backend app skeleton.
- Add Next.js frontend skeleton.
- Add typed config.
- Add `/api/health`.
- Add CI basics.
- Add thin Makefile.
- Add project README for clean rebuild branch.

Done when:

- Backend starts.
- Frontend starts.
- CI runs basic checks.
- No old monitor code is required for startup.

### Phase 2: Runtime Vertical Slice

Tasks:

- Implement deploy manifest loader.
- Implement `ModelRuntime` interface.
- Add mocked model runtime for tests.
- Add promoted TCN runtime loader.
- Add `/api/spec`.
- Add `/api/monitor/predict_window`.
- Add frontend Monitor page skeleton.
- Add one replay smoke path.

Done when:

- A replay window can flow from frontend to backend to typed prediction response.
- Runtime behavior is controlled by promoted manifest only.

### Phase 3: ML CLI Foundation

Tasks:

- Add `fd` CLI.
- Add config schemas.
- Add `fd data validate`.
- Add `fd windows build` or equivalent minimal windowing command.
- Add `fd runtime smoke`.
- Add run manifest format.

Done when:

- Makefile no longer contains ML pipeline logic.
- At least one ML pipeline command is tested.

### Phase 4: Full Monitor Product Surface

Tasks:

- Implement WebSocket prediction transport.
- Implement monitor session state.
- Implement event persistence.
- Implement settings.
- Implement skeleton clip storage.
- Add privacy controls.
- Add demo guard.

Done when:

- Monitor, Events, Settings, Dashboard all work through typed APIs.
- Write endpoints are guarded.

### Phase 5: Model Benchmarks

Tasks:

- Reintroduce TCN training in the new CLI.
- Reintroduce CTR-GCN benchmark.
- Add PoseC3D-lite experimental candidate.
- Add consistent event-level evaluation.
- Add calibration step.

Done when:

- Models are compared through one pipeline and one metrics schema.
- Runtime promotion is reproducible.

### Phase 6: Documentation and Release

Tasks:

- Rewrite README.
- Write user guide.
- Write developer guide.
- Write ML pipeline guide.
- Write runtime claims document.
- Clean old artifacts or move them to an archive strategy.

Done when:

- A new developer can run the project from README.
- Runtime claim matches deployed artifact and tests.

## 18. First ADRs To Create

Recommended ADR list:

```text
docs/ADR/0001-clean-rebuild-in-existing-repo.md
docs/ADR/0002-nextjs-frontend-fastapi-backend.md
docs/ADR/0003-keep-pytorch-behind-runtime-interface.md
docs/ADR/0004-typed-python-cli-replaces-large-makefile.md
docs/ADR/0005-runtime-reads-promoted-manifest-only.md
docs/ADR/0006-demo-mode-and-api-guard.md
```

## 19. Initial Definition of Done

The clean rebuild is successful when:

- The project starts with one documented command.
- The frontend is Next.js and builds cleanly.
- The backend is FastAPI and exposes typed contracts.
- The ML runtime is manifest-driven.
- The ML pipeline is CLI-driven, not Makefile-driven.
- Secrets are absent from the repo and shared working tree.
- Public write endpoints are guarded.
- Current runtime metrics and docs agree.
- CI catches API, frontend, and runtime regressions.
- Legacy code can be removed or clearly archived.

## 20. Immediate Next Steps

1. Commit this document.
2. Add ADR 0001 and ADR 0002.
3. Decide whether to keep existing files temporarily or replace the frontend/backend skeleton first.
4. Create a thin Makefile for the clean branch.
5. Scaffold backend clean app.
6. Scaffold Next.js frontend.
7. Add CI for the new skeleton.

