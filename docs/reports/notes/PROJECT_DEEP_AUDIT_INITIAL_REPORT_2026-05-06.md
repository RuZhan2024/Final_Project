# Project Deep Audit - Initial Findings - 2026-05-06

This is the first evidence-based audit pass. No cleanup has been applied.

## Executive Summary

The repository source code is moderate in size, but the working tree is noisy:
there are hundreds of untracked generated experiment specs/results, many
historical artifacts, and several compatibility layers that now obscure the
locked runtime story.

The biggest architectural issue is not one single bug. It is drift between
three stories:

1. Historical product/runtime: TCN, GCN, HYBRID, OP-1/2/3, dataset selection.
2. Current locked deployment: TCN, CaucaFall/LE2i controlled mix, OP-2.
3. Research direction: CTR-GCN full two-stream evaluation and possible deploy.

Those stories are all still represented in code and generated files. Cleanup
should happen in batches, with tests protecting runtime behavior.

## Inventory Evidence

- `git status --short`: 391 entries
- Modified tracked files: 31
- Untracked files in `git status`: 360
- Untracked but not ignored via `git ls-files --others --exclude-standard`: 802
- Ignored files: 1,930,949
- Largest ignored source: `data/` with about 1.8M ignored files

Targeted non-dependency directory scan:

| Path | Files | Size MB | Notes |
| --- | ---: | ---: | --- |
| `outputs` | 1,364 | 917.21 | Mostly old checkpoints and metrics; ignored but physically huge |
| `artifacts` | 1,393 | 161.32 | Mix of report evidence, eval outputs, logs, old runs |
| `ops` | 646 | 120.89 | Includes deploy checkpoints plus hundreds of generated OP YAMLs |
| `applications/frontend/public` | 19 | 50.31 | MediaPipe assets, likely intentional |
| `ml` | 166 | 1.45 | Source package is small |
| `qa` | 159 | 0.80 | Test suite is small |
| `applications/backend` | 135 | 0.77 | Backend source is small |
| `applications/frontend/src` | 63 | 0.27 | Frontend source is small |

Current untracked-not-ignored hotspots:

| Path | Count |
| --- | ---: |
| `artifacts` | 445 |
| `ops` | 334 |
| `configs` | 12 |
| `docs` | 7 |

Untracked `ops/configs/ops` families:

| Family | Count |
| --- | ---: |
| `ctr_gcn:caucafall` | 210 |
| `tcn:caucafall` | 100 |
| `tcn:le2i` | 10 |
| `ctr_gcn:le2i` | 10 |

## Findings

### F1 - Generated Sweep Specs Are Polluting Git Status

- Category: should-clean
- Area: ops, artifacts, git hygiene
- Evidence:
  - 334 untracked-not-ignored files under `ops`
  - 216 YAML files and 214 `.sweep.json` files currently in `ops/configs/ops`
  - 210 untracked CTR-GCN CaucaFall spec files
- Risk:
  - Hard to distinguish canonical deploy specs from one-off sweep output.
  - Future commits may accidentally include hundreds of exploratory YAMLs.
  - Runtime spec discovery reads `ops/configs/ops/*.yaml`, so generated files in
    that directory can change which model is selected if naming/sorting changes.
- Recommendation:
  - Keep only canonical deploy specs in `ops/configs/ops`.
  - Move exploratory sweep specs to an evidence/archive directory or generate
    them under `artifacts/ops_sweeps/`.
  - Add `.gitignore` patterns for generated sweep specs after canonical keepers
    are named explicitly.

### F2 - `outputs/` Is Huge and Historical

- Category: archive
- Area: outputs, checkpoints
- Evidence:
  - `outputs`: 917.21 MB, 1,364 files
  - Largest files are many `best.pt` / `last.pt` checkpoints from old TCN/GCN
    experiments.
- Risk:
  - Local project copies become slow and confusing.
  - The deployed checkpoint already lives under `ops/deploy_assets/checkpoints`,
    so old checkpoints are not needed for runtime.
- Recommendation:
  - Keep `outputs/` ignored.
  - Create an external/archive policy for old training outputs.
  - Keep a CSV/Markdown evidence index for report-relevant runs instead of
    relying on the raw output tree.

### F3 - `applications/backend/core.py` Is a Legacy Aggregator

- Category: should-clean
- Area: backend
- Evidence:
  - Production routes now import focused modules such as `deploy_ops`,
    `runtime_state`, `json_utils`, repositories, and services directly.
  - `core.py` duplicates wrappers for normalization, DB helpers, deploy op
    params, runtime session state, and JSON helpers.
  - Current references are primarily test-facing, especially
    `qa/tests/server/test_runtime_core.py`.
- Risk:
  - Tests can pass against old wrappers while production uses newer modules.
  - Cleanup/refactors must touch duplicate surfaces.
- Recommendation:
  - Migrate tests from `applications.backend.core` to the focused modules.
  - Mark `core.py` as deprecated or remove it in a dedicated backend cleanup
    batch after tests are updated.

### F4 - Runtime Model Taxonomy Has Drifted

- Category: must-fix before CTR-GCN deploy
- Area: backend, frontend, deploy runtime
- Evidence:
  - `applications/backend/code_normalization.py` supports `TCN`, `GCN`,
    `HYBRID`, but not `CTR_GCN`.
  - `applications/backend/deploy_runtime.py` discovers and can run `ctr_gcn`.
  - `applications/frontend/src/pages/Monitor.tsx` maps mode as `tcn/gcn/hybrid`.
  - Settings currently lock display to `TCN champion`.
- Risk:
  - CTR-GCN can exist in training/eval/deploy specs but cannot be cleanly
    selected through the product runtime taxonomy.
  - Old `GCN` and `HYBRID` paths may be mistaken for CTR-GCN support.
- Recommendation:
  - Decide product taxonomy explicitly: `TCN` and `CTR_GCN` if GCN/hybrid are no
    longer user-facing.
  - Add `CTR_GCN` normalization and frontend mode mapping only when online deploy
    is intended.
  - Remove or hide `HYBRID` from product settings if not part of the report.

### F5 - Monitor Runtime Still Carries Optional `dual_policy` Branches Without Config

- Category: should-clean
- Area: backend monitor
- Evidence:
  - `monitor_policy.load_dual_policy_cfg` looks for
    `applications/configs/ops/dual_policy/...`.
  - That directory does not exist in the current workspace.
  - `safe_state`, `recall_state`, and `policy_alerts` still pass through the
    response/UI contract.
- Risk:
  - The code path is optional but adds cognitive load and frontend state
    ambiguity.
  - It can confuse the report story now that OP-2 locked TCN is the active
    delivery path.
- Recommendation:
  - Either add real dual-policy configs and document their role, or remove the
    dual-policy branch from runtime and keep any safe/recall comparison as an
    offline experiment artifact.

### F6 - `monitor.py` Is Still a Route-Level Orchestrator Hotspot

- Category: should-clean
- Area: backend
- Evidence:
  - `applications/backend/routes/monitor.py`: 730 lines.
  - It still coordinates request prep, live/replay gates, stale handling,
    inference, policy, persistence, notification context, response shaping, and
    WebSocket transport.
- Risk:
  - Replay/live fixes have to be made in route code even though most logic has
    service modules.
  - It is easy for live-only behavior to leak into replay or vice versa.
- Recommendation:
  - Add a `monitor_pipeline_service` facade that owns the full prediction
    decision path.
  - Keep route modules focused on HTTP/WebSocket transport.
  - Preserve current tests before splitting.

### F7 - CTR-GCN Trainer Reuses GCN Trainer Internals

- Category: should-clean
- Area: ML training
- Evidence:
  - `train_ctr_gcn.py` imports `WindowDatasetGCN`, `collect_probs`, and other
    helpers from `train_gcn.py`.
  - `train_gcn.py` is over 1,000 lines and carries old GCN-specific CLI
    compatibility.
- Risk:
  - CTR-GCN changes can accidentally depend on old GCN assumptions.
  - GCN trainer cannot be cleaned without risking CTR-GCN.
- Recommendation:
  - Extract shared window dataset, dataloader, criterion, scheduler, and
    evaluation helpers into `fall_detection.training.common` or similar.
  - Keep `train_gcn.py` and `train_ctr_gcn.py` as thin CLI/model-specific
    wrappers.

### F8 - CTR-GCN Module Docstring Is Out of Date

- Category: should-clean
- Area: ML core
- Evidence:
  - `ml/src/fall_detection/core/ctr_gcn.py` still describes itself as
    "single-stream only in the first pass" and "not a faithful reproduction".
  - The same file now includes `TwoStreamCTRGCN`.
- Risk:
  - Report/code language will conflict with the actual full two-stream work.
- Recommendation:
  - Update the module docstring to match the current project-adapted two-stream
    implementation and explicitly distinguish it from official CTR-GCN.

### F9 - Makefile Is Overgrown and Still Carries Old Dataset/Model Targets

- Category: should-clean
- Area: ops
- Evidence:
  - `Makefile`: about 69 KB.
  - It includes old locked GCN, LE2i paper GCN, MUVIM, URFall, unlabeled,
    hard-negative, audit, report, and full pipeline targets.
  - `bootstrap-dev` still calls `bash ops/scripts/bootstrap_dev.sh`; Windows
    PowerShell scripts now exist separately.
- Risk:
  - The canonical workflow is not obvious.
  - Windows migration work is split from Makefile entrypoints.
- Recommendation:
  - Split into `ops/make/*.mk` or replace with explicit PowerShell/Python task
    scripts.
  - Add a short canonical workflow section for Windows and report reproduction.
  - Move old paper/legacy dataset targets behind an archive makefile if still
    needed.

### F10 - Many `ops/scripts` Files Are Thin Legacy Wrappers

- Category: ignore or should-clean
- Area: ops
- Evidence:
  - Many scripts under 250 bytes only forward into package entrypoints.
- Risk:
  - Low runtime risk, but it creates apparent duplication.
- Recommendation:
  - Keep wrappers if Makefile depends on stable script paths.
  - Otherwise replace with `pyproject.toml` console scripts and update Makefile.

### F11 - Historical Docs Need an Archive Index

- Category: archive
- Area: docs
- Evidence:
  - `docs/reports/notes` contains old Candidate A/D runtime analyses, multiple
    CTR-GCN planning docs, high-score report restructure notes, and new report
    inventory.
- Risk:
  - It is hard to tell which docs represent current claims.
- Recommendation:
  - Create `docs/reports/notes/ARCHIVE_INDEX.md`.
  - Mark current canonical docs: experiment inventory, locked result notes,
    final report plan, and this audit.
  - Move old candidate notes under a dated archive folder only after final
    report references are updated.

### F12 - Test Coverage Is Useful but Uneven

- Category: should-clean
- Area: tests
- Evidence:
  - Backend monitor tests passed after installing dev dependencies.
  - A new frontend unit test covers replay triage parsing.
  - There is no browser/e2e replay test asserting a known custom clip result.
- Risk:
  - Replay regressions can return through frontend timing/UI code.
- Recommendation:
  - Add one deterministic replay API fixture test using stored pose windows.
  - Add one frontend integration test for replay display state if the tooling is
    kept.

## Proposed Cleanup Batches

### Batch A - Git Hygiene, No Runtime Behavior Change

1. Define canonical keep-list for `ops/configs/ops`.
2. Move or ignore generated sweep YAML/JSON.
3. Add `.gitignore` patterns for current generated artifact families.
4. Create an experiment evidence index for report-relevant outputs.

### Batch B - Docs and Artifact Archive

1. Create docs archive index.
2. Mark canonical report notes.
3. Move old candidate/planning notes only after report references are checked.
4. Archive old outputs/checkpoints outside the source workflow.

### Batch C - Runtime Taxonomy Cleanup

1. Decide whether product runtime is `TCN only`, `TCN + CTR_GCN`, or
   `TCN + GCN + CTR_GCN`.
2. Update `code_normalization`, frontend mode mapping, and settings UI to match.
3. Remove/hide `HYBRID` and dual-policy paths if not part of the final product.

### Batch D - Backend Refactor

1. Migrate tests away from `applications.backend.core`.
2. Delete/deprecate `core.py` wrappers.
3. Move route-level monitor orchestration into a pipeline service.

### Batch E - ML Training Refactor

1. Extract GCN/CTR-GCN shared dataset and training helpers.
2. Update CTR-GCN docstring and training contracts.
3. Keep regression tests for checkpoint loading and two-stream input shape.

## Immediate Recommendation

Start with Batch A. It gives the fastest safety improvement: a clean git status
and clear separation between source code, canonical deploy configs, and
generated experiments. Do not remove code paths until the product taxonomy is
decided.
