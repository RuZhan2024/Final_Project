# Code Comment Tasks

Date: 2026-04-09

## Purpose

Plan a controlled pass to add targeted code comments to the project without turning the codebase into noisy or redundant prose.

This task is not to "comment everything".
It is to add comments only where they materially improve:
- handoff readability
- supervisor reviewability
- maintenance of non-obvious logic
- safety around workflow-sensitive code paths

## Scope

Commenting should focus on:
- non-obvious control flow
- hidden contracts
- workflow-sensitive defaults
- dataset-specific assumptions
- alert-policy and persistence semantics
- places where future edits are likely to break behavior

Do **not** add comments for:
- obvious assignments
- trivial JSX structure
- simple wrapper functions
- straightforward data containers
- anything already made clear by good naming

## Comment Policy

Allowed comment types:
- brief intent comments above a complex block
- contract comments where downstream code depends on exact semantics
- caution comments where a default or branch is easy to misuse
- short rationale comments for dataset- or deployment-specific decisions

Disallowed comment types:
- line-by-line narration
- comments that repeat the code literally
- historical diary comments
- vague comments like "important logic here"
- comments that describe behavior no longer guaranteed by tests/docs
- branding-style terms such as `AI`, `smart`, or `intelligent` when a neutral engineering term is available

## Target Areas

Priority order:

1. ML pipeline
- `src/fall_detection/data/pipeline.py`
- `src/fall_detection/data/datamodule.py`
- `src/fall_detection/deploy/common.py`
- `src/fall_detection/evaluation/metrics_eval.py`
- `src/fall_detection/evaluation/score_unlabeled_alert_rate.py`
- `src/fall_detection/training/train_tcn.py`
- `src/fall_detection/training/train_gcn.py`

2. Server runtime
- `server/services/monitor_runtime_service.py`
- `server/services/monitor_context_service.py`
- `server/repositories/monitor_repository.py`
- `server/repositories/dashboard_repository.py`
- `server/routes/events.py`
- `server/routes/notifications.py`
- `server/notifications/manager.py`
- `server/notifications/sqlite_store.py`

3. Frontend runtime
- `apps/src/pages/monitor/hooks/usePoseMonitor.js`
- `apps/src/monitoring/MonitoringContext.js`
- `apps/src/pages/Monitor.js`
- `apps/src/pages/settings/SettingsPage.js`
- `apps/src/features/monitor/api.js`

4. Tooling / scripts
- `scripts/run_canonical_tests.sh`
- `scripts/build_report.sh`
- `scripts/freeze_manifest.sh`

## Batch Plan

### Batch 1: ML comments

Goal:
- annotate dataset contracts, window metadata semantics, deploy/eval parity assumptions, and trainer provenance behavior

Exit criteria:
- comments added only where the logic is non-obvious
- no redundant narration introduced
- files still read cleanly after the pass

### Batch 2: Server comments

Goal:
- annotate persistence contracts, notification truth source, replay/realtime differences, and event status assumptions

Exit criteria:
- comments clarify runtime contracts without bloating route files
- storage/notification source-of-truth points are explicit

### Batch 3: Frontend comments

Goal:
- annotate state synchronization rules, fallback assumptions, and replay persistence behavior

Exit criteria:
- comments help explain cross-component state behavior
- no JSX clutter comments added

### Batch 4: Script comments

Goal:
- annotate canonical test modes, freeze manifest purpose, and report-build argument behavior

Exit criteria:
- script entrypoints are understandable to a new reviewer
- shell comments remain short and functional

## Review Standard

Each file touched in the comment pass should satisfy:
- comments explain "why" or "contract", not obvious "what"
- comments are shorter than the code block they explain
- comments do not contradict current tests, audit docs, or README behavior
- comments improve reviewability for supervisor handoff
- comments use neutral implementation language rather than promotional wording

## Deliverables

- code comment pass applied in batches
- updated commit(s) with limited scope
- optional follow-up note listing which files received meaningful comments

## Non-Goals

- no refactor just to make comments fit
- no behavior changes
- no renaming pass
- no doc rewrite under the banner of comments

## Current Status

Status: `not_started`

Next step:
- start with `Batch 1: ML comments`
