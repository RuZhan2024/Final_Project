# Full Stack Project Audit

Date: 2026-04-09  
Scope: full-chain audit from ML extraction/preprocessing through backend/runtime/frontend/report evidence.

## Audit Baseline

- Repository root: `/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2`
- Audit date: `2026-04-09`
- Audit mode: mixed-state repository audit, not a clean-worktree freeze audit
- Worktree state at audit time: dirty, with both tracked deletions and untracked generated artifacts
- Python environment: local `.venv`
- Key environment constraint observed during audit:
  - monitor-facing pytest collection can abort during local `torch` initialization
  - plain pytest collection also depends on explicit `PYTHONPATH="$(pwd)/src:$(pwd)"`
- Authoritative evidence intent during audit:
  - report figure root: `artifacts/figures/report`
  - report result root: `artifacts/reports`
  - claim/evidence control layer: `research_ops/CLAIMS.yaml` and `research_ops/EVIDENCE_INDEX.yaml`

This audit therefore evaluates the project as it exists in a **real mixed working state**, not as a clean tagged release snapshot.

## Executive Summary

This repository is still a strong end-to-end fall-detection project, but it is currently in a **mixed operational state** rather than a frozen final state.

The strongest parts are:

- the package-first ML code layout under `src/fall_detection`
- the live monitor runtime path
- the Telegram notification path in realtime mode
- the existence of a report build path and a research control layer

The weakest parts are:

- cross-dataset evidence reproducibility
- repo hygiene and authoritative-state clarity
- some document/runtime drift
- a brittle test contract

The project is therefore:

- **good as a working system**
- **good as a thesis project**
- **not yet clean enough as a final frozen repository**

Current release-readiness verdict: **not freeze-ready**.

## Assessment by Lens

| Lens | Current assessment | Meaning |
| --- | --- | --- |
| Thesis / academic quality | Strong | The project has enough technical substance, implementation depth, and bounded evidence to support a serious thesis submission. |
| Engineering architecture quality | Strong with caveats | Package structure, runtime decomposition, and full-stack integration are meaningfully above normal student-project quality. |
| Freeze / release hygiene quality | Weak to moderate | The repository does not yet present one clean, authoritative, reproducible frozen state. |

## Audit Surface

This audit covered:

- Makefile pipeline entrypoints
- `scripts/*` wrappers
- `src/fall_detection/*` package layout
- backend app assembly and route surface
- frontend page/API integration
- Docker/dev startup paths
- evidence and report figure pack
- testability and repo health

## Code-Audit Coverage Matrix

This upgraded audit now explicitly covers code-path ownership across the full stack:

| Layer | Representative files inspected | Audit intent |
| --- | --- | --- |
| ML data pipeline | `src/fall_detection/data/pipeline.py`, `src/fall_detection/pose/preprocess_pose_npz.py`, `src/fall_detection/data/labels/*`, `src/fall_detection/data/windowing/*` | Verify contract-first extraction/preprocess/labels/splits/windows flow and identify reproducibility risks |
| Evaluation / calibration | `src/fall_detection/evaluation/fit_ops.py`, `src/fall_detection/evaluation/metrics_eval.py`, `src/fall_detection/core/alerting.py` | Verify validation-only OP fitting logic and event-metric contract discipline |
| Backend runtime | `server/routes/monitor.py`, `server/services/monitor_*`, `server/deploy_runtime.py` | Verify live/replay inference plumbing, persistence, and runtime guard logic |
| Notification backend | `server/notifications/manager.py`, `server/notifications/config.py`, `server/routes/notifications.py` | Verify current Telegram-first behavior, fallback handling, and dispatch path cleanliness |
| Frontend monitor | `apps/src/pages/Monitor.js`, `apps/src/pages/monitor/hooks/usePoseMonitor.js`, `apps/src/features/monitor/*` | Verify page-to-hook-to-API layering, media/runtime coupling, and remaining debug surface |
| Settings / app shell | `apps/src/pages/settings/SettingsPage.js`, `apps/src/monitoring/MonitoringContext.js`, `apps/src/lib/*` | Verify settings persistence path and top-level state organization |
| Deployment / startup | `README.md`, `docker-compose.yml`, `server/app.py`, `server/main.py` | Verify boot path, env assumptions, and operator entrypoints |
| Tests | `tests/server/*`, smoke/contract entrypoints | Verify realistic test entrypoints and identify coverage/failure-mode gaps |

## Full-Process Code-Audit Conclusion

This audit now covers the code path from:

- pose extraction
- pose preprocessing and masking
- labels / splits / window generation
- model training and operating-point fitting
- backend runtime inference and event persistence
- notification dispatch
- frontend monitor, settings, events, and dashboard integration

Current code-level judgment:

- **ML core code quality:** strong
- **backend runtime code quality:** strong with one large hotspot
- **frontend application code quality:** good with one large hotspot
- **deployment/test code quality:** moderate because environment assumptions are still too implicit
- **freeze-quality of the codebase surface:** still mixed

This means the project now audits as a **real full-process codebase**, not just as a report/evidence bundle. The remaining weaknesses are mostly:

- code-surface concentration points
- environment-sensitive verification
- stale docs around the current implemented notification path
- unresolved mismatch between some root evidence docs and `research_ops`

## Findings

### 1. Critical: Cross-dataset evidence was previously unstable and is now materially improved, but still needs one final frozen snapshot pass

Relevant files:

- [artifacts/reports/cross_dataset_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_summary.csv)
- [artifacts/reports/cross_dataset_manifest.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_manifest.json)
- [docs/project_targets/CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md)
- [artifacts/reports/cross_dataset_summary_legacy_pre_refreeze_20260409.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_summary_legacy_pre_refreeze_20260409.csv)

What I verified:

- The previously stored summary was stale and did not match rerun behavior.
- A refreeze pass has now rebuilt the active manifest and summary from existing candidate roots.
- The live summary no longer claims both `CAUCAFall->LE2I` directions collapse to `F1=0`.
- The current report and cross-dataset docs have already been updated to the refrozen values.

Impact:

- This is no longer a fully broken evidence chain.
- But it still needs one clean branch+commit snapshot if it is to count as a final frozen layer rather than a repaired working state.

Required action:

- Keep the refrozen manifest/summary/figure set as the only authoritative cross-dataset layer.
- Record it against a clean commit snapshot in the final freeze pass.

Status: `partially_closed`  
Priority: `P0`  
Final release blocker: `yes`

### 2. Medium: Diagnostic-only replay analysis is now separated correctly, but still must not leak back into the main report surface

Relevant files:

- [artifacts/figures/report/diagnostic/le2i_per_clip_outcome_heatmap.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/diagnostic/le2i_per_clip_outcome_heatmap.png)
- [scripts/generate_report_figures.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/generate_report_figures.py)
- [artifacts/reports/diagnostic/online_replay_le2i_perclip_20260402.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/diagnostic/online_replay_le2i_perclip_20260402.json)

What I verified:

- This figure is generated from a pre-fix LE2I degradation-diagnosis artifact.
- It has now been moved under `artifacts/figures/report/diagnostic/`.
- `scripts/generate_report_figures.py` now excludes it by default and only emits it under `--include-diagnostics`.

Impact:

- The storage/layout problem is fixed.
- The remaining risk is only misuse in writing or citation.

Required action:

- Keep it explicitly diagnostic-only in the final report pack.

Status: `closed`  
Priority: `P2`  
Final release blocker: `no`

### 3. High: Repository authoritative state is unclear because the worktree is heavily mixed

Evidence:

- `git status --short`

What I found:

- Many tracked files are currently deleted in-place, especially under:
  - `.make/`
  - `configs/ops/`
  - tutorial material paths
- A first cleanup pass has improved structure in:
  - `artifacts/reports` -> `live / archive / diagnostic`
  - `artifacts/figures/report` -> main root plus `diagnostic/`
  - `configs/ops` -> active root plus `archive/muvim/`
  - `docs/project_targets` -> active root plus `archive/planning/` and `supporting/`
- Even after that, the worktree still contains a large amount of uncommitted and mixed-state material.

Impact:

- Live/archive/generated boundaries are better than before.
- The repository is still not presenting one clean, authoritative frozen state.

Required action:

- Finish the freeze pass against a clean worktree and commit snapshot.
- Use the new freeze inventory and allowlists as the boundary contract.

Status: `partially_closed`  
Priority: `P0`  
Final release blocker: `yes`

### 4. High: Test contract is brittle and environment-sensitive

Relevant files:

- [README.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/README.md)
- [server/main.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/main.py)
- [tests/server/test_monitor_predict.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_monitor_predict.py)

What I verified:

- Running the server-related tests without `PYTHONPATH` fails at import collection.
- Running with the recommended `PYTHONPATH="$(pwd)/src:$(pwd)"` gets further, but monitor-facing test collection aborts when local `torch` initializes.
- A canonical wrapper now exists at [run_canonical_tests.sh](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/run_canonical_tests.sh) with split modes:
  - `torch-free`
  - `contract`
  - `monitor`
  - `all`

Impact:

- The test entrypoint problem is reduced.
- The remaining issue is still local environment sensitivity in the torch-dependent subset.

Required action:

- Keep the scripted split entrypoint as the canonical path.
- Document the `torch/OpenMP` issue if it cannot be fixed.

Status: `partially_closed`  
Priority: `P1`  
Final release blocker: `yes`

### 5. Medium: Notification implementation is now aligned to Telegram-first delivery, but legacy channel scaffolding still exists in docs and package surface

Relevant files:

- [apps/src/pages/settings/SettingsPage.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/settings/SettingsPage.js)
- [server/notifications/telegram_client.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/telegram_client.py)
- [server/README.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/README.md)

What I verified:

- The implementation has clearly moved to Telegram as the real notification channel.
- Settings UI now centers Telegram Chat ID rather than the older multi-channel behavior.
- `server/README.md` and the root `README.md` have now been updated to describe Telegram + AI summary as the current path.
- Legacy Twilio/email client modules and webhook surface still exist as reserved integration scaffolding.

Impact:

- The main doc/runtime mismatch has been reduced.
- The remaining noise is architectural breadth rather than operator confusion.

Required action:

- Keep Telegram documented as the current defended path.
- Treat Twilio/email artifacts as legacy or future-work surface, not as the active release path.

Status: `partially_closed`  
Priority: `P2`  
Final release blocker: `no`

### 6. Medium: Demo preset drift across the root docs and research-control layer has now been resolved

Relevant files:

- [README.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/README.md)
- [docs/project_targets/THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)
- [research_ops/EVIDENCE_INDEX.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/research_ops/EVIDENCE_INDEX.yaml)

What I found:

- `README.md` now recommends `CAUCAFall + TCN + OP-2`.
- `THESIS_EVIDENCE_MAP.md` also now treats `CAUCAFall + TCN + OP-2` as the preferred live demo profile.
- `research_ops/EVIDENCE_INDEX.yaml` has now been updated to the same `CAUCAFall + TCN + OP-2` preset.

Impact:

- The main control docs no longer point to conflicting live demo defaults.

Required action:

- Keep future preset changes synchronized across README, thesis evidence map, and `research_ops`.

Status: `closed`  
Priority: `P3`  
Final release blocker: `no`

### 7. Medium: Root report/evidence control docs are now mostly aligned, with only low-grade duplication remaining between the thesis map and branch-only research controls

Relevant files:

- [docs/project_targets/THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)
- [research_ops/EVIDENCE_INDEX.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/research_ops/EVIDENCE_INDEX.yaml)
- [research_ops/CLAIMS.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/research_ops/CLAIMS.yaml)

What I found:

- `THESIS_EVIDENCE_MAP.md` now points at the newer online replay evidence and the refrozen cross-dataset layer.
- The runtime-preset mismatch in `research_ops` has now been corrected.
- The main remaining issue is duplicated control intent between the root thesis map and the branch-only research layer.

Impact:

- The worst evidence drift has been reduced.
- There is still more than one control surface trying to describe the final story.

Required action:

- Keep `THESIS_EVIDENCE_MAP.md` as the root final-evidence contract.
- Keep `research_ops` explicitly branch-supporting rather than competing as an equal root control layer.

Status: `partially_closed`  
Priority: `P2`  
Final release blocker: `no`

### 8. Low: Figure directory layout is now materially better, with only minor remaining semantic cleanup

Relevant files:

- [artifacts/figures/report](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report)
- [scripts/generate_report_figures.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/generate_report_figures.py)

What I verified:

- Figures are now generated into the correct canonical report directory.
- Diagnostic-only figures now live under `artifacts/figures/report/diagnostic/`.
- The main root still contains both strongly cited figures and secondary/supporting figures, but no longer mixes diagnostic artifacts directly into the same level.

Impact:

- The major ambiguity has been removed.
- Remaining role clarity is mostly a naming and citation-discipline issue.

Required action:

- Optionally add a simple `main vs supporting` naming or manifest layer.

Status: `partially_closed`  
Priority: `P3`  
Final release blocker: `no`

### 9. Low: Report build and freeze tooling are now materially stronger, but the repository still lacks a fully clean frozen snapshot

Relevant files:

- [scripts/build_report.sh](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/build_report.sh)
- [scripts/freeze_manifest.sh](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/freeze_manifest.sh)
- [docs/reports/audit/FREEZE_STATUS_2026-04-09.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/audit/FREEZE_STATUS_2026-04-09.md)

What I found:

- `build_report.sh` now validates required tools and input existence.
- It supports `--pdf-only` and `--docx-only`.
- It derives output names more cleanly and allows metadata overrides.
- `freeze_manifest.sh` now provides a machine-checkable freeze-core boundary and reports current dirty status.

Impact:

- Tooling is no longer the main weakness.
- The remaining issue is that the repository content itself is still not fully frozen.

Required action:

- Use `freeze_manifest.sh` as the final pre-freeze gate.
- Clear or intentionally explain the remaining freeze-core dirty paths.

Status: `partially_closed`  
Priority: `P3`  
Final release blocker: `no`

### 10. Low: Frontend monitor architecture remains good, and the most obvious debug API residue has now been removed

Relevant files:

- [apps/src/features/monitor/api.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/features/monitor/api.js)
- [apps/src/pages/Monitor.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/Monitor.js)

What I found:

- The monitor page is reasonably well separated into:
  - page container
  - feature API layer
  - per-card components
  - hooks
- The temporary `triggerTestNotification()` residue has now been removed from the feature API and hook flow.
- The remaining hotspot is complexity concentration in the monitor hook rather than leftover debug entrypoints.

Impact:

- Low runtime risk.
- Mild API-surface clutter.

Required action:

- Keep reducing responsibility concentration in the monitor hook during future changes.

Status: `closed`  
Priority: `P3`  
Final release blocker: `no`

### 11. Medium: Makefile pipeline is powerful but dense, and there is still too much live historical experiment surface

Relevant files:

- [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile)
- [configs/ops](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops)
- [outputs](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/outputs)

What I verified:

- The Makefile provides an impressive end-to-end DAG:
  - extract
  - preprocess
  - labels
  - splits
  - windows
  - train
  - fit ops
  - eval
  - release/audit
- The package-first wrapper approach is clean.
- But the live repository still exposes a large historical experiment surface in `configs/ops` and `outputs`, much of it not clearly demoted to archive status.

Impact:

- The pipeline itself is sound.
- The visible experiment surface is noisier than it should be for a final hand-in.

Required action:

- Move superseded experiment families into explicit archive paths or document them as historical.

Status: `open`  
Priority: `P1`  
Final release blocker: `no`

### 12. Medium: The ML pipeline code is architecturally strong, but still split between clean package contracts and legacy artifact-heavy orchestration

Relevant files:

- [src/fall_detection/data/pipeline.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/data/pipeline.py)
- [src/fall_detection/pose/preprocess_pose_npz.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/pose/preprocess_pose_npz.py)
- [src/fall_detection/evaluation/fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py)
- [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile)

What I verified:

- `src/fall_detection/data/pipeline.py` is contract-first and clearly intended to replace the legacy step-by-step Makefile workflow.
- `preprocess_pose_npz.py` is explicit about missing-value semantics, body-centric normalization, and mask generation, which is good for traceability.
- `fit_ops.py` keeps the core calibration contract bounded to temperature scaling and validation-only sweep logic, which is scientifically the right structure.
- The risk is not bad code architecture; it is that the repo still exposes many historical ops/config/result families that make it harder to tell which pipeline state is final.

Impact:

- The ML pipeline itself audits well.
- The surrounding experiment surface still weakens final reproducibility posture.

Required action:

- Keep the package-first ML modules as the authoritative implementation path.
- Continue reducing legacy experiment noise around them.

Status: `open`  
Priority: `P1`  
Final release blocker: `no`

### 13. Medium: Backend runtime code is well-layered, but `monitor.py` remains a high-complexity hotspot

Relevant files:

- [server/routes/monitor.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/monitor.py)
- [server/services/monitor_runtime_service.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/services/monitor_runtime_service.py)
- [server/services/monitor_response_service.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/services/monitor_response_service.py)

What I verified:

- Monitor runtime responsibilities have been pushed into services more than before.
- The route still carries a large amount of low-level policy resolution, live-guard behavior, and delivery gate logic.
- This is maintainable for the current project scale, but it is still the densest backend file and therefore the most likely place for future drift.

Impact:

- No immediate correctness blocker by itself.
- High future-maintenance risk if more runtime policy is added without further decomposition.

Required action:

- Treat `server/routes/monitor.py` as a refactor hotspot.
- Any future runtime feature should prefer adding service-level helpers over extending the route directly.

Status: `open`  
Priority: `P2`  
Final release blocker: `no`

### 14. Medium: Notification code has now converged on Telegram operationally, but the module surface still exposes legacy channels

Relevant files:

- [server/notifications/manager.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/manager.py)
- [server/notifications/email_client.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/email_client.py)
- [server/notifications/twilio_client.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/twilio_client.py)
- [server/notifications/telegram_client.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/telegram_client.py)

What I verified:

- `NotificationManager` now dispatches the real path through Telegram and AI summary generation.
- The notification package still contains email/Twilio clients and older multi-channel abstractions.
- That is acceptable as future-work scaffolding, but it means the package surface is broader than the currently defended implementation.

Impact:

- Runtime behavior is clean enough.
- Architectural story is slightly noisier than the implemented product story.

Required action:

- Document Telegram as the implemented path.
- Mark email/SMS/phone as reserved integrations or future work in code comments/docs where needed.

Status: `open`  
Priority: `P2`  
Final release blocker: `no`

### 15. Medium: Frontend monitor code is strong for a student project, but `usePoseMonitor` is still a concentration point

Relevant files:

- [apps/src/pages/monitor/hooks/usePoseMonitor.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/monitor/hooks/usePoseMonitor.js)
- [apps/src/features/monitor/api.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/features/monitor/api.js)
- [apps/src/features/monitor/media.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/features/monitor/media.js)
- [apps/src/features/monitor/prediction.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/features/monitor/prediction.js)

What I verified:

- The monitor feature is separated into API/media/prediction/windowing helpers, which is the right direction.
- `usePoseMonitor.js` still owns a very large amount of behavior:
  - MediaPipe lifecycle
  - camera/replay control
  - buffer management
  - websocket/http prediction transport
  - clip upload
  - timeline UI state
- That is workable, but it is the frontend equivalent of the backend `monitor.py` hotspot.

Impact:

- Strong functionality, but high cognitive load for future edits.
- Regressions are more likely to concentrate here.

Required action:

- Keep extracting pure helpers from the hook when changing monitor behavior.
- Avoid adding new cross-cutting responsibilities directly into `usePoseMonitor`.

Status: `open`  
Priority: `P2`  
Final release blocker: `no`

### 16. Medium: Test coverage exists across the stack, but code-audit confidence is still limited by runtime-heavy areas

Relevant files:

- [tests/server/test_monitor_predict.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_monitor_predict.py)
- [tests/server/test_runtime_core.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_runtime_core.py)
- [tests/server/test_notification_manager.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/server/test_notification_manager.py)

What I verified:

- There is meaningful server/runtime test coverage, not just tiny unit tests.
- The notification path has direct tests, which is good.
- The most valuable runtime tests are also the ones most exposed to local environment fragility.

Impact:

- The repo is better tested than many comparable projects.
- The audit still cannot claim one stable, universally reproducible full-stack test command.

Required action:

- Maintain split test entrypoints:
  - torch-free / contract / API-level
  - torch-dependent runtime subsets
- Record canonical commands in README or release docs.

Status: `open`  
Priority: `P1`  
Final release blocker: `yes`

## ML Pipeline Audit

### What audits well

- The main CLI entrypoints are thin wrappers into `src/fall_detection`, which is the correct architecture.
- The Makefile pipeline ordering is explicit and non-trivial.
- Dataset-specific split guards exist, especially for CAUCAFall subject-split enforcement.
- The project has clearly moved away from monolithic top-level scripts toward package implementations.
- `data/pipeline.py` is a real contract-first core, not just a collection of ad hoc script glue.
- `preprocess_pose_npz.py` explicitly encodes missing-value and normalization semantics, which is exactly what a pose-based pipeline needs for auditability.
- `fit_ops.py` keeps calibration bounded to a validation-only deploy-policy layer, which is methodologically sound.

### Main risks

- Cross-dataset result freezing is not complete.
- Historical ops/output surfaces are too live.
- Reproducibility depends on specific local paths and workspace state more than it should.
- The repo still presents more historical experiment state than a final frozen ML pipeline should expose.

### Full-flow verdict

| Stage | Code-audit verdict | Main note |
| --- | --- | --- |
| Extract | Good | Dedicated pose extract modules exist and fit the package structure. |
| Preprocess | Good | Normalization and missing semantics are explicit and defensible. |
| Labels / splits | Good | Dataset-specific logic exists with meaningful safeguards. |
| Windowing | Good | Implemented as package modules, not purely shell glue. |
| Train | Good with caveats | Strong package layout, but final run selection still depends on frozen artifact discipline. |
| Fit ops | Strong | One of the best-audited parts of the ML path. |
| Eval | Good with caveats | Scientifically fine, but final evidence freezing was the weak point. |

## Backend Audit

### What audits well

- `server/main.py` is a clean app assembly layer.
- Route grouping is reasonable:
  - health
  - specs
  - operating points
  - settings
  - events
  - dashboard
  - monitor
  - notifications
  - caregivers
- `server/routes/monitor.py` has been decomposed into services enough to remain auditable.
- Notification dispatch is now clear enough to follow end to end: event -> classifier -> store -> Telegram dispatch.
- Repository/service separation exists and is meaningful rather than purely cosmetic.

### Main risks

- `server/README.md` still documents the older notification worldview.
- Backend testability remains environment-sensitive.
- `monitor.py` is still a complexity hotspot.

### Full-flow verdict

| Area | Code-audit verdict | Main note |
| --- | --- | --- |
| App assembly | Strong | `server/app.py` and `server/main.py` are clean enough. |
| Routes | Good | Route grouping is sensible and mostly coherent. |
| Services | Good | Enough logic has been moved out of routes to help maintainability. |
| Notifications | Good with drift | Implemented path is solid; docs/module surface still lag. |
| Persistence | Good with caveats | Multiple DB/fallback paths exist, which helps demos but adds complexity. |

## Frontend Audit

### What audits well

- The SPA structure is sensible:
  - Dashboard
  - Monitor
  - Events
  - Settings
- The monitor page is assembled from hooks and cards rather than one huge page file.
- `MonitoringContext` gives a coherent global control surface.
- Feature-level helpers under `apps/src/features/monitor/` are a real positive; they prevent the page layer from being the only abstraction.

### Main risks

- Some debug/test helper surface remains in feature APIs.
- Final UI behavior and supporting docs are not yet fully synchronized.
- `usePoseMonitor` remains a high-responsibility hook.

### Full-flow verdict

| Area | Code-audit verdict | Main note |
| --- | --- | --- |
| App shell / routing | Good | Clear page structure for a project of this size. |
| Dashboard / Events / Settings | Good | Separation is readable and practical. |
| Monitor UI | Good with hotspot risk | Strong feature layering, but one large hook still carries too much. |
| API client layer | Good with minor clutter | Useful abstraction, but a little legacy surface remains. |

## Deployment / Ops Audit

### What audits well

- Docker stack exists and is practical.
- Backend receives the right mounted workspace and event clip volume.
- Frontend dev server and backend service wiring are straightforward.
- The project supports both lightweight demo mode and fuller persistent mode, which is genuinely useful for review and delivery.

### Main risks

- Environment-variable expectations are spread across README, server README, and `.env` usage rather than centralized.
- Notification docs lag behind the actual Telegram-first implementation.
- The repository still contains local runtime artifacts like SQLite DB files and cache-heavy generated surfaces that blur the deploy boundary.

### Full-flow verdict

| Area | Code-audit verdict | Main note |
| --- | --- | --- |
| Local dev startup | Good | Works conceptually and is well documented at a high level. |
| Docker compose | Good | Practical and aligned with project goals. |
| Env management | Moderate | Functional, but documentation and centralization still need work. |
| Freeze hygiene | Weak | Runtime and generated state are still too mixed. |

## Test Audit

### What audits well

- The repo contains real contract/runtime tests rather than only trivial unit tests.
- Notification and runtime core paths have targeted tests.
- Frontend and backend are both represented in the verification story.

### Main risks

- One canonical “run this and trust the result” command is still missing.
- Torch/OpenMP behavior constrains reproducible local verification.
- Some runtime-heavy paths remain harder to test than they should be.

### Full-flow verdict

| Area | Code-audit verdict | Main note |
| --- | --- | --- |
| Smoke tests | Moderate | Useful, but not yet one-click authoritative. |
| Server contract tests | Good with env caveats | Valuable but environment-sensitive. |
| Runtime monitor tests | Moderate | Important, but fragile under local torch init. |
| Notification tests | Good | Stronger than average for this kind of project. |

## Overall Verdict

Current state:

- **System completeness:** strong
- **Engineering structure:** strong
- **Evidence discipline:** mixed
- **Repository cleanliness:** weak
- **Final reproducibility confidence:** moderate

Bottom line:

This is already a serious, advanced, end-to-end project.  
The remaining problems are mostly not “missing core functionality.”  
They are:

- frozen evidence drift
- documentation/runtime inconsistency
- noisy repository state
- brittle final audit posture

Full-process code-audit verdict:

- **ML pipeline code:** strong and defensible
- **backend code:** strong with one major hotspot (`server/routes/monitor.py`)
- **frontend code:** good with one major hotspot (`usePoseMonitor.js`)
- **notification implementation:** working and demo-valid, but docs/module surface still lag the Telegram-first reality
- **test/deploy layer:** functional but not yet freeze-grade

## Security, Privacy, and Dependency Governance

### Security and secrets

Current judgment: `moderate risk`

What is good:

- active secrets are expected to live in private env files rather than tracked source
- the current implemented notification channel does not require exposing public inbound credentials to end users

What still needs explicit closure:

- audit and documentation should explicitly state which env files must never be committed
- generated report/build outputs should be checked to ensure they do not embed private local paths or tokens
- notification documentation should define the minimum private env surface for Telegram and AI integration

### Privacy

Current judgment: `moderate but acceptable for thesis scope`

What is good:

- the system is pose-based rather than raw-RGB-model based
- the event pipeline already distinguishes anonymized or bounded artifact handling in several places

What still needs explicit closure:

- document whether realtime demo recordings are retained, and if so where
- clarify event-clip retention expectations in the final user-facing docs
- ensure caregiver-facing notification wording avoids unsupported medical claims

### Dependency governance

Current judgment: `moderate risk`

What is good:

- Python code is package-structured and does not rely only on ad hoc notebooks
- frontend and backend dependency surfaces are separable

What still needs explicit closure:

- document the canonical install/test path more explicitly
- identify any lockfile or pinned-environment assumptions required for final reproduction
- call out the local `torch/OpenMP` environment sensitivity as an environment-level dependency risk

## Runtime Performance and Failure-Mode Review

Current judgment: `partially reviewed, not fully closed`

### What is already evident

- the system can run end to end in local dev and docker-backed modes
- realtime monitor + Telegram notification works
- frontend can survive some backend/settings failures by surfacing warnings rather than crashing immediately

### What is not yet fully closed in this audit

- single canonical latency budget for the current review preset
- memory/resource profile under live preview plus monitor loop
- explicit degraded-mode matrix for:
  - backend unavailable
  - DB unavailable
  - torch/model import failure
  - missing camera permission
  - missing Telegram config

### Required action

- add one short runtime-failure-mode matrix to the final docs
- keep one canonical latency artifact for the current review preset

## Freeze Criteria

The repository should only be treated as freeze-ready when all of the following are true:

1. Cross-dataset manifest is rebuilt from checkpoints that actually exist, and the summary/figure/report text are regenerated from that manifest.
2. The main report figure pack contains only figures with explicit status:
   - main
   - supporting
   - diagnostic
3. `README.md`, `THESIS_EVIDENCE_MAP.md`, `research_ops/CLAIMS.yaml`, and `research_ops/EVIDENCE_INDEX.yaml` agree on:
   - review preset
   - active evidence artifacts
   - bounded claims
4. The worktree is no longer in a mixed ambiguous state:
   - live files committed
   - generated files either tracked intentionally or moved out of the live surface
   - superseded historical surfaces archived or clearly marked
5. One canonical test command is documented, with torch-free and torch-dependent subsets clearly separated.
6. Notification docs consistently describe Telegram as the current implemented channel and email/SMS/phone as future work.
7. A final audit rerun is recorded against a specific branch + commit snapshot.

## Recommended Next Actions

1. Freeze and clean the repository surface.
2. Repair and regenerate the cross-dataset evidence chain.
3. Reconcile README, thesis evidence map, and research ops ledger.
4. Separate final figures from supporting/diagnostic figures.
5. Normalize notification docs around Telegram-first current behavior.
6. Add one canonical scripted test entrypoint and document the torch-dependent caveat.

## Canonical Commands

These are the most important commands that should remain valid and documented during freeze work.

### Local settings-aware test path

```bash
./scripts/run_canonical_tests.sh torch-free
```

Note:
- `./scripts/run_canonical_tests.sh contract` and `./scripts/run_canonical_tests.sh monitor` remain environment-sensitive because they import the server app and can still hit local `torch` initialization issues

### Report build

```bash
./scripts/build_report.sh
```

### Report figure generation

```bash
python3 scripts/generate_report_figures.py
```

### Cross-dataset rerun pattern

```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/eval_metrics.py \
  --win_dir data/processed/le2i/windows_eval_W48_S12/test \
  --ckpt outputs/caucafall_gcn_W48S12/best.pt \
  --ops_yaml configs/ops/gcn_caucafall.yaml \
  --out_json outputs/metrics/cross_gcn_caucafall_to_le2i_rerun_exact_20260409.json \
  --fps_default 23 \
  --thr_min 0.001 --thr_max 0.95 --thr_step 0.01
```

### Docker startup

```bash
docker compose up
```
