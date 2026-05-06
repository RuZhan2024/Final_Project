# Git Diff Staging Plan - 2026-05-07

This note records the recommended commit boundary for the post-CTR-GCN cleanup state.

Current branch: `feat/ctr-gcn-upgrade`

Current verification state:

- Backend QA: `215 passed`
- Frontend typecheck: passed
- Frontend unit tests: `2 passed`
- Frontend production build: passed
- Windows full-stack smoke on `127.0.0.1:8000` and `127.0.0.1:3000`: passed
- Active deploy specs observed from a fresh backend: `["caucafall_tcn"]`
- Active replay clips observed from backend: `24`

## Current Diff Shape

- Tracked changes: 219 files
- Tracked diff size: about 2.1k insertions and 83k deletions
- Most deletions are historical ops/sweep configs that were removed from active runtime folders after archiving.
- Several report, label, split, Windows script, graph-training, and test files are currently untracked and need an explicit keep/drop decision.

## Recommended Commit Boundary

### Commit 1: Windows Bootstrap And Developer Runtime

Purpose: make the copied project usable on Windows without relying on macOS `make`.

Include:

- `.gitignore`
- `ops/scripts/bootstrap_dev.ps1`
- `ops/scripts/start_fullstack.ps1`
- `ops/scripts/stop_fullstack.ps1`

Rationale:

- This is an environment/operations change.
- It can be reviewed independently from ML/runtime behavior.
- It documents the Windows path that fixed Git/bootstrap/start/stop issues.
- `Makefile` is intentionally kept out of this commit because its current diff also contains CTR-GCN target changes and old-GCN cleanup.
- `applications/backend/README.md` is intentionally kept for the runtime-contract commit.

### Commit 2: Runtime Cleanup And Single Promoted Deploy Path

Purpose: remove stale online runtime branching and lock the deploy app to the promoted TCN runtime.

Include:

- `applications/backend/**`
- `applications/frontend/src/**`
- `applications/backend/README.md`
- `ops/deploy_assets/manifest.json`
- `ops/deploy_assets/checkpoints/caucafall_tcn_best.pt`
- Deleted deploy checkpoints under `ops/deploy_assets/checkpoints/`
- Deleted `applications/backend/core.py`
- Deleted `ml/src/fall_detection/deploy/run_modes.py`
- `ops/scripts/replay_online_windows.py`

Rationale:

- This is the user-facing runtime contract.
- It removes GCN/HYBRID online compatibility paths instead of leaving old logic coupled to the current app.
- It keeps online behavior aligned with the manifest-driven deploy assets.

### Commit 3: CTR-GCN Training, Data, Labels, Splits, And Fit Utilities

Purpose: keep the offline experiment pipeline and CTR-GCN upgrade work together.

Include:

- `Makefile`
- `ml/src/fall_detection/core/ctr_gcn.py`
- `ml/src/fall_detection/core/features.py`
- `ml/src/fall_detection/core/models.py`
- `ml/src/fall_detection/training/train_ctr_gcn.py`
- `ml/src/fall_detection/training/graph_training_utils.py`
- Deleted `ml/src/fall_detection/training/train_gcn.py`
- Deleted `ops/scripts/train_gcn.py`
- `ml/src/fall_detection/data/**`
- `ml/src/fall_detection/evaluation/**`
- `ml/src/fall_detection/pose/**`
- `ops/configs/labels/*_c2*.json`
- `ops/configs/splits/*_c2*`
- updated split summary files
- `ops/configs/delivery/tcn_mix25_s42_op2_motion018.yaml`
- archived/deleted old delivery config if we choose to track the archive move

Rationale:

- CTR-GCN remains an offline training/evaluation capability.
- The simple GCN trainer is retired.
- C2 labels, splits, and motion/fit utilities are needed for report reproducibility.

### Commit 4: QA Realignment For Cleaned Contracts

Purpose: make tests describe the current system rather than removed compatibility paths.

Include:

- `qa/tests/**`
- `applications/frontend/src/features/monitor/prediction.test.ts`

Rationale:

- Many tests were rewritten from stale private helpers/routes to current service/repository contracts.
- Keeping this separate makes runtime implementation diffs easier to review.

### Commit 5: Report Notes, Evidence Index, And Paper Materials

Purpose: preserve report/thesis traceability without mixing it into executable code.

Include:

- `docs/reports/notes/*`
- `docs/reports/drafts/FULL_PROJECT_REPORT_HIGH_SCORE_RESTRUCTURED.md`
- `artifacts/figures/report/runtime_evidence_panel.png`
- `ops/configs/ops/archive/historical_profiles_20260506/**`
- `ops/configs/ops/archive/legacy_root_configs_20260427/**`
- `ops/configs/delivery/archive/legacy_20260427/**`

Rationale:

- These files explain experiment history and cleanup decisions.
- They are useful for report defense, but they are not runtime dependencies.
- If repository size becomes a concern, track only the index notes and keep bulky archives local-only.

## Local-Only Recommendation

Keep these local and ignored unless explicitly needed in the report repository:

- `artifacts/archive/**`
- generated sweep/eval/training logs
- legacy deploy checkpoint archives
- transient dev state under `.make/**`
- virtual environments and dependency caches

## Open Decisions Before Staging

1. Track full ops config archives, or track only archive indexes plus active configs?
2. Track the full high-score report draft now, or keep it local until the writing pass stabilizes?
3. Track the runtime evidence PNG now, or regenerate it later from scripted evidence?
4. Commit everything as one final cleanup PR, or keep the five logical commits above for reviewability?

## Suggested Next Action

Use the five-commit boundary above. Stage Commit 1 first, inspect `git diff --cached --stat`, then run a quick smoke only if the staged set unexpectedly changes runtime files. After Commit 1, proceed through runtime, offline ML, QA, and report/evidence groups.
