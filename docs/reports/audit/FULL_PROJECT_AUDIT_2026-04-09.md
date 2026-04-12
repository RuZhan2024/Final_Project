# Full Project Audit

Date: 2026-04-09  
Scope: end-to-end repository audit across runtime code, evaluation artifacts, report evidence, testability, and repository hygiene.

## Executive Summary

The project remains strong as a working system and thesis-grade engineering artifact, but it is not currently in a clean final-submission state.

The highest-risk issues are:

1. The cross-dataset evidence layer is not fully trustworthy in its current stored form.
2. Part of the figure pack was generated from stale or transitional artifacts.
3. The repository worktree is heavily polluted with uncommitted additions/deletions, making it hard to identify the authoritative state.
4. The server-side test path is fragile: without `PYTHONPATH` the suite fails at import time, and with the recommended import path the monitor-facing subset aborts on local `torch` initialization.

The core runtime path itself is still in relatively good shape:

- the monitor stack exists and runs
- Telegram notifications work in realtime
- deploy/runtime code is structured enough to audit
- the report build path exists and produces output

So the project is **functionally viable**, but **evidence closure and repository hygiene are not yet at final-audit quality**.

## Findings

### 1. Critical: Cross-dataset summary is not fully reproducible from the current workspace

Evidence:

- [artifacts/reports/cross_dataset_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_summary.csv)
- [artifacts/reports/cross_dataset_manifest.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_manifest.json)
- [docs/project_targets/CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md)

What I verified:

- The stored summary currently reports `CAUCAFall -> LE2I` cross-domain `F1 = 0.0` for both TCN and GCN.
- I reran the exact `GCN CAUCAFall -> LE2I` evaluation with the stored checkpoint `outputs/caucafall_gcn_W48S12/best.pt` and current [configs/ops/gcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/gcn_caucafall.yaml).
- New output: [cross_gcn_caucafall_to_le2i_rerun_exact_20260409.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/outputs/metrics/cross_gcn_caucafall_to_le2i_rerun_exact_20260409.json)
- That rerun no longer matches the stored summary. It produces non-zero event recall and multiple alerts.
- For TCN, the old summary points to `outputs/caucafall_tcn_W48S12/best.pt`, but that checkpoint path does not exist in the current workspace. So that summary row is not exactly reproducible now.

Impact:

- The current cross-dataset summary cannot be treated as fully authoritative.
- The cross-dataset figure pack and report section built on top of this summary are therefore partially unstable.

Required action:

- Freeze a reproducible cross-dataset manifest with checkpoints that actually exist.
- Regenerate [cross_dataset_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_summary.csv) from that manifest.
- Rebuild the cross-dataset figures and revise the report text to match the regenerated results.

### 2. High: One report figure still reflects a diagnostic-phase artifact, not the final state

Evidence:

- [artifacts/figures/report/diagnostic/le2i_per_clip_outcome_heatmap.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/diagnostic/le2i_per_clip_outcome_heatmap.png)
- [scripts/generate_report_figures.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/generate_report_figures.py)
- [artifacts/reports/diagnostic/online_replay_le2i_perclip_20260402.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/diagnostic/online_replay_le2i_perclip_20260402.json)

What I verified:

- The LE2I per-clip heatmap is generated from `artifacts/reports/diagnostic/online_replay_le2i_perclip_20260402.json`.
- That file corresponds to the earlier degradation diagnosis stage, before the later replay-FPS and final runtime fixes.
- Later files such as:
  - [online_replay_le2i_after_replayfpsfix_20260402.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/archive/replay_runtime_iterations_20260402/online_replay_le2i_after_replayfpsfix_20260402.json)
  - [online_replay_le2i_finalfix_20260402.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/archive/replay_runtime_iterations_20260402/online_replay_le2i_finalfix_20260402.json)
  are aggregate summaries, not per-clip artifacts.

Impact:

- This figure is valid only as a **diagnostic figure**.
- It should not be cited as a final runtime-result figure.

Required action:

- Either remove it from any final figure pack, or explicitly label it as pre-fix diagnostic evidence.

### 3. High: The repository is in a heavily mixed state with large-scale tracked deletions and many untracked artifacts

Evidence:

- `git status --short`

What I found:

- Large numbers of tracked files are deleted in-place across:
  - `.make/`
  - `configs/ops/`
  - tutorial/archive docs
- Many new artifacts are untracked across:
  - `artifacts/reports/`
  - `artifacts/figures/report/`
  - `artifacts/replay_eval/`
  - `artifacts/report_build/`
  - additional scripts and planning docs

Impact:

- It is difficult to identify the true authoritative project state.
- Audit conclusions can drift if later work accidentally depends on mixed staged/unstaged files.
- Merge safety and evidence reproducibility both degrade under this condition.

Required action:

- Split the repo into three explicit groups:
  - authoritative tracked code/config/docs
  - tracked archival evidence
  - generated or transient outputs
- Then either commit or archive the current intended keep-set, and move the rest out of the live surface.

### 4. High: Testability is fragile and environment-dependent

Evidence:

- Plain `pytest` collection fails without `PYTHONPATH`.
- With README-style `PYTHONPATH`, monitor-related test collection aborts on local `torch` initialization.
- [README.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/README.md)

What I verified:

- Running the server-related test subset without `PYTHONPATH` produces import errors against `server.*`.
- Running with:
  - `source .venv/bin/activate`
  - `PYTHONPATH="$(pwd)/src:$(pwd)"`
  gets past import resolution, but then the monitor-facing test path aborts during `torch` import from the local environment.

Impact:

- The test contract is real but brittle.
- A reviewer or fresh machine can easily conclude that the suite is broken even if the issue is partly environment-related.

Required action:

- Add a single canonical test command to the root README and to a script wrapper.
- Separate torch-free fast tests from torch-dependent runtime tests.
- Document the local `torch/OpenMP` caveat explicitly if it cannot be resolved before final closure.

### 5. Medium: The report evidence map is partially stale

Evidence:

- [docs/project_targets/THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)
- [research_ops/EVIDENCE_INDEX.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/research_ops/EVIDENCE_INDEX.yaml)

What I found:

- The thesis evidence map still references older replay artifacts for `E6`:
  - `archive/replay_matrix_legacy_20260402/mc_replay_matrix_20260401.csv`
  - `archive/replay_matrix_legacy_20260402/mc_replay_matrix_20260401.json`
- The research-ops evidence index has already been updated to the newer online replay artifacts:
  - `online_mc_replay_matrix_20260402.csv`
  - `online_mc_replay_matrix_20260402.json`

Impact:

- Two evidence-control layers are currently disagreeing.
- This is exactly the kind of drift that can leak into the final report.

Required action:

- Reconcile `THESIS_EVIDENCE_MAP.md` with `research_ops/EVIDENCE_INDEX.yaml`.
- Decide which file is the source of truth and downgrade the other to a generated or mirrored view.

### 6. Medium: Report build pipeline is functional but still low-rigor

Evidence:

- [scripts/build_report.sh](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/build_report.sh)

What I found:

- The script builds successfully and is useful.
- But it still hard-codes:
  - title
  - author
  - date
  - output basename
- It does not validate required tools or input file existence before invoking `pandoc`.

Impact:

- Good enough for internal iteration.
- Not ideal as a final reproducible build contract.

Required action:

- Parameterize date/output naming.
- Add checks for `pandoc` and `xelatex`.
- Move metadata to YAML or CLI flags derived from config.

### 7. Medium: Root README and live deploy defaults are not fully aligned

Evidence:

- [README.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/README.md)
- [research_ops/CLAIMS.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/research_ops/CLAIMS.yaml)
- [research_ops/EVIDENCE_INDEX.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/research_ops/EVIDENCE_INDEX.yaml)

What I found:

- The README still presents `CAUCAFall + TCN + OP2` as the recommended review path.
- The evidence layer separately records `LE2I TCN OP-2` as the current online review runtime profile.

Impact:

- Reviewers can be told two different “default demos” depending on which file they read.

Required action:

- Pick one canonical review preset and update both README and evidence docs.

### 8. Medium: Generated report figures now live in the correct directory, but not all of them are part of the active report pack

Evidence:

- [artifacts/figures/report](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report)
- [scripts/generate_report_figures.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/generate_report_figures.py)

What I found:

- Figure output has been standardized to `artifacts/figures/report`, which is correct.
- But the directory now contains a mix of:
  - actively cited report figures
  - auxiliary/generated figures
  - at least one diagnostic-only figure

Impact:

- The directory is structurally correct but semantically mixed.

Required action:

- Tag figures as:
  - `main_report`
  - `supporting`
  - `diagnostic_only`
- Or split generated supporting figures into a subfolder while keeping final cited figures at the top level.

### 9. Low: Notification implementation is currently stronger in code than in documentation

Evidence:

- [server/notifications/telegram_client.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/telegram_client.py)
- [server/notifications/manager.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/manager.py)
- [apps/src/pages/settings/SettingsPage.js](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/apps/src/pages/settings/SettingsPage.js)

What I verified:

- Telegram-based notification now exists and works in realtime.
- The runtime implementation and UI were updated, but the broader documentation pack has not yet been uniformly rewritten around “Telegram now, other channels future work.”

Impact:

- This is not a runtime blocker.
- It is a documentation coherence issue.

Required action:

- Update final report wording and user-facing docs so they consistently describe Telegram as the implemented channel and email/SMS/phone as future work.

## Working Areas That Audit Well

The following parts are in comparatively good condition:

- The monitor runtime path is structurally understandable.
- Telegram notifications are integrated end to end in the current realtime path.
- The research control layer under [research_ops](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/research_ops) is substantially more disciplined than the older project-target note sprawl.
- The report figure generation code now writes to the correct canonical figure directory.
- The report build script exists and is usable.

## Recommended Action Order

1. Freeze the repository surface.
   - Decide what stays live.
   - Commit or archive the current worktree split.

2. Repair the cross-dataset evidence chain.
   - Rebuild the manifest from checkpoints that actually exist.
   - Regenerate summary and figure.
   - Patch report text.

3. Reconcile evidence-control documents.
   - Align `THESIS_EVIDENCE_MAP.md`, `research_ops/EVIDENCE_INDEX.yaml`, and the draft.

4. Separate final figures from supporting/diagnostic figures.

5. Stabilize the test contract.
   - one canonical command
   - one torch-free subset
   - one torch-dependent subset

6. Clean up README/default profile drift and notification wording.

## Audit Verdict

Current state:

- **System viability:** good
- **Thesis viability:** good with caution
- **Repository cleanliness:** poor
- **Evidence closure:** mixed
- **Reproducibility confidence:** moderate, but not yet final-audit strong

Bottom line:

This is a strong advanced project, but not yet a cleanly frozen final repository.  
The biggest risk is no longer “the system does not work.”  
The biggest risk is “the stored evidence and repository state do not yet cleanly identify the one authoritative story.”
