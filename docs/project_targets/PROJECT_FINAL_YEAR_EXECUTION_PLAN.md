# Final-Year Research Execution Plan

## Scope
This document defines a thesis-defense-level execution plan for scientific rigor and deployment readiness.
Each task is documented with: issue, resolve method, execution steps, target, and acceptance evidence.

## Global Standards (Apply to All Tasks)
### 1) Canonical Metrics Schema
Use one JSON schema everywhere (train/val/test, in-domain/cross-dataset). Recommended keys:
- `dataset`, `split`, `arch`, `seed`
- `window`: `{W, S, fps}`
- `features`: `{motion, conf, bone, bone_len, ...}`
- `confirm`: `{enabled, confirm_s, min_lying, max_motion, ...}`
- `op_policy_id`, `thr`
- `AP`, `F1`, `Recall`, `Precision`, `FA24h`
- `delay_p50`, `delay_p95`, `latency_p50`, `latency_p95`
- `artifact_ptrs`: `{ckpt, ops_yaml, metrics_json, logs}`

### 2) Canonical Experiment ID
Use one stable parseable ID as primary key:
`EXP={arch}_{dataset}_{W}W{S}S_seed{seed}_feat{...}_confirm{...}_op{...}`

### 3) One-Command Reproduction Contract
Every reported table row/figure must map to a single reproduce command (or make target) and exact artifact path.

### 3.5) Mandatory Run-Change Logging (Execution Discipline)
Before every training/eval/sweep run that changes parameters, append one row to:
- `artifacts/reports/tuning/PARAM_CHANGELOG.csv`

Minimum fields:
- `timestamp_utc`, `exp_id`, `base_ref`, `changed_params`, `command`, `status`, `artifacts`, `notes`

Workflow rule:
1. Pre-run row: `status=running` with exact command and changed parameters.
2. Post-run update: switch to `done` (or `failed`) and fill produced artifact paths.
3. No result may be cited in thesis/report unless a matching changelog row exists.

### 4) Core Metrics Contract (Required)
Every final report table must include these metrics where applicable:
- `AP`, `F1`, `Recall`, `Precision`, `FA24h`
- `FPR@Recall>=0.95` (or `FA24h@Recall>=0.95` when FPR not directly available)
- event delay: `delay_p50`, `delay_p95`
- latency: `latency_p50`, `latency_p95`
- calibration: `ECE`, `Brier` (for uncertainty/calibration tasks)
- cost-sensitive utility: `U = C_fn*FN + C_fp*FP` with fixed pre-registered costs
- cross-domain deltas: `ΔF1`, `ΔRecall`, `ΔFA24h` (vs in-domain baseline)

### 5) Metric Reporting Rules
- Any missing metric must be explicitly marked `NA` with reason.
- No metric from test split can be used for threshold/policy tuning.
- Final thesis/report numbers must come only from artifacts tracked in `THESIS_EVIDENCE_MAP.md`.
- For unlabeled/weakly labeled sets, treat `FA24h` as an estimated alert-rate proxy only unless the set is verified fall-free. Do not use unlabeled sets for final recall/F1/AP claims.

### 6) Dataset Role Policy (Locked)
- Primary deployment target dataset: `CAUCAFall`
- Comparative/generalization dataset: `LE2i`
- Acceptance gates for deployment readiness are enforced on the primary dataset first.
- Comparative dataset results are mandatory to report, but not used as sole blocker for primary deployment acceptance.
- Decision lock date: `2026-03-03` (no role switching in final cycle without explicit plan revision and gate redefinition).

## Success Criteria (Program-Level)
The project is final-year-research ready when all are true:
1. Core claims are statistically supported (multi-seed + significance).
2. Primary-dataset deployment targets are met, and comparative/cross-dataset results are reported transparently.
3. Deployment evidence includes FA/24h, recall, latency, and failure handling.
4. Every reported result is reproducible from command + config + artifact mapping.
5. External examiner can run the demo from runbook on a clean machine.

## Reachability & Limitations Register (Live)
Purpose: keep claims realistic during execution and avoid over-claiming before closure tasks are complete.

### Currently Reachable (based on existing artifacts)
- `Recall/F1/AP/FA24h` reporting is implemented and reproducible for in-domain runs.
- CAUCAFall currently supports low-alarm operating points (`FA24h=0` in current selected runs).
- Cross-dataset protocol and artifacts are in place (`cross_*` metrics + transfer plot).
- Latency profiling artifacts and summary plotting are in place.

### Current Limitations (must be closed before final claims)
- LE2i false alerts remain high in current baseline outputs; keep LE2i in comparative role and do not claim low-alarm deployment readiness there.
- Cross-domain transfer is asymmetric; do not claim robust universal transfer yet.
- Real-world field validation dataset evidence is pending full completion.

### Closure Criteria for Limitations
1. LE2i low-alarm claim remains optional/comparative unless project scope is expanded beyond primary-dataset deployment target.
2. Cross-domain robustness claim unlocked only after reporting `ΔF1/ΔRecall/ΔFA24h` for both directions with discussion.
3. Stability claim unlocked only after 5-seed summary (`mean/std/95% CI`) is complete for final candidates.
4. Deployment claim unlocked only after field validation report artifacts are complete.

## Task 1 — Claim Table
### Issue
Claims are implicit and scattered.
### Resolve method
Create falsifiable claim table with explicit pass/fail rules.
### How to do
1. Create `docs/project_targets/CLAIM_TABLE.md`.
2. Keep only 3-4 high-value claims.
3. Use columns: `Claim`, `Metric(s)`, `Threshold`, `Dataset(s)`, `Protocol`, `Evidence artifact`, `Reproduce command`, `Failure condition`, `Pass/Fail rule`.
### Target
Claims are concise, falsifiable, and directly testable.
### How to get target
- No claim without threshold + failure condition + reproduce command.

## Task 2 — Experiment Registry
### Issue
Manual logging is error-prone.
### Resolve method
Auto-append run metadata from code, not by hand.
### How to do
1. Add `src/fall_detection/core/registry.py` with `append_row(...)`.
2. Auto-record fields: `exp_id`, `git_commit`, `git_dirty`, `command`, `env(python/torch/cuda)`, `seed`, `config_hash`, `artifact_root`.
3. Write rows to `artifacts/registry/experiments.csv`.
### Target
Registry is machine-generated and complete.
### How to get target
- Randomly sample 5 rows; reproduce each run from registry only on clean env.

## Task 2.5 — Final Candidate Freeze (New)
### Issue
Multi-seed/ablation can waste budget on weak configs.
### Resolve method
Freeze 2 final candidates per dataset (TCN + GCN) before heavy evaluation.
### How to do
1. Use existing sweep/stage2 outputs.
2. Select candidates by deployment criteria (Recall + FA/24h), not AP alone.
3. Record selected configs in `docs/project_targets/FINAL_CANDIDATES.md`.
### Target
Downstream evaluations run only on strong candidates.
### How to get target
- Candidate file includes command, config hash, artifact pointers.

## Task 3 — Multi-Seed Stability
### Issue
Single-seed best results are insufficient.
### Resolve method
Run stability on frozen candidates with practical seed budgets.
### How to do
1. Final candidates: 5 seeds each.
2. Early exploration: 3 seeds each.
3. Report mean/std/95% CI.
4. Decide and state metric level explicitly: window-level vs event-level (at least event-level must be covered).
5. Mandatory CI metrics: `F1`, `Recall`, `FA24h`, `delay_p95` (add `AP` as secondary).
### Target
Stability is quantified and protocol is explicit.
### How to get target
- `docs/project_targets/STABILITY_REPORT.md` contains both setup and CI tables.
- Execute:
  - `python tools/run_stability_manifest.py --manifest artifacts/registry/stability_manifest.csv --start_status todo --stop_on_fail 1`
  - `python scripts/plot_stability_metrics.py --glob "outputs/metrics/*seed*.json" --out_fig artifacts/figures/stability/fc_stability_boxplot.png`

## Task 4 — Cross-Dataset Generalization
### Issue
In-domain only evidence is insufficient.
### Resolve method
Run bidirectional cross-dataset evaluation under fixed invariants.
### How to do
1. Define invariants first: same `W/S`, feature flags, label mapping, OP-fit rule (val-only).
2. Run LE2i->CAUCAFall and CAUCAFall->LE2i.
3. Add top-3 error taxonomy: domain gap, FPS mismatch, skeleton quality/camera angle.
4. Report required deltas: `ΔF1`, `ΔRecall`, `ΔFA24h` against corresponding in-domain runs.
### Target
Generalization limits and causes are evidence-based.
### How to get target
- `docs/project_targets/CROSS_DATASET_REPORT.md` includes invariants + taxonomy.
- Execute:
  - `python scripts/plot_cross_dataset_transfer.py --manifest artifacts/reports/cross_dataset_manifest.json --out_fig artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png`

## Task 5 — Significance Testing
### Issue
Too many comparisons dilute statistical power.
### Resolve method
Pre-register 2-3 primary hypotheses with paired tests where possible.
### How to do
1. Define hypotheses only:
`TCN vs GCN`, `confirm on/off`, `feature on/off` (example).
2. Use paired tests on same videos/subjects if possible.
3. Report p-values and CI with interpretation.
4. Primary test metrics: event `F1`, `Recall`, `FA24h`; secondary: `AP`.
### Target
Key comparisons are statistically defensible.
### How to get target
- `docs/project_targets/SIGNIFICANCE_REPORT.md` covers pre-registered hypotheses only.

## Task 6 — Ablation Matrix
### Issue
Full factorial is expensive and low-yield.
### Resolve method
Use prioritized one-factor ablations + two interaction checks.
### How to do
1. Build `docs/project_targets/ABLATION_MATRIX.md` with `priority` column.
2. Run one-factor-at-a-time for `motion/conf/bone/confirm`.
3. Run only two interactions (example: `confirm×motion`, `bone×gcn`).
### Target
Maximum insight per compute budget.
### How to get target
- High-priority rows complete; low-priority rows optional and labeled.

## Task 6.5 — Evidence Plot Suite (Required)
### Issue
Current evidence relies too much on metric tables; thesis-grade reporting needs diverse plots for behavior, stability, and deployment tradeoffs.
### Resolve method
Define a required plot suite and generate each plot from reproducible scripts/artifacts.
### How to do
1. Create `docs/project_targets/PLOT_EVIDENCE_CHECKLIST.md` with one row per required figure.
2. Required plots (minimum):
- F1-vs-threshold (`tau`) for TCN/GCN on LE2i and CAUCAFall.
- Recall-vs-FA24h tradeoff curves for TCN/GCN on both datasets.
- PR curves (or AP summary plot) for all final candidates.
- Multi-seed box/violin plots for key event metrics (`F1`, `Recall`, `FA24h`).
- Cross-dataset transfer comparison bars (in-domain vs cross-domain).
- Runtime latency distribution (p50/p95/tail) for target deployment device.
- Calibration plot(s): reliability curve + ECE summary.
- Cost-sensitive utility plot under fixed (`C_fn`, `C_fp`) settings.
3. Standardize output location: `artifacts/figures/<exp_or_group>/...`.
4. Add one reproduce command per plot family (Make target or script command) into `THESIS_EVIDENCE_MAP.md`.
### Target
All critical claims are supported by both tables and diverse visual evidence.
### How to get target
- Every required plot exists on disk and is mapped in evidence map with command + commit.
- No thesis/report figure is manually edited outside reproducible scripts.

## Task 7 — OP Policy Validation
### Issue
Hidden leakage risk in threshold/policy tuning.
### Resolve method
Enforce strict protocol checklist.
### How to do
1. Fit OP on val only.
2. Forbid: threshold tuning on test, calibration on test, best-seed selection by test.
3. Add protocol checklist before publishing any number.
4. Include policy quality table with:
`Recall`, `FA24h`, `delay_p95`, and `Utility(C_fn,C_fp)` for OP1/OP2/OP3.
### Target
Policy selection is leakage-free and auditable.
### How to get target
- `docs/project_targets/OPS_POLICY_REPORT.md` includes completed checklist.

## Task 8 — Runtime Failure-Mode Robustness
### Issue
Static tables are not enough for runtime robustness claims.
### Resolve method
Make failure testing executable.
### How to do
1. Add `tools/fault_inject.py` to simulate: empty skeleton, dropped frames, low conf, camera end, missing files, DB/API failure.
2. Define expected behavior per fault (skip/log/cooldown/fallback).
3. Export pass/fail summary JSON + logs.
4. Include robustness metrics:
  - `failure_recovery_rate`
  - `false_alert_spike_after_fault`
  - `mean_recovery_time_s`
### Target
Robustness claims are test-backed, not narrative-only.
### How to get target
- `docs/project_targets/ROBUSTNESS_REPORT.md` + fault summary JSON artifacts.

## Task 9 — Latency and Resource Profiling
### Issue
Latency claims are device-dependent and currently under-specified.
### Resolve method
Define target device profile first, then measure full stack.
### How to do
1. Fix device profile(s): laptop CPU, GPU server, edge-class (choose applicable).
2. Measure preprocessing, model forward, end-to-end pipeline.
3. Report batch size 1 separately from offline batch mode.
4. Report p50/p95 + tail/worst-case.
5. Enforce latency target gates per profile (example):
  - `latency_p95 <= 200ms` for end-to-end stream step on target profile.
### Target
Real-time feasibility is quantitatively defensible on target hardware.
### How to get target
- `docs/project_targets/LATENCY_REPORT.md` + `artifacts/reports/infer_profile_*.json`.
- Execute:
  - `python scripts/profile_infer.py --win_dir <val_or_test_windows> --ckpt <best.pt> --io_only 0 --profile cpu_local --out_json artifacts/reports/infer_profile_cpu_local_<exp>.json`
  - `python scripts/plot_latency_profiles.py --reports artifacts/reports/infer_profile_*.json --out_fig artifacts/figures/latency/latency_profile_summary.png`

## Task 9.5 — Real-World Deployment Validation Set
### Issue
Benchmark datasets alone do not prove deployment reliability in your actual environment.
### Resolve method
Create a small real-world validation set (recorded locally) and evaluate end-to-end behavior.
### How to do
1. Record a deployment validation dataset:
- ADL clips: walking, sitting, bending, picking object, intentional lying down.
- Fall-like/safety events: controlled fast-sit/collapse simulations (safe protocol only).
- Variation axes: lighting, camera angle, distance, partial occlusion, single/multi-person.
2. Keep this dataset isolated from training/validation/test benchmark splits.
3. Run the deployed stack (camera/API/model/policy) and log:
- event detection, false alerts, delay, missed events, runtime failures.
- FPR@high-recall operating region and per-session alert counts.
4. Export report artifacts:
- `artifacts/reports/deployment_field_eval.json`
- `artifacts/reports/deployment_field_failures.json`
5. Summarize protocol and results in:
- `docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md`
### Target
Deployment claims are backed by realistic, local-environment evidence.
### How to get target
- At least 20-40 short clips collected and evaluated.
- Report includes FA rate, recall proxy, delay stats, and failure-mode summary.
- Evidence map links results to commands and artifact paths.
- Include calibration/uncertainty snapshot on field set (`ECE`, uncertainty histogram).

## Task 10 — Reproducible Demo Runbook
### Issue
Examiner setup risk remains high.
### Resolve method
Provide known-good minimal raw-free demo sample with expected outputs.
### How to do
1. Include tiny processed sample for demo path.
2. Add exact commands and expected checks (hash or metric ranges).
3. Add failure troubleshooting by stage.
### Target
Examiner can run demo without raw dataset.
### How to get target
- Clean-machine run succeeds using runbook and sample only.

## Task 11 — Raw-Required vs Raw-Free Modes
### Issue
Cloud mode still accidentally triggers extraction.
### Resolve method
Enforce mode boundaries in Makefile/scripts.
### How to do
1. Full mode: requires `data/raw`; Eval mode: no extraction allowed.
2. In Eval mode, extraction targets fail fast with clear message if `data/raw` missing.
3. Add `.env.example` cloud template with minimum required vars.
### Target
Raw-free cloud execution is reliable and explicit.
### How to get target
- Eval-mode sweeps/eval run successfully with no `data/raw`.

## Task 12 — Evidence Map
### Issue
Tables/figures may drift from artifact truth.
### Resolve method
Make evidence map the single source of truth with automated checks.
### How to do
1. Create `docs/project_targets/THESIS_EVIDENCE_MAP.md`.
2. Required columns: `Figure/Table ID`, `Artifact path`, `Command`, `Script`, `Inputs`, `Outputs`, `Commit`.
3. Add `tools/check_evidence_map.py` to validate all artifact paths exist.
### Target
Every reported number has a verifiable artifact pointer.
### How to get target
- Pre-submission check script passes with zero missing paths.

## Execution Order (Recommended)
1. Task 1-2 (claims + registry)
2. Task 2.5 (freeze final candidates)
3. Task 11 (mode separation)
4. Task 3-4-5 (stability + generalization + significance)
5. Task 6-6.5-7 (ablation + plots + OP policy)
6. Task 8-9-9.5 (robustness + latency + real-world deployment validation)
7. Task 10-12 (runbook + evidence map + final consistency pass)

## Weekly Milestones (8 Weeks)
1. Week 1: Task 1-2
2. Week 2: Task 2.5 + Task 11
3. Week 3-4: Task 3-4
4. Week 5: Task 5-6
5. Week 6: Task 7-8
6. Week 7: Task 6.5 + Task 9 + Task 9.5
7. Week 8: Task 12 + final thesis sync

## Final Acceptance Checklist
- [ ] Canonical metrics schema adopted across train/eval/report
- [ ] Canonical `EXP` id used as primary key
- [ ] Every table row has one reproduce command
- [ ] Claim table complete with pass/fail + failure conditions
- [ ] Registry auto-appends and supports replay for sampled rows
- [ ] Final candidate configs frozen before multi-seed
- [ ] Multi-seed + CI complete for core candidates
- [ ] Cross-dataset results and taxonomy reported
- [ ] Cross-domain deltas (`ΔF1`, `ΔRecall`, `ΔFA24h`) reported
- [ ] Significance tests completed for pre-registered hypotheses
- [ ] Priority ablations completed
- [ ] Required evidence plot suite generated and mapped
- [ ] OP policy leakage checklist passed
- [ ] Failure-mode robustness tests executable and passing
- [ ] Latency/resource targets measured and documented
- [ ] Calibration metrics (`ECE`, `Brier`) reported for final candidates
- [ ] Cost-sensitive utility table reported with fixed (`C_fn`, `C_fp`)
- [ ] Clean-machine runbook validated
- [ ] Raw-free mode validated
- [ ] Evidence map check script passes
