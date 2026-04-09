# Paper Publication Readiness Plan

## Goal
Close the gap from “strong dissertation project” to “submission-ready paper package”.

## Current Assessment
- Dissertation/report readiness: **high**
- Publication readiness: **partial** (needs stronger statistical and protocol closure)

---

## Unified Target Metrics

These targets are the single acceptance reference for optimization and publication readiness.

| Tier | Scope | Metric | Target |
|---|---|---|---|
| Primary (must hit) | CAUCAFall, OP1 | Recall | `>= 0.95` |
| Primary (must hit) | CAUCAFall, OP2 | F1 | `>= 0.90` (aspirational publish target), minimum acceptable `>= 0.80` |
| Primary (must hit) | CAUCAFall, OP2 | Precision | `>= 0.85` (target), minimum acceptable `>= 0.75` |
| Primary (must hit) | CAUCAFall, OP3 | FA24h | `<= 1.0` |
| Primary (must report) | CAUCAFall | AP | report + CI (no standalone pass/fail) |
| Stability (must hit) | Final candidates | multi-seed std(F1) | `<= 0.05` preferred (flag if higher) |
| Stability (must hit) | Final candidates | 95% CI | reported for F1/Recall/FA24h |
| Comparative (must report) | LE2i | OP2 F1/Recall/Precision/FA24h | report transparently (no hard gate for deployment pass) |
| Deployment (must hit) | Replay/field validation set | false-alert behavior | documented + within locked policy expectation |

Notes:
- Primary dataset for deployment claims remains **CAUCAFall**.
- LE2i is comparative/generalization evidence and must be reported, but does not block primary deployment claim if CAUCAFall gates pass.
- Any metric that does not meet target must be explicitly documented in limitations.

---

## Workstreams (Detailed)

## W1. Freeze Final Experimental Protocol (P0)
### Tasks
- Lock final datasets, splits, metrics, OP-fit rules, and seed lists in one immutable protocol file.
- Freeze policy: no post-lock metric-definition or tuning-rule changes.

### Files
- `docs/project_targets/CLAIM_TABLE.md`
- `docs/project_targets/FINAL_CANDIDATES.md`
- `docs/project_targets/supporting/OPS_POLICY_REPORT.md`
- `docs/project_targets/PROJECT_FINAL_YEAR_EXECUTION_PLAN.md`

### Commands
- `python tools/check_evidence_map.py` (if available; otherwise add as TODO)
- `python scripts/summarize_metrics_table.py` (refresh current table)

### Acceptance
- One protocol version/date is declared.
- Every final figure/table in evidence map references this protocol.

---

## W2. Statistical Strengthening (P0)
### Tasks
- Re-run or verify multi-seed outputs for final candidates with fixed seeds.
- Produce mean/std/95% CI for event metrics (`F1`, `Recall`, `FA24h`, plus `AP`).
- Run only pre-registered significance tests.

### Files
- `docs/project_targets/STABILITY_REPORT.md`
- `docs/project_targets/SIGNIFICANCE_REPORT.md`
- `artifacts/reports/stability_summary.csv`
- `artifacts/reports/significance_summary.json`

### Commands
- `python tools/run_stability_manifest.py --manifest artifacts/registry/stability_manifest.csv --start_status todo --stop_on_fail 1`
- `python scripts/plot_stability_metrics.py --glob "outputs/metrics/*_stb_s*.json" --out_fig artifacts/figures/stability/fc_stability_boxplot.png`

### Acceptance
- Final results table has CI columns.
- Significance report includes hypothesis, test, p-value, interpretation.

---

## W3. Real-World Validation Evidence (P0)
### Tasks
- Complete field observation CSV with real replay/live cases (fall + non-fall).
- Generate deployment event summary and failure taxonomy.

### Files
- `docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md`
- `docs/project_targets/FIELD_VALIDATION_RUNBOOK.md`
- `artifacts/reports/deployment_field_eval.json`
- `artifacts/reports/deployment_field_failures.json`
- `artifacts/reports/deployment_field_validation_summary.md`

### Commands
- `python tools/summarize_dual_policy_events.py --resident_id 1 --hours 24 --out_json artifacts/reports/deployment_dual_policy_events.json`
- `python tools/summarize_field_validation.py --obs_csv artifacts/reports/deployment_field_observations.csv --hours 1.0 --dual_policy_json artifacts/reports/deployment_dual_policy_events.json --out_eval_json artifacts/reports/deployment_field_eval.json --out_failures_json artifacts/reports/deployment_field_failures.json --out_markdown artifacts/reports/deployment_field_validation_summary.md`

### Acceptance
- Field validation report includes explicit pass/fail gate.
- Evidence map includes field-validation artifact row with command.

---

## W4. Novelty and Positioning (P1)
### Tasks
- Finalize 3-5 contribution claims with direct evidence links.
- Add “why this is new vs closest baselines” paragraph set.
- Add “limitations + failure modes” section as first-class result.

### Files
- `docs/project_targets/CLAIM_TABLE.md`
- `docs/project_targets/THESIS_EVIDENCE_MAP.md`
- manuscript draft (external or to be added under `docs/`)

### Acceptance
- Each contribution has: claim, metric threshold, artifact, reproduce command.

---

## W5. Reproducibility Package (P1)
### Tasks
- Ensure each final figure/table has one reproduce command.
- Freeze final tag and release bundle status.
- Run one clean-machine dry run.

### Files
- `docs/project_targets/SUBMISSION_PACK_INDEX.md`
- `docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md`
- `artifacts/reports/release_snapshot.md`
- `artifacts/reports/release_bundle_status.json`

### Commands
- `bash tools/run_deployment_lock_validation.sh`
- `python tools/check_release_bundle.py`

### Acceptance
- Checklist fully green for code + user guide + runbook + evidence map.

---

## W6. Writing Package for Submission (P1)
### Tasks
- Build venue-ready paper outline (IMRaD).
- Convert thesis-long sections into concise paper form.
- Prepare camera-ready figures from reproducible scripts only.

### Target Outputs
- `paper_draft_v1` (external doc or repo doc)
- finalized figure pack (from `artifacts/figures/`)
- bibliography synced with cited methods

### Acceptance
- Full manuscript draft complete and internally reviewable.

---

## 6-Week Execution Schedule

## Week 1
- Close W1 protocol lock.
- Refresh claim table and evidence map consistency.

## Week 2
- Close W2 stability + CI.
- Generate stability figures and significance summary.

## Week 3
- Close W3 field validation artifacts.
- Finalize failure taxonomy and deployment narrative.

## Week 4
- Close W4 novelty positioning draft.
- Map each claim to exact artifact/command.

## Week 5
- Close W5 reproducibility bundle.
- Run full submission checklist and clean-machine dry run.

## Week 6
- Close W6 manuscript pack.
- Final internal review and submission preparation.

---

## Definition of “Submission-Ready”
- W1-W3 complete with PASS artifacts.
- W4 novelty claims are evidence-backed and bounded by limitations.
- W5 reproducibility checks are green.
- W6 manuscript draft complete with final figure/table set.

---

## Publication Optimization Sprint (Executable)

## Sprint Objective
Lift the project from “strong report” to “paper-ready evidence quality” under a fixed protocol.

## Guardrails (Must Follow)
- No metric/protocol changes mid-sprint after lock.
- No unpublished numbers in draft text without artifact + command.
- No “best single run” claims without multi-seed summary.

## Week A — Protocol + Stable Core Results

### Day A1: Lock Protocol
Tasks:
- Freeze final comparison scope (primary: CAUCAFall; comparative: LE2i).
- Freeze metrics and OP selection rules.
- Freeze seed list for final candidates.

Commands:
- `python scripts/summarize_metrics_table.py`
- `python tools/check_evidence_map.py` (if present; else mark TODO)

Acceptance:
- Protocol lock section updated in:
  - `PROJECT_FINAL_YEAR_EXECUTION_PLAN.md`
  - `CLAIM_TABLE.md`
  - `THESIS_EVIDENCE_MAP.md`

### Day A2-A3: Multi-Seed Repro Runs
Tasks:
- Run fixed seed manifests for final TCN/GCN candidates.
- Ensure every run is recorded in registry/changelog.

Commands:
- `python tools/run_stability_manifest.py --manifest artifacts/registry/stability_manifest.csv --start_status todo --stop_on_fail 1`
- `python scripts/plot_stability_metrics.py --glob "outputs/metrics/*_stb_s*.json" --out_fig artifacts/figures/stability/fc_stability_boxplot.png`

Acceptance:
- Updated:
  - `artifacts/reports/stability_summary.csv`
  - `artifacts/reports/stability_summary.json`
  - `docs/project_targets/STABILITY_REPORT.md`

### Day A4: Significance + CI
Tasks:
- Recompute significance for pre-registered hypotheses only.
- Add CI/p-value lines to main result table.

Acceptance:
- `docs/project_targets/SIGNIFICANCE_REPORT.md` updated with final values.
- Main result table in evidence map aligned with CI references.

### Day A5: Freeze Best Publishable Profiles
Tasks:
- Choose final “paper profiles” (TCN + GCN) using locked criteria.
- Lock corresponding ckpt + ops + metrics pointers.

Acceptance:
- `FINAL_CANDIDATES.md` and `THESIS_EVIDENCE_MAP.md` fully synchronized.

## Week B — Deployment Evidence + Manuscript Package

### Day B1-B2: Field Validation Closure
Tasks:
- Complete observation CSV and generate field summary artifacts.
- Produce false-alarm/failure taxonomy section.

Commands:
- `python tools/summarize_dual_policy_events.py --resident_id 1 --hours 24 --out_json artifacts/reports/deployment_dual_policy_events.json`
- `python tools/summarize_field_validation.py --obs_csv artifacts/reports/deployment_field_observations.csv --hours 1.0 --dual_policy_json artifacts/reports/deployment_dual_policy_events.json --out_eval_json artifacts/reports/deployment_field_eval.json --out_failures_json artifacts/reports/deployment_field_failures.json --out_markdown artifacts/reports/deployment_field_validation_summary.md`

Acceptance:
- Field validation docs/artifacts referenced in evidence map.

### Day B3: Repro Bundle Finalization
Tasks:
- Run deployment lock validation and release bundle checks.
- Ensure runbooks and user guide match current runtime behavior.

Commands:
- `bash tools/run_deployment_lock_validation.sh`
- `python tools/check_release_bundle.py`

Acceptance:
- `FINAL_SUBMISSION_CHECKLIST.md` all critical items checked.
- `DELIVERY_ALIGNMENT_STATUS.md` updated.

### Day B4-B5: Paper Draft Assembly
Tasks:
- Write compact contribution section (3-5 claims with links).
- Insert final figures/tables from reproducible artifacts only.
- Add explicit limitations and threat-to-validity section.

Acceptance:
- Draft has zero unmapped numbers.
- Every figure/table has ID + artifact + command in evidence map.

## Final Go/No-Go Gate
- Go if all are true:
  - multi-seed + CI + significance complete
  - field validation report complete
  - reproducibility checks pass
  - manuscript claims all evidence-linked
- No-Go otherwise: extend sprint by one cycle with unresolved items only.
