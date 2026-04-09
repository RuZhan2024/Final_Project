# Branch Support Allowlist

Date: 2026-04-09  
Scope: material that should remain on the current branch for thesis/reporting depth, but is not part of the minimum frozen snapshot

## Rule

These files are intentionally retained on the branch because they support:

- thesis/report writing
- internal auditability
- engineering traceability
- experiment history

They should not be treated as the minimum release boundary by default.

## 1. Research Control Layer

Keep on branch:

- `research_ops/`

## 2. Supporting Project-Target Docs

Keep on branch:

- `docs/project_targets/supporting/`

Includes:

- `ABLATION_MATRIX.md`
- `EXPERIMENT_RECORDING_PROTOCOL.md`
- `LATENCY_REPORT.md`
- `OBJECTIVES_EVIDENCE_OUTCOMES.md`
- `OPS_POLICY_REPORT.md`
- `PARAM_PROMOTION_WORKFLOW.md`
- `ROBUSTNESS_REPORT.md`

## 3. Archived Planning Docs

Keep on branch:

- `docs/project_targets/archive/planning/`

Includes:

- `PAPER_SECTION_HEADINGS.md`
- `PAPER_SUBMISSION_WEEK_PLAN.md`
- `PLOT_SELECTION_FOR_REPORT.md`

## 4. Audit, Notes, and Readiness Layers

Keep on branch:

- `docs/reports/audit/`
- `docs/reports/notes/`
- `docs/reports/readiness/`

## 5. Supplemental Evidence and Engineering Traces

Keep on branch:

- `artifacts/reports/tuning/`
- `artifacts/reports/gcn_aug/`
- `artifacts/reports/gcn_overtake/`
- `artifacts/reports/hneg_cycle/`
- `artifacts/reports/fault_inject_summary.json`
- `artifacts/reports/infer_profile_cpu_local_tcn_le2i.json`
- `artifacts/reports/op123_per_seed.csv`
- `artifacts/reports/op123_stability_summary.csv`
- `artifacts/reports/op123_stability_summary.json`

## 6. Diagnostic and Archive Materials

Keep on branch:

- `artifacts/reports/archive/`
- `artifacts/reports/diagnostic/`
- `artifacts/figures/report/diagnostic/`
- `configs/ops/archive/muvim/`
- `configs/ops/diagnostic/`

## 7. Use Rule

When in doubt:

- if the file is needed to reproduce the exact final claim set, it belongs in `FREEZE_CORE_ALLOWLIST_2026-04-09.md`
- if it is mainly useful for explanation, history, diagnosis, or branch-local thesis management, it belongs here
