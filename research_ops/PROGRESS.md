# PROGRESS

## Version
- current: v2

## Research Ops Status
- branch_scope: `feature/monitor-architecture-refactor` only
- claim_ledger: active
- evidence_ledger: active
- draft_alignment: active

## Global Status
- topic_init: completed
- literature: bounded_for_thesis
- experiments: bounded_for_thesis
- paper: in_progress
- final_export: buildable

## Current Position

- The project is being written as a rigorous thesis/report, not a conference submission.
- The strongest evidence chain is now claim-safe and bound to `C1-C5` and `E1-E9`.
- The current draft builds to `docx` and `pdf` through `scripts/build_report.sh`.

## Active Risks

- Field-validation evidence remains sample-level only and keeps `C4` at `partially_supported`.
- The bibliography/export layer is still lightweight rather than publication-grade.
- Any future replay/runtime artifact changes must be reflected in both ledgers before draft wording is updated.

## Active Refine Triggers

- Refresh replay/runtime evidence if deploy profiles or online preprocessing change again.
- Re-run draft alignment if any claim status changes.
- Upgrade the report build pipeline if submission formatting becomes stricter.

## Stage Logs

### Stage 1 - TOPIC_INIT
- status: completed
- deliverables:
  - "Locked thesis framing around deployment-aware pose-based fall detection."
  - "Primary dataset role fixed to CAUCAFall; LE2i fixed as comparative evidence."
- decision:
  - "Treat the project as a deployment-oriented system study, not a universal robustness paper."
- risks:
  - "If the venue framing changes later, claim scope may need to narrow again."
- updated_files:
  - "research_ops/PROJECT_CONFIG.yaml"
  - "research_ops/CLAIMS.yaml"

### Stage 2 - PROBLEM_DECOMPOSE
- status: completed
- deliverables:
  - "Claim stack reduced to system validity, controlled offline comparison, cross-dataset limitation evidence, bounded runtime evidence, and alert-policy calibration."
- decision:
  - "Use `C1-C5` as the live paper-facing claim set."
- risks:
  - "Replay/runtime evidence must remain clearly separated from offline benchmark evidence."
- updated_files:
  - "research_ops/CLAIMS.yaml"

### Stage 14 - RESULT_ANALYSIS
- status: completed
- deliverables:
  - "Evidence-side ledger created and tied to concrete repo artifacts."
  - "Draft alignment audit completed and refreshed to current status."
  - "Calibration/operating-point claim registered as `C5` with evidence `E9`."
- decision:
  - "Use `E1-E9` as the controlled evidence vocabulary for paper-safe wording."
- risks:
  - "Ledger drift will invalidate the paper-control layer if updates are made casually."
- updated_files:
  - "research_ops/EVIDENCE_INDEX.yaml"
  - "research_ops/PAPER_ALIGNMENT_AUDIT_2026-04-05.md"
  - "research_ops/README.md"

### Stage 17 - PAPER_DRAFT
- status: in_progress
- deliverables:
  - "Current draft aligned with the active runtime/replay evidence."
  - "Operating-point trade-off table added to support `C5`."
  - "Build script created and validated."
- decision:
  - "Use the current markdown draft as the canonical thesis source."
- risks:
  - "Formatting quality is adequate for iteration but not yet polished for final submission packaging."
- updated_files:
  - "docs/reports/drafts/HIGH_STANDARD_FINAL_PROJECT_REPORT_DRAFT_2026-03-29.md"
  - "scripts/build_report.sh"
