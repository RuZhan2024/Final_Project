# PROGRESS

## Version
- current: v1

## Research Ops Status
- claim_ledger: active
- evidence_ledger: active

## Global Status
- topic_init: completed
- literature: in_progress
- experiments: in_progress
- paper: in_progress

## Loop Points
- possible_refine:
  - "Refresh current replay/runtime evidence after any deploy-profile or preprocessing change."
  - "Tighten paper wording whenever CLAIMS.yaml status changes."
- possible_pivot: []

## Stage Logs
### Stage 1 - TOPIC_INIT
- status: completed
- deliverables:
  - "Locked thesis topic framing around deployment-aware pose-based fall detection."
  - "Primary dataset role fixed to CAUCAFall; LE2i fixed as comparative/generalization evidence."
- decision:
  - "Treat this project as a deployment-oriented system study, not as a universal robustness paper."
- risks:
  - "Venue target may still change from thesis report framing to a conference-style paper framing."
- updated_files:
  - "research_ops/PROJECT_CONFIG.yaml"
  - "research_ops/CLAIMS.yaml"

### Stage 2 - PROBLEM_DECOMPOSE
- status: completed
- deliverables:
  - "Main claim stack decomposed into system validity, controlled TCN-vs-GCN comparison, cross-dataset limitation analysis, and bounded runtime/deployment evidence."
- decision:
  - "Use only claim set C1-C4 as paper-facing high-level claims under the current freeze."
- risks:
  - "Replay/runtime evidence must remain clearly separated from offline benchmark evidence."
- updated_files:
  - "research_ops/CLAIMS.yaml"

### Stage 14 - RESULT_ANALYSIS
- status: in_progress
- deliverables:
  - "Evidence-side ledger created to pair claim IDs with repository artifacts and reproduction commands."
  - "Draft-to-ledger alignment audit created for the current report draft."
- decision:
  - "Use E1-E8 as the controlled evidence vocabulary for paper-safe wording and future revisions."
  - "Register calibration/operating-point design as explicit claim C5 with evidence E9."
- risks:
  - "If new runtime or report artifacts supersede current ones, EVIDENCE_INDEX.yaml and CLAIMS.yaml must be updated together."
- updated_files:
  - "research_ops/EVIDENCE_INDEX.yaml"
  - "research_ops/PAPER_ALIGNMENT_AUDIT_2026-04-05.md"
  - "research_ops/README.md"
