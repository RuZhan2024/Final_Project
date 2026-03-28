# FYP Supervisor Checklist

Date: 2026-03-16

Purpose:
Provide a concise supervisor-facing checklist showing which evaluation expectations are already satisfied, which are incomplete, and what action is required before final submission.

## Summary

Overall status: `Strong, but not fully closed`

Current profile:
- technical quality: `done`
- methodology and evaluation design: `done`
- statistical reporting: `mostly done`
- dissertation-style reflection and ethics closure: `missing in part`
- field-validation closure: `missing`

## Checklist

| Area | Status | Evidence | Missing | Action |
|---|---|---|---|---|
| Project objectives are clear and measurable | Done | `PROJECT_FINAL_YEAR_EXECUTION_PLAN.md`, publication readiness plan | No concise examiner-facing summary table | Add a one-page `Objectives -> Metrics -> Evidence -> Outcome` table |
| Literature review is current and synthetic | Partial | `docs/Literature Review.docx`, paper outline | Cannot yet confirm quality of synthesis or recency from repo evidence alone | Revise literature review to emphasize themes, debates, gap, and recent sources |
| Methodology is justified and appropriate | Done | README, readiness report, execution plan, pipeline code | None at high level | Keep methodology chapter tightly aligned to actual pipeline and frozen protocol |
| Data collection and datasets are clearly described | Done | dataset docs, cross-dataset report, deployment field-validation plan | Field-data collection is planned but not fully closed | Finalize field-data description once remaining evidence is complete |
| Ethical considerations are addressed | Partial | ethics placeholder in paper structure | No strong standalone ethics/privacy discussion yet | Add `Ethics, Privacy, and Deployment Risk` subsection to dissertation/paper |
| Results are presented clearly and accurately | Done | stability, significance, cross-dataset, deployment docs | Some evidence files still need final cleanup | Ensure final tables and figures come only from current locked artifacts |
| Statistical analysis is appropriate | Mostly done | 5-seed stability reports, Wilcoxon significance report | Small-sample power limits; some tests degenerate | State limitations clearly and keep claims conservative |
| Findings are linked back to research questions | Partial | execution plan and outline imply this | No direct `RQ -> Result` mapping table yet | Add a table mapping each research question to metrics, sections, and artifacts |
| Limitations are honestly acknowledged | Done | readiness report, significance report, deployment validation task sheet | Needs final polished write-up in dissertation chapter | Consolidate current limitations into one formal section |
| Future work is clearly stated | Done | paper outline, readiness plan | Could be more concise and prioritized | Reduce future work to 3-5 concrete next steps |
| Reproducibility is demonstrated | Mostly done | Makefile workflows, manifests, reports, audit docs | Some stale placeholder files remain | Refresh claim/evidence files and verify all cited artifacts exist |
| Continuous evaluation is documented | Mostly done | audits, readiness reports, execution planning | Reflection narrative is not yet explicit | Add a short evaluation log summary or reflection section |
| Critical reflection is included | Partial | indirect evidence in plans and audits | No direct reflective chapter or subsection | Write a dedicated reflection section on decisions, failures, trade-offs, and lessons learned |
| External or practical validation is present | Partial | replay/deployment validation framework exists | Full field-validation closure still pending | Complete field-validation artifacts and summarize them clearly |

## Priority Actions

### Priority 1: Must Close
1. Complete field-validation artifacts and summary.
2. Add an examiner-facing `Objectives -> Evidence -> Outcome` table.
3. Add a direct `Research Question -> Result` mapping table.
4. Add a dedicated `Ethics, Privacy, and Deployment Risk` subsection.
5. Refresh stale claim/evidence placeholder documents.

### Priority 2: Strongly Recommended
1. Add a short critical reflection section.
2. Tighten the literature review into a more synthetic, gap-driven narrative.
3. Consolidate limitations into one final polished section.

### Priority 3: Polish
1. Reduce future work to the most defensible next steps.
2. Cross-check every figure and table against live artifact paths.
3. Ensure all final reported numbers come from locked, reproducible outputs.

## Supervisor-Facing Verdict

If assessed against the lecture standard, the project is already strong on technical substance and evaluation design.

If the remaining gaps above are closed, the project will present much more clearly as `First-class` work rather than strong technical work with a few presentation and evaluation-closure weaknesses.
