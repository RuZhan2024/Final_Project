# Full Report Master Plan

Date: 2026-04-09  
Target: full final-year project report  
Expected length: approximately 80-100 pages including figures, tables, and appendices

## Purpose

This plan controls the full-report line. It should expand the compact paper story into a supervisor-facing and submission-facing project report without changing the locked evidence hierarchy.

The report should not be written as a longer undergraduate-style project summary. The target is a research-grade technical report: analytically disciplined, evidence-controlled, explicit about limits, and strong enough in reasoning that length comes from substance rather than padding.

## Primary Control Files

1. [FULL_PROJECT_REPORT_FINAL_2026-04-11.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md)
2. [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)
3. [PAPER_SUBMISSION_READINESS_CHECKLIST.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_SUBMISSION_READINESS_CHECKLIST.md)
4. [SUPERVISOR_HANDOFF_SUMMARY_2026-04-09.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/audit/SUPERVISOR_HANDOFF_SUMMARY_2026-04-09.md)

## Structure Goal

The full report should cover:

1. motivation and problem framing
2. related work
3. requirements and scope
4. system architecture
5. data and protocol
6. model design and training
7. calibration and alert policy
8. implementation and refactoring
9. experiments and results
10. testing, audit, and verification
11. discussion
12. limitations
13. future work
14. conclusion
15. appendices

## Expansion Sources

Use these sources to expand beyond the paper line:

- `docs/project_targets/*`
- `docs/reports/audit/*`
- `docs/reports/readiness/*`
- `docs/reports/runbooks/*`
- active config files under `configs/ops/*`
- active evidence artifacts under `artifacts/reports/*`

## What the Full Report Adds

- fuller literature review
- system requirements and design rationale
- expanded implementation detail
- testing and audit chapters
- fuller deployment discussion
- reproducibility appendix
- auditability and freeze-state appendix

## Quality Standard

The quality bar for the full report is intentionally higher than the local sample-report baseline. The school samples are useful for understanding expected report length, chapter spread, and appendix-heavy structure, but they are not the writing-quality ceiling for this project.

The full report should therefore satisfy the following rules:

1. Every chapter must have a clear argumentative job.
   Each major chapter should answer a specific technical or methodological question, define a boundary, or support a research question. Chapters should not exist only to "cover" a topic.

2. Engineering sections must explain decisions, not just describe components.
   When discussing frontend, backend, pipeline, or deployment design, the report should explain why a design was chosen, what alternatives were implicitly rejected, what trade-offs were accepted, and how those choices affect the evidence base.

3. Results sections must analyse rather than merely report.
   Tables and figures should be introduced, interpreted, and bounded. The report should explain why the TCN is stronger under the frozen protocol, why transfer is asymmetric, why the uncertainty-aware path did not improve the bounded replay matrix, and why replay remains system evidence rather than benchmark evidence.

4. Discussion and limitation sections must be sharp rather than defensive.
   The report should actively define what cannot be claimed, which conclusions are strong, which are merely suggestive, and where the current evidence runs out.

5. Appendix material must carry research value.
   Appendices should support auditability, reproducibility, artifact traceability, and deeper technical context. They must not be used as a dumping ground for unintegrated notes.

6. Length must come from analytical depth, not repetition.
   Expanding to full-report scale should come from richer protocol explanation, system rationale, implementation detail, verification, and appendices. It should not come from repeating the same result statements in multiple chapters.

7. Language must remain controlled and non-promotional.
   The report should avoid product-style framing, inflated novelty language, or branding-driven phrasing. It should read like a careful technical investigation.

## How to Use Sample Reports

The university sample reports may be used for:

- understanding realistic full-report length
- observing common chapter spread
- checking where long reports typically place implementation, testing, and appendices
- calibrating expectations for figure/table density

The sample reports must not be used as:

- writing-quality targets
- evidence sources
- templates for technical claims
- justification for weak analysis, filler narrative, or business-plan padding

The correct use of those samples is structural only. The full report for this project should aim higher in evidence discipline, analytical clarity, and methodological rigor.

## What Must Stay the Same

- frozen evidence hierarchy
- `CAUCAFall` as the primary benchmark/deployment target
- `LE2i` as comparative generalisation evidence
- `MUVIM` as supporting exploratory evidence
- `CAUCAFall + TCN + OP-2` as the preferred live demo preset

## Risks to Avoid

- turning replay evidence into benchmark evidence
- letting full-report length create unsupported claims
- copying audit prose directly into the main narrative without integrating it
- overloading the main chapters with appendix-level detail
- drifting into undergraduate-style descriptive writing when analytical writing is needed
- using sample-report structure as an excuse for low-density argumentation
- padding the report with business-style or market-style sections that do not serve the technical research story

## Exit Criteria

- structurally appropriate for an 80-100 page report
- consistent with the compact paper line
- every long-form chapter still resolves to live artifacts
- appendices support, rather than replace, the main narrative
- the report reads as a research-grade technical report rather than a long project diary
