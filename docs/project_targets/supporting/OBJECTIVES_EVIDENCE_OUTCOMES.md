# Objectives, Evidence, and Outcomes

Purpose:
Provide a concise examiner-facing table that shows how each major project objective was addressed, what evidence exists, the current outcome, and what still needs to be closed.

## Summary Table

| Objective | Success Measure | Current Evidence | Current Outcome | Status | Next Action |
|---|---|---|---|---|---|
| Build an end-to-end pose-based fall detection pipeline | Working pipeline from data preparation to model evaluation and runtime integration | `README.md`, `docs/reports/readiness/READINESS_REPORT.md` | End-to-end pipeline exists across preprocessing, training, evaluation, backend, and frontend | Done | Keep final report aligned to the implemented pipeline only |
| Compare temporal and graph-based models for fall detection | Report comparable TCN and GCN results on the same protocol | `docs/project_targets/FINAL_CANDIDATES.md`, `docs/project_targets/SIGNIFICANCE_REPORT.md` | TCN and GCN have both been trained and compared; TCN currently trends stronger on final candidate evidence | Done | Present the comparison conservatively because statistical power remains limited |
| Achieve strong primary-dataset performance on CAUCAFall | Meet or approach locked targets for `Recall`, `F1`, `Precision`, and `FA24h` | `docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md`, `docs/project_targets/STABILITY_REPORT.md`, `docs/project_targets/SIGNIFICANCE_REPORT.md` | Primary dataset performance is strong enough to support the main technical story, with CAUCAFall serving as the deployment target dataset | Mostly Done | Final dissertation should quote only locked metrics and clearly distinguish target, minimum acceptable, and observed values |
| Evaluate generalization across datasets | Bidirectional transfer results with fixed invariants and transparent performance drop | `docs/project_targets/CROSS_DATASET_REPORT.md` | Cross-dataset evaluation is completed and shows generalization limits rather than robust universal transfer | Done | Emphasize the limitations of transfer rather than over-claiming robustness |
| Quantify stability of final candidates | 5-seed summaries with mean, std, and 95% CI | `docs/project_targets/STABILITY_REPORT.md` | Multi-seed stability evidence exists for all final candidates | Done | Use stability tables directly in the final evaluation/results chapter |
| Apply statistically defensible comparison methods | Pre-registered or limited-hypothesis significance tests with interpretation | `docs/project_targets/SIGNIFICANCE_REPORT.md` | Wilcoxon-based paired comparisons were carried out; results are informative but remain exploratory at `n=5` | Mostly Done | Keep statistical claims cautious and mention power limitations |
| Calibrate model outputs for operational alerting | Validation-only operating-point fitting with defined recall/balanced/false-alert profiles | `docs/project_targets/PROJECT_FINAL_YEAR_EXECUTION_PLAN.md`, `docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md` | Alert-policy calibration is part of the project design and a distinctive part of the system contribution | Done | Make this explicit in both methods and discussion |
| Evaluate deployment-oriented behaviour, not benchmark metrics alone | Replay or field evidence including false-alert behaviour, delay, and failure modes | `docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md`, `docs/project_targets/FIELD_VALIDATION_MINIMUM_PACK.md`, `artifacts/reports/deployment_field_validation_summary.md` | The reporting path and sample artifacts exist, but the current field set is still too small for final paper closure | Partial | Replace the sample-level field set with the intended `20-40` clip minimum pack and then update the evidence map row |
| Ensure reproducibility of reported results | Reproduce commands, manifests, artifact paths, and frozen protocol | `docs/project_targets/PROJECT_FINAL_YEAR_EXECUTION_PLAN.md`, `docs/reports/readiness/READINESS_REPORT.md` | Reproducibility is a clear project priority and is implemented well, but some stale claim/evidence placeholders remain | Mostly Done | Refresh stale claim/evidence files and verify all cited artifacts exist |
| Produce a dissertation/paper-ready evaluation package | Clear IMRaD structure, evidence-backed claims, limitations, and future work | `docs/project_targets/archive/planning/PAPER_SECTION_HEADINGS.md`, `docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md` | The structure and evidence plan are strong, but narrative closure on ethics, reflection, and field evidence is still incomplete | Partial | Add ethics/privacy, reflection, and final objective-to-result mapping into the dissertation |

## Suggested Use in Dissertation

Use this table near the start of the evaluation chapter or near the end of the introduction.

Recommended framing:
- introduce the table as a compact summary of how project aims were assessed
- use it to show that evaluation was planned against measurable outcomes
- refer back to it when concluding whether the project met its objectives

## Notes for Final Version

- Replace broad phrases such as `strong enough` with exact values once the final locked results table is frozen.
- Do not claim field-validation completion until the required artifacts are actually complete.
- Treat the current field-validation artifacts as sample-level workflow evidence unless the clip count is raised to the intended minimum.
- Keep the primary-dataset and comparative-dataset roles explicit throughout.
