# Paper Submission Week Plan

Date: 2026-03-22

Purpose:
Convert the six blocking submission items in `PAPER_SUBMISSION_READINESS_CHECKLIST.md` into a one-week execution plan with concrete outputs.

Planning rule:
- This plan assumes the goal is to reach "submission-ready internal freeze", not full camera-ready polish.
- If any day slips, preserve priority order rather than trying to do everything at once.

## Weekly Objective

By the end of the week, the project should have:
- one frozen paper protocol
- one final bounded claim set
- one closed evidence map for all paper figures/tables
- one resolved position on the TCN-vs-GCN statistics story
- one completed field-validation evidence bundle
- one recorded clean reproducibility check

## Day 1: Freeze Protocol and Scope

Primary goal:
- Stop protocol drift.

Tasks:
- Freeze one final candidate pair per dataset and one primary paper scope.
- Freeze the final seed list used for all statistical summaries.
- Freeze metric definitions and the exact role of each dataset:
  - `CAUCAFall`: primary dataset
  - `LE2i`: comparative/generalization dataset
- Update all source-of-truth docs so they agree.

Files to update:
- [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md)
- [CLAIM_TABLE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CLAIM_TABLE.md)
- [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)
- [RESEARCH_QUESTIONS_MAPPING.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md)

End-of-day acceptance:
- One protocol date/version is declared.
- No doc implies a broader claim than the others.

## Day 2: Resolve the Statistics Story

Primary goal:
- Remove ambiguity around the main comparison claim.

Decision gate:
- Choose one of these two routes and record it explicitly.

Route A:
- increase seeds for the final TCN-vs-GCN comparison
- recompute stability and significance artifacts

Route B:
- keep `n=5`
- narrow the paper claim from "TCN is superior" to "TCN trends stronger under the locked protocol, but evidence remains exploratory"

Chosen route:
- `Route B`
- The paper will keep the current `n=5` and use cautious directional wording rather than a stronger significance claim.

Files and artifacts:
- [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md)
- [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md)
- [stability_summary.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/stability_summary.json)
- [significance_summary.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/significance_summary.json)

End-of-day acceptance:
- The main claim language matches the available statistics.
- No remaining paper text requires a stronger significance conclusion than the artifacts support.
- Status: `Closed by wording decision`

## Day 3: Close the Field-Validation Bundle

Primary goal:
- Turn deployment validation from "framework exists" into actual evidence.

Tasks:
- Record or finalize the target field clips.
- Fill the observation CSV.
- Run the field summarization scripts.
- Generate the final markdown summary and failure JSON.

Files and artifacts:
- [DEPLOYMENT_FIELD_VALIDATION.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md)
- `artifacts/reports/deployment_field_observations.csv`
- `artifacts/reports/deployment_field_eval.json`
- `artifacts/reports/deployment_field_failures.json`
- `artifacts/reports/deployment_field_validation_summary.md`

End-of-day acceptance:
- Field-validation artifacts exist and are cited in the evidence map.
- The paper can state exactly what was tested, what passed, and what remains limited.

## Day 4: Close Evidence Mapping for Every Final Number

Primary goal:
- Make every final paper artifact traceable.

Tasks:
- Audit each final table and figure intended for the paper.
- Ensure each has:
  - an artifact file
  - a reproduce command
  - a bounded interpretation
- Remove any stale entries or references to legacy profiles.

Files to update:
- [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)
- [OBJECTIVES_EVIDENCE_OUTCOMES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/supporting/OBJECTIVES_EVIDENCE_OUTCOMES.md)
- [PLOT_EVIDENCE_CHECKLIST.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PLOT_EVIDENCE_CHECKLIST.md)

End-of-day acceptance:
- Every final figure/table in the draft appears in the evidence map.
- No figure/table in the evidence map is uncited or stale.

## Day 5: Write Final Claims and Limitations

Primary goal:
- Replace implicit positioning with explicit paper-ready language.

Tasks:
- Lock 3-5 bounded contribution claims.
- Write one consolidated limitations section covering:
  - small-sample statistical power
  - cross-dataset asymmetry
  - incomplete universality of deployment claims
  - field sample size and environment sensitivity
- Ensure the abstract, introduction, and discussion all use the same claim boundaries.

Files to update:
- [CLAIM_TABLE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CLAIM_TABLE.md)
- [PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md)
- paper draft or thesis draft outside this repo, if applicable

End-of-day acceptance:
- The paper no longer over-claims universal robustness or definitive superiority.
- Limitations are explicit and reviewer-safe.

## Day 6: Run a Clean Reproducibility Pass

Primary goal:
- Verify the submission path from an external-review perspective.

Tasks:
- Perform one clean-machine or clean-user run of the recommended demo/smoke path.
- Record:
  - exact environment
  - commands used
  - what worked
  - what failed
  - any required manual fixes

Files to update:
- [FINAL_SUBMISSION_CHECKLIST.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md)
- [READINESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/readiness/READINESS_REPORT.md)
- add one small release note or dry-run note under `artifacts/reports/`

End-of-day acceptance:
- One external-review path is documented as passing.
- Known environment caveats are written down instead of left implicit.

## Day 7: Final Go/No-Go Review

Primary goal:
- Decide honestly whether the paper should be submitted now.

Review checklist:
- Is one frozen protocol declared everywhere?
- Is the TCN-vs-GCN claim statistically aligned with the evidence?
- Is field validation complete enough to support the deployment story?
- Does every final table/figure have an artifact and reproduce command?
- Has one clean reproducibility pass been recorded?
- Are contribution claims and limitations consistent across abstract, results, and discussion?

Decision outcomes:
- `Go`: all blocking items closed
- `Conditional go`: one minor documentation-only blocker remains
- `No go`: any scientific or reproducibility blocker remains

## Suggested Command Backbone

Use these as the minimal command families during the week:

```bash
python tools/run_stability_manifest.py --manifest artifacts/registry/stability_manifest.csv --start_status todo --stop_on_fail 1
python scripts/plot_stability_metrics.py --glob "outputs/metrics/*_stb_s*.json" --out_fig artifacts/figures/stability/fc_stability_boxplot.png
python3 scripts/build_cross_dataset_summary.py --manifest artifacts/reports/cross_dataset_manifest.json --out_csv artifacts/reports/cross_dataset_summary.csv
python3 scripts/plot_cross_dataset_transfer.py --summary_csv artifacts/reports/cross_dataset_summary.csv --out_fig artifacts/figures/report/cross_dataset_transfer_summary.png
python tools/summarize_dual_policy_events.py --resident_id 1 --hours 24 --out_json artifacts/reports/deployment_dual_policy_events.json
python tools/summarize_field_validation.py --obs_csv artifacts/reports/deployment_field_observations.csv --hours 1.0 --dual_policy_json artifacts/reports/deployment_dual_policy_events.json --out_eval_json artifacts/reports/deployment_field_eval.json --out_failures_json artifacts/reports/deployment_field_failures.json --out_markdown artifacts/reports/deployment_field_validation_summary.md
```

## Expected End State

If the week succeeds, the project should be described as:
- a submission-ready paper package with bounded claims

If the week stalls before the blocking items close, keep the official description as:
- a strong dissertation/FYP project
- a paper draft with promising evidence, but not yet final-submission safe
