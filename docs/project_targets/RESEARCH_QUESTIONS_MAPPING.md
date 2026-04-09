# Research Questions, Metrics, Evidence, and Result Mapping

Purpose:
Provide a direct mapping from research questions to metrics, available evidence, and the dissertation or paper sections that should answer them.

Protocol Freeze:
- `Paper Protocol Freeze v1`
- Primary dataset: `CAUCAFall`
- Comparative/generalization dataset: `LE2i`
- Frozen seed set for the current final-candidate comparison:
  - `1337`, `17`, `2025`, `33724876`, `42`
- Final candidate roots are locked by [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md)

## Mapping Table

| Research Question | What Must Be Measured | Primary Metrics | Current Evidence | Expected Results Section | Current Status |
|---|---|---|---|---|---|
| RQ1. Can a pose-based pipeline detect falls effectively on the primary benchmark dataset? | In-domain fall-detection performance on `CAUCAFall` under the frozen protocol | `F1`, `Recall`, `Precision`, `FA24h`, `AP` | [PAPER_PUBLICATION_READINESS_PLAN.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md), [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md), [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md), [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md) | `Results -> In-Domain Benchmark Results on CAUCAFall` | Mostly Answered |
| RQ2. Which model family is more suitable for this task: TCN or GCN? | Controlled comparison of the two architectures on matched data, features, and evaluation rules under the frozen final-candidate protocol | `F1`, `Recall`, `Precision`, `AP`, `FA24h` | [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md), [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md), [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md) | `Results -> Comparison of TCN and GCN Performance` and `Discussion -> Interpretation of the Main Findings` | Mostly Answered |
| RQ3. How much does operating-point calibration affect practical alerting behaviour? | Performance under different alert-policy profiles fitted on validation data only | `Recall`, `Precision`, `F1`, `FA24h`, `delay_p50`, `delay_p95` | [PROJECT_FINAL_YEAR_EXECUTION_PLAN.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PROJECT_FINAL_YEAR_EXECUTION_PLAN.md), [PAPER_PUBLICATION_READINESS_PLAN.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md) | `Results -> Precision, Recall, and False-Alert Trade-Offs` | Partially Answered |
| RQ4. Do the models generalize across datasets, or are they strongly domain-dependent? | Bidirectional cross-dataset transfer with fixed invariants and comparison to in-domain performance | `F1`, `Recall`, `AP`, `FA24h`, cross-domain deltas | [CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md) | `Results -> Cross-Dataset Generalization Results` and `Discussion -> Cross-Dataset and Deployment Implications` | Answered |
| RQ5. Are the reported results stable across random seeds? | Run-to-run variability and confidence bounds on final candidates | `mean`, `std`, `95% CI` for `F1`, `Recall`, `Precision`, `FA24h`, optional `AP` | [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md) | `Results -> Multi-Seed Stability and Statistical Results` | Answered |
| RQ6. Are observed differences between final candidates statistically meaningful? | Paired significance testing under the final candidate protocol | Wilcoxon `p` values, paired differences, effect sizes | [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md) | `Results -> Multi-Seed Stability and Statistical Results` and `Discussion -> Comparison with Prior Literature` | Mostly Answered |
| RQ7. Does the system remain useful when evaluated in a deployment-oriented setting? | Replay or field-validation evidence beyond benchmark test performance, with `CAUCAFall` remaining the main deployment-facing dataset | `FA/day`, `FA24h`, `delay_p50`, `delay_p95`, failure counts, event recall proxy | [DEPLOYMENT_FIELD_VALIDATION.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md), [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md) | `Results -> Deployment Validation Results` and `Discussion -> Practical Trade-Offs for Real-World Alerting` | Partially Answered |
| RQ8. Is the project reproducible enough for external review and defense? | Traceability from reported claims to commands, configs, and artifacts | existence of manifests, reports, reproduce commands, locked profiles | [PROJECT_FINAL_YEAR_EXECUTION_PLAN.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PROJECT_FINAL_YEAR_EXECUTION_PLAN.md), [READINESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/readiness/READINESS_REPORT.md), [OBJECTIVES_EVIDENCE_OUTCOMES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/supporting/OBJECTIVES_EVIDENCE_OUTCOMES.md) | `Methods -> Reproducibility and Artifact Tracking` and `Appendix -> Training and Evaluation Commands` | Mostly Answered |

## Recommended Use

Use this table:
- at the end of the introduction after listing the research questions
- at the start of the evaluation/results chapter
- in viva or supervisor review materials to show that each question has evidence

## Writing Guidance

- Each results subsection should explicitly state which research question it answers.
- Each discussion subsection should interpret the answer rather than restating raw metrics.
- If a question is only partially answered, say so directly and explain why.
- Do not convert a partial answer into a full claim without the missing evidence.

## Remaining Gaps to Close

1. RQ3 needs a cleaner final write-up showing how operating-point choices change practical alert behaviour.
2. RQ7 remains partial until field-validation artifacts are completed.
3. RQ8 will be stronger once a clean reproducibility pass is recorded against the frozen paper protocol.
