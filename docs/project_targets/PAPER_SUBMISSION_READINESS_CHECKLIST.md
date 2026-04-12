# Paper Submission Readiness Checklist

Date: 2026-03-22

Purpose:
Turn the current "strong dissertation project, partial publication readiness" state into a concrete submission gate.

Bottom line:
- Current state: strong thesis/FYP readiness
- Current state: near paper-ready, but not yet safe to submit without further closure
- This checklist is the decision gate for moving from internal review to paper submission

## Decision Rule

Use these labels:
- `Must close before submission`: blocking item
- `Should close if time allows`: non-blocking, but materially strengthens the paper
- `Nice to have`: useful polish, not a submission blocker

Recommended submission rule:
- Do not submit while any `Must close before submission` item remains open.
- If more than two `Should close if time allows` items remain open, position the paper as internal draft only.

## Must Close Before Submission

| Item | Why It Blocks Submission | Current Evidence | Required Closure | Status |
|---|---|---|---|---|
| Freeze one final paper protocol | A paper cannot mix evolving profiles, datasets, and claim scopes | [PAPER_PUBLICATION_READINESS_PLAN.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md), [CLAIM_TABLE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CLAIM_TABLE.md), [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md) | Declare one frozen candidate set, one seed set, one metric policy, and one claim scope used by every final table/figure | Open |
| Strengthen statistical closure on the main comparison | Current architecture-comparison result is promising, but primary Wilcoxon results remain exploratory at `n=5` | [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md), [significance_summary.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/significance_summary.json) | Adopt the frozen Day 2 decision: keep `n=5` and explicitly frame the paper as a benchmark/deployment study without a strong superiority claim | Closed by wording decision |
| Close deployment/field-validation evidence | Without this, the real-world usefulness story remains incomplete | [DEPLOYMENT_FIELD_VALIDATION.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md), [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md) | Produce the pending field-validation artifacts and add the final row to the evidence map | Open |
| Lock contribution claims to what the artifacts actually support | Over-claiming is the fastest way to weaken a submission | [CLAIM_TABLE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CLAIM_TABLE.md), [RESEARCH_QUESTIONS_MAPPING.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md) | Final paper must claim: working deployment-oriented system, cautious TCN-vs-GCN comparison, asymmetric transfer limits, and bounded delivery evidence; do not claim universal robustness or definitive significance | Partially closed |
| Ensure every final number has an artifact and reproduce command | Reviewers and supervisors need traceability from claim to file to command | [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md), [OBJECTIVES_EVIDENCE_OUTCOMES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/supporting/OBJECTIVES_EVIDENCE_OUTCOMES.md) | Every figure/table used in the paper must appear in the evidence map with artifact path and reproduce command | Partially closed |
| Verify the paper package on a clean reproducible path | A submission is weak if the environment story remains fragile | [READINESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/readiness/READINESS_REPORT.md), [FINAL_SUBMISSION_CHECKLIST.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md) | Run one clean-machine or clean-user validation and record the result as part of the release evidence | Open |

## Should Close If Time Allows

| Item | Why It Matters | Current Evidence | Recommended Action | Status |
|---|---|---|---|---|
| Add a concise novelty-vs-baseline paragraph set | A paper needs positioning, not just results | [PAPER_PUBLICATION_READINESS_PLAN.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md) | Write 3-5 bounded contribution statements with nearest-baseline comparison | Open |
| Turn operating-point calibration into a clearer paper result | This is one of the more distinctive system contributions | [RESEARCH_QUESTIONS_MAPPING.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md), [OPS_POLICY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/supporting/OPS_POLICY_REPORT.md) | Add one compact result table or figure showing how OP choices shift recall, precision, FA24h, and delay behavior | Open |
| Consolidate limitations into one formal section | The repo is honest about limitations, but the paper should centralize them | [CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md), [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md), [DEPLOYMENT_FIELD_VALIDATION.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md) | Create one limitations subsection covering small-`n` stats, transfer asymmetry, field sample size, and environment sensitivity | Open |
| Stabilize the runtime verification story | Current local verification still depends on environment specifics | [README.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/README.md) | Document one exact smoke path that works end to end with pinned env assumptions | Open |

## Nice To Have

| Item | Why It Helps | Recommended Action | Status |
|---|---|---|---|
| Add one focused ablation table | Helps reviewers understand where gains come from | Keep it small: architecture, feature set, or confirm-stage policy only | Optional |
| Add one failure-case figure panel | Makes the discussion more concrete | Include 3-5 representative misses or false-alert cases | Optional |
| Add one release-style artifact bundle check for paper figures | Reduces final-week drift | Verify the cited figure/table files all exist before submission | Optional |

## Recommended Paper Positioning

If submitted after the must-close items are done, the paper should be framed as:
- a deployment-oriented pose-based fall-detection system study
- with a controlled TCN-vs-GCN comparison
- with explicit operating-point calibration
- with honest cross-dataset limitations
- and with bounded real-world validation evidence

The paper should not be framed as:
- definitive proof that TCN is universally superior to GCN
- a universal cross-dataset generalization result
- a fully closed real-world clinical or home deployment study

## Fast Go/No-Go Gate

You can treat the project as submission-ready only if all of the following are true:

- one frozen paper protocol is declared and used everywhere
- the main comparison claim is either statistically strengthened or carefully narrowed
- field-validation artifacts are complete
- every final figure/table is mapped to a file and reproduce command
- one clean reproducibility check is recorded

If any of those are false, keep the status as:
- strong dissertation/FYP project
- internal paper draft, not final submission
