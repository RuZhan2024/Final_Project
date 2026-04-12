# Paper Writing Master Plan

Date: 2026-04-09  
Project: Pose-Based Fall Detection System  
Purpose: consolidate the active report-writing plan, task list, draft control rules, and evidence entrypoints into one working document.

## 1. Primary Control Files

Use these three files as the main paper-writing control surface.

1. [PAPER_FINAL_2026-04-11.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/drafts/PAPER_FINAL_2026-04-11.md)
   - current paper draft
   - primary compact-paper editing target
2. [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)
   - claim-to-artifact map
   - reproducibility commands
   - active figure pack and status
3. [PAPER_SUBMISSION_READINESS_CHECKLIST.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_SUBMISSION_READINESS_CHECKLIST.md)
   - final paper completion gate
   - submission-facing open items

If these files disagree:
- prefer the current draft for wording state
- prefer the thesis evidence map for artifact truth
- prefer this master plan for chapter scope and writing order

## 2. Writing Objective

The report should present the project as:

- a deployment-aware pose-based fall-detection system study
- with a controlled offline comparison between TCN and custom GCN
- with alert-policy calibration treated as part of the method
- with deployment, replay, and limited realtime evidence kept methodologically separate from offline model evidence

The report should not present the project as:

- a claim of universal real-world fall-detection robustness
- a clinically validated system
- a replay-tuned system whose deployment settings substitute for formal unseen-test evidence

## 3. Locked Narrative

The report should be written around three locked research questions.

1. RQ1: Comparative Offline Performance
   - how TCN and GCN compare under the frozen offline protocol
2. RQ2: Calibration and Operational Alerting
   - how operating-point fitting and alert policy turn scores into decisions
3. RQ3: Deployment Feasibility and Runtime Limits
   - what replay and bounded realtime evidence show about practical use and current limits

Use this narrative split throughout:

- offline evaluation answers model-comparison questions
- calibration and operating-point fitting answer alert-policy questions
- replay and limited realtime answer deployment/runtime questions

## 4. Active Evidence Hierarchy

### Tier A: Main report evidence

Use directly in the report body.

- [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md)
- [stability_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/stability_summary.csv)
- [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md)
- [significance_summary.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/significance_summary.json)
- [CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md)
- [cross_dataset_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_summary.csv)
- [DEPLOYMENT_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_LOCK.md)
- [deployment_lock_validation.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_lock_validation.md)
- [FOUR_VIDEO_DELIVERY_PROFILE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md)
- [online_mc_replay_matrix_20260402.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/online_mc_replay_matrix_20260402.csv)
- [DEPLOYMENT_FIELD_VALIDATION.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md)
- [deployment_field_validation_summary.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_field_validation_summary.md)

### Tier B: Methods and implementation evidence

Use to explain method and system design.

- [tcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/tcn_caucafall.yaml)
- [gcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/gcn_caucafall.yaml)
- [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py)
- [common.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/deploy/common.py)
- [server/routes/monitor.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/routes/monitor.py)
- [server/notifications/manager.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/server/notifications/manager.py)

### Tier C: Supporting-only or cautionary evidence

Do not let these become the main results layer.

- replay-debug and diagnostic artifacts in `artifacts/reports/archive/` and `artifacts/reports/diagnostic/`
- archive planning docs under `docs/project_targets/archive/`
- supporting documents under `docs/project_targets/supporting/`
- exploratory `MUVIM` outputs

## 5. Active Figure Pack

Primary report figures:

- [offline_stability_comparison.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/offline_stability_comparison.png)
- [cross_dataset_transfer_summary.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/cross_dataset_transfer_summary.png)
- [system_architecture_diagram.svg](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/system_architecture_diagram.svg)
- [alert_policy_flow.svg](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/alert_policy_flow.svg)
- [online_replay_accuracy_heatmap.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/online_replay_accuracy_heatmap.png)
- [online_mc_dropout_delta.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/online_mc_dropout_delta.png)

Diagnostic-only figure:

- [le2i_per_clip_outcome_heatmap.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/diagnostic/le2i_per_clip_outcome_heatmap.png)

Rule:
- do not use diagnostic-only figures as final results figures without explicit diagnostic framing

## 6. Writing Order

Write and revise in this order.

1. Evidence control refresh
   - confirm every section still points to live artifacts
2. Results and discussion first
   - these constrain the rest of the wording
3. Methods and architecture
   - only after result boundaries are fixed
4. Introduction and abstract
   - last major prose pass
5. Limitations and future work
   - final honesty check
6. Final formatting, tables, captions, and appendix pointers

## 7. Section Execution Plan

### Phase 1: Result lock

Target sections:
- Section 9 Results
- Section 10 Discussion
- Section 11 Limitations

Required evidence:
- offline stability/significance
- cross-dataset summary
- replay/runtime matrix
- field-validation summary

Exit criteria:
- every quantitative statement maps to a live artifact
- no stale figure paths remain
- replay evidence is clearly labeled as deployment/runtime evidence

### Phase 2: Method and system lock

Target sections:
- Section 4 System Architecture
- Section 5 Data and Experimental Protocol
- Section 6 Model Design
- Section 7 Calibration and Alert Policy
- Section 8 Implementation and Refactoring

Required evidence:
- configs
- training/eval/deploy code
- system diagrams
- runbooks and deployment lock docs

Exit criteria:
- architecture reflects current implementation
- runtime preset is described as `CAUCAFall + TCN + OP-2`
- Telegram notification path is described accurately

### Phase 3: Framing and polish

Target sections:
- Abstract
- Introduction
- Background and Related Work
- Conclusion
- figure captions
- table notes

Exit criteria:
- framing matches evidence hierarchy
- no over-claiming language remains
- bounded realtime/deployment phrasing is consistent

## 8. Chapter-Level Writing Tasks

Use this as the active per-section execution sheet.

### Section 4: System Architecture

Objective:
- explain the actual responsibility split across frontend, backend, persistence, and notification paths

Tasks:
- describe frontend, backend, database, monitor flow, event persistence, and notification path
- explain local/on-device path versus cloud deployment path
- use one architecture figure only
- explain why replay and realtime are distinct runtime paths

Primary evidence:
- [system_architecture_diagram.svg](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/system_architecture_diagram.svg)
- [DEPLOYMENT_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_LOCK.md)
- [DELIVERY_ALIGNMENT_STATUS.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md)

Open caution:
- the architecture description must match the current code, not an idealised design

### Section 5: Data and Experimental Protocol

Objective:
- describe dataset roles, split policy, window contract, and evidence separation clearly

Tasks:
- state the role of `CAUCAFall`, `LE2i`, and `MUVIM`
- describe train/validation/test or equivalent locked split policy
- explain how windows are generated
- state what data was used for calibration and operating-point fitting
- explicitly separate formal evaluation data, replay clips, and field/deployment evidence

Primary evidence:
- [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md)
- [LOCKED_PARAMS_RUNBOOK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/LOCKED_PARAMS_RUNBOOK.md)
- [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)

Open caution:
- do not let replay tuning read like independent test evaluation
- do not let the `MUVIM` track read as co-equal primary evidence

### Section 6: Model Design

Objective:
- explain the actual TCN and custom GCN paths accurately

Tasks:
- describe TCN architecture
- describe the custom GCN architecture precisely
- document feature channels and preprocessing assumptions
- explain deployment-time windowed inference

Primary evidence:
- [models.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/models.py)
- [train_tcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_tcn.py)
- [train_gcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_gcn.py)
- current `configs/ops/*.yaml`

Open caution:
- describe the current graph model as a custom spatio-temporal GCN baseline unless a stricter claim is justified

### Section 7: Calibration and Alert Policy

Objective:
- explain how model outputs become alert decisions

Tasks:
- describe validation-side temperature scaling
- explain operating-point fitting
- explain `OP-1 / OP-2 / OP-3`
- explain EMA, `k/n`, cooldown, and confirmation logic
- explain why single-window probability is not the final alert decision

Primary evidence:
- [Compute_Threshold.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/Compute_Threshold.md)
- [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py)
- [alert_policy_flow.svg](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/alert_policy_flow.svg)
- relevant `configs/ops/*.yaml`

Open caution:
- distinguish calibration used in operating-point fitting from runtime alert policy using fitted thresholds

### Section 8: Implementation and Refactoring

Objective:
- justify the final software architecture as maintainable and traceable

Tasks:
- explain why refactoring became necessary
- summarise frontend modularisation
- summarise backend route/service/repository separation
- explain why the monitor path required focused restructuring
- keep this as architecture rationale, not a commit log

Primary evidence:
- current refactored codebase
- [READINESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/readiness/READINESS_REPORT.md)
- [REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md)

### Section 9: Results

Objective:
- present results in clearly separated evidence layers

Subsections:
- 9.1 offline model results
- 9.2 cross-dataset results
- 9.3 deployment and runtime results

Tasks:
- lock exact figures and tables before interpretation text
- define headline metrics per subsection
- report both strengths and failure modes

Primary evidence:
- [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md)
- [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md)
- [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md)
- [CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md)
- [PLOT_EVIDENCE_CHECKLIST.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PLOT_EVIDENCE_CHECKLIST.md)

Headline metrics policy:
- 9.1 offline model results:
  - keep headline metrics small and policy-relevant
  - use event-level or policy-relevant `F1 / Recall / Precision / AP`
- 9.2 cross-dataset results:
  - focus on transfer degradation and failure boundary, not isolated score values
- 9.3 deployment and runtime results:
  - focus on latency, alert delay, runtime consistency, and deployment feasibility
  - do not write this subsection as a substitute classifier benchmark

Open caution:
- do not mix offline model evidence with tuned replay evidence in the same claim layer

### Section 10: Discussion

Objective:
- interpret the results rather than repeat them

Tasks:
- explain TCN versus GCN differences
- explain cross-dataset asymmetry
- explain why latency and pose quality affect alert outcomes
- discuss the role of operating-point calibration in practical alerting
- connect findings back to each research question

Primary evidence:
- section 9 outputs
- [READINESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/readiness/READINESS_REPORT.md)
- supporting robustness and latency material if needed

### Section 11: Limitations

Objective:
- present the limits clearly and professionally

Tasks:
- document limited realtime validation
- document dependence on frontend pose quality
- document deployment sensitivity to runtime latency
- document cross-dataset generalisation limits
- document replay-specific tuning history only if needed for honest framing

Primary evidence:
- [PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md)
- [READINESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/readiness/READINESS_REPORT.md)

### Section 12: Future Work

Objective:
- propose a short, credible forward path

Tasks:
- keep to 3 to 5 concrete next steps
- include stronger realtime validation, broader field validation, stronger domain generalisation, and clearer runtime uncertainty handling only if directly relevant

### Section 13: Conclusion

Objective:
- close with precise, bounded takeaways

Tasks:
- restate the problem
- restate what was achieved
- restate what evidence supports
- state what remains unresolved

### Section 14: Appendices

Candidate appendix contents:
- config snapshots
- extra tables
- additional operating-point details
- runtime diagnostics
- replay/deployment notes
- reproducibility notes

## 9. Evidence Control Tasks

Before finalising report text:

1. Verify every reported metric against an artifact already on disk.
2. Verify every included figure has:
   - a source script or artifact path
   - a clear caption
   - a sentence in the main text interpreting it
3. Verify every major claim maps to:
   - one research question
   - one section
   - one artifact or evidence document
4. Remove any sentence that depends on:
   - untracked numbers
   - remembered but undocumented experiments
   - replay-only tuned results presented as general evaluation

## 10. Safe Wording Rules

Use:
- `trends stronger`
- `bounded replay validation`
- `limited realtime validation`
- `deployment evidence`
- `comparative evidence`
- `asymmetric transfer`

Avoid:
- `proves real-world effectiveness`
- `clinically reliable`
- `universally robust`
- `fully validated deployment`
- branding language such as `AI-powered`

For notification wording:
- use `generated summary` or `provider-backed summary`
- do not frame the notification path as a branded feature

## 11. Current Known Boundaries

These must stay visible in the report.

1. TCN-vs-GCN comparison is informative but statistically cautious at the current seed budget.
2. Cross-dataset transfer is asymmetric and functions as a limitation boundary.
3. Replay evidence is valid system evidence, but not unseen-test model evidence.
4. The current uncertainty-aware replay path did not improve the bounded 24-clip matrix.
5. Field/realtime evidence remains limited.
6. The current implemented notification path is Telegram-first, with other channels deferred.

## 12. Practical Next Writing Steps

Immediate sequence:

1. refresh Section 9 against current live figures and summaries
2. refresh Section 7 wording for alert policy and Telegram notification path
3. refresh Section 10 and Section 11 so the limitations match the current evidence
4. rebuild the report
5. run one final wording audit for over-claiming

## 13. Build and Verification Commands

Build:

```bash
./scripts/build_report.sh --pdf-only
```

Lightweight verification:

```bash
make release-check
./scripts/run_canonical_tests.sh torch-free
./scripts/run_canonical_tests.sh frontend
```

Figure regeneration:

```bash
python3 scripts/build_cross_dataset_summary.py --manifest artifacts/reports/cross_dataset_manifest.json --out_csv artifacts/reports/cross_dataset_summary.csv
python3 scripts/plot_cross_dataset_transfer.py --summary_csv artifacts/reports/cross_dataset_summary.csv --out_fig artifacts/figures/report/cross_dataset_transfer_summary.png
python3 scripts/generate_report_figures.py
```

## 14. Final Rule

If draft wording and evidence disagree, change the draft, not the evidence story.
