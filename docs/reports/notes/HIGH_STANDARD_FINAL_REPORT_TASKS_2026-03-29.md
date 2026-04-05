# High-Standard Final Project Report Tasks

Date: 2026-03-29  
Project: Pose-Based Fall Detection System  
Scope: Final year project report written to a high academic standard, closer to a research dissertation than a routine coursework report.

## 1. Purpose

This document breaks the report work into explicit tasks so the writing process stays disciplined, evidence-linked, and defensible.

The report must:

- present a clear research problem
- separate model claims from system/deployment claims
- distinguish formal evaluation from demo/deployment evidence
- keep all claims tied to reproducible artifacts already present in the repository
- remain conservative where evidence is still limited

This is not a generic writing checklist. It is a project-specific report plan.

## 2. Writing Standard

The report should be written to a high standard with the following expectations:

- clear research questions
- justified methodology
- explicit evaluation protocol
- careful claim boundaries
- honest limitations
- professional structure and figure/table integration
- no unsupported performance claims
- no mixing of tuned demo behavior with formal unseen-test evaluation

## 3. Non-Negotiable Reporting Rules

These rules must be followed throughout the report:

1. Formal offline evaluation, replay deployment validation, and limited realtime validation must be reported as separate evidence layers.
2. Replay clip tuning or replay-only deployment adjustments must never be presented as unseen-test generalization evidence.
3. Temperature scaling / calibration must be described accurately:
   - used in validation-side operating-point fitting
   - not overstated as guaranteed runtime probability calibration unless explicitly implemented
4. TCN vs GCN conclusions must be tied to the controlled experimental protocol, not to anecdotal demo behavior.
5. Cloud deployment findings must be framed as system/deployment analysis, not as direct model-quality evidence.
6. Claims about realtime effectiveness must stay limited unless supported by dedicated realtime validation evidence.
7. Every important number in the report must map to a reproducible artifact, config, or report file already tracked by the project.
8. `MUVIM` work may be acknowledged as a secondary exploratory track, but it must not displace the primary `CAUCAFall` / `LE2i` evidence hierarchy.

## 4. Target Report Structure

The planned report structure is:

1. Introduction
2. Background and Related Work
3. Research Questions and Scope
4. System Architecture
5. Data and Experimental Protocol
6. Model Design
7. Calibration and Alert Policy
8. Implementation and Refactoring
9. Results
10. Discussion
11. Limitations
12. Future Work
13. Conclusion
14. Appendices

## 4A. Locked Research Questions

The report will be organised around exactly three research questions.

### RQ1. Comparative Offline Performance

Under the locked offline evaluation protocol, how do the TCN and the custom spatio-temporal GCN compare on the primary fall-detection task?

Purpose:
- establish the main controlled model-comparison result
- keep the core model claim tied to formal offline evidence

Primary evidence type:
- offline evaluation only

Primary evidence scope:
- in-domain controlled evaluation on the primary dataset
- comparative model behaviour under fixed protocol conditions

### RQ2. Calibration and Operational Alerting

How does validation-side operating-point calibration influence the conversion of window-level model outputs into practical alert decisions?

Purpose:
- justify operating-point fitting as a first-class part of the system design
- explain how probabilities become operational alerts

Primary evidence type:
- fit_ops outputs
- operating-point configurations
- alert-policy results

Primary evidence scope:
- temperature scaling and operating-point fitting on validation data
- alert-policy behaviour under fitted operating points

### RQ3. Deployment Feasibility and Runtime Limits

What do replay deployment evidence and limited realtime validation show about the practical feasibility and current runtime limits of the system?

Purpose:
- present the engineering/runtime story without overstating generalisation
- connect deployment findings to system reliability rather than model superiority

Primary evidence type:
- replay deployment validation
- runtime latency analysis
- limited realtime/on-device validation

Primary evidence scope:
- local/on-device monitoring behaviour
- cloud deployment/runtime behaviour
- practical latency and pipeline constraints

Constraint:
- deployment/runtime evidence must not be presented as substitute offline model evidence

## 4B. Locked Terminology

The following terms must be used consistently throughout the report.

### Offline Evaluation

Controlled evaluation under the locked experimental protocol using formal train/validation/test or equivalent fixed evaluation splits.

### Replay Validation

System-level validation using fixed replay clips through the monitor pipeline. This is deployment evidence, not formal unseen-test evidence unless explicitly defined as such by the protocol.

### Limited Realtime Validation

Small-scale live/on-device validation used to assess practical runtime feasibility. This is not a claim of broad real-world effectiveness.

### Calibration

Validation-side temperature scaling used during operating-point fitting. This term must not be used loosely to imply guaranteed runtime trustworthiness.

### Operating Point

A fitted threshold/policy profile, such as OP-1 / OP-2 / OP-3, selected from validation-side analysis.

### Alert Policy

The operational decision layer that combines probabilities, thresholds, EMA, `k/n`, cooldown, and confirmation logic into alert states.

### Model Evidence

Evidence used to support claims about the comparative behaviour of TCN and GCN under the formal protocol.

### Deployment Evidence

Evidence used to support claims about system/runtime behaviour in replay or limited realtime conditions. This does not directly prove model superiority.

## 4C. Chapter-Claim-Artifact Map

This table is the working control sheet for the report narrative. A major claim should not appear in the report unless it is represented here.

| Claim | Research Question | Report Section | Evidence Artifact(s) | Planned Figure/Table |
| --- | --- | --- | --- | --- |
| Under the frozen primary-dataset protocol, the final CAUCAFall TCN candidate trends stronger than the matched CAUCAFall GCN candidate, but the comparative conclusion should remain statistically cautious at the current seed budget. | RQ1 | Results 9.1, Discussion 10 | [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md), [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md), [stability_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/stability_summary.csv), [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md), [significance_summary.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/significance_summary.json) | Main comparison table; stability figure if needed |
| Cross-dataset transfer is asymmetric and should be interpreted as a limitation boundary rather than evidence of universal robustness. | RQ1, RQ3 | Results 9.2, Discussion 10, Limitations 11 | [CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md), [cross_dataset_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_summary.csv), [cross_dataset_transfer_bars.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png) | Cross-dataset transfer figure |
| Validation-side operating-point calibration is a substantive part of the system because alert behaviour depends on fitted operating points rather than raw single-window probabilities alone. | RQ2 | Sections 7 and 9.1, Discussion 10 | [OPS_POLICY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/OPS_POLICY_REPORT.md), [Compute_Threshold.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/Compute_Threshold.md), `src/fall_detection/core/calibration.py`, `src/fall_detection/evaluation/fit_ops.py`, [tcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/tcn_caucafall.yaml), [gcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/gcn_caucafall.yaml) | OP/policy summary table |
| The deployable alert path is grounded in the CAUCAFall TCN OP-2 runtime configuration rather than an arbitrary or undocumented threshold profile. | RQ2, RQ3 | Sections 7, 8, and 9.3 | [tcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/tcn_caucafall.yaml), [DEPLOYMENT_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_LOCK.md), [README.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/README.md) | Deployment configuration table |
| The replay/deployment path provides valid system evidence, but tuned replay behaviour must be framed as deployment/demo calibration rather than unseen-test model generalisation. | RQ3 | Sections 5, 9.3, and 11 | [REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md), [DEPLOYMENT_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_LOCK.md), [REPLAY_LIVE_ACCEPTANCE_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/REPLAY_LIVE_ACCEPTANCE_LOCK.md), [deployment_lock_validation.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_lock_validation.md) | Deployment validation table |
| The system has a real deployment path and demonstrable software artifact, but field/realtime evidence remains bounded and should not be overstated. | RQ3 | Results 9.3, Limitations 11 | [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md), [DEPLOYMENT_FIELD_VALIDATION.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md), [FIELD_VALIDATION_MINIMUM_PACK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FIELD_VALIDATION_MINIMUM_PACK.md), [deployment_field_validation_summary.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_field_validation_summary.md) | Field/deployment evidence summary table |
| Runtime latency and path-specific pipeline behaviour matter to practical alerting performance and belong in the system discussion, not the pure model-comparison claim. | RQ3 | Results 9.3, Discussion 10 | [LATENCY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/LATENCY_REPORT.md), runtime logs and deployment diagnostics referenced from [REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md), [READINESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/readiness/READINESS_REPORT.md) | Latency/runtime table or figure |

Rule:
- no major claim may appear in the report unless it is represented in this mapping table and linked to a tracked evidence artifact

## 5. Section-by-Section Tasks

### 5.1 Introduction

Objective:
- define the problem, motivation, system goal, and project contribution clearly

Tasks:
- write the practical motivation for fall detection
- justify the choice of pose-based vision over alternative formulations
- explain why deployment-aware evaluation matters, not just raw classification accuracy
- state the project scope precisely
- draft 3 to 4 contributions only

Evidence inputs:
- [OBJECTIVES_EVIDENCE_OUTCOMES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/OBJECTIVES_EVIDENCE_OUTCOMES.md)
- [CLAIM_TABLE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CLAIM_TABLE.md)
- [PROJECT_FINAL_YEAR_EXECUTION_PLAN.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PROJECT_FINAL_YEAR_EXECUTION_PLAN.md)

Open caution:
- avoid claiming universal robustness
- avoid presenting the project as clinically validated

### 5.2 Background and Related Work

Objective:
- place the work in the context of fall detection, pose-based methods, temporal sequence models, graph-based methods, and deployment-aware alerting

Tasks:
- summarise wearable vs vision vs skeleton-based approaches
- summarise why TCN and GCN are relevant for pose sequence modelling
- position calibration / operating-point selection as part of an alerting system, not just a classifier
- identify the gap this project addresses

Evidence inputs:
- literature review material outside repo if available
- [RESEARCH_QUESTIONS_MAPPING.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md)
- [PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md)

Open caution:
- this section must synthesise, not just list papers

### 5.3 Research Questions and Scope

Objective:
- define exactly what the project answers and what it does not answer

Tasks:
- adopt the locked three-question structure above unless a concrete evidence gap forces revision
- map each research question to:
  - metrics
  - evidence source
  - report section
- write explicit scope exclusions

Evidence inputs:
- [RESEARCH_QUESTIONS_MAPPING.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md)
- [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)
- [FYP_SUPERVISOR_CHECKLIST.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/audit/FYP_SUPERVISOR_CHECKLIST.md)

Open caution:
- scope must explicitly separate:
  - offline evaluation
  - deployment replay analysis
  - limited realtime/on-device validation

### 5.4 System Architecture

Objective:
- explain how the system is built and where responsibilities lie

Tasks:
- describe frontend, backend, database, monitor flow, event persistence, and notification path
- explain local/on-device path vs cloud deployment path
- produce or select one architecture figure
- explain why replay and realtime are distinct runtime paths

Evidence inputs:
- codebase architecture after refactor
- [DEPLOYMENT_DEFAULT_PROFILE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_DEFAULT_PROFILE.md)
- [DELIVERY_ALIGNMENT_STATUS.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md)
- [REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md)

Open caution:
- system diagram must align with actual code, not idealised architecture

### 5.5 Data and Experimental Protocol

Objective:
- describe datasets, splits, protocol locking, and evidence separation clearly

Tasks:
- list datasets used and their roles
- describe train/validation/test or equivalent locked split policy
- explain how windows are generated
- state what data was used for calibration and operating-point fitting
- state explicitly that `MUVIM` is a secondary exploratory track rather than a primary result-bearing dataset
- explicitly separate:
  - formal evaluation data
  - replay demo clips
  - field/deployment evidence

Evidence inputs:
- dataset docs and config files
- [DEPLOYMENT_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_LOCK.md)
- [LOCKED_PARAMS_RUNBOOK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/LOCKED_PARAMS_RUNBOOK.md)
- [EXPERIMENT_RECORDING_PROTOCOL.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/EXPERIMENT_RECORDING_PROTOCOL.md)

Open caution:
- this section must prevent any impression that demo clip tuning equals independent test evaluation
- this section must also prevent any impression that the `MUVIM` track is co-equal with the locked primary comparative protocol

### 5.6 Model Design

Objective:
- explain the actual model architectures and feature pipeline accurately

Tasks:
- describe TCN architecture
- describe the custom GCN architecture precisely
- document feature channels and preprocessing
- explain deployment-time windowed inference

Evidence inputs:
- `src/fall_detection/core/models.py`
- training scripts
- current deploy configs

Open caution:
- do not call the current GCN a strict ST-GCN implementation unless that is actually true
- describe it as a custom spatio-temporal GCN baseline if needed

### 5.7 Calibration and Alert Policy

Objective:
- explain how model outputs become operational alert decisions

Tasks:
- describe validation-side temperature scaling
- explain operating-point fitting
- explain OP-1 / OP-2 / OP-3
- explain EMA, `k/n`, cooldown, confirm logic
- explain why single-window probability is not the final alert decision

Evidence inputs:
- [OPS_POLICY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/OPS_POLICY_REPORT.md)
- [Compute_Threshold.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/Compute_Threshold.md)
- `src/fall_detection/core/calibration.py`
- `src/fall_detection/evaluation/fit_ops.py`
- relevant `configs/ops/*.yaml`

Open caution:
- wording must distinguish:
  - calibration used in OP fitting
  - runtime alert policy using fitted operating points

### 5.8 Implementation and Refactoring

Objective:
- justify the final software architecture and show engineering maturity

Tasks:
- explain why refactoring became necessary
- summarise frontend feature modularisation
- summarise backend route/service/repository separation
- explain why the monitor path required focused restructuring
- keep this as architecture rationale, not a commit log

Evidence inputs:
- current refactored codebase
- [REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md)
- readiness / audit notes if needed

Open caution:
- keep the section concise and relevant to maintainability, traceability, and correctness

### 5.9 Results

Objective:
- present results in a layered and defensible way

Subsections:
- 9.1 Offline model results
- 9.2 Cross-dataset results
- 9.3 Deployment and runtime results

Tasks:
- identify the exact figures and tables to include
- define and lock headline metrics before drafting interpretation text
- prepare clean captions and in-text interpretation
- report both strengths and failure modes

Evidence inputs:
- [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md)
- [CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md)
- [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md)
- [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md)
- [LATENCY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/LATENCY_REPORT.md)
- [PLOT_SELECTION_FOR_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PLOT_SELECTION_FOR_REPORT.md)
- [PLOT_EVIDENCE_CHECKLIST.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PLOT_EVIDENCE_CHECKLIST.md)

Open caution:
- do not mix offline model evidence with tuned replay demo evidence in the same claim layer

Locked headline metrics policy:

- `9.1 Offline model results`
  - headline metrics should be a small set only, chosen to reflect the actual project objective
  - likely candidates: event-level or policy-relevant F1 / Recall / Precision / AP, depending on the locked evidence pack
- `9.2 Cross-dataset results`
  - headline focus should be transfer degradation and failure boundary, not just isolated score values
- `9.3 Deployment and runtime results`
  - headline focus should be latency, alert delay, runtime consistency, and deployment feasibility
  - this subsection must not be written as a substitute classifier benchmark

### 5.10 Discussion

Objective:
- interpret the results instead of merely repeating them

Tasks:
- explain TCN vs GCN differences
- explain cross-dataset asymmetry
- explain why deployment latency and pose quality affect alert outcomes
- discuss the role of operating-point calibration in practical alerting
- connect findings back to each research question

Evidence inputs:
- results section outputs
- [ROBUSTNESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/ROBUSTNESS_REPORT.md)
- [READINESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/readiness/READINESS_REPORT.md)

Open caution:
- this section must not oversell weak or limited evidence

### 5.11 Limitations

Objective:
- present limits clearly and professionally

Tasks:
- document limited realtime validation
- document dependence on frontend pose quality
- document deployment sensitivity to runtime latency
- document cross-dataset generalisation limits
- document the status of replay-specific tuning if referenced

Evidence inputs:
- [PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md)
- [READINESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/readiness/READINESS_REPORT.md)
- [FYP_SUPERVISOR_CHECKLIST.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/audit/FYP_SUPERVISOR_CHECKLIST.md)

Open caution:
- write this as a serious scientific limitations section, not as an apology

### 5.12 Future Work

Objective:
- propose a short, credible forward path

Tasks:
- list 3 to 5 concrete next steps only
- include:
  - stronger realtime validation
  - broader deployment/field validation
  - stronger domain generalisation work
  - possible graph-model extensions
  - clearer runtime uncertainty handling

Open caution:
- future work must be focused, not a wish list

### 5.13 Conclusion

Objective:
- close the report with precise, defensible takeaways

Tasks:
- restate the problem
- restate what was achieved
- restate what evidence supports
- state what remains unresolved

Open caution:
- do not end with exaggerated product claims

### 5.14 Appendices

Objective:
- keep the main report readable while preserving technical depth

Candidate appendix contents:
- detailed config snapshots
- extra tables
- additional operating-point details
- runtime diagnostics
- replay/deployment notes
- additional figures
- reproducibility notes

## 6. Evidence Control Tasks

Before finalising the report text:

1. Verify every reported metric against an artifact already on disk.
2. Verify every figure included in the report has:
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
   - replay-only tuned demo results presented as general evaluation

## 7. Writing Workflow

Recommended order:

1. Finalise report structure
2. Lock the three research questions
3. Complete the chapter-claim-artifact map
4. Draft Research Questions and Scope
5. Draft Introduction
6. Draft System Architecture
7. Draft Data and Experimental Protocol
8. Draft Model Design
9. Draft Calibration and Alert Policy
10. Draft Results
11. Draft Discussion
12. Draft Limitations
13. Draft Conclusion
14. Assemble appendices
15. Final consistency pass

## 8. Immediate Next Tasks

These are the next writing actions after this document is approved:

1. Confirm the locked three research questions
2. Fill the chapter-claim-artifact map with actual artifact paths
3. Write `Research Questions and Scope`
4. Write `Introduction`
5. Write `System Architecture`

## 9. Decision Log To Preserve During Writing

The following decisions must remain stable unless explicitly revised:

- The project is primarily an on-device / local monitoring system with cloud deployment analysed as an extension.
- Replay and realtime are separate evidence paths.
- Replay-only tuning, if discussed, must be framed as deployment/demo calibration.
- Calibration is part of the operating-point fitting pipeline.
- Runtime deployment claims must stay conservative where direct evidence is limited.
- The report should prioritise defensible claims over aggressive framing.

## 10. Completion Criteria For The Report Draft

The report drafting phase can be considered structurally complete only when:

- every chapter above has at least a first full draft
- every major claim is linked to evidence
- no key result depends on undocumented experiments
- limitations are explicitly written
- tuned replay evidence is clearly separated from formal evaluation
- the narrative from research question to conclusion is traceable and consistent
