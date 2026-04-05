# High-Standard Final Report Evidence Inventory

Date: 2026-03-29  
Purpose: identify the repo artifacts that are suitable for direct use in the final report, separate them from supporting-only material, and record current evidence gaps before drafting the report body.

## 1. Evidence Selection Rules

An artifact is suitable for direct report use only if it satisfies all of the following:

- it is already tracked in the repository
- it corresponds to the current frozen protocol or current delivery/deployment path
- its interpretation is clear and bounded
- it does not rely on undocumented ad hoc tuning

Artifacts that are useful for orientation, debugging, or historical context may still be listed here, but they should be marked `supporting only` or `do not cite directly`.

## 2. Primary Report Evidence

These are the main evidence files currently suitable for direct report use.

### 2.1 Claim and Protocol Control

- [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md)
  - role: locked candidate roots and candidate interpretation rules
- [CLAIM_TABLE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CLAIM_TABLE.md)
  - role: bounded paper-safe claims
- [RESEARCH_QUESTIONS_MAPPING.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md)
  - role: metrics and section mapping
- [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)
  - role: high-level evidence-to-claim mapping

### 2.2 Offline Comparative Evidence

- [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md)
- [stability_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/stability_summary.csv)
- [stability_summary.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/stability_summary.json)
- [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md)
- [significance_summary.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/significance_summary.json)

Use:
- RQ1
- main offline TCN-vs-GCN comparison
- stability and statistical caution framing

### 2.3 Cross-Dataset Evidence

- [CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md)
- [cross_dataset_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_summary.csv)
- [cross_dataset_transfer_bars.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png)

Use:
- RQ1 limitation boundary
- discussion of generalisation limits

### 2.4 Deployment and Replay Evidence

- [DEPLOYMENT_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_LOCK.md)
- [deployment_lock_validation.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_lock_validation.md)
- [REPLAY_LIVE_ACCEPTANCE_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/REPLAY_LIVE_ACCEPTANCE_LOCK.md)
- [FOUR_VIDEO_DELIVERY_PROFILE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md)
- [mc_replay_matrix_20260401.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/mc_replay_matrix_20260401.csv)
- [mc_replay_matrix_20260401.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/mc_replay_matrix_20260401.json)
- [REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/REPORT_RELEVANT_CHANGE_SUMMARY_2026-03-28.md)

Use:
- RQ3 deployment/replay narrative
- bounded deployment usefulness
- explanation of replay-vs-runtime differences
- bounded uncertainty-gate evaluation

Constraint:
- these files support deployment evidence only
- they must not be presented as unseen-test model evidence
- the MC replay matrix is supporting evidence for runtime interpretation, not for accuracy-improvement claims

### 2.5 Field and Limited Realtime Evidence

- [DEPLOYMENT_FIELD_VALIDATION.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md)
- [FIELD_VALIDATION_MINIMUM_PACK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FIELD_VALIDATION_MINIMUM_PACK.md)
- [deployment_field_eval.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_field_eval.json)
- [deployment_field_failures.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_field_failures.json)
- [deployment_field_validation_summary.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_field_validation_summary.md)

Use:
- bounded field/realtime validation discussion
- limitations section

Constraint:
- current field evidence is still bounded
- use only for limited feasibility/supporting claims

### 2.6 Calibration and Alert-Policy Evidence

- `src/fall_detection/core/calibration.py`
- `src/fall_detection/evaluation/fit_ops.py`
- [tcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/tcn_caucafall.yaml)
- [gcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/gcn_caucafall.yaml)
- [Compute_Threshold.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/Compute_Threshold.md)

Use:
- RQ2
- methods section on calibration and operating-point fitting
- explanation of alert policy

Constraint:
- this supports the statement that calibration is part of the operating-point fitting pipeline
- it should not be overstated as explicit runtime probability calibration unless runtime code is changed accordingly

### 2.7 Architecture and Implementation Evidence

- [README.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/README.md)
- [READINESS_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/readiness/READINESS_REPORT.md)
- current refactored frontend and backend codebase

Use:
- system overview
- implementation/refactoring section
- reproducibility context

## 3. Primary Figures Currently Suitable for Report Use

- [offline_stability_comparison.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/offline_stability_comparison.png)
  - role: report-ready offline comparison figure
- [cross_dataset_transfer_summary.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/cross_dataset_transfer_summary.png)
  - role: report-ready cross-dataset limitation figure
- [system_architecture_diagram.svg](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/system_architecture_diagram.svg)
  - role: report-ready system architecture figure
- [alert_policy_flow.svg](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/alert_policy_flow.svg)
  - role: report-ready alert-policy figure

## 4. Supporting-Only or Use-With-Caution Evidence

These files are useful, but should not be treated as primary final-report evidence without qualification.

### 4.1 Supporting Only

- [PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md)
  - good wording source, not primary evidence
- [OBJECTIVES_EVIDENCE_OUTCOMES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/OBJECTIVES_EVIDENCE_OUTCOMES.md)
  - useful framing summary, not a metrics source
- [PAPER_SECTION_HEADINGS.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_SECTION_HEADINGS.md)
  - structure aid only

### 4.2 Do Not Cite Directly As Main Evidence Without Refresh

- [OPS_POLICY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/OPS_POLICY_REPORT.md)
  - current file explicitly says OP summaries should be recomputed before use
- removed legacy analysis figures under the old `artifacts/figures/pr_curves`, `artifacts/figures/latency`, `artifacts/figures/stability`, and `artifacts/figures/cross_dataset` output paths
  - replaced by the report-specific figure pack under `artifacts/figures/report/`
- broad historical tuning logs under `artifacts/reports/tuning/`
  - useful for internal recall of the tuning process
  - not suitable as headline final-report evidence unless a specific artifact is explicitly promoted
- `artifacts/replay_eval/*.json`
  - useful for replay-policy debugging history
  - not suitable as formal unseen-test evidence

### 4.3 Secondary Exploratory Track: MUVIM

The repository also contains a real `MUVIM` experiment track. It should be acknowledged in the report, but it should not be elevated into the primary result narrative.

Primary supporting artifacts:

- [README.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/README.md)
  - explicitly classifies MUVIM configs as a separate experiment track
- [CONFIG_RESULT_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/runbooks/CONFIG_RESULT_EVIDENCE_MAP.md)
  - records the existence of the MUVIM track and its artifact family
- [muvim_metric_contract_fix.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/tuning/muvim_metric_contract_fix.md)
  - documents the metric-contract correction that materially changed how MUVIM event metrics should be interpreted
- [muvim_r2_summary.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/tuning/muvim_r2_summary.md)
  - concise corrected post-fix operating-point summary
- [muvim_r3_summary.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/tuning/muvim_r3_summary.md)
  - training-side exploration that was later rejected

Safe report role:

- acknowledge MUVIM as a secondary exploratory dataset track
- use it to show that the project did broader experimentation beyond the main `CAUCAFall` / `LE2i` protocol
- use it, if needed, to support discussion about metric-contract discipline and the distinction between score quality and alert-policy behaviour

Constraint:

- do not present MUVIM as part of the primary locked comparative result
- do not let older pre-fix notes override the corrected post-fix summaries
- do not use MUVIM to weaken the report's main evidence hierarchy

## 5. Current Evidence Gaps

These are the main gaps that still need cautious wording in the report.

1. Field/realtime closure is incomplete.
   - bounded evidence exists
   - broad real-world deployment closure does not

2. Statistical comparison remains exploratory at the current frozen `n=5`.
   - this affects how strongly TCN-vs-GCN can be claimed

3. Cross-dataset transfer does not support universal robustness.
   - this must remain a limitation statement

4. Replay-specific deployment tuning exists in project history.
   - if mentioned, it must be framed as deployment/demo calibration only

5. The refactored live uncertainty gate does not currently show bounded replay gains.
   - this should be reported as a negative or neutral finding, not omitted
   - it supports cautious runtime interpretation rather than an enhancement claim

## 6. Safe Use Summary

Most defensible primary narrative:

- end-to-end deployment-oriented software artifact exists
- TCN trends stronger than the custom GCN under the locked primary-dataset protocol
- this comparative conclusion must remain cautious
- operating-point calibration is central to the alerting design
- cross-dataset transfer is asymmetric and bounded
- deployment evidence supports practical usefulness in replay/demo settings
- the current uncertainty-aware live gate is methodologically relevant but does not improve the bounded custom replay matrix
- field/realtime validation remains limited and should be framed conservatively
