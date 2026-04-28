Date: 2026-04-28  
Purpose: filter the full ML pipeline into what is genuinely worth promoting in the full report, what should stay supporting-only, and what should not consume final-report space.

# ML Pipeline Value Filter

## 1. Use Rule

This document is not an inventory of every experiment in the repository.

It is a filter for deciding which ML-facing work materially strengthens the final report.

The standard is:

1. does this work explain an important technical decision?
2. does it materially support a defended result?
3. does it improve the marker’s confidence that the team understood model, policy, and deployment behaviour?
4. does it add insight rather than just proving that many runs were executed?

If the answer is no, the work should stay in appendix, notes, or archive.

## 2. Highest-Value ML Work for the Full Report

These items are worth explicit treatment in the main report body.

### A. Locked Offline Comparative Evaluation

Why it matters:

1. this is the core ML evidence for `RQ1`
2. it supports the `TCN vs custom GCN` comparison under a controlled protocol
3. it gives the report a formal performance backbone that is distinct from replay or demo evidence

What to keep:

1. the defended `CAUCAFall` and `LE2i` offline comparison
2. the distinction between in-domain evidence and cross-dataset evidence
3. the fact that runtime or replay results should not replace offline model evidence

Primary evidence:

1. `outputs/metrics/tcn_caucafall_stb_s*.json`
2. `outputs/metrics/gcn_caucafall_stb_s*.json`
3. `outputs/metrics/tcn_le2i_stb_s*.json`
4. `outputs/metrics/gcn_le2i_stb_s*.json`
5. `docs/reports/notes/WEEK1_RESULTS_AND_FRAMING_LOCK_2026-04-27.md`

Report value:

1. **main text**

### B. Operating-Point Fitting and Calibration Work

Why it matters:

1. this is one of the strongest methodological parts of the whole project
2. it shows that the team did not stop at raw classifier scores
3. it explains how offline outputs were converted into alert-worthy runtime policies

What to keep:

1. temperature-calibrated operating-point fitting
2. the role of `OP1 / OP2 / OP3`
3. the reason multi-window persistence is more realistic than single-window triggering
4. the separation between offline sweep quality and deployable alert policy

Primary evidence:

1. `ops/configs/ops/tcn_caucafall.yaml`
2. `ops/configs/ops/gcn_caucafall.yaml`
3. `ops/configs/ops/tcn_le2i.yaml`
4. `ops/configs/ops/gcn_le2i.yaml`
5. `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`
6. `ml/src/fall_detection/evaluation/metrics_eval.py`

Report value:

1. **main text**

### C. Online Ops Refit and Runtime-Recovery Sequence

Why it matters:

1. this is not generic tuning; it is deployment-path debugging and alignment
2. it explains why replay/runtime behaviour changed over time
3. it supports the report’s deployment-oriented contribution more strongly than extra model-family exploration would

What to keep:

1. baseline online replay failure state
2. direct-window gate fix
3. motion-support fix
4. online ops refit
5. why these were system-path repairs rather than mere threshold changes

Primary evidence:

1. `artifacts/ops_reverify_20260315/online_replay_summary.json`
2. `artifacts/ops_reverify_20260315_after_gatefix/online_replay_summary.json`
3. `artifacts/ops_reverify_20260315_after_motionfix/online_replay_summary.json`
4. `artifacts/online_ops_fit_20260315/`
5. `artifacts/online_ops_fit_20260315_verify/final_summary_v3.json`
6. `docs/reports/runbooks/CONFIG_RESULT_EVIDENCE_MAP.md`
7. `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`

Report value:

1. **main text**

### D. Cross-Dataset Transfer and Failure-Boundary Work

Why it matters:

1. it prevents overclaiming from in-domain results
2. it gives the report a mature limitation boundary
3. it turns “generalisation is weaker” into a concrete failure-mode discussion

What to keep:

1. cross-dataset transfer results as bounded evidence
2. the difference between `TCN` missed-fall collapse and `GCN` recall-through-false-alert trade-off
3. the fact that this is limitation evidence, not a hidden win

Primary evidence:

1. `outputs/metrics/cross_tcn_caucafall_r2_train_hneg_to_le2i_frozen_20260409.json`
2. `outputs/metrics/cross_gcn_caucafall_r2_recallpush_b_to_le2i_frozen_20260409.json`
3. `docs/reports/notes/TARGETED_FAILURE_ANALYSIS_2026-04-27.md`
4. `docs/reports/notes/cross_dataset_summary_2026-04-27.csv`

Report value:

1. **main text**

### E. Targeted Retraining Strengthening

Why it matters:

1. this is real project-strengthening work, not cosmetic rewriting
2. it shows targeted response to an identified weakness
3. it produced bounded runtime improvement that can be defended honestly

What to keep:

1. why the team stayed with `CAUCAFall + TCN`
2. why the retraining target was missed-fall reduction rather than architecture replacement
3. `Candidate A` as the lead strengthening result
4. `Candidate D` as corroboration rather than the new winner
5. the modest but real bounded runtime improvement from `13/24 -> 16/24` and `15/24 -> 16/24`
6. the caveat that both post-fit profiles still required `confirm-disabled` fallback

Primary evidence:

1. `docs/reports/notes/CAUCAFALL_TCN_OFFLINE_STRENGTHENING_PLAN_2026-04-27.md`
2. `docs/reports/notes/CANDIDATE_A_RUNTIME_EVAL_2026-04-27.md`
3. `docs/reports/notes/CANDIDATE_A_LOCKED_SURFACE_EVAL_2026-04-27.md`
4. `docs/reports/notes/CANDIDATE_D_RUNTIME_EVAL_2026-04-27.md`
5. `outputs/metrics/tcn_caucafall_rtA_hneg_union_recall.json`
6. `outputs/metrics/tcn_caucafall_rtD_hneg_plus_continue.json`

Report value:

1. **main text**

### F. Hard-Negative Mining as a Deliberate Optimisation Mechanism

Why it matters:

1. it shows that later training iterations were guided by identified difficult cases
2. it helps explain why the retraining track was plausible rather than random
3. it is a better optimisation story than simply saying “more tuning was done”

What to keep:

1. hard-negative lists existed and were promoted into later training families
2. the goal was to reduce specific missed-fall or confusing boundary cases
3. this optimisation was linked to bounded runtime weaknesses, not only to offline score chasing

Primary evidence:

1. `outputs/hardneg/tcn_caucafall_targeted_train_union.txt`
2. `outputs/hardneg/tcn_caucafall_trainmix_r2.txt`
3. `outputs/hardneg/tcn_caucafall_evalmix_p50.txt`
4. `outputs/caucafall_tcn_W48S12_r2_train_hneg/train_config.json`
5. `outputs/caucafall_tcn_W48S12_rtA_hneg_union_recall/train_config.json`

Report value:

1. **main text if short**
2. **appendix if expanded**

## 3. Medium-Value ML Work

These are worth keeping, but with less narrative weight.

### G. Stability / Multi-Seed Reliability Track

Why it matters:

1. it strengthens confidence that the main offline comparison is not a one-run accident
2. it supports a higher-standard report tone

What to keep:

1. only the high-level stability conclusion
2. not the full per-run diary

Primary evidence:

1. `outputs/metrics/tcn_caucafall_stb_s17.json`
2. `outputs/metrics/tcn_caucafall_stb_s2025.json`
3. `outputs/metrics/gcn_caucafall_stb_s17.json`
4. `outputs/metrics/tcn_le2i_stb_s42.json`
5. `outputs/metrics/gcn_le2i_stb_s42.json`

Report value:

1. **short main-text mention**
2. **appendix/supporting detail**

### H. Representative TCN and GCN Optimisation Families

Why it matters:

1. it proves the model comparison was not a single-shot baseline versus baseline contest
2. it shows that each family received meaningful optimisation effort

What to keep:

1. one short paragraph stating that both families were iteratively tuned
2. one or two representative examples per family
3. not the whole archive of family names

Representative evidence:

1. `outputs/caucafall_tcn_W48S12_r1_ctrl/`
2. `outputs/caucafall_tcn_W48S12_r2_train_hneg/`
3. `outputs/caucafall_tcn_W48S12_r2_train_hneg_plus/`
4. `outputs/caucafall_gcn_W48S12_r1_ctrl/`
5. `outputs/caucafall_gcn_W48S12_r1_recovery/`
6. `outputs/caucafall_gcn_W48S12_r2_recallpush_a/`
7. `outputs/caucafall_gcn_W48S12_r2_recallpush_b/`

Report value:

1. **main text if compressed**
2. **appendix if detailed**

### I. MUVIM as an Exploratory Side Track

Why it matters:

1. it shows breadth and additional experimentation
2. it can demonstrate that the team explored beyond the final defended line

What to keep:

1. only a short note that MUVIM existed as an exploratory or secondary track
2. keep it out of the main results hierarchy unless the report explicitly needs it

Primary evidence:

1. `outputs/muvim_tcn_W48S12*/`
2. `outputs/muvim_gcn_W48S12*/`
3. `ops/configs/ops/archive/muvim/`
4. `outputs/metrics/tcn_muvim*.json`
5. `outputs/metrics/gcn_muvim*.json`

Report value:

1. **appendix or brief supporting paragraph**

## 4. Low-Value or Non-Promotable ML Work

These should not consume main-report space.

### J. Full Sweep Logs and Per-Threshold Archives

Why not:

1. they are useful for reproducibility
2. but they do not add much interpretive value by themselves
3. they can quickly turn the report into an experiment diary

Examples:

1. `*.sweep.json`
2. large archived ops grids
3. detailed threshold-by-threshold diagnostic dumps

Report value:

1. **appendix only**

### K. Every Historical Candidate Name

Why not:

1. names such as `opt_m1mask`, `opt_m5_pickupboost`, `opt_m7_pickupfix_cg30`, or broad `smoke_sweep` families are too granular for the main story
2. the marker does not need the full naming archaeology

Report value:

1. **do not promote**

### L. Temporary Metric Dumps and Recheck Files

Why not:

1. they are useful for local validation
2. they are not part of the defended evidence chain

Examples:

1. `_tmp_*`
2. ad hoc recheck JSONs
3. smoke outputs without a promoted conclusion

Report value:

1. **do not promote**

## 5. Recommended Report Allocation

### Put in Main Text

1. locked offline comparison
2. operating-point fitting and calibration
3. online ops refit and runtime-recovery sequence
4. cross-dataset failure boundary
5. targeted retraining strengthening
6. short hard-negative optimisation rationale
7. short stability/reliability statement

### Put in Appendix or Supporting Notes

1. detailed stability tables
2. representative training-family lists
3. MUVIM secondary track detail
4. sweep logs and archived ops scans
5. extra per-candidate tuning history

### Do Not Promote

1. temporary metrics
2. full naming archaeology of experiments
3. unfiltered tuning diary material

## 6. Bottom Line

The ML work that most improves the full report is not “all tuning”.

The high-value story is:

1. a controlled offline comparison
2. a calibration and operating-point layer that turns scores into alert policies
3. an online runtime-recovery and deployment-alignment process
4. a bounded cross-dataset limitation boundary
5. a targeted retraining strengthening result that improved the defended runtime story without pretending to solve deployment

Everything else should support that story, not compete with it.
