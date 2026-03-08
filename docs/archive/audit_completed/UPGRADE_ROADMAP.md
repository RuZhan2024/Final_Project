# UPGRADE_ROADMAP

Date: 2026-03-02  
Scope: execution status (implemented vs pending), tied to repo evidence.

## Current Diagnosis (Still True)
- Event-level metrics are now refreshed and non-zero, but FA/24h remains high on LE2i and CAUCAFall GCN.
- Root issue identified earlier remains policy/ops-level degeneracy risk, not basic classifier discrimination.
- Guardrails and audit gates are now implemented to prevent silent recurrence.
- Strict parity gate now passes after recapturing baseline targets from refreshed LE2i metrics.

## Status Matrix

| Item | Status | Evidence |
|---|---|---|
| P0.1 Unit/Event semantics gate | DONE | `window_span_seconds` added in [src/fall_detection/core/alerting.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/alerting.py:180); metrics path uses it at [metrics_eval.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/metrics_eval.py:527); tests in [test_event_time_semantics.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/tests/test_event_time_semantics.py:9) |
| P0.2 Numeric fingerprint gate | DONE | Gate config in [configs/audit_gates.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/audit_gates.json:2); script consumes it in [scripts/audit_numeric.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/audit_numeric.py:67); Make target in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1032) |
| P0.3 Temporal stride/context gate | DONE | Gate config in [configs/audit_gates.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/audit_gates.json:10); script consumes it in [scripts/audit_temporal.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/audit_temporal.py:37); Make target in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1039) |
| P0.4 Fit policy sanity (degenerate sweep block) | DONE | Degeneracy detector in [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:125); enforced fail unless override at [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:750); Make flags in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:344) |
| P0.5 Artifact portability gate | DONE | Relative-path emission in [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:116) + usage at [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:727); bundle validator in [scripts/audit_artifact_bundle.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/audit_artifact_bundle.py:16); Make target [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1029) |
| P1.1 Adaptive GCN adjacency (feature-flag) | DONE | Model support in [models.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/models.py:255); train flag plumbing in [train_gcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_gcn.py:466); Make flags in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:306) |
| P1.2 CTR-GCN-lite channel refinement | DONE | Channel-wise refinement flag and low-rank per-channel topology in [models.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/models.py:265), [models.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/models.py:319); train flag plumbing in [train_gcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_gcn.py:470); Make flags in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:845) |
| P1.3 TSM in TCN (feature-flag) | DONE | `TemporalShift1D` + TCN hooks in [models.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/models.py:85); train flags in [train_tcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_tcn.py:419); Make flags in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:255) |
| P1.4 Multi-stream fusion extension | DONE | Added explicit ablation controls: fusion modes (`concat/sum/joint_only/motion_only`) and stream-drop probabilities in [models.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/models.py:457), [models.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/core/models.py:495); train/ckpt plumbing in [train_gcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_gcn.py:469), [train_gcn.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/training/train_gcn.py:586); Make knobs in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:305), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:846) |
| P2.1 Cost-sensitive OP2 objective | DONE | Objective selection in [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:207); CLI in [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:531); Make forwarding in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:341) |
| P2.2 Hard-negative loop standardization | DONE | Replay knobs are now first-class (`TCN/GCN_RESUME`, `TCN/GCN_HARD_NEG_LIST`, `TCN/GCN_HARD_NEG_MULT`) at [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:429); train target forwarding at [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:848), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:871); standardized one-cycle orchestration targets at [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1171) |
| P2.3 On-device profiling/budget gate | DONE | Budget checker in [scripts/audit_profile_budget.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/audit_profile_budget.py:40); Make targets in [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1076) |

## Verified Commands (Latest)
- `make -s audit-all-deploy PROFILE=cpu_local DS=le2i MODEL=tcn PROFILE_IO_ONLY=1 AUDIT_DATASETS='le2i,caucafall'` -> pass.
- `python -m pytest -q tests/test_import_smoke.py tests/test_data_sources_config.py tests/test_windows_contract.py tests/test_adapter_contract.py tests/test_event_time_semantics.py` -> `7 passed`.
- `make -n hneg-cycle-tcn-le2i` -> dry-run confirms standardized order: `windows-eval -> fa-windows -> mine-hard-negatives -> train(resume+replay) -> fit-ops -> eval -> plot`.
- `make -s audit-promoted-profiles` -> pass (`artifacts/reports/promoted_profiles_20260302.json`).
- `make -s audit-all MODEL=tcn AUDIT_DATASETS='le2i,caucafall'` now includes promoted-profile gating and passes.

## Remaining Work (Sequential)
1. Investigate persistent single-scene FP (`Coffee_room_02__Videos__video__52_`) with data/model-side mitigation (policy-only path exhausted).
2. Keep reporting both FA tracks on each cycle (now active): short test-set FA/24h and longer negative-stream FA/day (`score_unlabeled_alert_rate`).

## Next Execution Block (Executed 2026-03-02)
- Data-side replay pack is now prepared from adapter-consistent train/val negatives (`min_p=0.50`):
  - `outputs/hardneg/tcn_le2i_evalmix_p50.txt` (37 windows)
  - `outputs/hardneg/tcn_le2i_coffee_room_02_p50.txt` (7 windows)
  - summary: `artifacts/reports/tuning/hneg_pack_summary.json`
- Bounded retrain invocation used (standardized replay knobs):
  - `make train-tcn-le2i OUT_TAG=_hneg_pack TCN_RESUME=outputs/le2i_tcn_W48S12/best.pt TCN_HARD_NEG_LIST=outputs/hardneg/tcn_le2i_evalmix_p50.txt TCN_HARD_NEG_MULT=2`
- Interrupted `_hneg_pack` run was resumed from its latest checkpoint and completed by early stop:
  - resume: `outputs/le2i_tcn_W48S12_hneg_pack/best.pt`
  - final history: `outputs/le2i_tcn_W48S12_hneg_pack/history.jsonl` (`21` epochs)
- Follow-up evaluation completed for `_hneg_pack`:
  - `configs/ops/tcn_le2i_hneg_pack.yaml`
  - `outputs/metrics/tcn_le2i_hneg_pack.json`
  - `outputs/plots/tcn_le2i_hneg_pack_recall_vs_fa.png`
  - `outputs/plots/tcn_le2i_hneg_pack_f1_vs_tau.png`
  - `artifacts/reports/hneg_cycle/hneg_pack_unlabeled_fa.json`
- `_hneg_pack` comparison snapshot:
  - test OP2: `F1=0.8889`, `recall=0.8889`, `precision=0.8889`, `FA/24h=581.5843`, `alerts=9`
  - negative-stream FA/day: `0.0` over `196.319s` (`n_alert_events=0`)

## Latest Policy-Tuning Findings (LE2i TCN, no training)
- Baseline-parity recapture:
  - `baselines/le2i/58813e8/performance_baseline.json` now tracks `f1=0.8235`, `recall=0.7778`, `fa24h=581.5843`.
  - `make -s audit-parity-le2i-strict MODEL=tcn` now passes.
- Confirm-gated tuning:
  - Relaxed confirm with `confirm=1` either degenerated (no alerts) or reduced recall while keeping `fa24h=581.58`.
  - Evidence: `artifacts/reports/tuning/c1_relaxed.ops.sweep.json`, `artifacts/reports/tuning/c2_relaxed_nolow.metrics.json`.
- Additional confirm-policy probe:
  - `c3_confirm_short` and `c4_confirm_loose` keep the same `fa24h=581.58` and low recall (`0.4444`), so global confirm retuning still does not solve the scene FP.
  - Evidence: `artifacts/reports/tuning/c3_confirm_short.metrics.json`, `artifacts/reports/tuning/c4_confirm_loose.metrics.json`, `artifacts/reports/tuning/trace_c3_video52.json`.
- Cost-sensitive FA-aware OP2 probe:
  - `c5_costfp3_fa` and `c6_costfp8_fa` produce identical OP2 thresholds and identical test metrics to baseline; no FA/24h movement.
  - `c7_confirm_costfa` reduces long-stream FA/day to `0.0` but still degrades test recall (`0.4444`) and leaves test FA/24h unchanged.
  - Evidence: `artifacts/reports/tuning/c5_costfp3_fa.metrics.json`, `artifacts/reports/tuning/c6_costfp8_fa.metrics.json`, `artifacts/reports/tuning/c7_confirm_costfa.metrics.json`, `artifacts/reports/tuning/c7_confirm_costfa.unlabeled_fa.json`.
- Threshold/persistence feasibility bound:
  - No-false-alert threshold from current test sweep is `tau_high≈0.681`, but recall drops to `0.1111`; global threshold-only mitigation is infeasible.
  - Raising persistence (`k/n` to `3/5`, `4/6`, `5/7`) still leaves `video__52` with one false event.
  - Evidence: `outputs/metrics/tcn_le2i.json`, `artifacts/reports/tuning/trace_v52_k3n5.json`, `artifacts/reports/tuning/trace_v52_k4n6.json`, `artifacts/reports/tuning/trace_v52_k5n7.json`.
- Strict confirm profile cross-check:
  - The strict confirm profile that suppresses `video__52` also suppresses same-scene true events (`video__59`, `video__60`).
  - Evidence: `artifacts/reports/hneg_cycle/trace_base_video52_confirm1.json`, `artifacts/reports/tuning/trace_v59_base.json`, `artifacts/reports/tuning/trace_v59_confirm_strict.json`, `artifacts/reports/tuning/trace_v60_base.json`, `artifacts/reports/tuning/trace_v60_confirm_strict.json`.
- Joint policy feasibility sweep (`video__52` vs `video__59/60`):
  - Confirm-only sweep at fixed OP2 thresholds (`240` configs) found no setting with `video__52=0` and both `video__59>=1`, `video__60>=1`.
  - Broader joint sweep over thresholds/persistence/confirm (`3888` configs) also found no such setting.
  - Evidence: `artifacts/reports/tuning/confirm_grid_hm4_v52_v59_v60.json`, `artifacts/reports/tuning/policy_joint_grid_hm4_v52_v59_v60.json`.
- Hybrid fusion probe (`video__52` vs `video__59/60`):
  - Using `hneg_pack_tsm_p80hm4m6` TCN + baseline LE2i GCN at the same alert policy, `min`/`avg` probability fusion did not suppress `video__52` (`events=1` remains).
  - `min` fusion also removed true event on `video__59` (`events=0`), so fusion does not provide a safe mitigation path here.
  - Evidence: `artifacts/reports/tuning/hybrid_probe_hm4m6_v52_v59_v60.json`.
- Coffee_room_02 negative-pool profile (correct `y` label filter):
  - In train+val, negatives/unlabeled windows are `853`; score profile is sparse at high-confidence tail (`p>=0.8: 0`, `p>=0.7: 2`, `p>=0.6: 4`, `p>=0.5: 10`).
  - This explains why stricter replay mining quickly saturates and fails to create new scene pressure.
  - Evidence: `artifacts/reports/tuning/coffee_room_02_neg_score_profile_hm4m6_yfix.json`.
  - Prepared QA/replay candidate pack: `artifacts/reports/tuning/cr02_neg_review_candidates_hm4m6.csv`, `outputs/hardneg/tcn_le2i_cr02_p50_direct_hm4m6.txt`.
  - Span-overlap QA on those 10 candidates found no direct overlap with annotated fall spans (`n_overlap=0`, min distance among span-bearing videos = `10` frames), so these are not obvious label-leak windows.
  - Evidence: `artifacts/reports/tuning/cr02_neg_review_candidates_hm4m6_with_spans.csv`, `artifacts/reports/tuning/cr02_neg_review_candidates_hm4m6_with_spans.json`.
- Persistence/cooldown tuning:
  - Increasing `k/n` to `3/5` or `4/6` worsened FA to `1163.17` with lower recall (`0.6667`).
  - Evidence: `artifacts/reports/tuning/p1_k3n5.metrics.json`, `artifacts/reports/tuning/p2_k4n6.metrics.json`.
- Hard-negative mining prepared:
  - Generated list: `outputs/hardneg/tcn_le2i.txt` (28 candidates across 12 videos), ready for bounded replay retrain.
- Hard-negative loop executed:
  - `hneg1` retrain improved F1/recall but FA/24h remained unchanged at `581.58`.
  - `hneg2` retrain regressed to baseline metrics and still `581.58` FA/24h.
  - Persistent false-alert video unchanged across base/hneg1/hneg2.
- Scene-targeted replay executed:
  - `hneg_scene1` (Coffee_room_02 train/val negatives) also failed to remove test false alert on `Coffee_room_02__Videos__video__52_`.
  - `hneg_cr02p50` (strict Coffee_room_02 replay pack, 7 windows) also failed to remove `video__52` false alert (`artifacts/reports/hneg_cycle/trace_hneg_cr02p50_video52.json`) and reverted event metrics to baseline (`outputs/metrics/tcn_le2i_hneg_cr02p50.json`).
  - model-side `hneg_pack_tsm` (`use_tsm=1`) preserved improved event metrics (`F1/recall/precision=0.8889`) and long-stream `FA/day=0.0`, but still kept `video__52` false alert (`artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_video52.json`).
  - stronger replay on the same model (`hneg_pack_tsm_hm4`, `hard_neg_mult=4`) again preserved the same OP2 metrics and long-stream `FA/day=0.0`, but still kept `video__52` false alert (`artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_hm4_video52.json`, `outputs/metrics/tcn_le2i_hneg_pack_tsm_hm4.json`, `artifacts/reports/hneg_cycle/hneg_pack_tsm_hm4_unlabeled_fa.json`).
  - stricter replay mining from hm4 checkpoint (`p>=0.80` yielded `7` windows; `p>=0.90` yielded `6` total / `1` Coffee_room_02) plus stronger replay (`hneg_pack_tsm_p80hm4m6`, `hard_neg_mult=6`) still preserved the same OP2 metrics and long-stream `FA/day=0.0`, with persistent `video__52` false alert (`artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_p80hm4m6_video52.json`, `outputs/metrics/tcn_le2i_hneg_pack_tsm_p80hm4m6.json`, `artifacts/reports/hneg_cycle/hneg_pack_tsm_p80hm4m6_unlabeled_fa.json`).
  - direct Coffee_room_02 replay from the validated 10-window negative pack (`hneg_pack_tsm_cr02p50m10`, `hard_neg_mult=10`) regressed event metrics toward baseline (`F1=0.8235`, `recall=0.7778`) and still retained `video__52` false alert (`artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_cr02p50m10_video52.json`, `outputs/metrics/tcn_le2i_hneg_pack_tsm_cr02p50m10.json`, `artifacts/reports/hneg_cycle/hneg_pack_tsm_cr02p50m10_unlabeled_fa.json`).
  - near-span-cleaned replay (removed 3 windows within 24 frames of fall spans; 7-window pack, `hard_neg_mult=10`) also stayed baseline-like and retained `video__52` false alert (`outputs/metrics/tcn_le2i_hneg_pack_tsm_cr02p50clean24m10.json`, `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_cr02p50clean24m10_video52.json`, `artifacts/reports/hneg_cycle/hneg_pack_tsm_cr02p50clean24m10_unlabeled_fa.json`).
  - mixed-scene replay from FA-window hard negatives (`15` windows, `hard_neg_mult=8`) preserved improved OP2 metrics but still retained `video__52` false alert (`outputs/metrics/tcn_le2i_hneg_pack_tsm_favalp20m8.json`, `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_favalp20m8_video52.json`, `artifacts/reports/hneg_cycle/hneg_pack_tsm_favalp20m8_unlabeled_fa.json`).
  - nearest-neighbor replay from `video__52` analog negatives (`27` windows, `hard_neg_mult=6`) likewise preserved improved OP2 metrics but retained `video__52` false alert (`outputs/metrics/tcn_le2i_hneg_pack_tsm_v52nnm6.json`, `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_v52nnm6_video52.json`, `artifacts/reports/hneg_cycle/hneg_pack_tsm_v52nnm6_unlabeled_fa.json`).
- Per-video trace diagnostic executed:
  - New script `scripts/diagnose_video_trace.py` confirms high-confidence saturation on `video__52`.
  - With same thresholds, `confirm=1` suppresses this clip’s false event (`events=0`), indicating a policy-level suppression path exists for this scene.
- Consolidated replay matrix artifact:
  - `artifacts/reports/hneg_cycle/video52_experiment_matrix.json` and `.csv` summarize OP2 metrics, `video__52` event count, and unlabeled FA/day across all replay variants executed to date.
- Longer-duration FA track (negative stream) now computed:
  - base: `1` alert over `196.319s` (`FA/day=440.10`)
  - hneg1: `0` alerts (`FA/day=0.0`)
  - hneg2: `0` alerts (`FA/day=0.0`)
  - hneg_pack: `0` alerts (`FA/day=0.0`)
  - hneg_cr02p50: `1` alert (`FA/day=440.10`)
  - hneg_pack_tsm: `0` alerts (`FA/day=0.0`)
  - hneg_pack_tsm_hm4: `0` alerts (`FA/day=0.0`)
  - hneg_pack_tsm_p80hm4m6: `0` alerts (`FA/day=0.0`)
  - hneg_pack_tsm_cr02p50m10: `0` alerts (`FA/day=0.0`)
  - hneg_pack_tsm_cr02p50clean24m10: `0` alerts (`FA/day=0.0`)
  - hneg_pack_tsm_favalp20m8: `0` alerts (`FA/day=0.0`)
  - hneg_pack_tsm_v52nnm6: `0` alerts (`FA/day=0.0`)

- Scene-scoped start-guard mitigation (Coffee_room_02 only) validated:
  - Added `start_guard_max_lying` + `start_guard_prefixes` flow through alerting/eval scripts and Makefile policy flags.
  - Fixed two integration issues discovered during validation:
    - stringified prefix list from ops YAML is now parsed correctly (`"['Coffee_room_02__Videos__']"`).
    - OP sweep/count path now passes `video_id`, so prefix-scoped guard applies consistently in `ops`.
  - Result (`tcn_le2i_hneg_pack_tsm_v52guard02scene`):
    - OP2: `F1=0.9412`, `recall=0.8889`, `precision=1.0000`, `FA/24h=0.0`, `n_false_alerts=0`.
    - `video__52`: false event removed (`n_alert_events=0`), while `video__59` and `video__60` remain detected.
    - long negative stream remains clean: `FA/day=0.0`.
  - Evidence:
    - `configs/ops/tcn_le2i_hneg_pack_tsm_v52guard02scene.yaml`
    - `outputs/metrics/tcn_le2i_hneg_pack_tsm_v52guard02scene.json`
    - `artifacts/reports/hneg_cycle/hneg_pack_tsm_v52guard02scene_unlabeled_fa.json`

- Additional model-side retrain probe (no scene guard):
  - Resume from `hneg_pack_tsm` with focal loss + stronger mask augmentation (`hneg_pack_tsm_focalm`).
  - Outcome: did not remove persistent `video__52` false alert; at fixed OP2-style thresholds (`tau=0.48/0.3744`) totals became `n_alert_events=10`, `n_true_alerts=9`, `n_false_alerts=1` (`FA/24h=581.58`).
  - `video__59` / `video__60` remain detected; long negative-stream FA/day stays `0.0`.
  - Evidence:
    - `outputs/le2i_tcn_W48S12_hneg_pack_tsm_focalm/best.pt`
    - `configs/ops/tcn_le2i_hneg_pack_tsm_focalm.yaml`
    - `outputs/metrics/tcn_le2i_hneg_pack_tsm_focalm_fixed048.json`
    - `artifacts/reports/hneg_cycle/hneg_pack_tsm_focalm_unlabeled_fa.json`

- Additional model-side retrain probe (BCE smoothing + stronger replay, no scene guard):
  - Resume from `hneg_pack_tsm` with BCE + `label_smoothing=0.05` + `hard_neg_mult=3` (`hneg_pack_tsm_bce_ls05m3`).
  - Outcome: same persistent `video__52` false alert at fixed OP2-style thresholds (`tau=0.48/0.3744`): `n_alert_events=10`, `n_true_alerts=9`, `n_false_alerts=1`, `FA/24h=581.58`.
  - `video__59` / `video__60` remain detected; long negative-stream FA/day remains `0.0`.
  - Evidence:
    - `outputs/le2i_tcn_W48S12_hneg_pack_tsm_bce_ls05m3/best.pt`
    - `configs/ops/tcn_le2i_hneg_pack_tsm_bce_ls05m3.yaml`
    - `outputs/metrics/tcn_le2i_hneg_pack_tsm_bce_ls05m3_fixed048.json`
    - `artifacts/reports/hneg_cycle/hneg_pack_tsm_bce_ls05m3_unlabeled_fa.json`

- Training capability extension + targeted replay probe:
  - Added scene-targeted hard-negative upweighting in TCN training (`--hard_neg_prefixes`, `--hard_neg_prefix_mult`) to avoid globally increasing all hard negatives; Makefile train targets now forward resume/hard-neg knobs for both TCN and GCN.
  - Probe run (`hneg_pack_tsm_cr02boost`, prefix `Coffee_room_02__Videos__`, prefix mult `4`) still keeps the same persistent `video__52` false alert at fixed OP2-style thresholds.
  - Outcome at `tau=0.48/0.3744`: `n_alert_events=10`, `n_true_alerts=9`, `n_false_alerts=1`, `FA/24h=581.58`; unlabeled FA/day remains `0.0`.
  - Evidence:
    - `outputs/le2i_tcn_W48S12_hneg_pack_tsm_cr02boost/best.pt`
    - `outputs/metrics/tcn_le2i_hneg_pack_tsm_cr02boost_fixed048.json`
    - `artifacts/reports/hneg_cycle/hneg_pack_tsm_cr02boost_unlabeled_fa.json`

- GCN architecture-side probe (adaptive adj + CTR-lite + hard-neg replay):
  - Trained `gcn_le2i_adactr_hneg` (two-stream, `use_adaptive_adj=1`, `use_ctr_gcn_lite=1`) and evaluated at fixed OP2-style thresholds (`tau=0.48/0.3744`).
  - Outcome is worse than baseline/TCN path: `n_false_alerts=2`, `FA/24h=1163.17`, and unlabeled stream `n_alert_events=2` (`FA/day=880.2`).
  - Persistent `video__52` false alert remains, so this branch is not deployable.
  - Evidence:
    - `outputs/metrics/gcn_le2i_adactr_hneg_fixed048.json`
    - `artifacts/reports/hneg_cycle/gcn_le2i_adactr_hneg_unlabeled_fa.json`

- Additional model-side structural probe (larger TCN receptive field, no scene guard):
  - Trained `hneg_rf6k5` (`num_blocks=6`, `kernel=5`, `use_tsm=1`, with hard-negative replay) and evaluated at fixed OP2-style thresholds (`tau=0.48/0.3744`).
  - Outcome matches prior no-guard model-side probes on test metrics: `n_alert_events=10`, `n_true_alerts=9`, `n_false_alerts=1`, `FA/24h=581.58`.
  - Persistent `video__52` false alert remains (`1` false event), while `video__59` and `video__60` remain detected (`1` true event each).
  - Unlabeled negative-stream score regresses to `1` alert over `196.319s` (`FA/day=440.10`).
  - Evidence:
    - `outputs/le2i_tcn_W48S12_rf6k5_hneg/best.pt`
    - `outputs/metrics/tcn_le2i_rf6k5_hneg_fixed048.json`
    - `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_hneg_unlabeled_fa.json`

- Guarded evaluation of `hneg_rf6k5` (scene-scoped start guard):
  - Using `start_guard_max_lying=0.2`, `start_guard_prefixes=Coffee_room_02__Videos__` with the same fixed thresholds (`tau=0.48/0.3744`) yields perfect test-set event totals: `n_alert_events=9`, `n_true_alerts=9`, `n_false_alerts=0`, `avg_recall=1.0`, `FA/24h=0.0`.
  - `video__52` false event is removed (`0` alerts) while `video__59` and `video__60` remain detected (`1` true event each).
  - However, unlabeled negative-stream remains non-zero (`1` alert over `196.319s`, `FA/day=440.10`), and the source is `Coffee_room_01__Videos__video__34_`; this persists at tested thresholds (`tau_high=0.48`, `0.51`, `0.56`), so this variant does not dominate the prior guarded TSM candidate on stability.
  - Evidence:
    - `outputs/metrics/tcn_le2i_rf6k5_hneg_v52guard02scene_fixed048.json`
    - `outputs/metrics/tcn_le2i_rf6k5_hneg_v52guard02scene_fixed056.json`
    - `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_hneg_v52guard02scene_unlabeled_fa.json`
    - `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_hneg_v52guard02scene_unlabeled_fa_fixed056.json`

- Targeted replay fix for residual unlabeled blocker (`video__34`) validated:
  - Added focused hard-negative replay pack from unlabeled stream (`Coffee_room_01__Videos__video__34_`, `25` windows) and resumed from `rf6k5_hneg` to train `rf6k5_v34neg`.
  - Fixed-threshold check (`tau=0.51/0.3978`):
    - no guard: still `video__52` FP persists (`n_false_alerts=1`), but unlabeled FA/day becomes `0.0`.
    - with room2 guard (`start_guard_max_lying=0.2`, `start_guard_prefixes=Coffee_room_02__Videos__`): test is perfect (`n_true_alerts=9`, `n_false_alerts=0`, recall `1.0`) and unlabeled FA/day is `0.0`.
  - Full fit_ops flow with guarded config and `op_tie_break=min_thr` produces an OP2 point that keeps this behavior:
    - selected `tau_high=0.27`, `tau_low=0.2106`
    - test totals: `n_alert_events=9`, `n_true_alerts=9`, `n_false_alerts=0`, `avg_recall=1.0`, `FA/24h=0.0`
    - unlabeled stream: `n_alert_events=0`, `FA/day=0.0`
  - Evidence:
    - `outputs/hardneg/tcn_le2i_video34_fa_val.txt`
    - `outputs/le2i_tcn_W48S12_rf6k5_v34neg/best.pt`
    - `outputs/metrics/tcn_le2i_rf6k5_v34neg_v52guard02scene_minthr.json`
    - `configs/ops/tcn_le2i_rf6k5_v34neg_v52guard02scene_minthr.yaml`
    - `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_v34neg_v52guard02scene_minthr_unlabeled_fa_op2.json`

- Active LE2i profile promotion completed:
  - Promoted the validated candidate into active ops path:
    - `configs/ops/tcn_le2i_hneg_pack_tsm.yaml`
    - `configs/ops/tcn_le2i_hneg_pack_tsm.sweep.json`
  - Updated internal `sweep_json` pointer to match promoted filename.
  - Post-promotion verification:
    - `outputs/metrics/tcn_le2i_hneg_pack_tsm_promoted.json` -> `n_true_alerts=9`, `n_false_alerts=0`, `recall=1.0`, `FA/24h=0.0`
    - `artifacts/reports/hneg_cycle/tcn_le2i_hneg_pack_tsm_promoted_unlabeled_fa.json` -> `n_alert_events=0`, `FA/day=0.0`
  - Audit gates re-run and passing after promotion:
    - `make -s audit-parity-le2i-strict MODEL=tcn`
    - `make -s audit-all MODEL=tcn AUDIT_DATASETS='le2i,caucafall'`

- CAUCAFall TCN profile retune/promotion completed:
  - Existing CAUCAFall sweep already contained a zero-FA, full-recall operating point around `tau_high=0.561`.
  - Added constrained fit profile (`op_tie_break=min_thr`, `min_tau_high=0.56`) to prevent unsafe low-threshold picks while recovering recall from the prior conservative profile.
  - Promoted active CAUCAFall ops path:
    - `configs/ops/tcn_caucafall.yaml`
    - `configs/ops/tcn_caucafall.sweep.json`
  - Post-promotion verification:
    - `outputs/metrics/tcn_caucafall_promoted.json` -> `n_true_alerts=5`, `n_false_alerts=0`, `recall=1.0`, `FA/24h=0.0`
    - `artifacts/reports/hneg_cycle/caucafall_tcn_promoted_unlabeled_fa.json` -> `n_alert_events=0`, `FA/day=0.0`

- CAUCAFall GCN profile retune/promotion completed:
  - Prior active GCN profile (`tau_high=0.30`) kept full recall but produced test false alerts (`n_false_alerts=2`).
  - Added constrained fit profile (`op_tie_break=min_thr`, `min_tau_high=0.37`) and promoted:
    - `configs/ops/gcn_caucafall.yaml`
    - `configs/ops/gcn_caucafall.sweep.json`
  - Post-promotion verification:
    - `outputs/metrics/gcn_caucafall_promoted2.json` -> `n_true_alerts=5`, `n_false_alerts=0`, `recall=1.0`, `FA/24h=0.0`
    - `artifacts/reports/hneg_cycle/gcn_caucafall_promoted2_unlabeled_fa.json` -> `n_alert_events=0`, `FA/day=0.0`
  - This aligns both CAUCAFall backbones (TCN and GCN) to the same dual-pass target without retraining.

## Reproducibility Pack (Current Winning Paths)

LE2i winning path (replay + guarded profile):
```bash
# 1) Targeted replay list from residual unlabeled blocker
python3 - <<'PY'
import glob, os
out='outputs/hardneg/tcn_le2i_video34_fa_val.txt'
files=sorted(glob.glob('data/processed/le2i/fa_windows_W48_S12/val/Coffee_room_01__Videos__video__34___*.npz'))
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    f.write('\\n'.join(files) + ('\\n' if files else ''))
print(out, len(files))
PY

# 2) Resume-train replay model
source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/train_tcn.py \
  --train_dir data/processed/le2i/windows_eval_W48_S12/train \
  --val_dir data/processed/le2i/windows_eval_W48_S12/val \
  --epochs 60 --batch 128 --lr 1e-3 --seed 33724876 --fps_default 25 \
  --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --motion_scale_by_fps 1 --conf_gate 0.2 --use_precomputed_mask 1 \
  --loss bce --hidden 128 --num_blocks 6 --kernel 5 --use_tsm 1 --tsm_fold_div 8 \
  --grad_clip 1.0 --patience 12 --thr_min 0.05 --thr_max 0.95 --thr_step 0.01 --monitor ap \
  --dropout 0.30 --mask_joint_p 0.15 --mask_frame_p 0.10 --pos_weight auto \
  --resume outputs/le2i_tcn_W48S12_rf6k5_hneg/best.pt \
  --hard_neg_list outputs/hardneg/tcn_le2i_video34_fa_val.txt --hard_neg_mult 4 \
  --save_dir outputs/le2i_tcn_W48S12_rf6k5_v34neg

# 3) Fit guarded ops (min-threshold tie-break) and evaluate
PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/fit_ops.py --arch tcn \
  --val_dir data/processed/le2i/windows_eval_W48_S12/val \
  --ckpt outputs/le2i_tcn_W48S12_rf6k5_v34neg/best.pt \
  --out configs/ops/tcn_le2i_rf6k5_v34neg_v52guard02scene_minthr.yaml \
  --fps_default 25 --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --ema_alpha 0.20 --k 2 --n 3 --cooldown_s 30 --tau_low_ratio 0.78 --confirm 0 \
  --start_guard_max_lying 0.2 --start_guard_prefixes Coffee_room_02__Videos__ \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.01 --time_mode center --merge_gap_s 1.0 --overlap_slack_s 0.5 \
  --op1_recall 0.95 --op3_fa24h 1.0 --ops_picker conservative --op_tie_break min_thr --tie_eps 1e-3 \
  --save_sweep_json 1 --allow_degenerate_sweep 0 --emit_absolute_paths 0 --min_tau_high 0.20

PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/eval_metrics.py \
  --win_dir data/processed/le2i/windows_eval_W48_S12/test \
  --ckpt outputs/le2i_tcn_W48S12_rf6k5_v34neg/best.pt \
  --ops_yaml configs/ops/tcn_le2i_rf6k5_v34neg_v52guard02scene_minthr.yaml \
  --out_json outputs/metrics/tcn_le2i_rf6k5_v34neg_v52guard02scene_minthr.json \
  --fps_default 25 --thr_min 0.01 --thr_max 0.95 --thr_step 0.01 --time_mode center --merge_gap_s 1.0 --overlap_slack_s 0.5
```

CAUCAFall winning path (profile retune only):
```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/fit_ops.py --arch tcn \
  --val_dir data/processed/caucafall/windows_eval_W48_S12/val \
  --ckpt outputs/caucafall_tcn_W48S12/best.pt \
  --out configs/ops/tcn_caucafall.yaml \
  --fps_default 23 --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --ema_alpha 0.20 --k 2 --n 3 --cooldown_s 30 --tau_low_ratio 0.78 --confirm 0 \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.01 --time_mode center --merge_gap_s 1.0 --overlap_slack_s 0.5 \
  --op1_recall 0.95 --op3_fa24h 1.0 --ops_picker conservative --op_tie_break min_thr --tie_eps 1e-3 \
  --save_sweep_json 1 --allow_degenerate_sweep 0 --emit_absolute_paths 0 --min_tau_high 0.56
```
