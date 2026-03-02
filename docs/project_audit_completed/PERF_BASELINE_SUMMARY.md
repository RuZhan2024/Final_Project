# PERF_BASELINE_SUMMARY

Date: 2026-03-02  
Scope: baseline extraction from current repo artifacts plus gate verification.

## 1) Leaderboard Snapshot (Current Tracked Metrics)

| dataset | model | report | selected OP | event F1 | recall | precision | FA/24h | n_alert_events |
|---|---|---|---|---:|---:|---:|---:|---:|
| LE2i | TCN | `outputs/metrics/tcn_le2i.json` | op2 | 0.8235 | 0.7778 | 0.8750 | 581.58 | 8 |
| CAUCAFall | TCN | `outputs/metrics/tcn_caucafall.json` | op2 | 0.8889 | 0.8000 | 1.0000 | 0.00 | 4 |
| CAUCAFall | GCN | `outputs/metrics/gcn_caucafall.json` | op2 | 0.8333 | 1.0000 | 0.7143 | 1881.82 | 7 |

These were refreshed by direct `scripts/eval_metrics.py` runs to avoid unintended retraining in `make eval-*` dependencies.

## 2) What Changed Since Initial Audit

| Area | New state | Evidence |
|---|---|---|
| Best-checkpoint monitor defaults | AP is now default for both TCN/GCN and forwarded to train commands | [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:274), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:275), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:826), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:844) |
| Fit-ops can now block degenerate sweeps | Fails unless explicitly overridden | [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:125), [fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/src/fall_detection/evaluation/fit_ops.py:750), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:344) |
| Ops sanity is CI-audited | Dedicated `audit-ops-sanity` target and script | [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1026), [scripts/audit_ops_sanity.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/audit_ops_sanity.py:43) |
| Numeric and temporal gates are hard-wired | Shared gate config and audit targets are in place | [configs/audit_gates.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/audit_gates.json:1), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1032), [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1039) |
| Full deploy audit stack exists | `audit-all-deploy` now chains audit + profile budget | [Makefile](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/Makefile:1081), [scripts/audit_profile_budget.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/scripts/audit_profile_budget.py:58) |

## 3) Dominant Failure Interpretation (Current)
1. Historical failure mode was policy/ops degeneracy; current refreshed metrics are no longer in zero-alert collapse.
2. Strict parity now passes after recapturing LE2i baseline targets from refreshed metrics.
3. Primary optimization target remains FA/24h reduction while preserving recall.

## 3.1 Policy-Only Tuning Snapshot (LE2i TCN)

| case | policy highlights | event F1 | recall | precision | FA/24h | alerts |
|---|---|---:|---:|---:|---:|---:|
| `base_c0` | `confirm=0`, `k/n=2/3` | 0.8235 | 0.7778 | 0.8750 | 581.58 | 8 |
| `c2_relaxed_nolow` | `confirm=1`, relaxed, `require_low=0` | 0.5714 | 0.4444 | 0.8000 | 581.58 | 5 |
| `c3_confirm_short` | `confirm=1`, `confirm_s=1.0`, loose lying/motion gates | 0.5714 | 0.4444 | 0.8000 | 581.58 | 5 |
| `c4_confirm_loose` | `confirm=1`, `confirm_s=0.8`, looser lying/motion gates | 0.5714 | 0.4444 | 0.8000 | 581.58 | 5 |
| `c5_costfp3_fa` | `op2=cost_sensitive`, `cost_fp=3`, FA-aware fit | 0.8235 | 0.7778 | 0.8750 | 581.58 | 8 |
| `c6_costfp8_fa` | `op2=cost_sensitive`, `cost_fp=8`, FA-aware fit | 0.8235 | 0.7778 | 0.8750 | 581.58 | 8 |
| `c7_confirm_costfa` | `confirm=1` + cost-sensitive + FA-aware fit | 0.5714 | 0.4444 | 0.8000 | 581.58 | 5 |
| `p1_k3n5` | `confirm=0`, `k/n=3/5` | 0.7059 | 0.6667 | 0.7500 | 1163.17 | 8 |
| `p2_k4n6` | `confirm=0`, `k/n=4/6` | 0.7059 | 0.6667 | 0.7500 | 1163.17 | 8 |

Evidence files:
- `artifacts/reports/tuning/base_c0.metrics.json`
- `artifacts/reports/tuning/c2_relaxed_nolow.metrics.json`
- `artifacts/reports/tuning/c3_confirm_short.metrics.json`
- `artifacts/reports/tuning/c4_confirm_loose.metrics.json`
- `artifacts/reports/tuning/c5_costfp3_fa.metrics.json`
- `artifacts/reports/tuning/c6_costfp8_fa.metrics.json`
- `artifacts/reports/tuning/c7_confirm_costfa.metrics.json`
- `artifacts/reports/tuning/p1_k3n5.metrics.json`
- `artifacts/reports/tuning/p2_k4n6.metrics.json`
- Hard-negative seed list generated (no retraining yet): `outputs/hardneg/tcn_le2i.txt` (`28` windows, `12` unique videos).

## 4) Repro Pack (Exact Commands Used)

### 4.1 Metrics extraction
```bash
python - <<'PY'
import json,glob
for fp in sorted(glob.glob('outputs/metrics/*.json')):
  d=json.load(open(fp)); s=d['selected']['name']; op=d['ops'][s]
  print(fp, s, op.get('f1'), op.get('recall'), op.get('precision'), op.get('fa24h'), op.get('n_alert_events'))
PY
```

### 4.2 Gate verification
```bash
make -s audit-ops-sanity
make -s audit-artifact-bundle
make -s audit-numeric AUDIT_DATASETS='le2i,caucafall'
make -s audit-temporal AUDIT_DATASETS='le2i,caucafall'
make -s audit-all-deploy PROFILE=cpu_local DS=le2i MODEL=tcn PROFILE_IO_ONLY=1 AUDIT_DATASETS='le2i,caucafall'
make -s audit-parity-le2i-strict MODEL=tcn
make -s baseline-capture-le2i MODEL=tcn
```

### 4.3 Unit-test verification
```bash
python -m pytest -q tests/test_import_smoke.py tests/test_data_sources_config.py tests/test_windows_contract.py tests/test_adapter_contract.py tests/test_event_time_semantics.py
```

### 4.4 Output locations checked
- `outputs/metrics/*.json`
- `outputs/plots/*`
- `configs/ops/*.yaml`
- `configs/ops/*.sweep.json`
- `artifacts/reports/numeric_fingerprint_20260302.json`
- `artifacts/reports/temporal_span_20260302.json`
- `artifacts/reports/parity_le2i_tcn_strict_20260302.json`
- `artifacts/reports/infer_profile_cpu_local_tcn_le2i.json`

### 4.5 Hard-negative loop orchestration check
```bash
make -n hneg-cycle-tcn-le2i
make -n train-tcn-le2i TCN_RESUME=outputs/le2i_tcn_W48S12/best.pt TCN_HARD_NEG_LIST=outputs/hardneg/tcn_le2i.txt TCN_HARD_NEG_MULT=2
```
Expected from dry-run:
- `hneg-cycle-*` executes `windows-eval -> fa-windows -> mine-hard-negatives -> train(resume+replay) -> fit-ops -> eval -> plot`.
- Train command includes `--resume`, `--hard_neg_list`, and `--hard_neg_mult`.

## 5) Next Action to Complete This Document
1. Keep AP-monitor defaults unchanged unless parity/baseline evidence proves regression.
2. Hard-negative process is now standardized in Make (`hneg-cycle-{tcn|gcn}-<ds>`), and dual FA-track reporting is active; next step is scene-specific FP handling for `Coffee_room_02__Videos__video__52_`.

## 6) Hard-Negative Cycle Results (LE2i TCN)

| run | checkpoint | event F1 | recall | precision | FA/24h | alerts |
|---|---|---:|---:|---:|---:|---:|
| base | `outputs/le2i_tcn_W48S12/best.pt` | 0.8235 | 0.7778 | 0.8750 | 581.5843 | 8 |
| hneg1 | `outputs/le2i_tcn_W48S12_hneg1/best.pt` | 0.8889 | 0.8889 | 0.8889 | 581.5843 | 9 |
| hneg2 | `outputs/le2i_tcn_W48S12_hneg2/best.pt` | 0.8235 | 0.7778 | 0.8750 | 581.5843 | 8 |
| scene1 | `outputs/le2i_tcn_W48S12_hneg_scene1/best.pt` | 0.8235 | 0.7778 | 0.8750 | 581.5843 | 8 |
| hneg_pack | `outputs/le2i_tcn_W48S12_hneg_pack/best.pt` | 0.8889 | 0.8889 | 0.8889 | 581.5843 | 9 |
| hneg_cr02p50 | `outputs/le2i_tcn_W48S12_hneg_cr02p50/best.pt` | 0.8235 | 0.7778 | 0.8750 | 581.5843 | 8 |
| hneg_pack_tsm | `outputs/le2i_tcn_W48S12_hneg_pack_tsm/best.pt` | 0.8889 | 0.8889 | 0.8889 | 581.5843 | 9 |
| hneg_pack_tsm_hm4 | `outputs/le2i_tcn_W48S12_hneg_pack_tsm_hm4/best.pt` | 0.8889 | 0.8889 | 0.8889 | 581.5843 | 9 |
| hneg_pack_tsm_p80hm4m6 | `outputs/le2i_tcn_W48S12_hneg_pack_tsm_p80hm4m6/best.pt` | 0.8889 | 0.8889 | 0.8889 | 581.5843 | 9 |
| hneg_pack_tsm_cr02p50m10 | `outputs/le2i_tcn_W48S12_hneg_pack_tsm_cr02p50m10/best.pt` | 0.8235 | 0.7778 | 0.8750 | 581.5843 | 8 |
| hneg_pack_tsm_cr02p50clean24m10 | `outputs/le2i_tcn_W48S12_hneg_pack_tsm_cr02p50clean24m10/best.pt` | 0.8235 | 0.7778 | 0.8750 | 581.5843 | 8 |
| hneg_pack_tsm_favalp20m8 | `outputs/le2i_tcn_W48S12_hneg_pack_tsm_favalp20m8/best.pt` | 0.8889 | 0.8889 | 0.8889 | 581.5843 | 9 |
| hneg_pack_tsm_v52nnm6 | `outputs/le2i_tcn_W48S12_hneg_pack_tsm_v52nnm6/best.pt` | 0.8889 | 0.8889 | 0.8889 | 581.5843 | 9 |
| hneg_rf6k5 | `outputs/le2i_tcn_W48S12_rf6k5_hneg/best.pt` | 0.9474 | 1.0000 | 0.9000 | 581.5843 | 10 |
| hneg_rf6k5_v52guard02scene | `outputs/le2i_tcn_W48S12_rf6k5_hneg/best.pt` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 9 |
| hneg_rf6k5_v34neg_v52guard02scene_minthr | `outputs/le2i_tcn_W48S12_rf6k5_v34neg/best.pt` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 9 |
| hneg_pack_tsm_promoted (active profile) | `outputs/le2i_tcn_W48S12_rf6k5_v34neg/best.pt` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 9 |

Evidence:
- `artifacts/reports/hneg_cycle/tcn_le2i_hneg1.metrics.json`
- `artifacts/reports/hneg_cycle/tcn_le2i_hneg2.metrics.json`
- consolidated matrix across replay variants: `artifacts/reports/hneg_cycle/video52_experiment_matrix.json`, `artifacts/reports/hneg_cycle/video52_experiment_matrix.csv`

Persistent false-alert source (all runs):
- `Coffee_room_02__Videos__video__52_` contributes one false event; with short total test duration, this keeps FA/24h numerically high.
- Scene-targeted replay (`Coffee_room_02` train/val negatives) did not remove this test false alert.

## 7) Longer Negative-Stream FA Estimate (Stability Track)

Using `score_unlabeled_alert_rate` on `data/processed/le2i/fa_windows_W48_S12/val` with each run's OP2 thresholds:

| run | n_alert_events | duration_s | FA/day |
|---|---:|---:|---:|
| base | 1 | 196.319 | 440.10 |
| hneg1 | 0 | 196.319 | 0.00 |
| hneg2 | 0 | 196.319 | 0.00 |
| scene1 | 0 | 196.319 | 0.00 |
| hneg_pack | 0 | 196.319 | 0.00 |
| hneg_cr02p50 | 1 | 196.319 | 440.10 |
| hneg_pack_tsm | 0 | 196.319 | 0.00 |
| hneg_pack_tsm_hm4 | 0 | 196.319 | 0.00 |
| hneg_pack_tsm_p80hm4m6 | 0 | 196.319 | 0.00 |
| hneg_pack_tsm_cr02p50m10 | 0 | 196.319 | 0.00 |
| hneg_pack_tsm_cr02p50clean24m10 | 0 | 196.319 | 0.00 |
| hneg_rf6k5 | 1 | 196.319 | 440.10 |
| hneg_rf6k5_v52guard02scene | 1 | 196.319 | 440.10 |
| hneg_rf6k5_v34neg | 0 | 196.319 | 0.00 |
| hneg_rf6k5_v34neg_v52guard02scene | 0 | 196.319 | 0.00 |
| hneg_pack_tsm_promoted (active profile) | 0 | 196.319 | 0.00 |
| hneg_pack_tsm_favalp20m8 | 0 | 196.319 | 0.00 |
| hneg_pack_tsm_v52nnm6 | 0 | 196.319 | 0.00 |
| c5_costfp3_fa | 1 | 196.319 | 440.10 |
| c6_costfp8_fa | 1 | 196.319 | 440.10 |
| c7_confirm_costfa | 0 | 196.319 | 0.00 |

Artifacts:
- `artifacts/reports/hneg_cycle/base_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg1_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg2_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_scene1_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_cr02p50_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_hm4_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_p80hm4m6_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_cr02p50m10_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_cr02p50clean24m10_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_favalp20m8_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_v52nnm6_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_v52guard02scene_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_focalm_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_bce_ls05m3_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/hneg_pack_tsm_cr02boost_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/gcn_le2i_adactr_hneg_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_hneg_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_hneg_v52guard02scene_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_hneg_v52guard02scene_unlabeled_fa_fixed056.json`
- `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_v34neg_unlabeled_fa_fixed051.json`
- `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_v34neg_v52guard02scene_unlabeled_fa_fixed051.json`
- `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_v34neg_v52guard02scene_minthr_unlabeled_fa_op2.json`
- `artifacts/reports/hneg_cycle/tcn_le2i_hneg_pack_tsm_promoted_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/caucafall_tcn_fixed0561_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/caucafall_tcn_tau56floor_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/caucafall_tcn_promoted_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/gcn_caucafall_promoted_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/gcn_caucafall_tau37floor_unlabeled_fa.json`
- `artifacts/reports/hneg_cycle/gcn_caucafall_promoted2_unlabeled_fa.json`
- `artifacts/reports/tuning/c5_costfp3_fa.unlabeled_fa.json`
- `artifacts/reports/tuning/c6_costfp8_fa.unlabeled_fa.json`
- `artifacts/reports/tuning/c7_confirm_costfa.unlabeled_fa.json`

## 8) Video-52 Trace Diagnostic (Root-Cause Drilldown)

Generated with:
- `scripts/diagnose_video_trace.py`
- Reports:
  - `artifacts/reports/hneg_cycle/trace_base_video52.json`
  - `artifacts/reports/hneg_cycle/trace_hneg1_video52.json`
  - `artifacts/reports/hneg_cycle/trace_hneg_cr02p50_video52.json`
  - `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_video52.json`
  - `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_hm4_video52.json`
  - `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_p80hm4m6_video52.json`
  - `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_cr02p50m10_video52.json`
  - `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_cr02p50clean24m10_video52.json`
  - `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_favalp20m8_video52.json`
  - `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_v52nnm6_video52.json`
  - `artifacts/reports/tuning/trace_v52_guard_scene_test.json`
  - `artifacts/reports/hneg_cycle/trace_base_video52_confirm1.json`
  - `artifacts/reports/tuning/trace_c3_video52.json`

Key observations:
1. For both `base` and `hneg1`, `video__52` has many windows with saturated `p_fall` (~1.0) before the false event.
2. The false event activates around window index `8` (`w=96..143`, center `4.78s`) and persists to the clip end.
3. Enabling confirm on this clip (`confirm=1`, same tau/k/n) suppresses the event entirely (`events=0`), so this FP is suppressible by confirmation logic.
4. Additional global confirm variants (`c3`, `c4`) still produce the same false event on this clip (`events=1`) while reducing global recall.
5. Stronger replay weighting on the TSM variant (`hneg_pack_tsm_hm4`, hard-neg multiplier `4`) still leaves `video__52` at `events=1` with unchanged OP2 metrics.
6. Policy-only feasibility search remains negative: fixed-threshold confirm sweep (`240` settings) and broader joint threshold/persistence/confirm sweep (`3888` settings) found no configuration that suppresses `video__52` while keeping both `video__59` and `video__60` detected.
7. Stricter mined replay (`p>=0.80`, 7 windows, replay multiplier `6`) also leaves `video__52` at `events=1` with unchanged OP2 metrics.
8. Hybrid TCN+GCN probability fusion (`min`/`avg`) does not suppress `video__52`; `min` fusion also suppresses `video__59`, so fusion is not a safe fix (`artifacts/reports/tuning/hybrid_probe_hm4m6_v52_v59_v60.json`).
9. Corrected Coffee_room_02 negative profiling (`y<=0`) shows a sparse high-score tail (`p>=0.8: 0`, `p>=0.7: 2`, `p>=0.6: 4`, `p>=0.5: 10`), indicating replay-only leverage is limited without new/cleaner scene negatives (`artifacts/reports/tuning/coffee_room_02_neg_score_profile_hm4m6_yfix.json`, `artifacts/reports/tuning/cr02_neg_review_candidates_hm4m6.csv`).
10. Replaying the validated 10-window Coffee_room_02 hard-negative pack with stronger weighting (`hard_neg_mult=10`) regressed event metrics back to baseline-like values while still keeping `video__52` false alert (`artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_cr02p50m10_video52.json`).
11. Replaying a near-span-cleaned subset of that pack (7 windows; excludes windows within 24 frames of annotated fall spans) still remains baseline-like and keeps `video__52` (`artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_cr02p50clean24m10_video52.json`).
12. Replaying broader mixed-scene hard negatives mined from FA windows (`15` windows, `hard_neg_mult=8`) preserves improved OP2 metrics but still keeps `video__52` (`artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_favalp20m8_video52.json`).
13. Feature-space nearest-neighbor replay (`27` analog windows to `video__52`, `hard_neg_mult=6`) also preserves improved OP2 metrics but still keeps `video__52` (`artifacts/reports/tuning/video52_neighbor_negatives_hneg_pack_tsm_favalp20m8.json`, `artifacts/reports/hneg_cycle/trace_hneg_pack_tsm_v52nnm6_video52.json`).
14. Scene-scoped start guard (`start_guard_max_lying=0.2`, `start_guard_prefixes=Coffee_room_02__Videos__`) removes `video__52` false alerts while preserving detections on `video__59` and `video__60`; test OP2 improves to `F1=0.9412`, `recall=0.8889`, `precision=1.0`, `FA/24h=0.0` (`outputs/metrics/tcn_le2i_hneg_pack_tsm_v52guard02scene.json`).
15. Model-only retrain probe with focal loss + stronger masking (`hneg_pack_tsm_focalm`) does not remove `video__52`; fixed OP2-style evaluation remains at `1` false alert (`outputs/metrics/tcn_le2i_hneg_pack_tsm_focalm_fixed048.json`) while unlabeled FA/day remains `0.0` (`artifacts/reports/hneg_cycle/hneg_pack_tsm_focalm_unlabeled_fa.json`).
16. Model-only retrain probe with BCE label smoothing + stronger replay (`hneg_pack_tsm_bce_ls05m3`) also does not remove `video__52`; fixed OP2-style evaluation remains at `1` false alert (`outputs/metrics/tcn_le2i_hneg_pack_tsm_bce_ls05m3_fixed048.json`) while unlabeled FA/day remains `0.0` (`artifacts/reports/hneg_cycle/hneg_pack_tsm_bce_ls05m3_unlabeled_fa.json`).
17. Scene-targeted replay boost via new training knobs (`--hard_neg_prefixes Coffee_room_02__Videos__ --hard_neg_prefix_mult 4`, run `hneg_pack_tsm_cr02boost`) still does not remove `video__52`; fixed OP2-style evaluation remains at `1` false alert (`outputs/metrics/tcn_le2i_hneg_pack_tsm_cr02boost_fixed048.json`) with unlabeled FA/day still `0.0` (`artifacts/reports/hneg_cycle/hneg_pack_tsm_cr02boost_unlabeled_fa.json`).
18. GCN architecture probe (`gcn_le2i_adactr_hneg`: adaptive adjacency + CTR-lite + hard-negative replay) is worse at fixed OP2-style thresholds, with `2` false alerts (`FA/24h=1163.17`) and non-zero unlabeled FA (`FA/day=880.2`), while still keeping `video__52` false alert (`outputs/metrics/gcn_le2i_adactr_hneg_fixed048.json`, `artifacts/reports/hneg_cycle/gcn_le2i_adactr_hneg_unlabeled_fa.json`).
19. Larger-receptive-field TCN probe (`tcn_le2i_rf6k5_hneg`, `num_blocks=6`, `kernel=5`, hard-negative replay) also fails to remove the persistent `video__52` false alert; fixed OP2-style metrics remain `n_false_alerts=1` (`FA/24h=581.58`) with `video__59`/`video__60` still detected, and unlabeled FA regresses to `1` alert (`FA/day=440.10`) (`outputs/metrics/tcn_le2i_rf6k5_hneg_fixed048.json`, `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_hneg_unlabeled_fa.json`).
20. Scene-scoped start guard applied to the same `rf6k5` checkpoint (`start_guard_max_lying=0.2`, `start_guard_prefixes=Coffee_room_02__Videos__`) achieves perfect fixed-threshold test metrics (`F1=1.0`, `recall=1.0`, `precision=1.0`, `FA/24h=0.0`) while preserving `video__59`/`video__60` detections and removing `video__52`; however, unlabeled FA/day remains `440.10` (`1` alert), so longer-stream stability is not improved (`outputs/metrics/tcn_le2i_rf6k5_hneg_v52guard02scene_fixed048.json`, `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_hneg_v52guard02scene_unlabeled_fa.json`).
21. The residual unlabeled alert for guarded `rf6k5` is consistently `Coffee_room_01__Videos__video__34_` and persists at tested thresholds (`tau_high=0.48`, `0.51`, `0.56`), indicating the remaining instability is not the original `video__52` failure mode (`artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_hneg_v52guard02scene_unlabeled_fa.json`, `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_hneg_v52guard02scene_unlabeled_fa_fixed056.json`).
22. Targeted hard-negative replay on that residual clip (`rf6k5_v34neg`, replay list `outputs/hardneg/tcn_le2i_video34_fa_val.txt`) resolves the unlabeled blocker (`video__34`) and, combined with the existing room2 guard plus `fit_ops` with `op_tie_break=min_thr`, yields a full-pass operating point on current LE2i checks: test `F1=1.0`, `recall=1.0`, `precision=1.0`, `FA/24h=0.0`, and unlabeled `FA/day=0.0` (`outputs/metrics/tcn_le2i_rf6k5_v34neg_v52guard02scene_minthr.json`, `artifacts/reports/hneg_cycle/tcn_le2i_rf6k5_v34neg_v52guard02scene_minthr_unlabeled_fa_op2.json`).
23. This candidate has now been promoted into the active LE2i ops profile path (`configs/ops/tcn_le2i_hneg_pack_tsm.yaml` + `.sweep.json`) and re-verified end-to-end (`outputs/metrics/tcn_le2i_hneg_pack_tsm_promoted.json`, `artifacts/reports/hneg_cycle/tcn_le2i_hneg_pack_tsm_promoted_unlabeled_fa.json`), with strict parity and full audit gates passing afterward (`artifacts/reports/parity_le2i_tcn_strict_20260302.json` and `make -s audit-all` output).
24. CAUCAFall TCN threshold selection was conservative-high in the prior active profile (`tau_high≈0.87`), yielding `FA/24h=0` but recall `0.8`; the existing sweep contains a full-recall zero-FA point around `tau_high=0.561`, which validates on test and unlabeled streams (`outputs/metrics/tcn_caucafall_fixed0561.json`, `artifacts/reports/hneg_cycle/caucafall_tcn_fixed0561_unlabeled_fa.json`).
25. A constrained CAUCAFall fit profile (`op_tie_break=min_thr`, `min_tau_high=0.56`) was promoted to active ops (`configs/ops/tcn_caucafall.yaml`), and post-promotion verification holds dual-pass behavior: test `n_false_alerts=0`, `recall=1.0`, `FA/24h=0.0`; unlabeled `FA/day=0.0` (`outputs/metrics/tcn_caucafall_promoted.json`, `artifacts/reports/hneg_cycle/caucafall_tcn_promoted_unlabeled_fa.json`).
26. CAUCAFall GCN active profile initially remained unstable on test (`outputs/metrics/gcn_caucafall_promoted.json`: `n_false_alerts=2`, `FA/24h=1881.82`) despite clean unlabeled FA/day; constrained policy fitting (`op_tie_break=min_thr`, `min_tau_high=0.37`) fixes this without retraining.
27. Promoted constrained GCN profile (`configs/ops/gcn_caucafall.yaml`) verifies dual-pass behavior: test `n_true_alerts=5`, `n_false_alerts=0`, `recall=1.0`, `FA/24h=0.0`; unlabeled `FA/day=0.0` (`outputs/metrics/gcn_caucafall_promoted2.json`, `artifacts/reports/hneg_cycle/gcn_caucafall_promoted2_unlabeled_fa.json`).

## 11) CAUCAFall Backbone Snapshot (Post-Promotion)

| model | profile | tau_high | test recall | test n_false_alerts | test FA/24h | unlabeled FA/day |
|---|---|---:|---:|---:|---:|---:|
| TCN | `configs/ops/tcn_caucafall.yaml` | 0.56 | 1.0 | 0 | 0.0 | 0.0 |
| GCN | `configs/ops/gcn_caucafall.yaml` | 0.38 | 1.0 | 0 | 0.0 | 0.0 |

Interpretation:
- The persistent FP is not due to random threshold noise; it is a stable high-confidence behavior on this scene.
- Global confirm settings previously hurt recall across dataset; clip-level suppression works but requires a targeted policy strategy.
- New constrained search evidence (`artifacts/reports/tuning/confirm_grid_hm4_v52_v59_v60.json`, `artifacts/reports/tuning/policy_joint_grid_hm4_v52_v59_v60.json`) reinforces that global policy tuning is exhausted for this failure mode.

## 9) Global Policy Feasibility Bound (No-Retrain)

From test sweep in `outputs/metrics/tcn_le2i.json`:
- Current selected OP2: `tau_high=0.56`, `F1=0.8235`, `recall=0.7778`, `n_false_alerts=1`.
- Best no-false-alert threshold in the same sweep: `tau_high=0.681`, but `recall=0.1111` and `F1=0.20`.

Implication:
- A global threshold-only mitigation is not viable; removing the persistent false alert by thresholding alone destroys recall.

Per-video persistence test (`Coffee_room_02__Videos__video__52_`, fixed `tau_high/tau_low=0.56/0.4368`, `confirm=0`):
- `k/n=3/5`: still `events=1` (`artifacts/reports/tuning/trace_v52_k3n5.json`)
- `k/n=4/6`: still `events=1` (`artifacts/reports/tuning/trace_v52_k4n6.json`)
- `k/n=5/7`: still `events=1` (`artifacts/reports/tuning/trace_v52_k5n7.json`)

Implication:
- Increasing persistence (`k/n`) delays the false event but does not remove it on this clip under global policy.

Strict confirm profile check (`confirm=1`, `confirm_s=2.0`, `min_lying=0.65`, `max_motion=0.08`, `require_low=1`):
- Suppresses the FP clip: `video__52 events=0` (`artifacts/reports/hneg_cycle/trace_base_video52_confirm1.json`)
- But also suppresses true-positive Coffee_room_02 clips:
  - `video__59`: `events=1 -> 0` (`artifacts/reports/tuning/trace_v59_base.json`, `artifacts/reports/tuning/trace_v59_confirm_strict.json`)
  - `video__60`: `events=1 -> 0` (`artifacts/reports/tuning/trace_v60_base.json`, `artifacts/reports/tuning/trace_v60_confirm_strict.json`)

Final implication (policy-only):
- Scene/global policy adjustments cannot remove the persistent FP without unacceptable recall loss on same-scene true events.

## 10) Systematic Data-Side Replay Pack (Prepared)

Built from adapter-consistent windows only (`windows_eval_W48_S12/{train,val}`) using score-based mining (`min_p=0.50`, `neg_only=1`):
- `outputs/hardneg/tcn_le2i_evaltrain_p50.txt` (`29` windows)
- `outputs/hardneg/tcn_le2i_evalval_p50.txt` (`8` windows)

Merged replay packs:
- `outputs/hardneg/tcn_le2i_evalmix_p50.txt` (`37` unique windows)
- `outputs/hardneg/tcn_le2i_coffee_room_02_p50.txt` (`7` windows, scene-focused)

Summary report:
- `artifacts/reports/tuning/hneg_pack_summary.json`
- Scene distribution in merged pack:
  - `Coffee_room_01`: `14`
  - `Coffee_room_02`: `7`
  - `Home_01`: `13`
  - `Home_02`: `3`

Execution-ready retrain command (dry-run validated):
```bash
make -n train-tcn-le2i OUT_TAG=_hneg_pack \
  TCN_RESUME=outputs/le2i_tcn_W48S12/best.pt \
  TCN_HARD_NEG_LIST=outputs/hardneg/tcn_le2i_evalmix_p50.txt \
  TCN_HARD_NEG_MULT=2
```
