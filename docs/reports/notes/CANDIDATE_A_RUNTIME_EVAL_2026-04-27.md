Date: 2026-04-27  
Owner: report/paper finalization branch  
Purpose: record the first real project-strengthening result from the `CAUCAFall + TCN` offline retraining track.

# Candidate A Runtime Evaluation

## 1. Candidate

Training run:

- `outputs/caucafall_tcn_W48S12_rtA_hneg_union_recall`

Best checkpoint:

- `outputs/caucafall_tcn_W48S12_rtA_hneg_union_recall/best.pt`

Training summary:

- best validation checkpoint remained at `ep14`
- headline validation values at the saved best:
  - `AP=0.9952`
  - `AUC=0.997`
  - `F1=0.967`
  - `P=0.937`
  - `R=1.000`
  - `FPR=0.047`

## 2. Offline Post-Fit Result

Post-fit ops file:

- `configs/ops/tcn_caucafall_rtA_hneg_union_recall.yaml`

Offline metrics artifact:

- `outputs/metrics/tcn_caucafall_rtA_hneg_union_recall.json`

Observed post-fit behaviour:

- `fit_ops` first produced a degenerate sweep under confirmation
- the fitting process retried with confirmation disabled
- the written ops file therefore records:
  - `confirm=false`
  - `confirm_fallback=disabled_confirm_after_degenerate_sweep`

This is important. It means Candidate A cannot be described as a straightforward stronger replacement of the active runtime line just because its post-fit offline sweep looks perfect.

Key fitted `OP-2` values:

- `tau_high=0.86`
- `tau_low=0.6708`
- fitted validation replay summary inside the sweep:
  - `precision=1.0`
  - `recall=1.0`
  - `f1=1.0`
  - `fa24h=0.0`

## 3. Four-Folder Canonical Replay Check

Evaluation command family:

- `ops/scripts/eval_delivery_videos.py`
- windows:
  - `artifacts/fall_test_eval_20260315/windows/unsplit`
- fall folders:
  - `/corridor/`
  - `/kitchen/`
- non-fall folders:
  - `/corridor_adl/`
  - `/kitchen_adl/`

Candidate A runtime artifacts:

- `artifacts/fall_test_eval_20260427_rtA/unified_tcn_caucafall_rtA_op2.csv`
- `artifacts/fall_test_eval_20260427_rtA/unified_tcn_caucafall_rtA_op2.json`
- `artifacts/fall_test_eval_20260427_rtA/unified_tcn_caucafall_rtA_op2_metrics.json`

Baseline runtime artifacts used for same-surface comparison:

- `artifacts/fall_test_eval_20260427_baseline/unified_tcn_caucafall_baseline_op2.csv`
- `artifacts/fall_test_eval_20260427_baseline/unified_tcn_caucafall_baseline_op2.json`
- `artifacts/fall_test_eval_20260427_baseline/unified_tcn_caucafall_baseline_op2_metrics.json`

Comparison summary:

- baseline canonical `TCN + OP-2`
  - `TP=3`
  - `TN=10`
  - `FP=2`
  - `FN=9`
  - `13/24`
- Candidate A `TCN + OP-2`
  - `TP=6`
  - `TN=10`
  - `FP=2`
  - `FN=6`
  - `16/24`

Machine-readable comparison:

- `docs/reports/notes/candidate_a_four_folder_compare_2026-04-27.csv`

## 4. Folder-Level Breakdown

Baseline:

- `corridor`: `TP=2`, `FN=4`
- `corridor_adl`: `TN=6`
- `kitchen`: `TP=1`, `FN=5`
- `kitchen_adl`: `FP=2`, `TN=4`

Candidate A:

- `corridor`: `TP=4`, `FN=2`
- `corridor_adl`: `TN=6`
- `kitchen`: `TP=2`, `FN=4`
- `kitchen_adl`: `FP=2`, `TN=4`

Interpretation:

1. the gain came entirely from lower false negatives on the fall folders
2. ADL control did not improve, but it also did not get worse on this surface
3. the improvement is real and useful, but it is not yet a clean promotion result because the candidate needed confirm fallback during `fit_ops`

## 5. Current Judgment

Candidate A is the first retraining result that clearly improves the bounded canonical four-folder runtime check without increasing ADL false alarms.

That makes it a meaningful project-strengthening result.

However, it should not be promoted immediately as the new defended runtime line until one more check is done:

1. compare it against the existing bounded custom replay surface used in the `15/24` discussion
2. confirm that the `confirm=false` fallback does not weaken the deployment narrative too much

Update:

1. the locked-surface comparison has now been completed in:
   - `docs/reports/notes/CANDIDATE_A_LOCKED_SURFACE_EVAL_2026-04-27.md`
2. that comparison shows:
   - baseline locked surface `15/24`
   - Candidate A locked surface `16/24`
3. therefore Candidate A has now improved both bounded replay checks that matter most in the current report/paper narrative

## 6. Safe Report/Paper Wording

Safe wording:

- a recall-oriented continuation of the `CAUCAFall + TCN` line improved the canonical four-folder bounded replay check from `13/24` to `16/24` without increasing ADL false positives

Unsafe wording:

- Candidate A fully solves runtime performance
- Candidate A is already the new deploy profile
- Candidate A proves stable deployment superiority
