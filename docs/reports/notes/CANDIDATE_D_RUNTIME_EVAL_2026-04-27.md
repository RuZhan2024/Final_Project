Date: 2026-04-27  
Owner: report/paper finalization branch  
Purpose: record the runtime-side evaluation of `Candidate D` and compare it against both the baseline and `Candidate A`.

# Candidate D Runtime Evaluation

## 1. Candidate

Training run:

- `outputs/caucafall_tcn_W48S12_rtD_hneg_plus_continue`

Best checkpoint:

- `outputs/caucafall_tcn_W48S12_rtD_hneg_plus_continue/best.pt`

Best observed validation point:

- `ep14`
- `AP=0.9938`
- `AUC=0.996`
- `F1=0.947`
- `P=0.982`
- `R=0.915`
- `FPR=0.012`

## 2. Post-Fit Ops Result

Ops file:

- `configs/ops/tcn_caucafall_rtD_hneg_plus_continue.yaml`

Offline metrics:

- `outputs/metrics/tcn_caucafall_rtD_hneg_plus_continue.json`

Important fitting note:

1. `fit_ops` again produced a degenerate sweep under confirmation
2. the fitter retried with confirmation disabled
3. the written ops file therefore records:
   - `confirm=false`
   - `confirm_fallback=disabled_confirm_after_degenerate_sweep`

This means Candidate D does not beat Candidate A on runtime-profile robustness. Both candidates currently depend on the same fallback mechanism.

## 3. Canonical Four-Folder Replay Result

Artifacts:

- `artifacts/fall_test_eval_20260427_rtD/unified_tcn_caucafall_rtD_op2.csv`
- `artifacts/fall_test_eval_20260427_rtD/unified_tcn_caucafall_rtD_op2.json`
- `artifacts/fall_test_eval_20260427_rtD/unified_tcn_caucafall_rtD_op2_metrics.json`

Result:

- `TP=6`
- `TN=10`
- `FP=2`
- `FN=6`
- `16/24`

Comparison:

- baseline canonical profile: `13/24`
- Candidate A canonical profile: `16/24`
- Candidate D canonical profile: `16/24`

## 4. Locked 24-Clip Replay Surface

Artifacts:

- `artifacts/fall_test_eval_20260427_locked_candidateD/summary_tcn_caucafall_rtD_lockedsurface_op2.csv`
- `artifacts/fall_test_eval_20260427_locked_candidateD/summary_tcn_caucafall_rtD_lockedsurface_op2.json`
- `artifacts/fall_test_eval_20260427_locked_candidateD/summary_tcn_caucafall_rtD_lockedsurface_op2_metrics.json`

Result:

- `TP=6`
- `TN=10`
- `FP=2`
- `FN=6`
- `16/24`

Comparison:

- baseline locked profile: `15/24`
- Candidate A locked profile: `16/24`
- Candidate D locked profile: `16/24`

## 5. Aggregate Judgment

Candidate D is not a failed run. It does produce a better bounded runtime result than the original baseline.

However, it does **not** currently beat Candidate A on the runtime surfaces that matter most for the report and paper.

Current comparison:

1. both Candidate A and Candidate D improve the canonical four-folder bounded replay check from `13/24` to `16/24`
2. both Candidate A and Candidate D improve the historical locked 24-clip surface from `15/24` to `16/24`
3. both still rely on confirm-disabled fallback during `fit_ops`

So the practical conclusion is:

1. the retraining track has produced a real bounded-runtime improvement over baseline
2. Candidate D does not currently justify replacing Candidate A as the lead strengthening result
3. Candidate A remains the simpler lead story because it reached the same runtime outcome first

## 6. Comparison Artifact

Unified comparison table:

- `docs/reports/notes/candidate_ad_runtime_surface_compare_2026-04-27.csv`
