Date: 2026-04-27  
Owner: report/paper finalization branch  
Purpose: compare `Candidate A` against the existing `15/24` bounded custom replay line on the same 24-clip surface.

# Candidate A Locked-Surface Evaluation

## 1. Comparison Surface

This note evaluates the same bounded 24-clip custom replay surface that underlies the current `15/24` runtime discussion in the report and paper.

Baseline artifact:

- `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op2.csv`

Candidate A artifact:

- `artifacts/fall_test_eval_20260427_locked_candidateA/summary_tcn_caucafall_rtA_lockedsurface_op2.csv`

Machine-readable comparison:

- `docs/reports/notes/candidate_a_locked_surface_compare_2026-04-27.csv`

## 2. Aggregate Result

Baseline `tcn_caucafall_locked_op2`:

- `TP=5`
- `TN=10`
- `FP=2`
- `FN=7`
- `15/24`

Candidate A on the same surface:

- `TP=6`
- `TN=10`
- `FP=2`
- `FN=6`
- `16/24`

Interpretation:

1. Candidate A improves the bounded custom replay line by one additional recovered fall clip
2. the gain comes without worsening the ADL false-positive burden
3. this is a real improvement, but it is still modest rather than decisive

## 3. Folder-Level Breakdown

Baseline:

- `corridor`: `TP=3`, `FN=3`
- `corridor_adl`: `TN=6`
- `kitchen`: `TP=2`, `FN=4`
- `kitchen_adl`: `FP=2`, `TN=4`

Candidate A:

- `corridor`: `TP=4`, `FN=2`
- `corridor_adl`: `TN=6`
- `kitchen`: `TP=2`, `FN=4`
- `kitchen_adl`: `FP=2`, `TN=4`

So the improvement is concentrated in the corridor fall subset. The kitchen fall subset does not improve on this surface.

## 4. Reproducibility Note

The historical locked ops file:

- `ops/configs/ops/tcn_caucafall_locked.yaml`

does not currently replay cleanly on this branch because its relative checkpoint path resolves to:

- `ops/outputs/caucafall_tcn_W48S12_r1_ctrl/best.pt`

which does not exist in the current branch layout.

This does not invalidate the baseline comparison because the original locked replay artifact already exists and remains readable:

- `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op2.csv`
- `artifacts/fall_test_eval_20260315/metrics_tcn_caucafall_locked_op2.json`

But it does mean the historical locked replay path is not currently one-command reproducible without either:

1. repairing the checkpoint reference in the old ops YAML, or
2. recreating an equivalent locked config under the current branch layout

## 5. Promotion Judgment

This result strengthens Candidate A, but it still does not justify immediate promotion as the new defended runtime line.

Reason:

1. it improves both bounded runtime surfaces checked so far
2. but only modestly:
   - `13/24 -> 16/24` on the canonical four-folder replay check
   - `15/24 -> 16/24` on the historical locked 24-clip surface
3. and its fitted ops still depend on a confirm-disabled fallback during `fit_ops`

Safe wording:

- a recall-oriented continuation of the `CAUCAFall + TCN` line improved both bounded custom replay checks by reducing false negatives without increasing false positives, but the gain remained modest and the post-fit runtime profile still relied on a confirm-disabled fallback
