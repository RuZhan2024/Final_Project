# GCN Round-2 Train Plan (CAUCAFall)

Date: 2026-03-05
Branch: `opt_gcn_caucafall_opscan`

## Objective

Try to improve CAUCAFall GCN event recall beyond 0.8 while preserving deploy-safe false alerts.

## Current baseline (must beat)

- Metrics: `outputs/metrics/gcn_caucafall.json`
- OP2: `Recall=0.8`, `F1=0.8889`, `FA24h=0.0`, `Precision=1.0`

## Round-2 candidates

### A) `exp_gcn_caucafall_r2_recallpush_a_seed33724876`
- Resume from current canonical checkpoint: `outputs/caucafall_gcn_W48S12/best.pt`
- No hard negatives (avoid further recall suppression)
- Recall-oriented train knobs:
  - `loss=focal`, `focal_alpha=0.35`, `focal_gamma=1.5`
  - `dropout=0.20`
  - `mask_joint_p=0.05`, `mask_frame_p=0.03`
  - `weight_decay=2e-4`, `label_smoothing=0.0`
- Policy fit guard remains enabled: `min_tau_high=0.40`

### B) `exp_gcn_caucafall_r2_recallpush_b_seed33724876`
- Resume from current canonical checkpoint
- Mild hard negatives (`hard_neg_mult=1`) + BCE for stability
- Keep same policy fit guard (`min_tau_high=0.40`)

## Acceptance criteria

Promote only if all true on test OP2:
- `FA24h <= 0.0`
- `Recall > 0.8` (strictly better than baseline)
- `F1 >= 0.8889`

If no candidate passes: keep current canonical unchanged.

## Execution Status

- Completed candidate A:
  - metrics: `outputs/metrics/gcn_caucafall_r2_recallpush_a.json`
  - ops: `configs/ops/gcn_caucafall_r2_recallpush_a.yaml`
- Completed candidate B:
  - metrics: `outputs/metrics/gcn_caucafall_r2_recallpush_b.json`
  - ops: `configs/ops/gcn_caucafall_r2_recallpush_b.yaml`
- Summary:
  - `artifacts/reports/tuning/gcn_round2_summary.md`
  - `artifacts/reports/tuning/gcn_round2_summary.csv`

## Result

- Both Round-2 train-side candidates improved AP, but OP2 recall dropped to `0.2` at `FA24h=0`.
- Neither candidate met acceptance criteria against the canonical baseline (`Recall=0.8`, `F1=0.8889`, `FA24h=0`).
- Decision: keep canonical GCN deployment profile unchanged (`configs/ops/gcn_caucafall.yaml` + `outputs/caucafall_gcn_W48S12/best.pt`).
