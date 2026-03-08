# TCN Round-3 Plan (CAUCAFall)

Date: 2026-03-04

## Objective

Recover/maintain high event recall while preserving zero false alerts, using milder regularization than Round-2.

## Baseline to beat

- `outputs/metrics/tcn_caucafall_r1_augreg.json`
- Target: keep `FA/24h=0` and improve or match event recall/F1.

## Experiments

1. `exp_tcn_caucafall_r3_mild_hneg_seed33724876`
   - Resume from `r1_augreg`.
   - Train-only hard negatives enabled (`hard_neg_mult=1`).
   - Milder regularization:
     - `dropout=0.35`
     - `mask_joint_p=0.10`
     - `mask_frame_p=0.06`
     - `weight_decay=5e-4`
     - `label_smoothing=0.01`

2. `exp_tcn_caucafall_r3_mild_nohneg_seed33724876`
   - Resume from `r1_augreg`.
   - No hard negatives.
   - Same mild regularization as above.

## Protocol

- Train on: `data/processed/caucafall/windows_W48_S12/train`
- Fit OPs on: `.../val`
- Eval on: `.../test`
- Record all status transitions in:
  - `artifacts/registry/overfit_experiment_registry.csv`

## Acceptance

- Promote candidate only if all hold:
  - event recall >= baseline recall
  - event F1 >= baseline F1
  - FA/24h <= baseline FA/24h

## Execution Status

- Completed on 2026-03-04:
  - `exp_tcn_caucafall_r3_mild_hneg_seed33724876`
  - `exp_tcn_caucafall_r3_mild_nohneg_seed33724876`
- Output summaries:
  - `artifacts/reports/tuning/tcn_round3_summary.csv`
  - `artifacts/reports/tuning/tcn_round3_summary.md`

## Result

- Round-3 did **not** pass promotion acceptance.
- Compared with `r1_augreg`, Round-3 variants increased AP/AUC but reduced OP2 event recall/F1.
- Decision: keep `r1_augreg` as deployment default; retain Round-3 artifacts as negative-control evidence.
