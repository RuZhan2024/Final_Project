# GCN Round-1 Recovery Plan (CAUCAFall)

Date: 2026-03-05

## Objective

Improve CAUCAFall GCN event recall from current deploy-level baseline while keeping false alarms controlled.

## Baseline (current deploy-friendly points)

- `outputs/metrics/gcn_caucafall_stb_s17.json` (OP2 recall=0.8, FA/24h=0)
- `outputs/metrics/gcn_caucafall_cauc_hneg1.json` (OP2 recall=0.8, FA/24h=0)

## Candidate R1

`exp_gcn_caucafall_r1_recovery_seed33724876`

- Resume from `outputs/caucafall_gcn_W48S12_r1_augreg/best.pt` (33-joint compatible)
- Add train-split hard negatives:
  - list: `outputs/hardneg/gcn_caucafall_train_p50.txt`
  - mult: `2`
- Mild regularization:
  - `dropout=0.25`
  - `mask_joint_p=0.08`
  - `mask_frame_p=0.05`
  - `weight_decay=5e-4`
  - `label_smoothing=0.01`
- Stability:
  - `use_ema=1`, `ema_decay=0.995`
  - `monitor=ap`

## Protocol

- Train: `data/processed/caucafall/windows_W48_S12/train`
- Val: `data/processed/caucafall/windows_W48_S12/val`
- Fit OPs: `data/processed/caucafall/windows_eval_W48_S12/val`
- Eval: `data/processed/caucafall/windows_eval_W48_S12/test`

## Promotion Criteria

Promote only if all hold against `gcn_caucafall_stb_s17`:

- OP2 recall >= 0.8
- OP2 F1 >= baseline F1
- OP2 FA/24h <= baseline FA/24h

## Execution Status

- First attempt failed due resume contract mismatch:
  - resumed from a 17-joint checkpoint while current pipeline is 33-joint.
  - hidden dim also mismatched (default 128 vs checkpoint 96).
- Restarted with compatible settings:
  - resume: `outputs/caucafall_gcn_W48S12_r1_augreg/best.pt`
  - `--hidden 96`
- Completed train -> fit_ops -> eval successfully.

## Result

- Summary artifacts:
  - `artifacts/reports/tuning/gcn_round1_recovery_summary.md`
  - `artifacts/reports/tuning/gcn_round1_recovery_summary.csv`
- Outcome:
  - Initial fit (`r1_recovery`) improved AP but OP2 picked a low threshold and produced very high FA/24h.
  - Recalibrated OP fit with `min_tau_high=0.40` (`r1_recovery_min40`) restored deploy-safe OP2 metrics.
  - `r1_recovery_min40` matches `stb_s17` on OP2 recall/F1/FA24h while keeping higher AP.
- Decision:
  - promote `r1_recovery_min40` as the 33-joint GCN deploy candidate.
