# MUVIM R3A (Training-side) Summary

## Goal
- Test whether training-side change (disable balanced sampler) improves MUVIM under corrected metric contract.

## Change Set (R3A)
- Keep architecture/features unchanged.
- Primary change: no `--balanced_sampler`.
- Save dirs:
  - `outputs/muvim_tcn_W48S12_r3a_unbalanced`
  - `outputs/muvim_gcn_W48S12_r3a_unbalanced`

## Evaluation Artifacts
- `outputs/metrics/tcn_muvim_r3a_eval.json`
- `outputs/metrics/gcn_muvim_r3a_eval.json`

## Comparison vs R2 Baseline

| run | AP | Precision | Recall | F1 | FA/24h |
|---|---:|---:|---:|---:|---:|
| TCN R2 | 0.0460 | 0.5200 | 0.9091 | 0.6616 | 797.9067 |
| TCN R3A | 0.4311 | 0.4839 | 0.4545 | 0.4687 | 531.9378 |
| GCN R2 | 0.0572 | 0.6000 | 0.9697 | 0.7413 | 598.4300 |
| GCN R3A | 0.1067 | 0.0968 | 0.0909 | 0.0938 | 930.8912 |

## Decision
- Reject R3A for deployment-oriented event metrics.
- Keep R2 ops + quick checkpoints as current MUVIM baseline.
- Notes:
  - TCN-R3A improved score AP but harmed event recall/F1 substantially.
  - GCN-R3A degraded both recall/F1 and FA.
