# MUVIM R3B (Fine-tune) Summary

## Intent
- Keep sampling strategy stable.
- Try small-step fine-tuning from quick checkpoints:
  - lower LR
  - mild label smoothing
  - mild weight decay

## Runs
- TCN:
  - checkpoint: `outputs/muvim_tcn_W48S12_r3b/best.pt`
  - metrics: `outputs/metrics/tcn_muvim_r3b_eval.json`
- GCN (smoke):
  - checkpoint: `outputs/muvim_gcn_W48S12_r3b_smoke/best.pt`
  - metrics: `outputs/metrics/gcn_muvim_r3b_smoke_eval.json`

## Comparison vs R2/R3A

| run | AP | Precision | Recall | F1 | FA/24h |
|---|---:|---:|---:|---:|---:|
| TCN R2 | 0.0460 | 0.5200 | 0.9091 | 0.6616 | 797.9067 |
| TCN R3A | 0.4311 | 0.4839 | 0.4545 | 0.4687 | 531.9378 |
| TCN R3B | 0.3144 | 0.2029 | 0.4848 | 0.2861 | 1828.5362 |
| GCN R2 | 0.0572 | 0.6000 | 0.9697 | 0.7413 | 598.4300 |
| GCN R3A | 0.1067 | 0.0968 | 0.0909 | 0.0938 | 930.8912 |
| GCN R3B-smoke | 0.2635 | 0.1622 | 0.2121 | 0.1838 | 1030.6295 |

## Decision
- Reject R3B for deployment metrics.
- Keep R2 as MUVIM default baseline.
