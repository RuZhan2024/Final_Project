# MUVIM Tuning R1 (train-side)

Goal: improve deployment-oriented outcomes without changing data format/model family.

## Exp-TCN-R1
- Base: outputs/muvim_tcn_W48S12_quick
- Changes:
  - epochs 20 (from 5)
  - lr 1e-4
  - dropout 0.30
  - mask_joint_p 0.08, mask_frame_p 0.05
  - weight_decay 3e-4
  - label_smoothing 0.02
- Output dir: outputs/muvim_tcn_W48S12_tune_r1

## Exp-GCN-R1
- Base: outputs/muvim_gcn_W48S12_quick
- Changes:
  - epochs 12
  - lr 1e-4
  - dropout 0.25
  - x_noise_std 0.01
  - temporal_dropout_p 0.10
  - weight_decay 3e-4
  - label_smoothing 0.02
- Output dir: outputs/muvim_gcn_W48S12_tune_r1

## Eval pipeline per exp
1. fit_ops (confirm off to avoid degenerate sweep fallback variance)
2. eval_metrics on windows_eval test
3. compare vs quick baselines:
   - tcn_muvim_quick.json
   - gcn_muvim_quick.json

## R1 execution outcome

### Ops-only sweep
- Executed variants:
  - `configs/ops/tcn_muvim_quick_c{1..4}_*.yaml`
  - `configs/ops/gcn_muvim_quick_c{1..4}_*.yaml`
- Outcome: all 8 variants worse than `*_quick` baselines.

### Train-side probes
- TCN probe outputs:
  - `outputs/muvim_tcn_W48S12_tune_r1/`
  - `outputs/muvim_tcn_W48S12_hneg_r1/`
- GCN probe output:
  - `outputs/muvim_gcn_W48S12_hneg_r1_smoke/`
- Outcome: all probes degraded vs quick baselines under same eval entrypoint.

### Comparison (selected policy in each metrics JSON)

| run | ap | precision | recall | f1 | fa24h | false_alerts | alert_events |
|---|---:|---:|---:|---:|---:|---:|---:|
| tcn_quick | 0.8850 | 0.8571 | 1.2000 | 1.0000 | 189.4737 | 1 | 7 |
| tcn_tune_r1 | 0.2403 | 0.0000 | 0.0000 | 0.0000 | 66.4922 | 2 | 2 |
| tcn_hneg_r1 | 0.2941 | 0.0000 | 0.0000 | 0.0000 | 66.4922 | 2 | 2 |
| gcn_quick | 0.8663 | 0.9000 | 1.8000 | 1.2000 | 189.4737 | 1 | 10 |
| gcn_hneg_r1_smoke | 0.1860 | 0.0000 | 0.0000 | 0.0000 | 66.4922 | 2 | 2 |

## Decision
- Keep current MUVIM `*_quick` checkpoints/ops as baseline.
- Reject R1 tuning candidates.
- Next step before more tuning: metric-contract audit for event counting (current outputs show recall/f1 > 1.0 in some files), then rerun tuning under corrected contract.
