# CAUCAFall TCN Optimization Progress (Step 2)

| tag | AP | F1(op2) | Recall(op2) | Precision(op2) | FA24h | ΔF1 | ΔRecall | ΔFA24h | metrics |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.9819 | 0.8889 | 0.8000 | 1.0000 | 0.00 | +0.0000 | +0.0000 | +0.00 | `outputs/metrics/tcn_caucafall_cauc_hneg1_confirm0.json` |
| m1_maskfix | 0.9790 | 0.7500 | 0.6000 | 1.0000 | 0.00 | -0.1389 | -0.2000 | +0.00 | `outputs/metrics/tcn_caucafall_opt_m1maskfix.json` |
| m2_bce | 0.9809 | 0.7500 | 0.6000 | 1.0000 | 0.00 | -0.1389 | -0.2000 | +0.00 | `outputs/metrics/tcn_caucafall_opt_m2_bce.json` |
| m2_focal | 0.9840 | 0.8889 | 0.8000 | 1.0000 | 0.00 | +0.0000 | +0.0000 | +0.00 | `outputs/metrics/tcn_caucafall_opt_m2_focal.json` |
| m3_hn3 | 0.9866 | 0.7500 | 0.6000 | 1.0000 | 0.00 | -0.1389 | -0.2000 | +0.00 | `outputs/metrics/tcn_caucafall_opt_m3_hn3.json` |

## Current Selection Rule
- Keep safety constraint: `FA24h <= 5`.
- Among safe candidates, maximize `Recall`, then `F1`, then `AP`.

## Current Best Safe Candidate
- tag: `m2_focal`
- ckpt: `outputs/caucafall_tcn_W48S12_opt_m2_focal/best.pt`
- ops: `configs/ops/tcn_caucafall_opt_m2_focal.yaml`
- metrics: `outputs/metrics/tcn_caucafall_opt_m2_focal.json`