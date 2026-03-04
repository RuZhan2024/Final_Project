# CAUCAFall TCN P0 Ablation Results

| Profile | Selected OP | AP | F1 | Recall | Precision | FA24h | Delta AP vs FC1 | Delta F1 | Delta Recall | Delta FA24h | Metrics JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FC1_promoted | op2 | 0.9819 | 0.8889 | 0.8000 | 1.0000 | 0.00 | +0.0000 | +0.0000 | +0.0000 | +0.00 | `outputs/metrics/tcn_caucafall_cauc_hneg1_confirm0.json` |
| A1_nomotion | op2 | 0.9795 | 0.8889 | 0.8000 | 1.0000 | 0.00 | -0.0024 | +0.0000 | +0.0000 | +0.00 | `outputs/metrics/tcn_caucafall_ablate_nomotion.json` |
| A2_nobone | op2 | 0.9787 | 0.8889 | 0.8000 | 1.0000 | 0.00 | -0.0031 | +0.0000 | +0.0000 | +0.00 | `outputs/metrics/tcn_caucafall_ablate_nobone.json` |
