# CAUCAFall Targeted Retrain: Policy Comparison

| tag | AP | F1 | Recall | Precision | FA24h | tau_high | fall_left_recall | metrics |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| old_baselineops | 0.9840 | 0.8889 | 0.8000 | 1.0000 | 0.00 | 0.850 | 0.0 | `outputs/metrics/tcn_caucafall_opt_m2_focal.json` |
| old_midplateauops | 0.9840 | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.430 | 1.0 | `outputs/metrics/tcn_caucafall_opt_m2_focal_midplateau.json` |
| new_baselineops | 0.9812 | 0.8889 | 0.8000 | 1.0000 | 0.00 | 0.850 | 0.0 | `outputs/metrics/tcn_caucafall_opt_m4_hn_targeted_baselineops.json` |
| new_midplateauops | 0.9812 | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.430 | 1.0 | `outputs/metrics/tcn_caucafall_opt_m4_hn_targeted_midplateauops.json` |

## Delta (new - old)
- baseline ops: ΔF1=+0.0000, ΔRecall=+0.0000, ΔFA24h=+0.00
- midplateau ops: ΔF1=+0.0000, ΔRecall=+0.0000, ΔFA24h=+0.00