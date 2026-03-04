# Model-Side Minimal Grid Results (TCN, CAUCAFall)

| model | AP | F1 | Recall | Precision | FA24h | tau_high | fall_left_recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_best | 0.9840 | 0.8889 | 0.8000 | 1.0000 | 0.00 | 0.850 | 0.0 |
| m8_bce_balanced | 0.9753 | 0.8889 | 0.8000 | 1.0000 | 0.00 | 0.860 | 0.0 |
| m9_focal_balanced | 0.9827 | 0.8889 | 0.8000 | 1.0000 | 0.00 | 0.870 | 0.0 |

## Delta vs current_best
- m8_bce_balanced: ΔF1=+0.0000, ΔRecall=+0.0000, ΔFA24h=+0.00, ΔAP=-0.0088
- m9_focal_balanced: ΔF1=+0.0000, ΔRecall=+0.0000, ΔFA24h=+0.00, ΔAP=-0.0014