# CAUCAFall TCN Dual-Policy Deployment Profile

Policies:
- `safe`: `configs/ops/dual_policy/tcn_caucafall_dual_safe.yaml`
- `recall`: `configs/ops/dual_policy/tcn_caucafall_dual_recall.yaml`

## Main Checkpoint Comparison (m2_focal)

| policy | AP | F1 | Recall | Precision | FA24h | tau_high | tau_low |
|---|---:|---:|---:|---:|---:|---:|---:|
| safe | 0.9840 | 0.8889 | 0.8000 | 1.0000 | 0.00 | 0.850 | 0.663 |
| recall | 0.9840 | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.430 | 0.335 |

## Stability Check on 4 Seed Checkpoints

| policy | mean_AP | mean_F1 | mean_Recall | mean_Precision | mean_FA24h | max_FA24h |
|---|---:|---:|---:|---:|---:|---:|
| safe | 0.9827 | 0.8889 | 0.8000 | 1.0000 | 0.00 | 0.00 |
| recall | 0.9827 | 0.9091 | 1.0000 | 0.8333 | 940.91 | 940.91 |

## Deployment Recommendation

1. Primary alarm channel uses `safe` policy.
2. Secondary triage channel uses `recall` policy for operator review only (no automatic emergency action).
3. Keep both channels logged in production for next retraining cycle evidence.
4. Gate for promoting `recall` into auto-alert path: `FA24h <= 5` on multi-seed stress checks.
