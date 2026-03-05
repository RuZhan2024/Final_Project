# GCN Overtake Round (Caucafall, 33-joint)

## Key Findings

- Existing deployed GCN config (`configs/ops/gcn_caucafall.yaml`) on test: recall **0.80**, precision **1.00**, f1 **0.889**, fa24h **0.0**.
- Existing deployed TCN config (`configs/ops/tcn_caucafall.yaml`) on test: recall **1.00**, precision **1.00**, f1 **1.000**, fa24h **0.0**.
- This confirms your memory: GCN has a practical deployment point around **recall=0.80** (not 0.60) when using current ops yaml.
- In this round, all 33-joint GCN candidates re-fit with the same conservative settings collapsed to low OP2 threshold (~0.22-0.23), causing large false alerts on test (fa24h ~4704).

## Candidate Refit Results (same fit_ops settings)

| tag | recall | precision | f1 | fa24h | n_gt | n_true | n_alert | tau_high(OP2) | tau_low(OP2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.00 | 0.50 | 0.667 | 4704.5 | 5 | 5 | 10 | 0.23000000417232513 | 0.1793999969959259 |
| r1_augreg | 1.00 | 0.50 | 0.667 | 4704.5 | 5 | 5 | 10 | 0.23000000417232513 | 0.1793999969959259 |
| r1_ctrl | 1.00 | 0.50 | 0.667 | 4704.5 | 5 | 5 | 10 | 0.23000000417232513 | 0.1793999969959259 |
| r1_recovery | 1.00 | 0.50 | 0.667 | 4704.5 | 5 | 5 | 10 | 0.23000000417232513 | 0.1793999969959259 |
| r2_recallpush_a | 1.00 | 0.50 | 0.667 | 4704.5 | 5 | 5 | 10 | 0.2199999988079071 | 0.17159999907016754 |
| r2_recallpush_b | 1.00 | 0.50 | 0.667 | 4704.5 | 5 | 5 | 10 | 0.23000000417232513 | 0.1793999969959259 |

## Why GCN still lags TCN in frontend
- Training AP is high, but deployment behavior depends on OP thresholds + online gating.
- Refit without stronger FA constraints selected overly low tau_high on val; that did not generalize to test/live.
- Current stable GCN deployment point is still below TCN event performance.

## Next executable steps (to truly overtake)
1. Run GCN-only 33-joint sweep with explicit FA-oriented OP fitting (`op2_objective=cost_sensitive`, higher `cost_fp`, `min_tau_high>=0.35`).
2. Evaluate on replay set (same videos you use in frontend) and lock ops from replay+val intersection, not val-only.
3. Keep TCN as primary auto-alert and use GCN as recall/watch channel until GCN reaches >= TCN F1 at FA24h<=TCN.
