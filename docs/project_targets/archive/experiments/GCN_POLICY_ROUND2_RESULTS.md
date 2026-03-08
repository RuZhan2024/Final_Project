# GCN Policy Round-2 Results (CAUCAFall)

Date: 2026-03-04

## Objective

Test whether GCN false-alert behavior can be fixed by policy-layer tuning only (no retraining).

## Base checkpoint

- `outputs/caucafall_gcn_W48S12_r1_augreg/best.pt`

## Experiments

1. `exp_gcn_caucafall_r2_policy_confirm_strict_seed33724876`
   - Policy changes:
     - `confirm=1`
     - `confirm_min_lying=0.70`
     - `confirm_max_motion=0.03`
     - `confirm_require_low=1`
   - Output:
     - `configs/ops/gcn_caucafall_r2_confirm_strict.yaml`
     - `outputs/metrics/gcn_caucafall_r2_confirm_strict.json`
   - Result:
     - `TP=5, FP=5, Recall=1.0, Precision=0.5, Event F1=0.6667, FA/24h=4704.5455`
   - Note:
     - fit_ops hit degenerate sweep with strict confirm and auto-fell back to `confirm=0`; outcome unchanged vs baseline.

2. `exp_gcn_caucafall_r2_policy_ultra_high_thr_seed33724876`
   - Policy changes:
     - `thr_min=0.60, thr_max=0.999, min_tau_high=0.85`
     - `op1_recall=0.80, op3_fa24h=0.20`
   - Output:
     - `configs/ops/gcn_caucafall_r2_ultra_high_thr.yaml`
     - `outputs/metrics/gcn_caucafall_r2_ultra_high_thr.json`
   - Result:
     - `TP=0, FP=0, Recall=0.0, Precision=NA, Event F1=NA, FA/24h=0.0`
   - Note:
     - false alerts are suppressed, but true alerts are also completely suppressed.

## Decision

Policy-only tuning is insufficient for current GCN checkpoint behavior on CAUCAFall:

- Normal thresholds: recall stays high but false-alert rate remains unacceptable.
- Very strict thresholds: false alerts disappear but recall collapses.

Therefore, production alerting should remain:

- `TCN` as primary alerting model.
- `GCN` as auxiliary signal only (or disabled for autonomous alert trigger) until train-side changes produce a better precision/FA tradeoff.

## Next actionable commands

1. Keep deploy route on TCN OP-2:
   - `curl -s -X PUT "http://127.0.0.1:8000/api/settings?resident_id=1" -H "Content-Type: application/json" -d '{"active_dataset_code":"caucafall","active_model_code":"TCN","active_op_code":"OP-2","mc_enabled":true,"mc_M":10,"mc_M_confirm":25}'`
2. (Optional) Run GCN train-side Round-3 only after a new plan is recorded in registry.
