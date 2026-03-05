# GCN CAUCAFall OP Fine Scan (Round 1)

- Checkpoint: `outputs/caucafall_gcn_W48S12/best.pt`
- Validation scan: `data/processed/caucafall/windows_eval_W48_S12/val`
- Threshold grid: 0.3..0.6 (step mostly 0.02)

## Selected by Val (deploy-safe objective)
- tau_high: `0.36`
- val: precision=1.0000, recall=1.0000, f1=1.0000, fa24h=0.0000

## Test Confirmation (single shot)
- json: `artifacts/reports/tuning/gcn_opscan_round1/test_selected_tau_0360.json`
- test precision=1.0000, recall=0.8000, f1=0.8889, fa24h=0.0000, tp=4, fp=0, fn=1

## Test Robustness Check (nearby taus)
- `tau=0.36`: precision=1.0000, recall=0.8000, f1=0.8889, fa24h=0.0
- `tau=0.38`: precision=1.0000, recall=0.8000, f1=0.8889, fa24h=0.0
- `tau=0.40`: precision=1.0000, recall=0.8000, f1=0.8889, fa24h=0.0
- `tau=0.41`: precision=1.0000, recall=0.8000, f1=0.8889, fa24h=0.0
- Conclusion: threshold fine-scan alone does not improve current test recall beyond 0.8.

## Recommendation
- Keep current canonical GCN OP (`configs/ops/gcn_caucafall.yaml`) unchanged.
- To improve beyond recall 0.8 at zero FA/24h, next step requires training-side changes (not threshold-only tuning).
