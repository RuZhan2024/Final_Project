# OP Policy Report

## Protocol Checklist (must all be true)
- [x] OP fit uses validation split only.
- [x] No threshold/policy tuning on test split.
- [x] No calibration on test split.
- [x] No best-seed selection by test score.
- [x] Per-seed OP stability summary attached (`artifacts/reports/op123_stability_summary.csv`).

## Current Promoted Policies
| Profile | Arch | Dataset | Ops YAML | Selected OP | Test Metrics Snapshot |
|---|---|---|---|---|---|
| FC1 promoted | TCN | CAUCAFall | `configs/ops/tcn_caucafall_cauc_hneg1_confirm0.yaml` | `op2` | `AP=0.9819, F1=0.8889, Recall=0.8000, FA24h=0.0` |
| FC2 comparative | GCN | CAUCAFall | `configs/ops/gcn_caucafall.yaml` | `op2` | `AP=0.9640, F1=0.8889, Recall=0.8000, FA24h=0.0` |

## OP1/OP2/OP3 Stability Summary (Completed)
Artifacts:
- `artifacts/reports/op123_per_seed.csv`
- `artifacts/reports/op123_stability_summary.csv`
- `artifacts/reports/op123_stability_summary.json`

Key observations from OP2 means (5 seeds):
- `tcn/caucafall`: `F1_mean=0.8611`, `Recall_mean=0.7600`, `FA24h_mean=0.0`
- `gcn/caucafall`: `F1_mean=0.5873`, `Recall_mean=0.4400`, `FA24h_mean=0.0`
- `tcn/le2i`: `F1_mean=0.8235`, `Recall_mean=0.7778`, `FA24h_mean=581.58`
- `gcn/le2i`: `F1_mean=0.7500`, `Recall_mean=0.6667`, `FA24h_mean=581.58`

Policy interpretation:
- CAUCAFall supports low-alert operation (`FA24h=0`) for both architectures under OP2.
- LE2i remains high-alert under current OP2 selection and is therefore comparative/generalization evidence, not deployment acceptance evidence.

## Cost Utility (Template)
Use fixed costs from fit-ops contract:
- `C_fn=5.0`
- `C_fp=1.0`
- `U = C_fn * FN + C_fp * FP`

Implementation note:
- Utility can be computed from event counts in metrics JSON (`n_gt_events`, `n_true_alerts`, `n_false_alerts`) without changing model behavior.

## Validation Commands
```bash
python scripts/eval_metrics.py --win_dir data/processed/caucafall/windows_eval_W48_S12/test --ckpt outputs/caucafall_tcn_W48S12_cauc_hneg1/best.pt --ops_yaml configs/ops/tcn_caucafall_cauc_hneg1_confirm0.yaml --out_json outputs/metrics/tcn_caucafall_cauc_hneg1_confirm0.json
python scripts/eval_metrics.py --win_dir data/processed/caucafall/windows_eval_W48_S12/test --ckpt outputs/caucafall_gcn_W48S12/best.pt --ops_yaml configs/ops/gcn_caucafall.yaml --out_json outputs/metrics/gcn_caucafall.json
```
