# Ablation Matrix

Primary dataset focus: `CAUCAFall`
Comparative dataset: `LE2i` (selected rows only)

## Priority Rules
- `P0`: required before final write-up freeze.
- `P1`: strong evidence improvement, run if compute/time allows.
- `P2`: optional exploratory.

## Matrix
| Ablation ID | Priority | Dataset | Arch | Change vs promoted baseline | Expected Effect | Status | Command (template) | Acceptance Signal |
|---|---|---|---|---|---|---|---|---|
| A1 | P0 | CAUCAFall | TCN | `use_motion:1->0` | recall drop, AP drop | Done | `make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_ablate_nomotion FEAT_USE_MOTION=0 EPOCHS=120 TCN_PATIENCE=20 && make fit-ops-caucafall eval-caucafall ADAPTER_USE=1 OUT_TAG=_ablate_nomotion FEAT_USE_MOTION=0 ALERT_CONFIRM=0` | `artifacts/reports/tuning/caucafall_tcn_p0_ablation_table.md` (A1 vs FC1 deltas) |
| A2 | P0 | CAUCAFall | TCN | `use_bone:1->0,use_bone_length:1->0` | spatial robustness drop | Done | `make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_ablate_nobone FEAT_USE_BONE=0 FEAT_USE_BONE_LEN=0 EPOCHS=120 TCN_PATIENCE=20 && make fit-ops-caucafall eval-caucafall ADAPTER_USE=1 OUT_TAG=_ablate_nobone FEAT_USE_BONE=0 FEAT_USE_BONE_LEN=0 ALERT_CONFIRM=0` | `artifacts/reports/tuning/caucafall_tcn_p0_ablation_table.md` (A2 vs FC1 deltas) |
| A3 | P0 | CAUCAFall | TCN | `confirm:0->1` (strict policy) | possible FA drop, recall tradeoff | Done | `python scripts/fit_ops.py ... --confirm 1 --out configs/ops/tcn_caucafall_cauc_hneg1_confirm15.yaml` | Already measured in tuning artifacts |
| A4 | P0 | CAUCAFall | GCN | `confirm:1->0` | recall/FA tradeoff check | Done | `python scripts/fit_ops.py ... --confirm 0 --out configs/ops/gcn_caucafall_cauc_hneg1_confirm0.yaml` | Already measured in tuning artifacts |
| A5 | P1 | CAUCAFall | GCN | `use_motion:1->0` | recall/AP drop | Todo | `python scripts/train_gcn.py ... --use_motion 0 --save_dir outputs/caucafall_gcn_ablate_nomotion` | Delta table for C2 support |
| A6 | P1 | LE2i | TCN | repeat A1 | domain sensitivity evidence | Todo | `python scripts/train_tcn.py ... --train_dir data/processed/le2i/... --use_motion 0` | Comparative delta narrative |
| A7 | P1 | LE2i | GCN | repeat A5 | domain sensitivity evidence | Todo | `python scripts/train_gcn.py ... --train_dir data/processed/le2i/... --use_motion 0` | Comparative delta narrative |
| A8 | P2 | CAUCAFall | TCN+GCN | interaction: `confirm x motion` | interaction effect | Todo | run A1/A5 + confirm variants | Interaction figure optional |

## Notes
- Do not tune on test split.
- Keep `W=48`, `S=12`, seed set policy aligned with final candidates.
- For P0 rows, run 3 seeds minimum if compute allows.
