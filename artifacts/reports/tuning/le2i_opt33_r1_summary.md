# LE2i Optimization Round-1 (33-joint Contract)

## Scope
- Dataset: `le2i`
- Contract: `W48/S12`, 33 joints, adapter pipeline unchanged
- Goal: improve event-level `F1/Recall` without changing model family (TCN/GCN only)

## Commands Executed
```bash
make train-tcn-le2i ADAPTER_USE=1 OUT_TAG=_opt33_r1 \
  EPOCHS=120 TCN_PATIENCE=20 \
  TCN_DROPOUT_le2i=0.30 TCN_MASK_JOINT_P_le2i=0.10 TCN_MASK_FRAME_P_le2i=0.05

make train-gcn-le2i ADAPTER_USE=1 OUT_TAG=_opt33_r1 \
  EPOCHS_GCN=120 GCN_PATIENCE=20 GCN_MIN_EPOCHS=10 \
  GCN_DROPOUT_le2i=0.25 MASK_JOINT_P_le2i=0.08 MASK_FRAME_P_le2i=0.04

make eval-le2i eval-gcn-le2i ADAPTER_USE=1 OUT_TAG=_opt33_r1
```

## Artifacts
- TCN ckpt: `outputs/le2i_tcn_W48S12_opt33_r1/best.pt`
- GCN ckpt: `outputs/le2i_gcn_W48S12_opt33_r1/best.pt`
- TCN ops: `configs/ops/tcn_le2i_opt33_r1.yaml`
- GCN ops: `configs/ops/gcn_le2i_opt33_r1.yaml`
- TCN eval: `outputs/metrics/tcn_le2i_opt33_r1.json`
- GCN eval: `outputs/metrics/gcn_le2i_opt33_r1.json`

## Baseline vs Round-1 (Test, event-level)
| Model | AP | F1 | Recall | Precision | FA24h |
|---|---:|---:|---:|---:|---:|
| TCN baseline (`outputs/metrics/tcn_le2i.json`) | 0.8226 | 0.6667 | 0.5556 | 0.8333 | 581.58 |
| TCN round-1 (`outputs/metrics/tcn_le2i_opt33_r1.json`) | 0.8377 | 0.8235 | 0.7778 | 0.8750 | 581.58 |
| GCN baseline (`outputs/metrics/gcn_le2i.json`) | 0.8641 | 0.7500 | 0.6667 | 0.8571 | 581.58 |
| GCN round-1 (`outputs/metrics/gcn_le2i_opt33_r1.json`) | 0.8314 | 0.8889 | 0.8889 | 0.8889 | 581.58 |

## Interpretation
- `F1/Recall` improved for both TCN and GCN on LE2i.
- `FA24h` unchanged because test set duration is short and both baseline/round-1 still have one false alert event.
- For deployment-style low-false policy on LE2i, next round should target suppressing the remaining false alert with policy tuning and/or hard-negative replay.

## Next Suggested Round (Round-2)
1. Keep round-1 checkpoints fixed.
2. Run targeted policy grid on LE2i to trade `recall` vs `fa24h` (OP2/OP3 behavior).
3. If still one false alert, run one light hard-negative replay round on LE2i train windows only and re-evaluate.
