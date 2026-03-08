# LE2i Optimization Round-2 (Hard-Negative Replay)

## Executed
- Train-split hard-negative mining (from round-1 checkpoints)
- TCN round-2 retrain + fit_ops + eval
- GCN round-2 retrain + fit_ops + eval

## Commands
```bash
python scripts/mine_hard_negatives.py --ckpt outputs/le2i_tcn_W48S12_opt33_r1/best.pt \
  --windows_dir data/processed/le2i/windows_W48_S12/train \
  --out_txt outputs/hardneg/tcn_le2i_train_opt33r1_p35_top300.txt \
  --batch 256 --min_p 0.35 --top_k 300 --max_per_clip 20 --neg_only 1 --dedup_shift_frames 12

python scripts/mine_hard_negatives.py --ckpt outputs/le2i_gcn_W48S12_opt33_r1/best.pt \
  --windows_dir data/processed/le2i/windows_W48_S12/train \
  --out_txt outputs/hardneg/gcn_le2i_train_opt33r1_p35_top300.txt \
  --batch 256 --min_p 0.35 --top_k 300 --max_per_clip 20 --neg_only 1 --dedup_shift_frames 12

make train-tcn-le2i ADAPTER_USE=1 OUT_TAG=_opt33_r2 \
  TCN_RESUME=outputs/le2i_tcn_W48S12_opt33_r1/best.pt \
  TCN_HARD_NEG_LIST=outputs/hardneg/tcn_le2i_train_opt33r1_p35_top300.txt \
  TCN_HARD_NEG_MULT=2 TCN_DROPOUT_le2i=0.28 TCN_MASK_JOINT_P_le2i=0.08 TCN_MASK_FRAME_P_le2i=0.03 \
  EPOCHS=120 TCN_PATIENCE=20
make eval-le2i ADAPTER_USE=1 OUT_TAG=_opt33_r2

make train-gcn-le2i ADAPTER_USE=1 OUT_TAG=_opt33_r2 \
  GCN_RESUME=outputs/le2i_gcn_W48S12_opt33_r1/best.pt \
  GCN_HARD_NEG_LIST=outputs/hardneg/gcn_le2i_train_opt33r1_p35_top300.txt \
  GCN_HARD_NEG_MULT=2 GCN_DROPOUT_le2i=0.22 MASK_JOINT_P_le2i=0.06 MASK_FRAME_P_le2i=0.03 \
  GCN_TEMPORAL_DROPOUT_P=0.02 EPOCHS_GCN=120 GCN_PATIENCE=20 GCN_MIN_EPOCHS=10
make eval-gcn-le2i ADAPTER_USE=1 OUT_TAG=_opt33_r2
```

## Results (event-level test)
| Model | Round-1 | Round-2 | Decision |
|---|---:|---:|---|
| TCN AP | 0.8377 | 0.8656 | keep r2 for TCN |
| TCN F1 | 0.8235 | 0.8235 | tie |
| TCN Recall | 0.7778 | 0.7778 | tie |
| TCN Precision | 0.8750 | 0.8750 | tie |
| TCN FA24h | 581.58 | 581.58 | tie |
| GCN AP | 0.8314 | 0.8250 | reject r2 |
| GCN F1 | 0.8889 | 0.6667 | reject r2 |
| GCN Recall | 0.8889 | 0.5556 | reject r2 |
| GCN Precision | 0.8889 | 0.8333 | reject r2 |
| GCN FA24h | 581.58 | 581.58 | tie |

## Lock Recommendation
- LE2i TCN: use `outputs/le2i_tcn_W48S12_opt33_r2/best.pt`
- LE2i GCN: keep `outputs/le2i_gcn_W48S12_opt33_r1/best.pt` (do not promote r2)
