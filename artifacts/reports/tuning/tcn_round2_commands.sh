#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)"

SEED=33724876

echo "[R2-1] Mine train-only hard negatives (leakage-safe)"
python3 scripts/mine_hard_negatives.py \
  --ckpt outputs/caucafall_tcn_W48S12_r1_augreg/best.pt \
  --windows_dir data/processed/caucafall/windows_W48_S12/train \
  --out_txt outputs/hardneg/tcn_caucafall_trainmix_r2.txt \
  --batch 256 \
  --min_p 0.50 \
  --top_k 500 \
  --max_per_clip 3 \
  --neg_only 1 \
  --dedup_shift_frames 12 \
  --verbose 1

echo "[R2-1] Train TCN with train-hardneg replay"
make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_r2_train_hneg \
  SPLIT_SEED=${SEED} \
  TCN_RESUME=outputs/caucafall_tcn_W48S12_r1_augreg/best.pt \
  TCN_HARD_NEG_LIST=outputs/hardneg/tcn_caucafall_trainmix_r2.txt \
  TCN_HARD_NEG_MULT=2 \
  TCN_DROPOUT=0.40 \
  TCN_MASK_JOINT_P=0.12 \
  TCN_MASK_FRAME_P=0.08

echo "[R2-1] Fit ops + eval"
python3 scripts/fit_ops.py --arch tcn \
  --val_dir data/processed/caucafall/windows_W48_S12/val \
  --ckpt outputs/caucafall_tcn_W48S12_r2_train_hneg/best.pt \
  --out configs/ops/tcn_caucafall_r2_train_hneg.yaml \
  --fps_default 23 \
  --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --ema_alpha 0.20 --k 2 --n 3 --cooldown_s 30 --tau_low_ratio 0.78 \
  --confirm 0 --confirm_s 2.0 --confirm_min_lying 0.65 --confirm_max_motion 0.08 --confirm_require_low 1 \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.01 \
  --time_mode center --merge_gap_s 1.0 --overlap_slack_s 0.5 \
  --op1_recall 0.95 --op3_fa24h 1.0 \
  --ops_picker conservative --op_tie_break max_thr --tie_eps 1e-3 \
  --save_sweep_json 1 --allow_degenerate_sweep 0 --emit_absolute_paths 0 --min_tau_high 0.20

python3 scripts/eval_metrics.py \
  --win_dir data/processed/caucafall/windows_W48_S12/test \
  --ckpt outputs/caucafall_tcn_W48S12_r2_train_hneg/best.pt \
  --ops_yaml configs/ops/tcn_caucafall_r2_train_hneg.yaml \
  --out_json outputs/metrics/tcn_caucafall_r2_train_hneg.json \
  --fps_default 23

echo "[R2-2] Train TCN with stronger regularization + train-hardneg"
make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_r2_train_hneg_plus \
  SPLIT_SEED=${SEED} \
  TCN_RESUME=outputs/caucafall_tcn_W48S12_r1_augreg/best.pt \
  TCN_HARD_NEG_LIST=outputs/hardneg/tcn_caucafall_trainmix_r2.txt \
  TCN_HARD_NEG_MULT=2 \
  TCN_DROPOUT=0.45 \
  TCN_MASK_JOINT_P=0.16 \
  TCN_MASK_FRAME_P=0.10 \
  TCN_WEIGHT_DECAY=0.0015 \
  TCN_LABEL_SMOOTHING=0.04

echo "[R2-2] Fit ops + eval"
python3 scripts/fit_ops.py --arch tcn \
  --val_dir data/processed/caucafall/windows_W48_S12/val \
  --ckpt outputs/caucafall_tcn_W48S12_r2_train_hneg_plus/best.pt \
  --out configs/ops/tcn_caucafall_r2_train_hneg_plus.yaml \
  --fps_default 23 \
  --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --ema_alpha 0.20 --k 2 --n 3 --cooldown_s 30 --tau_low_ratio 0.78 \
  --confirm 0 --confirm_s 2.0 --confirm_min_lying 0.65 --confirm_max_motion 0.08 --confirm_require_low 1 \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.01 \
  --time_mode center --merge_gap_s 1.0 --overlap_slack_s 0.5 \
  --op1_recall 0.95 --op3_fa24h 1.0 \
  --ops_picker conservative --op_tie_break max_thr --tie_eps 1e-3 \
  --save_sweep_json 1 --allow_degenerate_sweep 0 --emit_absolute_paths 0 --min_tau_high 0.20

python3 scripts/eval_metrics.py \
  --win_dir data/processed/caucafall/windows_W48_S12/test \
  --ckpt outputs/caucafall_tcn_W48S12_r2_train_hneg_plus/best.pt \
  --ops_yaml configs/ops/tcn_caucafall_r2_train_hneg_plus.yaml \
  --out_json outputs/metrics/tcn_caucafall_r2_train_hneg_plus.json \
  --fps_default 23

echo "[ok] TCN Round-2 done"
