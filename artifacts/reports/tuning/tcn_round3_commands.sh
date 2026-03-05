#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)"

SEED=33724876
TRAIN_DIR="data/processed/caucafall/windows_W48_S12/train"
VAL_DIR="data/processed/caucafall/windows_W48_S12/val"
TEST_DIR="data/processed/caucafall/windows_W48_S12/test"
BASE_CKPT="outputs/caucafall_tcn_W48S12_r1_augreg/best.pt"
HNEG_LIST="outputs/hardneg/tcn_caucafall_trainmix_r2.txt"

echo "[R3-A] mild + train hard-neg"
python3 scripts/train_tcn.py \
  --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" \
  --epochs 200 --batch 128 --lr 1e-3 --seed "$SEED" --fps_default 23 \
  --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --motion_scale_by_fps 1 --conf_gate 0.20 --use_precomputed_mask 1 \
  --resume "$BASE_CKPT" \
  --hard_neg_list "$HNEG_LIST" --hard_neg_mult 1 \
  --loss bce --focal_alpha 0.25 --focal_gamma 2.0 \
  --hidden 128 --num_blocks 4 --kernel 3 --use_tsm 0 --tsm_fold_div 8 \
  --grad_clip 1.0 --patience 30 --monitor ap \
  --thr_min 0.05 --thr_max 0.95 --thr_step 0.01 \
  --dropout 0.35 --mask_joint_p 0.10 --mask_frame_p 0.06 \
  --weight_decay 0.0005 --label_smoothing 0.01 \
  --pos_weight auto \
  --save_dir outputs/caucafall_tcn_W48S12_r3_mild_hneg

python3 scripts/fit_ops.py --arch tcn \
  --val_dir "$VAL_DIR" \
  --ckpt outputs/caucafall_tcn_W48S12_r3_mild_hneg/best.pt \
  --out configs/ops/tcn_caucafall_r3_mild_hneg.yaml \
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
  --win_dir "$TEST_DIR" \
  --ckpt outputs/caucafall_tcn_W48S12_r3_mild_hneg/best.pt \
  --ops_yaml configs/ops/tcn_caucafall_r3_mild_hneg.yaml \
  --out_json outputs/metrics/tcn_caucafall_r3_mild_hneg.json \
  --fps_default 23

echo "[R3-B] mild + no hard-neg"
python3 scripts/train_tcn.py \
  --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" \
  --epochs 200 --batch 128 --lr 1e-3 --seed "$SEED" --fps_default 23 \
  --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --motion_scale_by_fps 1 --conf_gate 0.20 --use_precomputed_mask 1 \
  --resume "$BASE_CKPT" \
  --loss bce --focal_alpha 0.25 --focal_gamma 2.0 \
  --hidden 128 --num_blocks 4 --kernel 3 --use_tsm 0 --tsm_fold_div 8 \
  --grad_clip 1.0 --patience 30 --monitor ap \
  --thr_min 0.05 --thr_max 0.95 --thr_step 0.01 \
  --dropout 0.35 --mask_joint_p 0.10 --mask_frame_p 0.06 \
  --weight_decay 0.0005 --label_smoothing 0.01 \
  --pos_weight auto \
  --save_dir outputs/caucafall_tcn_W48S12_r3_mild_nohneg

python3 scripts/fit_ops.py --arch tcn \
  --val_dir "$VAL_DIR" \
  --ckpt outputs/caucafall_tcn_W48S12_r3_mild_nohneg/best.pt \
  --out configs/ops/tcn_caucafall_r3_mild_nohneg.yaml \
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
  --win_dir "$TEST_DIR" \
  --ckpt outputs/caucafall_tcn_W48S12_r3_mild_nohneg/best.pt \
  --ops_yaml configs/ops/tcn_caucafall_r3_mild_nohneg.yaml \
  --out_json outputs/metrics/tcn_caucafall_r3_mild_nohneg.json \
  --fps_default 23

echo "[ok] TCN Round-3 completed"
