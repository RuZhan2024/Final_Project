#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEED=33724876
TRAIN_DIR=data/processed/caucafall/windows_W48_S12/train
VAL_DIR=data/processed/caucafall/windows_W48_S12/val
EVAL_VAL_DIR=data/processed/caucafall/windows_eval_W48_S12/val
EVAL_TEST_DIR=data/processed/caucafall/windows_eval_W48_S12/test
BASE_CKPT=outputs/caucafall_gcn_W48S12/best.pt

# -------------------------
# A) Recall-push focal, no hard-neg
# -------------------------
python3 scripts/train_gcn.py \
  --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" \
  --save_dir outputs/caucafall_gcn_W48S12_r2_recallpush_a \
  --resume "$BASE_CKPT" \
  --epochs 80 --min_epochs 10 --patience 18 \
  --batch 128 --lr 3e-4 --seed "$SEED" --fps_default 23 \
  --monitor ap \
  --loss focal --focal_alpha 0.35 --focal_gamma 1.5 \
  --hidden 96 --num_blocks 6 --temporal_kernel 9 --base_channels 48 \
  --two_stream 1 --fuse concat --use_adaptive_adj 0 --adaptive_adj_embed 16 \
  --dropout 0.20 --mask_joint_p 0.05 --mask_frame_p 0.03 \
  --weight_decay 0.0002 --label_smoothing 0.0 \
  --use_conf 1 --use_motion 1 --use_bone 1 --use_bonelen 1 \
  --normalize torso --include_centered 1 --include_abs 0 --include_vel 1 \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.05 \
  --use_ema 1 --ema_decay 0.995

python3 scripts/fit_ops.py --arch gcn \
  --val_dir "$EVAL_VAL_DIR" \
  --ckpt outputs/caucafall_gcn_W48S12_r2_recallpush_a/best.pt \
  --out configs/ops/gcn_caucafall_r2_recallpush_a.yaml \
  --fps_default 23 \
  --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --ema_alpha 0.20 --k 2 --n 3 --cooldown_s 30 --tau_low_ratio 0.78 \
  --confirm 0 --confirm_s 2.0 --confirm_min_lying 0.65 --confirm_max_motion 0.08 --confirm_require_low 1 \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.01 \
  --time_mode center --merge_gap_s 1.0 --overlap_slack_s 0.5 \
  --op1_recall 0.95 --op3_fa24h 1.0 --op2_objective f1 \
  --ops_picker conservative --op_tie_break max_thr --tie_eps 1e-3 \
  --save_sweep_json 1 --allow_degenerate_sweep 0 --emit_absolute_paths 0 \
  --min_tau_high 0.40

python3 scripts/eval_metrics.py \
  --win_dir "$EVAL_TEST_DIR" \
  --ckpt outputs/caucafall_gcn_W48S12_r2_recallpush_a/best.pt \
  --ops_yaml configs/ops/gcn_caucafall_r2_recallpush_a.yaml \
  --out_json outputs/metrics/gcn_caucafall_r2_recallpush_a.json \
  --fps_default 23

# -------------------------
# B) Recall-push bce + mild hard-neg
# -------------------------
python3 scripts/train_gcn.py \
  --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" \
  --save_dir outputs/caucafall_gcn_W48S12_r2_recallpush_b \
  --resume "$BASE_CKPT" \
  --hard_neg_list outputs/hardneg/gcn_caucafall_train_p50.txt --hard_neg_mult 1 \
  --epochs 80 --min_epochs 10 --patience 18 \
  --batch 128 --lr 3e-4 --seed "$SEED" --fps_default 23 \
  --monitor ap \
  --loss bce --focal_alpha 0.25 --focal_gamma 2.0 \
  --hidden 96 --num_blocks 6 --temporal_kernel 9 --base_channels 48 \
  --two_stream 1 --fuse concat --use_adaptive_adj 0 --adaptive_adj_embed 16 \
  --dropout 0.22 --mask_joint_p 0.05 --mask_frame_p 0.03 \
  --weight_decay 0.0002 --label_smoothing 0.0 \
  --use_conf 1 --use_motion 1 --use_bone 1 --use_bonelen 1 \
  --normalize torso --include_centered 1 --include_abs 0 --include_vel 1 \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.05 \
  --use_ema 1 --ema_decay 0.995

python3 scripts/fit_ops.py --arch gcn \
  --val_dir "$EVAL_VAL_DIR" \
  --ckpt outputs/caucafall_gcn_W48S12_r2_recallpush_b/best.pt \
  --out configs/ops/gcn_caucafall_r2_recallpush_b.yaml \
  --fps_default 23 \
  --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --ema_alpha 0.20 --k 2 --n 3 --cooldown_s 30 --tau_low_ratio 0.78 \
  --confirm 0 --confirm_s 2.0 --confirm_min_lying 0.65 --confirm_max_motion 0.08 --confirm_require_low 1 \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.01 \
  --time_mode center --merge_gap_s 1.0 --overlap_slack_s 0.5 \
  --op1_recall 0.95 --op3_fa24h 1.0 --op2_objective f1 \
  --ops_picker conservative --op_tie_break max_thr --tie_eps 1e-3 \
  --save_sweep_json 1 --allow_degenerate_sweep 0 --emit_absolute_paths 0 \
  --min_tau_high 0.40

python3 scripts/eval_metrics.py \
  --win_dir "$EVAL_TEST_DIR" \
  --ckpt outputs/caucafall_gcn_W48S12_r2_recallpush_b/best.pt \
  --ops_yaml configs/ops/gcn_caucafall_r2_recallpush_b.yaml \
  --out_json outputs/metrics/gcn_caucafall_r2_recallpush_b.json \
  --fps_default 23
