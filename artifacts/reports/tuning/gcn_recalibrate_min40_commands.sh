#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 scripts/fit_ops.py --arch gcn \
  --val_dir data/processed/caucafall/windows_eval_W48_S12/val \
  --ckpt outputs/caucafall_gcn_W48S12_r1_recovery/best.pt \
  --out configs/ops/gcn_caucafall_r1_recovery_min40.yaml \
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
  --win_dir data/processed/caucafall/windows_eval_W48_S12/test \
  --ckpt outputs/caucafall_gcn_W48S12_r1_recovery/best.pt \
  --ops_yaml configs/ops/gcn_caucafall_r1_recovery_min40.yaml \
  --out_json outputs/metrics/gcn_caucafall_r1_recovery_min40.json \
  --fps_default 23
