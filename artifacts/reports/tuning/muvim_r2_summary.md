# MUVIM R2 (Post Metric-Contract Fix) Summary

## Scope
- No retraining.
- Re-fit operating points on validation set with corrected event-metric contract.
- Evaluate on test set using quick checkpoints.

## Commands Used
```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 scripts/fit_ops.py --arch tcn \
  --val_dir data/processed/muvim/windows_eval_W48_S12/val \
  --ckpt outputs/muvim_tcn_W48S12_quick/best.pt \
  --out configs/ops/tcn_muvim_quick_r2.yaml \
  --fps_default 30 --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --ema_alpha 0.20 --k 2 --n 3 --cooldown_s 30 --tau_low_ratio 0.78 --confirm 0 \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.01 \
  --time_mode center --merge_gap_s 1.0 --overlap_slack_s 0.5 \
  --op1_recall 0.95 --op3_fa24h 1.0 --op2_objective f1 \
  --cost_fn 5.0 --cost_fp 1.0 --ops_picker conservative --op_tie_break max_thr --tie_eps 1e-3 \
  --save_sweep_json 1 --allow_degenerate_sweep 0 --emit_absolute_paths 0 --min_tau_high 0.20

python3 scripts/fit_ops.py --arch gcn \
  --val_dir data/processed/muvim/windows_eval_W48_S12/val \
  --ckpt outputs/muvim_gcn_W48S12_quick/best.pt \
  --out configs/ops/gcn_muvim_quick_r2.yaml \
  --fps_default 30 --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --ema_alpha 0.20 --k 2 --n 3 --cooldown_s 30 --tau_low_ratio 0.78 --confirm 0 \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.01 \
  --time_mode center --merge_gap_s 1.0 --overlap_slack_s 0.5 \
  --op1_recall 0.95 --op3_fa24h 1.0 --op2_objective f1 \
  --cost_fn 5.0 --cost_fp 1.0 --ops_picker conservative --op_tie_break max_thr --tie_eps 1e-3 \
  --save_sweep_json 1 --allow_degenerate_sweep 0 --emit_absolute_paths 0 --min_tau_high 0.20

python3 scripts/eval_metrics.py --win_dir data/processed/muvim/windows_eval_W48_S12/test \
  --ckpt outputs/muvim_tcn_W48S12_quick/best.pt \
  --ops_yaml configs/ops/tcn_muvim_quick_r2.yaml \
  --out_json outputs/metrics/tcn_muvim_quick_r2_eval.json \
  --fps_default 30

python3 scripts/eval_metrics.py --win_dir data/processed/muvim/windows_eval_W48_S12/test \
  --ckpt outputs/muvim_gcn_W48S12_quick/best.pt \
  --ops_yaml configs/ops/gcn_muvim_quick_r2.yaml \
  --out_json outputs/metrics/gcn_muvim_quick_r2_eval.json \
  --fps_default 30
```

## Result vs Fixed Baseline

| run | AP | Precision | Recall | F1 | FA/24h |
|---|---:|---:|---:|---:|---:|
| TCN fixed baseline | 0.0460 | 0.2982 | 0.5152 | 0.3778 | 1329.8445 |
| TCN R2 ops | 0.0460 | 0.5200 | 0.9091 | 0.6616 | 797.9067 |
| GCN fixed baseline | 0.0572 | 0.2000 | 0.3333 | 0.2500 | 1329.8445 |
| GCN R2 ops | 0.0572 | 0.6000 | 0.9697 | 0.7413 | 598.4300 |

Notes:
- AP is model score quality (threshold-independent), unchanged here because no retraining.
- Event metrics improved significantly due better operating-point selection under corrected metric contract.
- OP3 remains weak because `op3_fa24h=1.0` is not reachable on this dataset slice.

## Promotion
- Promoted R2 ops to quick defaults:
  - `configs/ops/tcn_muvim_quick.yaml`
  - `configs/ops/gcn_muvim_quick.yaml`
