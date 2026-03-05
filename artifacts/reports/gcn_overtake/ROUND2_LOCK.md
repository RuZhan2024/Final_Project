# GCN Round-2 Lock (Caucafall)

Locked deployment ops file:
- `configs/ops/gcn_caucafall.yaml`

Locked checkpoint:
- `outputs/caucafall_gcn_W48S12_r2_recallpush_b/best.pt`

Lock fit command:
```bash
python scripts/fit_ops.py --arch gcn \
  --val_dir data/processed/caucafall/windows_eval_W48_S12/val \
  --ckpt outputs/caucafall_gcn_W48S12_r2_recallpush_b/best.pt \
  --out configs/ops/gcn_caucafall.yaml \
  --fps_default 23 --center pelvis --use_motion 1 --use_conf_channel 1 --use_bone 1 --use_bone_length 1 \
  --ema_alpha 0.20 --k 2 --n 3 --cooldown_s 30 --tau_low_ratio 0.78 \
  --confirm 0 --confirm_s 2.0 --confirm_min_lying 0.65 --confirm_max_motion 0.08 --confirm_require_low 1 \
  --thr_min 0.01 --thr_max 0.95 --thr_step 0.01 --time_mode center --merge_gap_s 1.0 --overlap_slack_s 0.5 \
  --op1_recall 0.95 --op3_fa24h 1.0 --op2_objective cost_sensitive --cost_fn 5 --cost_fp 10 \
  --ops_picker conservative --op_tie_break max_thr --tie_eps 1e-3 \
  --save_sweep_json 1 --allow_degenerate_sweep 0 --emit_absolute_paths 0 --min_tau_high 0.30
```

Verification eval:
- `outputs/metrics/gcn_caucafall_round2_locked.json`
- Test totals: recall=1.0, precision=1.0, f1=1.0, fa24h=0.0, n_gt=5, n_true=5, n_false=0.

Notes:
- `ops.OP2` values in YAML come from val-fit objective (not test totals).
- Runtime must be restarted to load the updated YAML.
