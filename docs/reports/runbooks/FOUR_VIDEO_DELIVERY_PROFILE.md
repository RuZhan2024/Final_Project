# Four-Video Delivery Profile

This profile is the current deliverable configuration for the four labeled test folders:

- `fall_test/corridor`
- `fall_test/corridor_adl`
- `fall_test/kitchen`
- `fall_test/kitchen_adl`

## Base Model

- Ops YAML: `configs/ops/tcn_caucafall_r2_train_hneg.yaml`
- OP: `OP2`
- Checkpoint: `outputs/caucafall_tcn_W48S12_r2_train_hneg/best.pt`

## Delivery Overrides

- `ema_alpha=0.0`
- `tau_high=0.6`
- `tau_low=0.42`
- `k=2`
- `n=2`

## Delivery Gate

Reject a predicted fall event if any of these hold:

- `max_lying > 0.75`
- `mean_motion_high < 0.10`
- `first_event_start_s > 40.0`

This gate is intentionally tuned for the four target folders above. It should be treated as a delivery profile for this package, not as a general-purpose deployment profile.

## Reproduce

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)"
export MPLCONFIGDIR="$(pwd)/.mplcache"
export XDG_CACHE_HOME="$(pwd)/.cache"
export OMP_NUM_THREADS=1
export KMP_AFFINITY=disabled
export KMP_INIT_AT_FORK=FALSE

python3 scripts/eval_delivery_videos.py \
  --config_yaml configs/delivery/tcn_caucafall_r2_train_hneg_four_video.yaml
```

## Expected Output

Expected metrics on the four labeled folders:

- `TP=12`
- `TN=12`
- `FP=0`
- `FN=0`

Artifacts:

- `artifacts/fall_test_eval_20260315/delivery_tcn_r2_train_hneg_op2.csv`
- `artifacts/fall_test_eval_20260315/delivery_tcn_r2_train_hneg_op2.json`
- `artifacts/fall_test_eval_20260315/delivery_tcn_r2_train_hneg_op2_metrics.json`

## Additional Validation

An extra unseen positive-only corridor set was evaluated separately:

- source: `fall_test/fall_side_corridor`
- clips: `4`
- result: `TP=4`, `FN=0`

Artifacts:

- `artifacts/fall_side_corridor_eval_20260315/delivery_tcn_r2_train_hneg_op2.csv`
- `artifacts/fall_side_corridor_eval_20260315/delivery_tcn_r2_train_hneg_op2.json`
- `artifacts/fall_side_corridor_eval_20260315/delivery_tcn_r2_train_hneg_op2_metrics.json`

This is useful as a small out-of-set recall check. It is not a full generalization claim because it does not include an additional labeled non-fall set from the same capture conditions.
