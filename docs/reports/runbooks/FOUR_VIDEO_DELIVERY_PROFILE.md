# Four-Folder Custom Replay Check

This runbook now evaluates the four labeled custom folders against the same canonical online monitor profile used by the main `CAUCAFall TCN OP-2` path. It is no longer treated as a delivery-only profile with separate overrides.

- `fall_test/corridor`
- `fall_test/corridor_adl`
- `fall_test/kitchen`
- `fall_test/kitchen_adl`

## Canonical Runtime Profile

- Ops YAML: `configs/ops/tcn_caucafall.yaml`
- OP: `OP2`
- Checkpoint: `deploy_assets/checkpoints/caucafall_tcn_best.pt`

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

## Current Unified-Profile Output

Current metrics on the four labeled folders under the canonical online profile:

- `TP=3`
- `TN=10`
- `FP=2`
- `FN=9`

Artifacts:

- `artifacts/fall_test_eval_20260330/unified_tcn_caucafall_op2.csv`
- `artifacts/fall_test_eval_20260330/unified_tcn_caucafall_op2.json`
- `artifacts/fall_test_eval_20260330/unified_tcn_caucafall_op2_metrics.json`

Interpretation:

- this check is now aligned with the same policy path used by the monitor
- it should be treated as a bounded custom replay check
- it is no longer valid to describe this set as `24/24` under a special delivery-only profile

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
