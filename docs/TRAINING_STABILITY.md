# Training Stability Upgrades

This document summarizes minimal-diff stability controls added to:
- `src/fall_detection/training/train_tcn.py`
- `src/fall_detection/training/train_gcn.py`

Defaults preserve prior behavior unless a new flag is enabled.

## P0 (always active, low risk)

### Non-finite guard
- If `loss` is NaN/Inf, the step is skipped.
- Logs warning with epoch/step/lr/loss.
- `history.jsonl` now includes `nonfinite_skips` (per epoch).

### Gradient norm telemetry
- Captures pre-clip grad norm each train step.
- `history.jsonl` now includes:
  - `train_grad_norm_mean`
  - `train_grad_norm_p95`
  - `train_grad_norm_max`

### GCN fixed-threshold metrics
- GCN now logs fixed-threshold metrics (like TCN):
  - `val_f1_fixed`, `val_p_fixed`, `val_r_fixed`, `val_fpr_fixed`, `val_thr_fixed`

## P1 (default OFF unless changed)

### TCN dataloader controls
New TCN flags:
- `--num_workers` (default `0`)
- `--persistent_workers` (default `0`)
- `--prefetch_factor` (default `None`)

Notes:
- On MPS devices, `num_workers` is forced to `0` for runtime stability.

### Scheduler input stability
New flags (both TCN/GCN):
- `--scheduler_metric {val_loss,val_ap,val_f1}` (default `val_loss`)
- `--scheduler_ema_beta` (default `0.0`, off)

Behavior:
- Checkpoint/best-model selection is unchanged (raw monitor metric).
- `scheduler_metric` and optional EMA smoothing affect only scheduler stepping.

## P2 (default OFF)

### Optional CUDA AMP
New flag (both TCN/GCN):
- `--amp 0|1` (default `0`)

Behavior:
- Enabled only when `--amp 1` and CUDA is available.
- Uses `autocast` + `GradScaler`.

### Scheduler choices
New flags (both TCN/GCN):
- `--scheduler {plateau,cosine,onecycle}` (default `plateau`)
- `--max_lr` (used by `onecycle`, default falls back to `--lr`)

Notes:
- `onecycle` steps per batch.
- `plateau` uses `--scheduler_metric`/`--scheduler_ema_beta`.

## Recommended starting points

Conservative (reproducibility-first):
- `--scheduler plateau --scheduler_metric val_loss --scheduler_ema_beta 0.8 --amp 0`

Throughput on CUDA:
- `--amp 1 --scheduler cosine`

Aggressive optimization:
- `--scheduler onecycle --max_lr <2x-5x base_lr>`

## Smoke commands (1-2 epoch)

### TCN
```bash
python scripts/train_tcn.py \
  --train_dir data/processed/le2i/windows_eval_W48_S12/train \
  --val_dir data/processed/le2i/windows_eval_W48_S12/val \
  --save_dir /tmp/tcn_smoke \
  --epochs 2 --batch 16 \
  --scheduler plateau --scheduler_metric val_loss --scheduler_ema_beta 0.8 \
  --amp 0
```

### GCN
```bash
python scripts/train_gcn.py \
  --train_dir data/processed/le2i/windows_eval_W48_S12/train \
  --val_dir data/processed/le2i/windows_eval_W48_S12/val \
  --save_dir /tmp/gcn_smoke \
  --epochs 2 --batch 16 \
  --scheduler plateau --scheduler_metric val_loss --scheduler_ema_beta 0.8 \
  --amp 0
```

## Validation commands

```bash
python -m compileall src/fall_detection/training
python -m compileall scripts
pytest -q
```
