# Training Upgrades

## Summary
This upgrade improves training correctness and reproducibility for TCN/GCN without changing model families.

### Key changes
- Unified checkpoint schema normalization in `core/ckpt.py`.
- Added `ckpt_version` marker (`2`) and atomic checkpoint writes (`.tmp` + `os.replace`).
- GCN best-checkpoint saving now uses EMA-averaged weights when EMA is enabled.
- GCN now stores and restores `ema_state` on resume when present.
- Added GCN `last.pt` raw-weight checkpoint (best remains selection checkpoint).
- Added TCN `--min_epochs` early-stop gate (parity with GCN behavior).
- Added GCN DataLoader knobs: `--prefetch_factor`, `--persistent_workers`.
- Added `--deterministic` to both trainers.
- `scheduler_metric` default is now **auto** (resolved from `monitor`) in both trainers.

## New / Changed Flags

### Both TCN and GCN
- `--deterministic {0,1}` (default `1`)
  - `1`: `cudnn.deterministic=True`, `cudnn.benchmark=False`
  - `0`: `cudnn.deterministic=False`, `cudnn.benchmark=True`
  - Seed application is unchanged and always applied.

- `--scheduler_metric {val_loss,val_f1,val_ap}` (default: auto)
  - If omitted:
    - `--monitor ap` => `scheduler_metric=val_ap`
    - `--monitor f1` => `scheduler_metric=val_f1`
  - If explicitly set, explicit value is preserved.

### TCN
- `--min_epochs` (default `0`)
  - Patience-based early stop cannot trigger before `ep >= min_epochs`.

### GCN
- `--prefetch_factor` (default `2`)
- `--persistent_workers` (default `1`)
  - Applied only when `num_workers > 0`.

## EMA + Checkpoint Behavior

### GCN
- `best.pt` uses EMA weights when `--use_ema 1`.
- `best.pt` includes `ema_state` when EMA is enabled.
- `last.pt` stores raw model weights (non-EMA) for debugging/comparison.

### TCN
- Existing EMA save behavior remains: best checkpoint is saved under EMA context when enabled.

## Resume Semantics
- Resume restores model weights from checkpoint.
- If checkpoint contains `ema_state` and EMA is enabled, EMA state is restored (warns but does not crash on mismatch).
- Current trainers do **not** restore optimizer/scheduler/scaler states; resume is model (+ optional EMA) focused.

## Backward Compatibility
- Old checkpoints without `ema_state` load safely.
- `save_ckpt` supports both styles:
  - Canonical: `save_ckpt(path, **kwargs)`
  - Legacy: `save_ckpt(path, bundle_dict)`
- Both styles are normalized to the same on-disk key schema.

## Smoke checks
```bash
python -m py_compile src/fall_detection/core/ckpt.py
python -m py_compile src/fall_detection/core/ema.py
python -m py_compile src/fall_detection/training/train_tcn.py
python -m py_compile src/fall_detection/training/train_gcn.py
python -m py_compile scripts/train_tcn.py
python -m py_compile scripts/train_gcn.py
python tools/smoke_ckpt_ema.py
```
