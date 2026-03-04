# Deployment Lock (Current Recommended)

Date: 2026-03-04

## Purpose

Freeze one known-good deployment profile so frontend, backend, and ML artifacts stay aligned.

## Locked profile

- Dataset: `caucafall`
- Primary model: `TCN`
- Operating point: `OP-2`
- OP-2 thresholds (from locked ops YAML):
  - `tau_high = 0.7099999785`
  - `tau_low = 0.5537999868`
- Confirmation: `off` (as stored in `alert_cfg.confirm=false`)

## Locked artifacts

- Ops YAML (runtime canonical):
  - `configs/ops/tcn_caucafall.yaml`
- Checkpoint referenced by canonical ops:
  - `outputs/caucafall_tcn_W48S12_r1_augreg/best.pt`

## Runtime defaults synced

- Backend default settings:
  - `active_dataset_code = caucafall`
  - `active_model_code = TCN`
  - `active_op_code = OP-2`
  - `fall_threshold ~= 0.71` (fallback)
- Frontend settings fallback threshold:
  - `~0.71`

## Notes

- `GCN` remains available for comparison/auxiliary signal, but not recommended as autonomous alert source in current lock.
- If lock changes, update this file and re-run a quick replay sanity check before release.
