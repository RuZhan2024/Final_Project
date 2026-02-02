# Deploy modes (v1)

This folder provides a lightweight **CPU-friendly triage runner** that simulates deployment logic
using offline-generated **window NPZ** files.

## Modes

- **tcn**: baseline (more conservative)
- **gcn**: more sensitive
- **dual**: run TCN + GCN and confirm using agreement rules

## Recommended latency targets (typical defaults)

With `FPS=30`, `W=48` (1.6s context), `S=12` (0.4s step):

- **Possible fall**: ~2–3s
- **Confirmed fall**: ~4–6s worst-case (often earlier)

These come from the triage settings:
- possible: `possible_k` within `possible_T_s`
- confirm window: `confirm_T_s` and confirm votes

## Run on a windows folder

Example (dual):

```bash
python deploy/run_modes.py \
  --mode dual \
  --win_dir data/processed/le2i/windows_W48_S12/test_unlabeled \
  --ckpt_tcn outputs/le2i_tcn_W48S12/best.pt \
  --ckpt_gcn outputs/le2i_gcn_W48S12/best.pt \
  --cfg configs/deploy_modes.yaml \
  --prefer_ema 1
