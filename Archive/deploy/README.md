# Deploy modes (v1)

This folder adds a lightweight **3-mode triage runner** for CPU testing.

## Modes

- **tcn**: baseline (conservative)
- **gcn**: sensitive
- **dual**: best overall (TCN + GCN) using agreement confirmation

## Latency targets (recommended defaults)

With `FPS=30`, `W=48` (1.6s context), `S=12` (0.4s step):

- **Possible fall (push only)**: <= **2.5s**
- **Confirmed fall**: <= **6.0s** worst-case (often earlier)

Defaults implement this via:
- possible: `K=3` within `2.0s`
- confirm window: `3.6s`

## Run on a windows folder

Example (dual):

```bash
python deploy/run_modes.py \
  --mode dual \
  --win_dir data/processed/le2i/windows_W48_S12/test_unlabeled \
  --ckpt_tcn outputs/le2i_tcn_W48S12/best.pt \
  --ckpt_gcn outputs/le2i_gcn_W48S12/best.pt \
  --cfg configs/deploy_modes.yaml
```

## MC Dropout (Option 2)

CPU-friendly: enable MC dropout **only during confirm**:

```yaml
mc:
  M: 1
  M_confirm: 12
```

This keeps normal streaming fast while improving confirmation quality.
