# LE2i GCN Optimization Round-8 (Data-Side Noise)

## Goal
Improve LE2i robustness without test leakage by adding lightweight input perturbation during training.

## Config
- Base checkpoint: `outputs/le2i_gcn_W48S12_opt33_r4_recallpush_promoted/best.pt`
- Run output: `outputs/le2i_gcn_W48S12_opt33_r8_dataside_noise`
- Key params:
  - `x_noise_std=0.01`
  - `x_quant_step=0.002`
  - `loss=bce`
  - `dropout=0.18`
  - `mask_joint_p=0.00`, `mask_frame_p=0.00`

## Result (fixed paper-profile policy)
- Baseline r4: `AP=0.8435, Recall=0.8889, Precision=1.0000, F1=0.9412, FA24h=0.0`
- r8: `AP=0.8451, Recall=0.8889, Precision=1.0000, F1=0.9412, FA24h=0.0`

## Decision
- r8 is a small but valid improvement under identical evaluation policy.
- Promote r8 as LE2i paper-profile locked checkpoint candidate.
