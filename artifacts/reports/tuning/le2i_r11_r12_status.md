# LE2i r11/r12 Status

## Summary
- r11a (pos_weight + stronger perturbation): no gain.
- r11b (one-cycle schedule): partial no gain.
- r12 (augmented positive train set): partial no gain.

## Current best remains
- `outputs/le2i_gcn_W48S12_opt33_r8_dataside_noise/best.pt`
- `outputs/metrics/gcn_le2i_opt33_r8_dataside_noise_paperops.json`
- Metrics: AP 0.8451 / Recall 0.8889 / Precision 1.0 / F1 0.9412 / FA24h 0.0

## Implication
Hyperparameter and in-split augmentation changes are no longer moving event recall.
Further improvement likely needs new labeled data diversity or protocol-level redesign.
