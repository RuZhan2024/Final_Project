# LE2i GCN Optimization Round-6 Summary

## Run
- Checkpoint base: `outputs/le2i_gcn_W48S12_opt33_r4_recallpush_promoted/best.pt`
- New training run: `outputs/le2i_gcn_W48S12_opt33_r6_recallpush_focal`
- Key changes: focal loss (`alpha=0.6`, `gamma=1.0`), lower dropout (`0.15`), no masking augmentation, LR `3e-4`, balanced sampler.

## Validation behavior
- Best val AP reached `0.9428` (higher than prior val AP), val recall in training logs around `0.947`.

## Test evaluation (fixed policy for fair compare)
- Policy: `configs/ops/gcn_le2i_paper_profile.yaml`
- Output: `outputs/metrics/gcn_le2i_opt33_r6_recallpush_focal_paperops.json`
- Result: `AP=0.8426, Recall=0.8889, Precision=1.0000, F1=0.9412, FA24h=0.0`

## Compare to promoted r4
- r4 (fixed policy): `AP=0.8435, Recall=0.8889, Precision=1.0000, F1=0.9412, FA24h=0.0`
- r6 (fixed policy): `AP=0.8426, Recall=0.8889, Precision=1.0000, F1=0.9412, FA24h=0.0`

## Fit-ops variant check
- r6 + its fitted ops (`configs/ops/gcn_le2i_opt33_r6_recallpush_focal.yaml`) produced one false alert on test (`FA24h=581.58`), so not acceptable for promotion.

## Conclusion
- Round-6 did not beat r4 on test under the locked policy.
- Keep `r4` as promoted LE2i GCN checkpoint.
