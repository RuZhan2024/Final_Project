# GCN Robust Augmentation Experiments (v1-v3)

Branch: `exp_gcn_robust_augment`

## Summary
All three augmentation variants underperformed the current locked deployment model.

## Test-set comparison
- locked: recall=1.0, precision=1.0, f1=1.0, fa24h=0.0
- aug_v1: recall=0.8, precision=1.0, f1=0.8889, fa24h=0.0
- aug_v2: recall=0.8, precision=1.0, f1=0.8889, fa24h=0.0
- aug_v3: recall=0.8, precision=1.0, f1=0.8889, fa24h=0.0

CSV:
- `artifacts/reports/gcn_aug/compare_locked_vs_aug_v123.csv`

## Decision
- Keep current locked GCN deployment (`r2_recallpush_b`) unchanged.
- Do not merge augmentation settings into production training defaults.

## Notes
- Augmentations tested here reduced event recall on the current evaluation protocol.
- If further optimization is needed, next step should target thresholding/gating on replay-domain rather than stronger train-time perturbation.
