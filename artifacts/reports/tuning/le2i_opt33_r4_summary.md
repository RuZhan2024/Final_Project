# LE2i GCN Optimization Round-4 (Recall-Push)

## Objective
Improve LE2i GCN while preserving deployment-safe event metrics (`FA24h=0` under paper-profile ops).

## Training config
- Base: `outputs/le2i_gcn_W48S12_opt33_r1/best.pt`
- New run: `outputs/le2i_gcn_W48S12_opt33_r4_recallpush`
- Key changes: `balanced_sampler=1`, `loss=bce`, `dropout=0.20`, `mask_joint_p=0.04`, `mask_frame_p=0.02`, `lr=5e-4`
- Contract-matching flags kept: `hidden=96`, `two_stream=1`, `use_motion=1`, `use_conf=1`, `use_bone=1`, `use_bonelen=1`

## Evaluation protocol
- Test windows: `data/processed/le2i/windows_eval_W48_S12/test`
- Policy: `configs/ops/gcn_le2i_paper_profile.yaml`
- Output: `outputs/metrics/gcn_le2i_opt33_r4_recallpush_promoted_paperops.json`

## Result vs previous best profile
- Previous (`gcn_le2i_paper_profile.json`):
  - AP `0.8314`, Recall `0.8889`, Precision `1.0000`, F1 `0.9412`, FA24h `0.0`
- Round-4 promoted (`gcn_le2i_opt33_r4_recallpush_promoted_paperops.json`):
  - AP `0.8435`, Recall `0.8889`, Precision `1.0000`, F1 `0.9412`, FA24h `0.0`

## Interpretation
- Improvement achieved on ranking quality (AP) without degrading event-level deployment metrics.
- Remaining recall gap (`8/9`) is not policy-only fixable without severe FA inflation (diagnostic showed recall can hit 1.0 only with very low thresholds and FA24h > 2000).
- Next gain requires targeted training/data interventions for the persistent FN clip.
