# Cross-Dataset Report Task Sheet

## Scope
This document defines the exact execution plan and acceptance criteria for cross-dataset evidence (`P7`).

## Objective
Measure generalization gap by evaluating models trained on one dataset and tested on the other under fixed invariants.

## Protocol Invariants (Must Not Change)
- Same window settings: `W=48`, `S=12`.
- Same feature flags as final candidates.
- Same label mapping and event metric definitions.
- Same OP-fit policy rule: fit on source validation split only.
- No tuning on target test split.

## Required Directions
1. LE2i -> CAUCAFall
2. CAUCAFall -> LE2i

For each direction, run both:
- TCN
- GCN

## Execution Tasks
1. Prepare cross-eval run matrix:
   - `artifacts/reports/cross_dataset_manifest.json`
2. For each matrix row:
   - load source-trained checkpoint
   - fit OP on source-val
   - evaluate on target-test
3. Save outputs:
   - `outputs/metrics/cross_<arch>_<src>_to_<tgt>.json`
4. Build comparison table:
   - in-domain vs cross-domain for `AP`, `F1`, `Recall`, `FA24h`
   - output: `artifacts/reports/cross_dataset_summary.csv`
5. Generate transfer plot:
   - `artifacts/figures/report/cross_dataset_transfer_summary.png`

## Command Templates
```bash
# Template (replace variables per row)
python scripts/eval_metrics.py \
  --win_dir data/processed/<target>/windows_eval_W48_S12/test \
  --ckpt outputs/<source>_<arch>_W48S12/best.pt \
  --ops_yaml configs/ops/<arch>_<source>.yaml \
  --out_json outputs/metrics/cross_<arch>_<source>_to_<target>.json
```

## Plot Task (`P7`)
- Status: `DONE`
- Scripts:
  - `scripts/build_cross_dataset_summary.py`
  - `scripts/plot_cross_dataset_transfer.py`
- Input:
  - `artifacts/reports/cross_dataset_manifest.json`
  - frozen in-domain metrics JSON (4 baseline files)
  - frozen cross-domain metrics JSON (4 transfer files)
- Output:
  - `artifacts/reports/cross_dataset_summary.csv`
  - `artifacts/figures/report/cross_dataset_transfer_summary.png`

## Error Taxonomy Section (Required)
For each direction, report top 3 failure causes:
1. Domain/camera gap
2. FPS or motion-scale mismatch
3. Skeleton quality/occlusion difference

Store as:
- `artifacts/reports/cross_dataset_error_taxonomy.md`

## Acceptance Criteria
- All 4 cross-domain evaluations completed and saved.
- Summary table includes absolute and relative performance drop vs in-domain.
- Transfer bar plot exists and is mapped in `THESIS_EVIDENCE_MAP.md`.
- Error taxonomy documented with concrete examples.

## Current Status
- Frozen candidate roots are defined in [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md):
  - `outputs/caucafall_tcn_W48S12_r2_train_hneg`
  - `outputs/caucafall_gcn_W48S12_r2_recallpush_b`
  - `outputs/le2i_tcn_W48S12_opt33_r2`
  - `outputs/le2i_gcn_W48S12_opt33_r2`
- Frozen cross-domain evaluation artifacts produced:
  - `outputs/metrics/cross_tcn_le2i_opt33_r2_to_caucafall_frozen_20260409.json`
  - `outputs/metrics/cross_gcn_le2i_opt33_r2_to_caucafall_frozen_20260409.json`
  - `outputs/metrics/cross_tcn_caucafall_r2_train_hneg_to_le2i_frozen_20260409.json`
  - `outputs/metrics/cross_gcn_caucafall_r2_recallpush_b_to_le2i_frozen_20260409.json`
- Manifest and summary produced:
  - `artifacts/reports/cross_dataset_manifest.json`
  - `artifacts/reports/cross_dataset_summary.csv`
  - `artifacts/reports/cross_dataset_summary_legacy_pre_refreeze_20260409.csv`
  - `artifacts/reports/cross_dataset_error_taxonomy.md`
- Plot generated:
  - `artifacts/figures/report/cross_dataset_transfer_summary.png`
- Current evidence interpretation:
  - transfer is strongly directional
  - `TCN CAUCAFall -> LE2i` remains a hard failure under the frozen policy (`cross_f1 = 0.0`)
  - `GCN CAUCAFall -> LE2i` no longer supports the old "complete collapse" story because its frozen rerun recovers event-level recall but at very poor `AP` and very high `FA24h`
