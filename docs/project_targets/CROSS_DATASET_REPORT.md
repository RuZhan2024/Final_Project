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
   - `artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png`

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
- Script: `scripts/plot_cross_dataset_transfer.py`
- Input:
  - In-domain metrics JSON (4 baseline files)
  - Cross-domain metrics JSON (4 transfer files)
- Output:
  - `artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png`

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
- Cross-domain evaluation artifacts produced:
  - `outputs/metrics/cross_tcn_le2i_to_caucafall.json`
  - `outputs/metrics/cross_tcn_caucafall_to_le2i.json`
  - `outputs/metrics/cross_gcn_le2i_to_caucafall.json`
  - `outputs/metrics/cross_gcn_caucafall_to_le2i.json`
- Manifest and summary produced:
  - `artifacts/reports/cross_dataset_manifest.json`
  - `artifacts/reports/cross_dataset_summary.csv`
  - `artifacts/reports/cross_dataset_error_taxonomy.md`
- Plot generated:
  - `artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png`
