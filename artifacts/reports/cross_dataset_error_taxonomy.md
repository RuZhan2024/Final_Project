# Cross-Dataset Error Taxonomy

Source summary: `artifacts/reports/cross_dataset_summary.csv`

## Observed transfer pattern
- LE2i -> CAUCAFall transfer remained strong for both TCN/GCN in current runs.
- CAUCAFall -> LE2i transfer degraded strongly (`F1=0`, `Recall=0`) with large FA24h increase.

## Top 3 failure modes (prioritized)
1. Domain/camera context mismatch
- Evidence: severe asymmetry between transfer directions, especially `CAUCAFall -> LE2i` collapse.
- Likely factors: scene layout, viewpoint, subject motion style, background dynamics.

2. Temporal/FPS and motion-scale mismatch
- Evidence: policy fitted on source domain does not transfer robustly to target in one direction.
- Likely factors: effective motion magnitude and timing features shift across datasets.

3. Skeleton quality/occlusion distribution shift
- Evidence: cross-domain alert policy appears over/under-sensitive depending on direction.
- Likely factors: pose confidence profile, missing joints, occlusion frequency differences.

## Mitigation plan (next iteration)
- Add source-target calibration guardrails using val-only transfer validation.
- Add robustness checks on low-confidence/missing-joint heavy clips.
- Add mixed-domain threshold sensitivity analysis before deployment lock.
