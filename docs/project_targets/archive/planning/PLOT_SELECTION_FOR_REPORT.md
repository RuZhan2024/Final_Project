# Plot Selection For Report

This document maps lecture-style plot techniques to the current fall-detection project,
with executable commands for reproducible figure generation.

## 1) Priority Plot Set (Use In Report/Paper)

### P1. Threshold Trade-off Curves
- Purpose: show decision-threshold behavior and operating-point rationale.
- Plot types:
  - F1 vs tau
  - Recall vs FA24h
- Why it fits this project:
  - OP-1/OP-2/OP-3 selection is threshold-driven.
  - Directly supports deployment policy discussion.
- Commands:
```bash
make plot-caucafall ADAPTER_USE=1
make plot-gcn-caucafall ADAPTER_USE=1
make plot-le2i ADAPTER_USE=1
make plot-gcn-le2i ADAPTER_USE=1
```

### P2. In-domain vs Cross-dataset Bars
- Purpose: show generalization gap.
- Plot type:
  - grouped bar charts (in-domain vs cross-domain) for F1/Recall/FA24h.
- Why it fits this project:
  - Explicitly tests transfer robustness across datasets.
- Command:
```bash
python scripts/plot_cross_dataset_transfer.py \
  --summary_csv artifacts/reports/cross_dataset_summary.csv \
  --out_fig artifacts/figures/report/cross_dataset_transfer_summary.png
```

### P3. Stability Distribution Plot (Multi-seed)
- Purpose: show reproducibility variance, not only single-run peak.
- Plot type:
  - boxplot (or equivalent distribution chart) per candidate/config.
- Why it fits this project:
  - Strengthens scientific credibility and thesis defensibility.
- Command:
```bash
python scripts/plot_stability_metrics.py \
  --glob 'outputs/metrics/*_stb_s*.json' \
  --out_fig artifacts/figures/stability/fc_stability_boxplot.png
```

### P4. Latency Profile Summary
- Purpose: show deployment readiness and tail latency behavior.
- Plot type:
  - latency summary chart (p50/p95/tail).
- Why it fits this project:
  - Links model quality to real-time constraints.
- Command:
```bash
python scripts/plot_latency_profiles.py
```

### P5. Candidate Comparison Plot
- Purpose: compact summary of final candidate set.
- Plot type:
  - multi-model comparison figure (AP/F1/Recall/FA24h as designed by script).
- Why it fits this project:
  - Good for result chapter overview figure.
- Command:
```bash
python scripts/plot_candidate_metrics.py
```

## 2) Secondary Plot Set (Optional But Valuable)

### S1. Dataset/Class Balance Bars
- Purpose: show class/video distribution and split sanity.
- Lecture mapping: `plt.bar`.
- Suggested implementation:
  - use labels/splits JSON/TXT to plot per-split class counts.
- Command:
```bash
make plot-balance-caucafall ADAPTER_USE=1
make plot-balance-le2i ADAPTER_USE=1
```

### S2. Skeleton Window Visual Inspection
- Purpose: qualitative evidence for data quality and failure modes.
- Lecture mapping: `plt.imshow` multi-panel inspection style.
- Suggested implementation:
  - show representative normal/fall/uncertain windows with key frames.

### S3. Confusion Matrix Heatmap
- Purpose: complement event metrics with classification error structure.
- Lecture mapping: `sklearn.confusion_matrix` + heatmap style.
- Suggested implementation:
  - compute at selected OP policy on eval/test split.
- Command:
```bash
make plot-confmat-caucafall ADAPTER_USE=1
make plot-confmat-gcn-caucafall ADAPTER_USE=1
```

### S4. Scatter Plot For Failure Analysis
- Purpose: expose relation between confidence, delay, and false alarms.
- Lecture mapping: `plt.scatter`.
- Suggested implementation:
  - points: event-level predictions, grouped by TP/FP/FN.
- Command:
```bash
make plot-failure-caucafall ADAPTER_USE=1
make plot-failure-gcn-caucafall ADAPTER_USE=1
```

## 3) Lecture Techniques That Map Well To This Project

From `LECTURE_TECHNIQUES_AND_PLOTS.md`, the most reusable patterns are:
- `plt.plot`: training/eval curves
- `plt.subplots`: side-by-side model/dataset comparisons
- `plt.bar`: grouped comparison of metrics
- `plt.imshow`: sample/frame-level qualitative checks
- `plt.scatter`: diagnostic relations and error clusters

These are already aligned with the project's current plotting scripts and artifact structure.

## 4) Recommended Figure Bundle For Final Submission

Use this as the minimum high-quality set:
1. F1-vs-tau + Recall-vs-FA24h for CAUCAFall and LE2i (TCN/GCN)
2. Cross-dataset transfer bars
3. Stability distribution plot (multi-seed)
4. Latency summary plot
5. Candidate comparison overview

## 5) Reproducibility Notes

- Prefer make/script commands above over notebook-only figure generation.
- Save outputs under `artifacts/figures/` or `artifacts/figures/plots/` and reference them in:
  - `docs/project_targets/THESIS_EVIDENCE_MAP.md`
- If a figure is updated, record command + commit in evidence map.
