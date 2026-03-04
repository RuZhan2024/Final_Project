# Plot Evidence Checklist

Use this checklist to track required figure generation for final report quality.
Each row must map to reproducible artifacts and one command.

| Plot ID | Plot Type | Scope | Script/Target | Output Path | Status | Notes |
|---|---|---|---|---|---|---|
| P1 | F1-vs-threshold (`tau`) | TCN + GCN on LE2i | `make plot-le2i ADAPTER_USE=1` + `make plot-gcn-le2i ADAPTER_USE=1` | `outputs/plots/tcn_le2i_f1_vs_tau.png`, `outputs/plots/gcn_le2i_f1_vs_tau.png` | DONE | Generated and verified on disk. |
| P2 | F1-vs-threshold (`tau`) | TCN + GCN on CAUCAFall | `make plot-caucafall ADAPTER_USE=1` + `make plot-gcn-caucafall ADAPTER_USE=1` | `outputs/plots/tcn_caucafall_f1_vs_tau.png`, `outputs/plots/gcn_caucafall_f1_vs_tau.png` | DONE | `gcn_caucafall` generated in latest run. |
| P3 | Recall-vs-FA24h | TCN + GCN on LE2i | `make plot-le2i ADAPTER_USE=1` + `make plot-gcn-le2i ADAPTER_USE=1` | `outputs/plots/tcn_le2i_recall_vs_fa.png`, `outputs/plots/gcn_le2i_recall_vs_fa.png` | DONE | Generated and verified on disk. |
| P4 | Recall-vs-FA24h | TCN + GCN on CAUCAFall | `make plot-caucafall ADAPTER_USE=1` + `make plot-gcn-caucafall ADAPTER_USE=1` | `outputs/plots/tcn_caucafall_recall_vs_fa.png`, `outputs/plots/gcn_caucafall_recall_vs_fa.png` | DONE | `gcn_caucafall` generated in latest run. |
| P5 | PR curve / AP comparison | FC1-FC4 final candidates | `python scripts/plot_candidate_metrics.py` | `artifacts/figures/pr_curves/fc1_fc4_ap_comparison.png` | DONE | Current metrics JSON provides AP + selected OP metrics; this figure is AP-comparison oriented (not full PR trace). |
| P6 | Multi-seed distribution (box/violin) | `F1`, `Recall`, `FA24h` | `python scripts/plot_stability_metrics.py --glob "outputs/metrics/*_stb_s*.json" --out_fig artifacts/figures/stability/fc_stability_boxplot.png` | `artifacts/figures/stability/fc_stability_boxplot.png` | DONE | 20 true seed-run metrics files aggregated; summary artifacts generated (`stability_summary.csv/json`). |
| P7 | Cross-dataset transfer bars | in-domain vs cross-dataset | `python scripts/plot_cross_dataset_transfer.py --manifest artifacts/reports/cross_dataset_manifest.json` | `artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png` | DONE | True cross-domain eval artifacts generated (`outputs/metrics/cross_*`), with summary + taxonomy reports. |
| P8 | Latency distribution (p50/p95/tail) | target deploy device | `python scripts/plot_latency_profiles.py` | `artifacts/figures/latency/latency_profile_summary.png` | DONE | Current figure is based on available profile reports; expand with more device profiles as they are collected. |

## Acceptance Rules
- Every `TODO` becomes `DONE` with output path filled.
- Every plot is registered in `docs/project_targets/THESIS_EVIDENCE_MAP.md`.
- No manual figure editing outside reproducible scripts.
