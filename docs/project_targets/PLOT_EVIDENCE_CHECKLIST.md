# Plot Evidence Checklist

Date: 2026-03-22

Purpose:
Track paper-facing figures under `Paper Protocol Freeze v1`.

Rule:
- Only plots that exist on disk and are intended for the frozen paper pack should be marked `DONE`.
- A plot can be scientifically useful but still be marked `Deferred` if it is not in the current frozen figure pack.

| Plot ID | Plot Type | Scope | Script/Target | Output Path | Status | Notes |
|---|---|---|---|---|---|---|
| P1 | F1-vs-threshold (`tau`) | TCN + GCN on LE2i | `make plot-le2i ADAPTER_USE=1` + `make plot-gcn-le2i ADAPTER_USE=1` | `artifacts/figures/plots/tcn_le2i_f1_vs_tau.png`, `artifacts/figures/plots/gcn_le2i_f1_vs_tau.png` | Deferred | Useful diagnostic figure, but not currently present in the frozen paper figure pack on disk. |
| P2 | F1-vs-threshold (`tau`) | TCN + GCN on CAUCAFall | `make plot-caucafall ADAPTER_USE=1` + `make plot-gcn-caucafall ADAPTER_USE=1` | `artifacts/figures/plots/tcn_caucafall_f1_vs_tau.png`, `artifacts/figures/plots/gcn_caucafall_f1_vs_tau.png` | Deferred | Useful diagnostic figure, but not currently present in the frozen paper figure pack on disk. |
| P3 | Recall-vs-FA24h | TCN + GCN on LE2i | `make plot-le2i ADAPTER_USE=1` + `make plot-gcn-le2i ADAPTER_USE=1` | `artifacts/figures/plots/tcn_le2i_recall_vs_fa.png`, `artifacts/figures/plots/gcn_le2i_recall_vs_fa.png` | Deferred | Useful trade-off figure, but not currently present in the frozen paper figure pack on disk. |
| P4 | Recall-vs-FA24h | TCN + GCN on CAUCAFall | `make plot-caucafall ADAPTER_USE=1` + `make plot-gcn-caucafall ADAPTER_USE=1` | `artifacts/figures/plots/tcn_caucafall_recall_vs_fa.png`, `artifacts/figures/plots/gcn_caucafall_recall_vs_fa.png` | Deferred | Useful trade-off figure, but not currently present in the frozen paper figure pack on disk. |
| P5 | PR curve / AP comparison | FC1-FC4 final candidates | `python scripts/plot_candidate_metrics.py` | `artifacts/figures/pr_curves/fc1_fc4_ap_comparison.png` | DONE | In frozen paper figure pack. Current figure is AP-comparison oriented, not a full PR trace. |
| P6 | Multi-seed distribution (box/violin) | `F1`, `Recall`, `FA24h` | `python scripts/plot_stability_metrics.py --glob "outputs/metrics/*_stb_s*.json" --out_fig artifacts/figures/stability/fc_stability_boxplot.png` | `artifacts/figures/stability/fc_stability_boxplot.png` | DONE | In frozen paper figure pack. Seed-run metrics and summary artifacts exist. |
| P7 | Cross-dataset transfer bars | in-domain vs cross-dataset | `python3 scripts/build_cross_dataset_summary.py --manifest artifacts/reports/cross_dataset_manifest.json --out_csv artifacts/reports/cross_dataset_summary.csv && python3 scripts/plot_cross_dataset_transfer.py --summary_csv artifacts/reports/cross_dataset_summary.csv --out_fig artifacts/figures/report/cross_dataset_transfer_summary.png` | `artifacts/figures/report/cross_dataset_transfer_summary.png` | DONE | In frozen paper figure pack. Cross-domain eval artifacts and summary exist. |
| P8 | Latency distribution (p50/p95/tail) | target deploy device | `python scripts/plot_latency_profiles.py` | `artifacts/figures/latency/latency_profile_summary.png` | DONE | In frozen paper figure pack. Expand with more device profiles only if the paper needs stronger runtime coverage. |

## Acceptance Rules
- Every `DONE` plot must exist on disk.
- Every frozen paper plot is registered in `docs/project_targets/THESIS_EVIDENCE_MAP.md` or explicitly referenced by a frozen evidence row.
- No manual figure editing outside reproducible scripts.

## Frozen Paper Figure Pack

Current frozen paper-facing figure pack:
- `P5`
- `P6`
- `P7`
- `P8`

Plots `P1-P4` remain useful optional diagnostics, but they are not part of the current frozen paper pack unless regenerated and explicitly promoted.
