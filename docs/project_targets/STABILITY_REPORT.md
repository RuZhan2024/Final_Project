# Stability Report Task Sheet

## Scope
This document defines the exact execution plan and acceptance criteria for multi-seed stability evidence (`P6`).

## Objective
Quantify run-to-run stability for final frozen candidates (FC1-FC4) with reproducible metrics and plots.

## Protocol
- Candidate set: FC1-FC4 from `docs/project_targets/FINAL_CANDIDATES.md`.
- Seed policy:
  - Exploration: 3 seeds.
  - Final reporting: 5 seeds.
- Metrics to report:
  - Event-level: `F1`, `Recall`, `Precision`, `FA24h`.
  - Optional window-level: `AP`.
- Selection policy:
  - OP fitting remains val-only per run.
  - No test-based parameter selection.

## Execution Tasks
1. Create run manifest with canonical `EXP` IDs:
   - `artifacts/registry/stability_manifest.csv`
2. Train/eval each candidate with seed set:
   - Example seed set: `33724876, 1337, 2025, 42, 17`
3. For each run, produce:
   - `outputs/<exp>/best.pt`
   - `configs/ops/<exp>.yaml` (or mapped ops filename)
   - `outputs/metrics/<exp>.json`
4. Aggregate all run metrics to:
   - `artifacts/reports/stability_summary.json`
   - `artifacts/reports/stability_summary.csv`
5. Generate stability figure(s):
   - `artifacts/figures/stability/fc_stability_boxplot.png`
   - `artifacts/figures/stability/fc_stability_violin.png` (optional)

## Command Templates
Use one command family per candidate and vary only `--seed` and output tag.

```bash
# Example pattern (adapt target names already in Makefile)
make train-<arch>-<dataset> ADAPTER_USE=1 SPLIT_SEED=<seed> OUT_TAG=_s<seed>
make fit-ops-<arch>-<dataset> ADAPTER_USE=1 SPLIT_SEED=<seed> OUT_TAG=_s<seed>
make eval-<arch>-<dataset> ADAPTER_USE=1 SPLIT_SEED=<seed> OUT_TAG=_s<seed>
```

If using direct scripts instead of Make, keep canonical `EXP` naming and record full command in registry.

## Plot Task (`P6`)
- Status: `DONE`
- Script: `scripts/plot_stability_metrics.py`
- Input: all per-seed metrics JSON files (`outputs/metrics/*_stb_s*.json`)
- Output: `artifacts/figures/stability/fc_stability_boxplot.png`

## Acceptance Criteria
- [x] 5-seed results exist for each final candidate.
- [x] Summary table includes `mean`, `std`, `95% CI` for required metrics.
- [x] Stability plot exists and is mapped in `THESIS_EVIDENCE_MAP.md`.
- [x] Rows are reproducible from manifest + metrics artifacts.

## Current Status
- Stability execution completed for all manifest rows:
  - `artifacts/registry/stability_manifest.csv` (`done=20`)
- Per-seed artifacts generated:
  - `outputs/metrics/*_stb_s*.json` (20 files)
  - `configs/ops/*_stb_s*.yaml` (20 files)
- Aggregated reports generated:
  - `artifacts/reports/stability_summary.json`
  - `artifacts/reports/stability_summary.csv`
- Stability figure regenerated from true seed artifacts:
  - `artifacts/figures/stability/fc_stability_boxplot.png`
- OP1/OP2/OP3 per-seed stability artifacts are also available:
  - `artifacts/reports/op123_per_seed.csv`
  - `artifacts/reports/op123_stability_summary.csv`
  - `artifacts/reports/op123_stability_summary.json`
