# LE2i Baseline Bundle (Commit 58813e8)

This folder stores reproducible baseline artifacts extracted from commit `58813e8` without requiring model retraining.

Included:
- `ops_tcn.yaml`: tracked operating-point config from `58813e8`.
- `ops_gcn.yaml`: tracked operating-point config from `58813e8`.
- `dataset_contract.json`: current-window dataset contract snapshot used for gate checks (counts/pos/neg/fps).
- `performance_baseline.json`: performance parity schema placeholder (targets currently unset).
- `make baseline-capture-le2i MODEL=tcn|gcn`: helper command to populate `performance_baseline.json` targets from a pinned metrics file + checkpoint path.

Important:
- `outputs/metrics/*.json` are gitignored and were not tracked in git history, so model-performance baseline metrics (F1/Recall/FA from committed reports) are unavailable from git alone.
- Performance parity strict mode (`make audit-parity-le2i-strict`) should only be enabled once a pinned checkpoint + provenance-backed metrics artifact is added and `performance_baseline.json` targets are populated.
