# Simple sweeps (style matches your sweep_tcn_le2i.py)

These four scripts run Make targets with parameter overrides, parse `history.jsonl`,
rank runs by `monitor_score` (default), and write out the best rerun command.

## Files
- sweep_tcn_le2i.py
- sweep_tcn_caucafall.py
- sweep_gcn_le2i.py
- sweep_gcn_caucafall.py
- sweep_lib_min.py (shared helpers)

## One-time prerequisites
Build windows for RS 30Hz (recommended):
  make pipeline-le2i-30
  make pipeline-caucafall-30

## Run
TCN LE2i:
  python3 sweep_tcn_le2i.py --exp s1

TCN CAUCAFall:
  python3 sweep_tcn_caucafall.py --exp s1

GCN LE2i:
  python3 sweep_gcn_le2i.py --exp s1

GCN CAUCAFall:
  python3 sweep_gcn_caucafall.py --exp s1

## Outputs
Each writes into:
  outputs/sweeps/<arch>/<dataset>/<exp>/

including:
  results.tsv
  best.json
  best_command.sh
  best_overrides.json

Re-run best:
  bash outputs/sweeps/tcn/le2i/<exp>/best_command.sh
