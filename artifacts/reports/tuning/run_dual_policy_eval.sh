#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

CKPT="outputs/caucafall_tcn_W48S12_opt_m2_focal/best.pt"
WIN_DIR="data/processed/caucafall/windows_eval_W48_S12/test"

PYTHONPATH="$(pwd)/src:$(pwd)" python scripts/eval_metrics.py \
  --win_dir "$WIN_DIR" \
  --ckpt "$CKPT" \
  --ops_yaml configs/ops/dual_policy/tcn_caucafall_dual_safe.yaml \
  --out_json outputs/metrics/tcn_caucafall_dual_safe.json \
  --fps_default 23

PYTHONPATH="$(pwd)/src:$(pwd)" python scripts/eval_metrics.py \
  --win_dir "$WIN_DIR" \
  --ckpt "$CKPT" \
  --ops_yaml configs/ops/dual_policy/tcn_caucafall_dual_recall.yaml \
  --out_json outputs/metrics/tcn_caucafall_dual_recall.json \
  --fps_default 23

echo "[ok] dual policy eval complete"
