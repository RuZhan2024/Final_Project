#!/usr/bin/env bash
set -euo pipefail

# CAUCAFall-only tuning bundle (TCN primary)
# Usage:
#   bash artifacts/reports/tuning/caucafall_next_commands.sh

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

BASE_TAG="_cauc_hneg1"
BASE_CKPT="outputs/caucafall_tcn_W48S12/best.pt"
WIN_TRAIN="data/processed/caucafall/windows_W48_S12/train"
WIN_EVAL_TEST="data/processed/caucafall/windows_eval_W48_S12/test"
HNEG_TXT="outputs/hardneg/tcn_caucafall_evalmix_p50.txt"

mkdir -p artifacts/reports/tuning outputs/hardneg

echo "[1/6] Mine hard negatives from CAUCAFall train windows (leak-safe)"
python scripts/mine_hard_negatives.py \
  --ckpt "$BASE_CKPT" \
  --windows_dir "$WIN_TRAIN" \
  --out_txt "$HNEG_TXT" \
  --batch 128 \
  --min_p 0.50 \
  --top_k 2000 \
  --max_per_clip 50 \
  --neg_only 1 \
  --dedup_shift_frames 12 \
  --verbose 1

echo "[2/6] Retrain TCN on CAUCAFall with hard-negative replay"
make train-tcn-caucafall ADAPTER_USE=1 \
  TCN_RESUME="$BASE_CKPT" \
  TCN_HARD_NEG_LIST="$HNEG_TXT" \
  TCN_HARD_NEG_MULT=2 \
  OUT_TAG="$BASE_TAG"

echo "[3/6] Fit + eval baseline policy on replayed checkpoint"
make fit-ops-caucafall ADAPTER_USE=1 OUT_TAG="$BASE_TAG"
make eval-caucafall ADAPTER_USE=1 OUT_TAG="$BASE_TAG"

echo "[4/6] Confirmation-policy ablations (same checkpoint, no retraining)"
# Variant A: confirm off
make fit-ops-caucafall ADAPTER_USE=1 OUT_TAG="${BASE_TAG}_confirm0" ALERT_CONFIRM=0
make eval-caucafall ADAPTER_USE=1 OUT_TAG="${BASE_TAG}_confirm0"

# Variant B: shorter confirm window
make fit-ops-caucafall ADAPTER_USE=1 OUT_TAG="${BASE_TAG}_confirm15" \
  ALERT_CONFIRM=1 ALERT_CONFIRM_S=1.5 ALERT_CONFIRM_MIN_LYING=0.65 ALERT_CONFIRM_MAX_MOTION=0.08
make eval-caucafall ADAPTER_USE=1 OUT_TAG="${BASE_TAG}_confirm15"

# Variant C: stricter/longer confirm window
make fit-ops-caucafall ADAPTER_USE=1 OUT_TAG="${BASE_TAG}_confirm30" \
  ALERT_CONFIRM=1 ALERT_CONFIRM_S=3.0 ALERT_CONFIRM_MIN_LYING=0.72 ALERT_CONFIRM_MAX_MOTION=0.06
make eval-caucafall ADAPTER_USE=1 OUT_TAG="${BASE_TAG}_confirm30"

echo "[5/6] Summarize tuned CAUCAFall metrics"
python scripts/summarize_metrics_table.py \
  outputs/metrics/tcn_caucafall${BASE_TAG}.json \
  outputs/metrics/tcn_caucafall${BASE_TAG}_confirm0.json \
  outputs/metrics/tcn_caucafall${BASE_TAG}_confirm15.json \
  outputs/metrics/tcn_caucafall${BASE_TAG}_confirm30.json \
  | tee artifacts/reports/tuning/caucafall_tcn_hneg_ablation_table.md

echo "[6/6] Reminder: promote best profile by deployment gate"
echo "Gate recommendation: maximize Recall subject to FA24h <= target, then tie-break by delay_p95/F1."
echo "Done."
