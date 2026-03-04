#!/usr/bin/env bash
set -euo pipefail

# Next cycle: data-centric hard-negative expansion (train split only, no leakage).
# Rule: log parameter changes before each run.

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

LOG_CSV="artifacts/reports/tuning/PARAM_CHANGELOG.csv"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[1/4] Mine additional train-only hard negatives (TCN, CAUCAFall)"
python tools/log_run_change.py \
  --csv "$LOG_CSV" \
  --timestamp_utc "$TS" \
  --exp_id "m10_trainonly_hneg_mine" \
  --base_ref "m2_focal" \
  --changed_params "hardneg_source=train,top_k=200,min_p=0.35,max_per_clip=20,dedup_shift_frames=12" \
  --command "python scripts/mine_hard_negatives.py --ckpt outputs/caucafall_tcn_W48S12_opt_m2_focal/best.pt --windows_dir data/processed/caucafall/windows_W48_S12/train --neg_only 1 --out_txt outputs/hardneg/tcn_caucafall_train_p35_top200.txt --top_k 200 --max_per_clip 20 --min_p 0.35 --dedup_shift_frames 12" \
  --status running \
  --notes "pre-run"

source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" \
python scripts/mine_hard_negatives.py \
  --ckpt outputs/caucafall_tcn_W48S12_opt_m2_focal/best.pt \
  --windows_dir data/processed/caucafall/windows_W48_S12/train \
  --neg_only 1 \
  --out_txt outputs/hardneg/tcn_caucafall_train_p35_top200.txt \
  --top_k 200 --max_per_clip 20 --min_p 0.35 --dedup_shift_frames 12

echo "[2/4] Retrain TCN with expanded train-only hard negatives"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python tools/log_run_change.py \
  --csv "$LOG_CSV" \
  --timestamp_utc "$TS" \
  --exp_id "m10_trainonly_hneg_retrain" \
  --base_ref "m2_focal" \
  --changed_params "hard_neg_list=tcn_caucafall_train_p35_top200,hard_neg_mult=2,resume_use_ckpt_feat_cfg=0" \
  --command "make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m10_hn_trainp35 TCN_RESUME=outputs/caucafall_tcn_W48S12_opt_m2_focal/best.pt TCN_HARD_NEG_LIST=outputs/hardneg/tcn_caucafall_train_p35_top200.txt TCN_HARD_NEG_MULT=2 TCN_RESUME_USE_CKPT_FEAT_CFG=0" \
  --status running \
  --notes "pre-run"

make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m10_hn_trainp35 \
  TCN_RESUME=outputs/caucafall_tcn_W48S12_opt_m2_focal/best.pt \
  TCN_HARD_NEG_LIST=outputs/hardneg/tcn_caucafall_train_p35_top200.txt \
  TCN_HARD_NEG_MULT=2 \
  TCN_RESUME_USE_CKPT_FEAT_CFG=0

echo "[3/4] Fit OP + Eval for m10"
make fit-ops-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m10_hn_trainp35 ALERT_CONFIRM=0
make eval-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m10_hn_trainp35

echo "[4/4] Mark run done in changelog (manual reminder)"
echo "Update rows m10_* to status=done and fill artifacts:"
echo "  outputs/caucafall_tcn_W48S12_opt_m10_hn_trainp35/best.pt"
echo "  configs/ops/tcn_caucafall_opt_m10_hn_trainp35.yaml"
echo "  outputs/metrics/tcn_caucafall_opt_m10_hn_trainp35.json"
