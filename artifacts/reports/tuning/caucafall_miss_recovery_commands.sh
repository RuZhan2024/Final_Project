#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[1/2] Eval with midpoint plateau policy"
python scripts/eval_metrics.py \
  --win_dir data/processed/caucafall/windows_eval_W48_S12/test \
  --ckpt outputs/caucafall_tcn_W48S12_opt_m2_focal/best.pt \
  --ops_yaml configs/ops/tcn_caucafall_opt_m2_focal_midplateau.yaml \
  --out_json outputs/metrics/tcn_caucafall_opt_m2_focal_midplateau.json \
  --fps_default 23 --thr_min 0.001 --thr_max 0.95 --thr_step 0.01

echo "[2/2] Compare baseline vs midpoint policy"
python - <<'PY'
import json
pairs = [
    ("baseline", "outputs/metrics/tcn_caucafall_opt_m2_focal.json"),
    ("midplateau", "outputs/metrics/tcn_caucafall_opt_m2_focal_midplateau.json"),
]
for tag, path in pairs:
    d = json.load(open(path))
    op = (d.get("ops") or {}).get("op2", {})
    miss = (
        d.get("detail", {})
        .get("per_video", {})
        .get("Subject.6/Fall left", {})
        .get("event_metrics", {})
        .get("event_recall")
    )
    print(
        f"{tag}: tau_high={op.get('tau_high')}, "
        f"f1={op.get('f1')}, recall={op.get('recall')}, "
        f"precision={op.get('precision')}, fa24h={(d.get('totals') or {}).get('fa24h', op.get('fa24h'))}, "
        f"fall_left_recall={miss}"
    )
PY

echo "[done]"
