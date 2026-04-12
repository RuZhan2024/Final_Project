#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
RESIDENT_ID="${RESIDENT_ID:-1}"
OUT_MD="${OUT_MD:-artifacts/reports/deployment_lock_validation.md}"

mkdir -p "$(dirname "$OUT_MD")"

health_json="$(curl -s "${API_BASE}/api/health" || true)"
settings_json="$(curl -s "${API_BASE}/api/settings?resident_id=${RESIDENT_ID}" || true)"
openapi_json="$(curl -s "${API_BASE}/openapi.json" || true)"

health_ok="no"
if echo "$health_json" | grep -qi '"ok"\|"status"\|"healthy"\|"api_online"'; then
  health_ok="yes"
fi

has_predict_window="no"
if echo "$openapi_json" | grep -q '"/api/monitor/predict_window"'; then
  has_predict_window="yes"
fi

active_model="$(echo "$settings_json" | python3 -c 'import json,sys
try:
 d=json.load(sys.stdin); print(((d.get("system") or {}).get("active_model_code")) or "")
except Exception: print("")')"
active_dataset="$(echo "$settings_json" | python3 -c 'import json,sys
try:
 d=json.load(sys.stdin); print(((d.get("system") or {}).get("active_dataset_code")) or "")
except Exception: print("")')"
active_op="$(echo "$settings_json" | python3 -c 'import json,sys
try:
 d=json.load(sys.stdin); print(((d.get("system") or {}).get("active_op_code")) or "")
except Exception: print("")')"
fall_thr="$(echo "$settings_json" | python3 -c 'import json,sys
try:
 d=json.load(sys.stdin); print(((d.get("system") or {}).get("fall_threshold")))
except Exception: print("")')"

now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

cat > "$OUT_MD" <<EOF
# Deployment Lock Validation

- Generated at (UTC): ${now_utc}
- API base: \`${API_BASE}\`
- Resident ID: \`${RESIDENT_ID}\`

## Auto checks

| Check | Result | Notes |
|---|---|---|
| API health reachable | ${health_ok} | GET \`${API_BASE}/api/health\` |
| Predict endpoint exists | ${has_predict_window} | \`/api/monitor/predict_window\` in OpenAPI |
| Active model | ${active_model:-N/A} | expected: \`TCN\` |
| Active dataset | ${active_dataset:-N/A} | expected: \`caucafall\` |
| Active OP | ${active_op:-N/A} | expected: \`OP-2\` |
| Fall threshold | ${fall_thr:-N/A} | expected around \`0.71\` |

## Manual replay checks (fill after test)

| Clip type | File | Expected | Observed | Pass/Fail | Notes |
|---|---|---|---|---|---|
| Non-fall |  | No alert / no repeated false events |  |  |  |
| Fall |  | Single clear fall event |  |  |  |

## Verdict

- [ ] PASS: lock profile usable for demo
- [ ] FAIL: requires further tuning/fixes
EOF

echo "[ok] wrote ${OUT_MD}"
