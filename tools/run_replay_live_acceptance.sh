#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
RESIDENT_ID="${RESIDENT_ID:-1}"
OUT_MD="${OUT_MD:-artifacts/reports/replay_live_acceptance.md}"

mkdir -p "$(dirname "$OUT_MD")"

health_json="$(curl -s "${API_BASE}/api/health" || true)"
settings_json="$(curl -s "${API_BASE}/api/settings?resident_id=${RESIDENT_ID}" || true)"
specs_json="$(curl -s "${API_BASE}/api/spec" || true)"

health_ok="no"
if echo "$health_json" | grep -qi '"ok"\|"status"\|"healthy"\|"api_online"'; then
  health_ok="yes"
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
mc_enabled="$(echo "$settings_json" | python3 -c 'import json,sys
try:
 d=json.load(sys.stdin); print(bool((d.get("system") or {}).get("mc_enabled")))
except Exception: print("False")')"

spec_count="$(echo "$specs_json" | python3 -c 'import json,sys
try:
 d=json.load(sys.stdin); print(len((d.get("specs") or [])))
except Exception: print(0)')"

ops_guard_ok="$(python3 - <<'PY'
import yaml
ok = True
for p in ("configs/ops/tcn_caucafall.yaml", "configs/ops/gcn_caucafall.yaml"):
    try:
        d = yaml.safe_load(open(p, "r", encoding="utf-8")) or {}
        ops = d.get("ops") or {}
        for k in ("OP1", "OP2", "OP3"):
            lg = (ops.get(k) or {}).get("live_guard")
            if not isinstance(lg, dict):
                ok = False
    except Exception:
        ok = False
print("yes" if ok else "no")
PY
)"

monitor_summary_polling="no"
if rg -q "useApiSummary" apps/src/pages/Monitor.js; then
  monitor_summary_polling="yes"
fi

now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

cat > "$OUT_MD" <<EOF
# Replay + Live Acceptance Report

- Generated at (UTC): ${now_utc}
- API base: \`${API_BASE}\`
- Resident ID: \`${RESIDENT_ID}\`

## Auto checks

| Check | Result | Notes |
|---|---|---|
| API health reachable | ${health_ok} | GET \`${API_BASE}/api/health\` |
| Deploy specs available | ${spec_count} | GET \`${API_BASE}/api/spec\` |
| Active model | ${active_model:-N/A} | expected test profile: \`TCN\` or \`GCN\` |
| Active dataset | ${active_dataset:-N/A} | expected test profile: \`caucafall\` |
| Active OP | ${active_op:-N/A} | expected test profile: \`OP-2\` |
| MC enabled | ${mc_enabled} | usually \`True\` for monitor |
| OP live_guard present (TCN/GCN OP1/2/3) | ${ops_guard_ok} | checks \`configs/ops/*_caucafall.yaml\` |
| Monitor page summary polling enabled | ${monitor_summary_polling} | expected: \`no\` |

## Manual replay acceptance (fixed clips)

| Case | Mode/Profile | Expected | Observed | Pass/Fail | Notes |
|---|---|---|---|---|---|
| Non-fall replay | caucafall + TCN + OP-2 | no fall event |  |  |  |
| Fall replay | caucafall + TCN + OP-2 | clear fall event |  |  |  |
| Non-fall replay | caucafall + GCN + OP-2 | no fall event |  |  |  |
| Fall replay | caucafall + GCN + OP-2 | clear fall event |  |  |  |

## Manual live acceptance (high-performance machine)

| Case | Profile | Expected | Observed | Pass/Fail | Notes |
|---|---|---|---|---|---|
| Standing / ADL | caucafall + TCN + OP-2 | no repeated false fall |  |  |  |
| Controlled fall | caucafall + TCN + OP-2 | fall detected |  |  |  |
| Standing / ADL | caucafall + GCN + OP-2 | no repeated false fall |  |  |  |
| Controlled fall | caucafall + GCN + OP-2 | fall detected |  |  |  |

## Acceptance gate

- [ ] Replay baseline stable and repeatable
- [ ] Live baseline acceptable on target hardware
- [ ] Ready for demo lock
EOF

echo "[ok] wrote ${OUT_MD}"
