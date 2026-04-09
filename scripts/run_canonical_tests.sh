#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "error: .venv not found at $ROOT/.venv" >&2
  exit 1
fi

source .venv/bin/activate
export PYTHONPATH="$ROOT/src:$ROOT"

mode="${1:-torch-free}"

case "$mode" in
  torch-free)
    pytest -q \
      tests/test_import_smoke.py \
      tests/test_server_integration_contract.py \
      tests/test_audit_api_contract.py \
      tests/test_audit_api_v1_parity.py \
      tests/server/test_notification_manager.py \
      tests/server/test_safe_guard_classifier.py \
      tests/server/test_safe_guard_store.py \
      tests/server/test_ai_report.py
    ;;
  monitor)
    pytest -q \
      tests/server/test_runtime_core.py \
      tests/server/test_monitor_predict.py
    ;;
  all)
    "$0" torch-free
    "$0" monitor
    ;;
  *)
    echo "usage: $0 [torch-free|monitor|all]" >&2
    exit 2
    ;;
esac
