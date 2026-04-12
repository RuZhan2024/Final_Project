#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "error: .venv not found at $ROOT/.venv" >&2
  exit 1
fi

source .venv/bin/activate
export PYTHONPATH="$ROOT/ml/src:$ROOT"

mode="${1:-torch-free}"

require_torch_importable() {
  # Fail fast here so known local torch import issues do not surface later as
  # less actionable pytest collection aborts.
  if bash -lc 'python3 -c "import torch; print(torch.__version__)" >/dev/null 2>&1'; then
    return 0
  fi

  echo "error: torch-backed canonical tests are not runnable in this environment" >&2
  echo "hint: importing torch failed before pytest collection; use torch-free/frontend modes here or rerun in a clean torch environment" >&2
  exit 1
}

case "$mode" in
  torch-free)
    # Keep this slice runnable on the lowest-friction review environment while
    # still covering the freeze-core regression tests that do not need torch.
    pytest -q \
      qa/tests/test_import_smoke.py \
      qa/tests/test_audit_api_contract.py \
      qa/tests/test_audit_api_v1_parity.py \
      qa/tests/test_datamodule_split_contract.py \
      qa/tests/test_pose_preprocess_config.py \
      qa/tests/test_data_pipeline_window_metadata.py \
      qa/tests/test_data_pipeline_caucafall_label_fps.py \
      qa/tests/server/test_notification_manager.py \
      qa/tests/server/test_notifications_routes.py \
      qa/tests/server/test_safe_guard_classifier.py \
      qa/tests/server/test_safe_guard_store.py \
      qa/tests/server/test_ai_report.py \
      qa/tests/server/test_op_code_normalization.py \
      qa/tests/server/test_events_test_fall_status_contract.py \
      qa/tests/server/test_monitor_repository_event_status.py \
      qa/tests/server/test_dashboard_repository_counts.py
    ;;
  contract)
    require_torch_importable
    pytest -q \
      qa/tests/test_server_integration_contract.py
    ;;
  monitor)
    require_torch_importable
    pytest -q \
      qa/tests/server/test_runtime_core.py \
      qa/tests/server/test_monitor_predict.py
    ;;
  frontend)
    if ! command -v npm >/dev/null 2>&1; then
      echo "error: npm is required for frontend canonical tests" >&2
      exit 1
    fi
    (
      cd "$ROOT/applications/frontend"
      CI=1 npm test -- --runInBand --watchAll=false --watchman=false \
        src/features/monitor/api.test.js
    )
    ;;
  all)
    "$0" torch-free
    "$0" contract
    "$0" monitor
    "$0" frontend
    ;;
  *)
    echo "usage: $0 [torch-free|contract|monitor|frontend|all]" >&2
    exit 2
    ;;
esac
