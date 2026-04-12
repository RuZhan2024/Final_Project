#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CORE_PATHS=(
  ".gitignore"
  "Makefile"
  "README.md"
  "docker-compose.yml"
  "requirements.txt"
  "requirements_server.txt"
  "pyproject.toml"
  "applications/frontend/src/App.js"
  "applications/frontend/src/pages/monitor/hooks/usePoseMonitor.js"
  "applications/frontend/package.json"
  "applications/backend/db.py"
  "applications/backend/deploy_runtime.py"
  "applications/backend/routes/events.py"
  "applications/backend/routes/monitor.py"
  "applications/backend/application.py"
  "applications/backend/config.py"
  "ops/scripts/bootstrap_dev.sh"
  "ops/scripts/release_doctor.sh"
  "ops/scripts/run_canonical_tests.sh"
  "ops/scripts/start_fullstack.sh"
  "ops/configs/ops/tcn_caucafall.yaml"
  "ops/configs/ops/gcn_caucafall.yaml"
  "qa/tests/conftest.py"
)

echo "[release-core]"
printf '%s\n' "${CORE_PATHS[@]}"
echo
echo "[git-status-subset]"
git status --short -- "${CORE_PATHS[@]}"
