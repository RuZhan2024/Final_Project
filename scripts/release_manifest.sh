#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CORE_PATHS=(
  ".gitignore"
  "Makefile"
  "README.md"
  "apps/src/App.js"
  "apps/src/pages/monitor/hooks/usePoseMonitor.js"
  "server/db.py"
  "server/deploy_runtime.py"
  "server/routes/events.py"
  "server/routes/monitor.py"
  "scripts/bootstrap_dev.sh"
  "scripts/release_doctor.sh"
  "scripts/start_fullstack.sh"
  "tests/conftest.py"
)

DOC_PATHS=(
  "docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md"
  "docs/reports/runbooks/ONLINE_FRONTEND_SMOKE_CHECKLIST.md"
  "docs/reports/runbooks/DELIVERY_RELEASE_BOUNDARY.md"
)

echo "[release-core]"
printf '%s\n' "${CORE_PATHS[@]}"
echo
echo "[release-docs]"
printf '%s\n' "${DOC_PATHS[@]}"
echo
echo "[git-status-subset]"
git status --short -- "${CORE_PATHS[@]}" "${DOC_PATHS[@]}"
