#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_BIN="${PY_BIN:-python3}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[err] missing required command: $1" >&2
    exit 1
  fi
}

require_cmd "${PY_BIN}"
require_cmd npm

cd "${ROOT_DIR}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[bootstrap] creating .venv"
  "${PY_BIN}" -m venv .venv
fi

source ".venv/bin/activate"

if ! python3 -c "import fastapi, uvicorn, yaml, numpy" >/dev/null 2>&1; then
  echo "[bootstrap] installing python dependencies"
  python3 -m pip install -r requirements.txt
  python3 -m pip install -e . --no-build-isolation
fi

if [[ ! -d "applications/frontend/node_modules" ]]; then
  echo "[bootstrap] installing frontend dependencies"
  (cd applications/frontend && npm install)
fi

exec bash ops/scripts/start_fullstack.sh
