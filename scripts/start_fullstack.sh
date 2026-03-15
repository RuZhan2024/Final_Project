#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BROWSER_MODE="${BROWSER:-none}"
BACKEND_LOG="${BACKEND_LOG:-/tmp/fall_detection_backend.log}"
HEALTH_URL="http://${BACKEND_HOST}:${BACKEND_PORT}/api/health"
API_BASE="http://${BACKEND_HOST}:${BACKEND_PORT}"
BACKEND_PID=""

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[err] missing required command: $1" >&2
    exit 1
  fi
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "[err] missing required path: $1" >&2
    exit 1
  fi
}

port_in_use() {
  local host="$1"
  local port="$2"
  python3 - "$host" "$port" <<'PY'
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((host, port))
except OSError:
    sys.exit(0)
finally:
    s.close()
sys.exit(1)
PY
}

wait_for_health() {
  local attempts="${1:-30}"
  local i
  for ((i=1; i<=attempts; i++)); do
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

cleanup() {
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" >/dev/null 2>&1; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
    wait "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

require_cmd python3
require_cmd npm
require_cmd curl
require_path "${ROOT_DIR}/.venv/bin/activate"
require_path "${ROOT_DIR}/apps/node_modules"

if port_in_use "${BACKEND_HOST}" "${BACKEND_PORT}"; then
  echo "[err] backend port already in use: ${BACKEND_HOST}:${BACKEND_PORT}" >&2
  exit 1
fi

if port_in_use "${FRONTEND_HOST}" "${FRONTEND_PORT}"; then
  echo "[err] frontend port already in use: ${FRONTEND_HOST}:${FRONTEND_PORT}" >&2
  exit 1
fi

echo "[dev] starting backend on ${BACKEND_HOST}:${BACKEND_PORT}"
(
  cd "${ROOT_DIR}"
  source ".venv/bin/activate"
  export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}"
  exec python3 -m uvicorn server.app:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}"
) >"${BACKEND_LOG}" 2>&1 &
BACKEND_PID="$!"

if ! wait_for_health 30; then
  echo "[err] backend failed health check: ${HEALTH_URL}" >&2
  echo "[err] backend log: ${BACKEND_LOG}" >&2
  exit 1
fi

echo "[dev] backend healthy: ${HEALTH_URL}"
echo "[dev] backend log: ${BACKEND_LOG}"
echo "[dev] starting frontend on ${FRONTEND_HOST}:${FRONTEND_PORT}"

cd "${ROOT_DIR}/apps"
HOST="${FRONTEND_HOST}" \
PORT="${FRONTEND_PORT}" \
BROWSER="${BROWSER_MODE}" \
REACT_APP_API_BASE="${API_BASE}" \
npm start
