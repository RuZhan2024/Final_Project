#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

command_exists() {
  local cmd="$1"
  if [[ "$cmd" == */* ]]; then
    [[ -x "$cmd" ]]
  else
    command -v "$cmd" >/dev/null 2>&1
  fi
}

python_is_compatible() {
  local py_bin="$1"
  "$py_bin" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

python_is_preferred_310() {
  local py_bin="$1"
  "$py_bin" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
}

emit_if_compatible() {
  local py_bin="$1"
  if command_exists "$py_bin" && python_is_compatible "$py_bin"; then
    printf '%s\n' "$py_bin"
    return 0
  fi
  return 1
}

choose_python_bin() {
  if [[ -n "${PY_BIN:-}" ]]; then
    printf '%s\n' "${PY_BIN}"
    return 0
  fi

  local candidate pyenv_root ver pyenv_bin

  for candidate in python3.10 /opt/homebrew/bin/python3.10 /usr/local/bin/python3.10; do
    if emit_if_compatible "$candidate"; then
      return 0
    fi
  done

  if command -v pyenv >/dev/null 2>&1; then
    pyenv_root="$(pyenv root 2>/dev/null || true)"
    if [[ -n "$pyenv_root" && -d "$pyenv_root/versions" ]]; then
      while IFS= read -r ver; do
        pyenv_bin="${pyenv_root}/versions/${ver}/bin/python"
        if command_exists "$pyenv_bin" && python_is_preferred_310 "$pyenv_bin"; then
          printf '%s\n' "$pyenv_bin"
          return 0
        fi
      done < <(find "$pyenv_root/versions" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort -V)

      while IFS= read -r ver; do
        pyenv_bin="${pyenv_root}/versions/${ver}/bin/python"
        if emit_if_compatible "$pyenv_bin"; then
          return 0
        fi
      done < <(find "$pyenv_root/versions" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort -V)
    fi
  fi

  for candidate in python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    if emit_if_compatible "$candidate"; then
      return 0
    fi
  done

  printf '%s\n' "python3"
}

PY_BIN="$(choose_python_bin)"

require_cmd() {
  if ! command_exists "$1"; then
    echo "[err] missing required command: $1" >&2
    exit 1
  fi
}

require_python_compatible() {
  local py_bin="$1"
  if ! python_is_compatible "$py_bin"; then
    echo "[err] ${py_bin} must be Python 3.10 or newer" >&2
    echo "[hint] install Python 3.10.x and rerun with: PY_BIN=python3.10 make bootstrap-dev" >&2
    exit 1
  fi
}

warn_if_node_not_22() {
  local node_major
  node_major="$(node -p 'process.versions.node.split(\".\")[0]' 2>/dev/null || true)"
  if [[ "${node_major}" != "22" ]]; then
    echo "[warn] detected Node.js $(node --version 2>/dev/null || echo unknown)" >&2
    echo "[warn] Node.js 22.x is recommended for frontend parity" >&2
  fi
}

require_cmd "${PY_BIN}"
require_cmd npm
require_python_compatible "${PY_BIN}"
warn_if_node_not_22

cd "${ROOT_DIR}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[bootstrap] creating .venv"
  "${PY_BIN}" -m venv .venv
fi

source ".venv/bin/activate"

echo "[bootstrap] upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

if ! python3 -c "import fastapi, uvicorn, yaml, numpy" >/dev/null 2>&1; then
  echo "[bootstrap] installing python dependencies"
  python -m pip install -r requirements.txt
  python -m pip install -e . --no-build-isolation
fi

if [[ ! -d "applications/frontend/node_modules" ]]; then
  echo "[bootstrap] installing frontend dependencies"
  (cd applications/frontend && npm install)
fi

echo "[bootstrap] syncing frontend MediaPipe assets"
(cd applications/frontend && npm run sync-mediapipe-assets)

exec bash ops/scripts/start_fullstack.sh
