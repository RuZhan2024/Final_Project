#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/applications/frontend"

cd "${FRONTEND_DIR}"

echo "[parity] frontend clean install"
npm ci

echo "[parity] frontend production build"
npm run build
