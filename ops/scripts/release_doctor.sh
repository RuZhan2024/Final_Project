#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "[check] git status"
git status --short

echo "[check] python compile"
python3 -m compileall ml/src/fall_detection applications/backend ops/scripts >/dev/null

echo "[check] release boundary"
bash ops/scripts/release_manifest.sh >/dev/null

echo "[check] frontend build"
(cd applications/frontend && npm run build >/dev/null)

echo "[ok] static release checks passed"
