#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[check] git status"
git status --short

echo "[check] release boundary"
bash scripts/release_manifest.sh >/dev/null

echo "[check] python compile"
python3 -m compileall src/fall_detection server scripts >/dev/null

echo "[check] frontend build"
(cd apps && npm run build >/dev/null)

echo "[ok] static release checks passed"
