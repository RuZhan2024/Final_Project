#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-/tmp/assignment3_bundle}"
ZIP_PATH="${2:-/tmp/assignment3_bundle.zip}"

SELECTED_CLIPS=(
  "ops/deploy_assets/replay_clips/corridor/corridor_back_2.mp4"
  "ops/deploy_assets/replay_clips/corridor/corridor_front_1.mp4"
  "ops/deploy_assets/replay_clips/corridor/corridor_front_2.mp4"
  "ops/deploy_assets/replay_clips/corridor/corridor_side_1.mp4"
  "ops/deploy_assets/replay_clips/corridor/corridor_side_2.mp4"
  "ops/deploy_assets/replay_clips/kitchen/kitchen_front_1.mp4"
  "ops/deploy_assets/replay_clips/kitchen/kitchen_front_2.mp4"
  "ops/deploy_assets/replay_clips/kitchen/kitchen_side_1.mp4"
  "ops/deploy_assets/replay_clips/kitchen/kitchen_side_2.mp4"
  "ops/deploy_assets/replay_clips/corridor_adl/corridor_walk_1.mp4"
  "ops/deploy_assets/replay_clips/kitchen_adl/kitchen_sit_2.mp4"
)

copy_path() {
  local rel="$1"
  local src="${ROOT_DIR}/${rel}"
  local dst="${OUT_DIR}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  if [[ -d "${src}" ]]; then
    cp -R "${src}" "${dst}"
  else
    cp "${src}" "${dst}"
  fi
}

remove_path() {
  local rel="$1"
  rm -rf "${OUT_DIR:?}/${rel}"
}

rm -rf "${OUT_DIR}" "${ZIP_PATH}"
mkdir -p "${OUT_DIR}"

copy_path "README.md"
copy_path ".env.example"
copy_path ".gitignore"
copy_path ".dockerignore"
copy_path ".coveragerc"
copy_path ".nvmrc"
copy_path "Makefile"
copy_path "pyproject.toml"
copy_path "pytest.ini"
copy_path "requirements.in"
copy_path "requirements-dev.txt"
copy_path "requirements.lock.txt"
copy_path "requirements.txt"
copy_path "requirements_server.txt"
copy_path "Dockerfile.backend"
copy_path "docker-compose.yml"
copy_path "render.yaml"
copy_path "qa"
copy_path "applications/backend"
copy_path "applications/frontend/.gitignore"
copy_path "applications/frontend/.dockerignore"
copy_path "applications/frontend/README.md"
copy_path "applications/frontend/Dockerfile.dev"
copy_path "applications/frontend/package-lock.json"
copy_path "applications/frontend/src"
copy_path "applications/frontend/package.json"
copy_path "applications/frontend/tsconfig.json"
copy_path "applications/frontend/scripts"
copy_path "applications/frontend/public/index.html"
copy_path "applications/frontend/public/logo_dark.png"
copy_path "applications/frontend/public/manifest.json"
copy_path "applications/frontend/public/robots.txt"
copy_path "ml/src"
copy_path "ops/scripts"
copy_path "ops/deploy_assets/checkpoints"
copy_path "ops/deploy_assets/manifest.json"
copy_path "ops/configs"

for clip in "${SELECTED_CLIPS[@]}"; do
  copy_path "${clip}"
done

# Mirror the known-good lightweight bundle: rely on bootstrap to install frontend
# dependencies and sync MediaPipe assets, and strip local caches / runtime state.
remove_path "applications/frontend/public/mediapipe"
find "${OUT_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${OUT_DIR}" -name ".DS_Store" -delete
remove_path "applications/backend/event_clips"
remove_path "applications/backend/safe_guard_notifications.sqlite3"
remove_path "ml/src/fall_detection/data"

(
  cd "$(dirname "${OUT_DIR}")"
  zip -qr "${ZIP_PATH}" "$(basename "${OUT_DIR}")"
)

echo "Bundle directory: ${OUT_DIR}"
echo "Bundle zip: ${ZIP_PATH}"
du -sh "${OUT_DIR}" "${ZIP_PATH}"
