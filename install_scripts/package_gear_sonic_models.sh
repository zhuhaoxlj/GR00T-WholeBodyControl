#!/usr/bin/env bash
# Package downloaded GEAR-SONIC deployment models for offline installation.
#
# Usage:
#   bash install_scripts/package_gear_sonic_models.sh [output_tar_gz]
#
# Run this on a machine that has already downloaded the models via
# download_from_hf.py. The generated tarball can be copied to an offline
# machine and installed with install_gear_sonic_models_offline.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${1:-$REPO_ROOT/gear_sonic_models_${TIMESTAMP}.tar.gz}"

REQUIRED_FILES=(
  "gear_sonic_deploy/policy/release/model_encoder.onnx"
  "gear_sonic_deploy/policy/release/model_decoder.onnx"
  "gear_sonic_deploy/policy/release/observation_config.yaml"
  "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
)

OPTIONAL_FILES=(
  "gear_sonic_deploy/policy/low_latency/model_encoder.onnx"
  "gear_sonic_deploy/policy/low_latency/model_decoder.onnx"
  "gear_sonic_deploy/policy/low_latency/observation_config.yaml"
)

echo "[INFO] Repository: $REPO_ROOT"
echo "[INFO] Output tarball: $OUTPUT"

missing=0
for file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "[ERROR] Missing required file: $file" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "" >&2
  echo "Download the required models first, for example:" >&2
  echo "  uv run --python 3.10 --with huggingface-hub --with socksio \\" >&2
  echo "    python download_from_hf.py" >&2
  exit 1
fi

STAGING_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

FILES_TO_PACKAGE=()
for file in "${REQUIRED_FILES[@]}"; do
  FILES_TO_PACKAGE+=("$file")
done

for file in "${OPTIONAL_FILES[@]}"; do
  if [[ -f "$file" ]]; then
    FILES_TO_PACKAGE+=("$file")
  fi
done

echo "[INFO] Files included:"
for file in "${FILES_TO_PACKAGE[@]}"; do
  echo "  - $file"
  mkdir -p "$STAGING_DIR/$(dirname "$file")"
  cp -a "$file" "$STAGING_DIR/$file"
done

if command -v sha256sum &>/dev/null; then
  (
    cd "$STAGING_DIR"
    sha256sum "${FILES_TO_PACKAGE[@]}" > GEAR_SONIC_MODELS.sha256
  )
fi

mkdir -p "$(dirname "$OUTPUT")"
tar -C "$STAGING_DIR" -czf "$OUTPUT" .

echo ""
echo "[OK] Offline model package created:"
echo "  $OUTPUT"
echo ""
echo "Copy this tarball and the installer script to the offline machine:"
echo "  install_scripts/install_gear_sonic_models_offline.sh"
