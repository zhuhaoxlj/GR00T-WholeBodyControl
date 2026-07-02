#!/usr/bin/env bash
# Install a packaged GEAR-SONIC model tarball on an offline machine.
#
# Usage:
#   bash install_scripts/install_gear_sonic_models_offline.sh /path/to/models.tar.gz
#
# Run from anywhere inside the GR00T-WholeBodyControl checkout, or keep this
# script under install_scripts/ in that checkout. The tarball should be created
# with package_gear_sonic_models.sh.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash install_scripts/install_gear_sonic_models_offline.sh /path/to/models.tar.gz" >&2
  exit 2
fi

PACKAGE="$1"
if [[ ! -f "$PACKAGE" ]]; then
  echo "[ERROR] Package not found: $PACKAGE" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

REQUIRED_FILES=(
  "gear_sonic_deploy/policy/release/model_encoder.onnx"
  "gear_sonic_deploy/policy/release/model_decoder.onnx"
  "gear_sonic_deploy/policy/release/observation_config.yaml"
  "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
)

echo "[INFO] Repository: $REPO_ROOT"
echo "[INFO] Installing model package: $PACKAGE"

tar -xzf "$PACKAGE" -C "$REPO_ROOT"

if [[ -f "GEAR_SONIC_MODELS.sha256" ]] && command -v sha256sum &>/dev/null; then
  echo "[INFO] Verifying checksums..."
  sha256sum -c GEAR_SONIC_MODELS.sha256
else
  echo "[WARN] Checksum file or sha256sum not available; skipping checksum verification."
fi

missing=0
for file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "[ERROR] Missing required file after install: $file" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

echo ""
echo "[OK] GEAR-SONIC models installed successfully."
echo ""
echo "You can now run, for example:"
echo "  cd $REPO_ROOT/gear_sonic_deploy"
echo "  bash deploy.sh --input-type keyboard --motion-data ./reference/example sim"
