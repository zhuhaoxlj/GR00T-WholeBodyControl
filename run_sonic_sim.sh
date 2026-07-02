#!/usr/bin/env bash
# Launch GEAR-SONIC simulation from this checkout without hard-coded paths.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$REPO_ROOT/gear_sonic_deploy/deploy.sh" \
  --input-type keyboard \
  --motion-data reference/example \
  sim
