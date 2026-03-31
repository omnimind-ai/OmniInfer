#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/turboquant-mac/build.sh"

if [[ ! -f "${INNER_SCRIPT}" ]]; then
  echo "macOS TurboQuant build script not found: ${INNER_SCRIPT}" >&2
  exit 1
fi

echo "Running macOS TurboQuant backend build script:"
echo "  bash ${INNER_SCRIPT} $*"

exec bash "${INNER_SCRIPT}" "$@"
