#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/omniinfer-native-linux/build.sh"

if [[ ! -f "${INNER_SCRIPT}" ]]; then
  echo "OmniInfer Native Linux build script not found: ${INNER_SCRIPT}" >&2
  exit 1
fi

echo "Running OmniInfer Native Linux (EAGLE3) build script:"
echo "  bash ${INNER_SCRIPT} $*"

exec bash "${INNER_SCRIPT}" "$@"
