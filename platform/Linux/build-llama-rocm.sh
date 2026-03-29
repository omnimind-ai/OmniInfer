#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/llama.cpp-linux-rocm/scripts/build-llama-linux-rocm.sh"

if [[ ! -f "${INNER_SCRIPT}" ]]; then
  echo "ROCm build script not found: ${INNER_SCRIPT}" >&2
  exit 1
fi

echo "Running Linux ROCm backend build script:"
echo "  bash ${INNER_SCRIPT} $*"

exec bash "${INNER_SCRIPT}" "$@"
