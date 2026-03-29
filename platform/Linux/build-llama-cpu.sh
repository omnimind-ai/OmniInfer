#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/llama.cpp-linux/scripts/build-llama-linux.sh"

if [[ ! -f "${INNER_SCRIPT}" ]]; then
  echo "CPU build script not found: ${INNER_SCRIPT}" >&2
  exit 1
fi

echo "Running Linux CPU backend build script:"
echo "  bash ${INNER_SCRIPT} $*"

exec bash "${INNER_SCRIPT}" "$@"
