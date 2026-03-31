#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/llama.cpp-mac-intel/build.sh"

if [[ ! -f "${INNER_SCRIPT}" ]]; then
  echo "macOS Intel build script not found: ${INNER_SCRIPT}" >&2
  exit 1
fi

echo "Running macOS Intel backend build script:"
echo "  bash ${INNER_SCRIPT} $*"

exec bash "${INNER_SCRIPT}" "$@"
