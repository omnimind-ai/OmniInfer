#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/omniinfer-native/package.sh"

if [[ ! -f "${INNER_SCRIPT}" ]]; then
  echo "Android omniinfer-native package script not found: ${INNER_SCRIPT}" >&2
  exit 1
fi

echo "Running Android omniinfer-native package script:"
echo "  bash ${INNER_SCRIPT} $*"

exec bash "${INNER_SCRIPT}" "$@"
