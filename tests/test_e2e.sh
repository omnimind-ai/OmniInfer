#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "[omniinfer-flow] shell host: $(uname -s) $(uname -m)"
exec python3 "$SCRIPT_DIR/test_e2e.py" "$@"
