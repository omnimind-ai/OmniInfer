#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/macos/mlx-mac"
VENV_ROOT="${PACKAGE_ROOT}/venv"
MODELS_ROOT="${PACKAGE_ROOT}/models"
LOGS_ROOT="${PACKAGE_ROOT}/logs"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
PYTHON_BIN=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: build-mlx-mac.sh [options]

Options:
  --python <path>  Python 3.10+ interpreter used to create the venv
  --dry-run        Print actions without executing them
  -h, --help       Show this help message
EOF
}

run_cmd() {
  echo "+ $*"
  if [[ ${DRY_RUN} -eq 0 ]]; then
    "$@"
  fi
}

pick_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return
    fi
  done
  echo ""
}

while (($# > 0)); do
  case "$1" in
    --python)
      PYTHON_BIN="${2:?missing value for --python}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

PYTHON_BIN="$(pick_python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "A Python 3.10+ interpreter is required to build the mlx-mac runtime." >&2
  exit 1
fi

if [[ ${DRY_RUN} -eq 0 ]]; then
  "${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("mlx-mac requires Python 3.10 or newer")
PY
fi

run_cmd mkdir -p "${PACKAGE_ROOT}" "${MODELS_ROOT}" "${LOGS_ROOT}"
run_cmd "${PYTHON_BIN}" -m venv "${VENV_ROOT}"
run_cmd "${VENV_ROOT}/bin/python" -m pip install --upgrade pip setuptools wheel
run_cmd "${VENV_ROOT}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"

echo
echo "mlx-mac runtime prepared:"
echo "  ${PACKAGE_ROOT}"
echo
echo "Use the OmniInfer launcher normally:"
echo "  ./omniinfer backend list"
echo "  ./omniinfer select mlx-mac"
