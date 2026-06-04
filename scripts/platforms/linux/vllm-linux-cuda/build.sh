#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
DRY_RUN=0
CLEAN_BUILD=0
SMOKE_TEST=0
PYTHON_BIN="${OMNIINFER_VLLM_PYTHON:-python3}"
PIP_PACKAGE="${OMNIINFER_VLLM_PIP_PACKAGE:-vllm}"
INDEX_URL="${OMNIINFER_VLLM_INDEX_URL:-}"
EXTRA_INDEX_URL="${OMNIINFER_VLLM_EXTRA_INDEX_URL:-}"
PIP_EXTRA_ARGS=()

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ROOT}/../../../.." && pwd)"
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/linux/vllm-linux-cuda"
BIN_ROOT="${PACKAGE_ROOT}/bin"
LOG_ROOT="${PACKAGE_ROOT}/logs"
MODELS_ROOT="${REPO_ROOT}/.local/models"

usage() {
  cat <<'EOF'
Usage: build.sh [options]

Options:
  --build-type <type>        Accepted for CLI consistency; vLLM is installed from Python wheels
  --python <path>            Python interpreter used to create the local venv
  --package <spec>           pip package spec, default: vllm
                             Example: 'vllm==0.9.2'
  --index-url <url>          pip index URL
  --extra-index-url <url>    extra pip index URL
  --pip-extra-arg <arg>      pass one extra argument to pip/uv pip
  --clean                    Remove the previous vLLM runtime venv first
  --smoke-test               Run 'vllm --help' after installation
  --dry-run                  Print actions without executing them
  -h, --help                 Show this help message

Environment:
  OMNIINFER_VLLM_PYTHON          Default Python interpreter
  OMNIINFER_VLLM_PIP_PACKAGE     Default pip package spec
  OMNIINFER_VLLM_INDEX_URL       Default pip index URL
  OMNIINFER_VLLM_EXTRA_INDEX_URL Default extra pip index URL
EOF
}

check_deps() {
  local rc=0
  _dep() {
    local cmd="$1" desc="$2" hint="$3" pkg="${4:-}"
    if command -v "${cmd}" >/dev/null 2>&1; then
      printf 'ok|%s|%s|%s|%s\n' "${cmd}" "${desc}" "${hint}" "${pkg}"
    else
      printf 'missing|%s|%s|%s|%s\n' "${cmd}" "${desc}" "${hint}" "${pkg}"
      rc=1
    fi
  }
  _dep "${PYTHON_BIN}" "Python 3 interpreter" "sudo apt install python3 python3-venv" python3
  return ${rc}
}

while (($# > 0)); do
  case "$1" in
    --build-type)
      BUILD_TYPE="${2:?missing value for --build-type}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?missing value for --python}"
      shift 2
      ;;
    --package)
      PIP_PACKAGE="${2:?missing value for --package}"
      shift 2
      ;;
    --index-url)
      INDEX_URL="${2:?missing value for --index-url}"
      shift 2
      ;;
    --extra-index-url)
      EXTRA_INDEX_URL="${2:?missing value for --extra-index-url}"
      shift 2
      ;;
    --pip-extra-arg)
      PIP_EXTRA_ARGS+=("${2:?missing value for --pip-extra-arg}")
      shift 2
      ;;
    --clean)
      CLEAN_BUILD=1
      shift
      ;;
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --check-deps)
      check_deps
      exit $?
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' was not found in PATH." >&2
    exit 1
  fi
}

run_cmd() {
  echo "+ $*"
  if [[ ${DRY_RUN} -eq 0 ]]; then
    "$@"
  fi
}

require_command "${PYTHON_BIN}"

PIP_ARGS=()
if [[ -n "${INDEX_URL}" ]]; then
  PIP_ARGS+=(--index-url "${INDEX_URL}")
fi
if [[ -n "${EXTRA_INDEX_URL}" ]]; then
  PIP_ARGS+=(--extra-index-url "${EXTRA_INDEX_URL}")
fi
PIP_ARGS+=("${PIP_EXTRA_ARGS[@]}")
PIP_ARGS+=("${PIP_PACKAGE}")

echo "Preparing vLLM Linux CUDA runtime..."
echo "  runtime: ${PACKAGE_ROOT}"
echo "  python: ${PYTHON_BIN}"
echo "  package: ${PIP_PACKAGE}"
echo "  build type: ${BUILD_TYPE} (accepted for CLI consistency)"

if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  echo "Cleaning previous runtime: ${PACKAGE_ROOT}"
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "+ mkdir -p ${LOG_ROOT} ${MODELS_ROOT}"
  [[ ${CLEAN_BUILD} -eq 1 ]] && echo "+ rm -rf ${PACKAGE_ROOT}"
  echo "+ ${PYTHON_BIN} -m venv ${PACKAGE_ROOT}"
  if command -v uv >/dev/null 2>&1; then
    echo "+ uv pip install --python ${BIN_ROOT}/python ${PIP_ARGS[*]}"
  else
    echo "+ ${BIN_ROOT}/python -m pip install --upgrade pip"
    echo "+ ${BIN_ROOT}/python -m pip install ${PIP_ARGS[*]}"
  fi
  [[ ${SMOKE_TEST} -eq 1 ]] && echo "+ ${BIN_ROOT}/vllm --help"
  exit 0
fi

if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  rm -rf "${PACKAGE_ROOT}"
fi

mkdir -p "${LOG_ROOT}" "${MODELS_ROOT}"

if [[ ! -x "${BIN_ROOT}/python" ]]; then
  run_cmd "${PYTHON_BIN}" -m venv "${PACKAGE_ROOT}"
fi

if [[ ! -x "${BIN_ROOT}/python" ]]; then
  echo "Failed to create vLLM Python environment at ${PACKAGE_ROOT}" >&2
  echo "Install python3-venv or pass --python to an interpreter with venv support." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  run_cmd uv pip install --python "${BIN_ROOT}/python" "${PIP_ARGS[@]}"
else
  run_cmd "${BIN_ROOT}/python" -m ensurepip --upgrade
  run_cmd "${BIN_ROOT}/python" -m pip install --upgrade pip
  run_cmd "${BIN_ROOT}/python" -m pip install "${PIP_ARGS[@]}"
fi

if [[ ! -x "${BIN_ROOT}/vllm" ]]; then
  echo "vLLM installation completed, but launcher was not found: ${BIN_ROOT}/vllm" >&2
  exit 1
fi

if [[ ${SMOKE_TEST} -eq 1 ]]; then
  run_cmd "${BIN_ROOT}/vllm" --help
fi

echo "vLLM Linux CUDA runtime is ready:"
echo "  ${BIN_ROOT}/vllm"
