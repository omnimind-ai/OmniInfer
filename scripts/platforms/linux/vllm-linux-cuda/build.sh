#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
DRY_RUN=0
CLEAN_BUILD=0
SMOKE_TEST=0
PYTHON_BIN="${OMNIINFER_VLLM_PYTHON:-python3}"
PIP_PACKAGE="${OMNIINFER_VLLM_PIP_PACKAGE:-}"
INDEX_URL="${OMNIINFER_VLLM_INDEX_URL:-}"
EXTRA_INDEX_URL="${OMNIINFER_VLLM_EXTRA_INDEX_URL:-}"
PYTORCH_INDEX_URL="${OMNIINFER_VLLM_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
UV_INDEX_STRATEGY="${OMNIINFER_VLLM_UV_INDEX_STRATEGY:-unsafe-best-match}"
USE_PRECOMPILED="${OMNIINFER_VLLM_USE_PRECOMPILED:-1}"
PRECOMPILED_WHEEL_COMMIT="${OMNIINFER_VLLM_PRECOMPILED_WHEEL_COMMIT:-nightly}"
PIP_EXTRA_ARGS=()

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ROOT}/../../../.." && pwd)"
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/linux/vllm-linux-cuda"
BIN_ROOT="${PACKAGE_ROOT}/bin"
LOG_ROOT="${PACKAGE_ROOT}/logs"
MODELS_ROOT="${REPO_ROOT}/.local/models"
SOURCE_DIR="${OMNIINFER_VLLM_SOURCE_DIR:-}"
SOURCE_REF="${OMNIINFER_VLLM_SOURCE_REF:-main}"
SOURCE_URL="${OMNIINFER_VLLM_SOURCE_URL:-https://github.com/vllm-project/vllm.git}"
SOURCE_ROOT="${OMNIINFER_VLLM_SOURCE_ROOT:-${REPO_ROOT}/.local/source/vllm}"

usage() {
  cat <<'EOF'
Usage: build.sh [options]

Options:
  --build-type <type>        Accepted for CLI consistency; vLLM is installed from Python wheels
  --python <path>            Python interpreter used to create the local venv
  --package <spec>           Override the default pip package set
                             Example: 'vllm==0.9.2'
  --index-url <url>          pip index URL
  --extra-index-url <url>    extra pip index URL
  --pip-extra-arg <arg>      pass one extra argument to pip/uv pip
  --source <path>            Install vLLM from a local source checkout
  --source-ref <ref>         Git ref used when cloning/updating the default source checkout
                             Defaults to main
  --source-url <url>         Git URL for the default source checkout
  --source-root <path>       Default source checkout path
  --no-precompiled           Do not request vLLM precompiled extension wheels for source installs
  --clean                    Remove the previous vLLM runtime venv first
  --smoke-test               Run 'vllm --help' after installation
  --dry-run                  Print actions without executing them
  -h, --help                 Show this help message

Environment:
  OMNIINFER_VLLM_PYTHON          Default Python interpreter
  OMNIINFER_VLLM_PIP_PACKAGE     Default pip package spec
  OMNIINFER_VLLM_INDEX_URL       Default pip index URL
  OMNIINFER_VLLM_EXTRA_INDEX_URL Default extra pip index URL
  OMNIINFER_VLLM_PYTORCH_INDEX_URL Default PyTorch CUDA wheel index for the pinned install
  OMNIINFER_VLLM_UV_INDEX_STRATEGY uv index strategy for the pinned install
  OMNIINFER_VLLM_SOURCE_DIR      Local vLLM source checkout to install from
  OMNIINFER_VLLM_SOURCE_REF      Git ref for the managed source checkout
  OMNIINFER_VLLM_SOURCE_URL      Git URL for the managed source checkout
  OMNIINFER_VLLM_SOURCE_ROOT     Managed source checkout path
  OMNIINFER_VLLM_USE_PRECOMPILED Set to 0 to disable precompiled source install wheels
  OMNIINFER_VLLM_PRECOMPILED_WHEEL_COMMIT Precompiled wheel commit selector
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
    --source)
      SOURCE_DIR="${2:?missing value for --source}"
      shift 2
      ;;
    --source-ref)
      SOURCE_REF="${2:?missing value for --source-ref}"
      shift 2
      ;;
    --source-url)
      SOURCE_URL="${2:?missing value for --source-url}"
      shift 2
      ;;
    --source-root)
      SOURCE_ROOT="${2:?missing value for --source-root}"
      shift 2
      ;;
    --no-precompiled)
      USE_PRECOMPILED=0
      shift
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

create_venv() {
  if "${PYTHON_BIN}" -m venv "${PACKAGE_ROOT}"; then
    return
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to create venv with ${PYTHON_BIN}, and uv was not found for fallback." >&2
    echo "Install python3-venv or pass --python to an interpreter with venv support." >&2
    exit 1
  fi

  echo "Falling back to uv venv because ${PYTHON_BIN} could not create a venv."
  rm -rf "${PACKAGE_ROOT}"
  local resolved_python="${PYTHON_BIN}"
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    resolved_python="$(command -v "${PYTHON_BIN}")"
  fi
  run_cmd uv venv --python "${resolved_python}" "${PACKAGE_ROOT}"
}

require_command "${PYTHON_BIN}"

if [[ -z "${PIP_PACKAGE}" && -z "${SOURCE_DIR}" ]]; then
  SOURCE_DIR="${SOURCE_ROOT}"
fi

if [[ -n "${SOURCE_DIR}" && -n "${PIP_PACKAGE}" ]]; then
  echo "Use either --source/OMNIINFER_VLLM_SOURCE_DIR or --package/OMNIINFER_VLLM_PIP_PACKAGE, not both." >&2
  exit 1
fi

PIP_PACKAGES=()
if [[ -n "${SOURCE_DIR}" ]]; then
  PIP_PACKAGES+=("${SOURCE_DIR}")
elif [[ -n "${PIP_PACKAGE}" ]]; then
  PIP_PACKAGES+=("${PIP_PACKAGE}")
else
  PIP_PACKAGES+=(
    "torch==2.5.1+cu121"
    "torchvision==0.20.1+cu121"
    "transformers==4.46.3"
    "setuptools"
    "vllm==0.6.6"
  )
  if [[ -z "${EXTRA_INDEX_URL}" ]]; then
    EXTRA_INDEX_URL="${PYTORCH_INDEX_URL}"
  fi
fi

PIP_ARGS=()
if [[ -n "${INDEX_URL}" ]]; then
  PIP_ARGS+=(--index-url "${INDEX_URL}")
fi
if [[ -n "${EXTRA_INDEX_URL}" ]]; then
  PIP_ARGS+=(--extra-index-url "${EXTRA_INDEX_URL}")
fi
PIP_ARGS+=("${PIP_EXTRA_ARGS[@]}")
PIP_ARGS+=("${PIP_PACKAGES[@]}")

UV_PIP_ARGS=()
if [[ -z "${PIP_PACKAGE}" && -n "${UV_INDEX_STRATEGY}" ]]; then
  UV_PIP_ARGS+=(--index-strategy "${UV_INDEX_STRATEGY}")
fi
UV_PIP_ARGS+=("${PIP_ARGS[@]}")

ensure_source_checkout() {
  if [[ -z "${SOURCE_DIR}" ]]; then
    return
  fi
  if [[ -d "${SOURCE_DIR}/.git" ]]; then
    run_cmd git -C "${SOURCE_DIR}" fetch origin "${SOURCE_REF}"
    run_cmd git -C "${SOURCE_DIR}" checkout "${SOURCE_REF}"
    run_cmd git -C "${SOURCE_DIR}" pull --ff-only origin "${SOURCE_REF}"
    return
  fi
  if [[ -e "${SOURCE_DIR}" ]]; then
    echo "vLLM source path exists but is not a Git checkout: ${SOURCE_DIR}" >&2
    exit 1
  fi
  run_cmd mkdir -p "$(dirname "${SOURCE_DIR}")"
  run_cmd git clone --depth 1 --branch "${SOURCE_REF}" "${SOURCE_URL}" "${SOURCE_DIR}"
}

install_source_runtime() {
  if [[ ${USE_PRECOMPILED} -eq 1 ]]; then
    run_cmd env \
      VLLM_USE_PRECOMPILED=1 \
      VLLM_PRECOMPILED_WHEEL_COMMIT="${PRECOMPILED_WHEEL_COMMIT}" \
      uv pip install --python "${BIN_ROOT}/python" --editable "${SOURCE_DIR}" --torch-backend=auto
  else
    run_cmd uv pip install --python "${BIN_ROOT}/python" --editable "${SOURCE_DIR}" --torch-backend=auto
  fi
}

echo "Preparing vLLM Linux CUDA runtime..."
echo "  runtime: ${PACKAGE_ROOT}"
echo "  python: ${PYTHON_BIN}"
if [[ -n "${SOURCE_DIR}" ]]; then
  echo "  source: ${SOURCE_DIR} (${SOURCE_REF})"
  echo "  precompiled: ${USE_PRECOMPILED}"
else
  echo "  packages: ${PIP_PACKAGES[*]}"
fi
echo "  build type: ${BUILD_TYPE} (accepted for CLI consistency)"

if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  echo "Cleaning previous runtime: ${PACKAGE_ROOT}"
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "+ mkdir -p ${LOG_ROOT} ${MODELS_ROOT}"
  [[ ${CLEAN_BUILD} -eq 1 ]] && echo "+ rm -rf ${PACKAGE_ROOT}"
  if [[ -n "${SOURCE_DIR}" ]]; then
    echo "+ git clone/fetch ${SOURCE_URL} ${SOURCE_DIR} (${SOURCE_REF})"
  fi
  echo "+ ${PYTHON_BIN} -m venv ${PACKAGE_ROOT}"
  if [[ -n "${SOURCE_DIR}" ]]; then
    echo "+ VLLM_USE_PRECOMPILED=${USE_PRECOMPILED} VLLM_PRECOMPILED_WHEEL_COMMIT=${PRECOMPILED_WHEEL_COMMIT} uv pip install --python ${BIN_ROOT}/python --editable ${SOURCE_DIR} --torch-backend=auto"
  elif command -v uv >/dev/null 2>&1; then
    echo "+ uv pip install --python ${BIN_ROOT}/python ${UV_PIP_ARGS[*]}"
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

if [[ -n "${SOURCE_DIR}" ]]; then
  require_command git
  require_command uv
fi

ensure_source_checkout

if [[ ! -x "${BIN_ROOT}/python" ]]; then
  create_venv
fi

if [[ ! -x "${BIN_ROOT}/python" ]]; then
  echo "Failed to create vLLM Python environment at ${PACKAGE_ROOT}" >&2
  echo "Install python3-venv or pass --python to an interpreter with venv support." >&2
  exit 1
fi

if [[ -n "${SOURCE_DIR}" ]]; then
  install_source_runtime
elif command -v uv >/dev/null 2>&1; then
  run_cmd uv pip install --python "${BIN_ROOT}/python" "${UV_PIP_ARGS[@]}"
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
