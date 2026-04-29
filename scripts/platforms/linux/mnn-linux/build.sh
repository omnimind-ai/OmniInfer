#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
DRY_RUN=0
BOOTSTRAP_SUBMODULE=1
WITH_OPENCL=0
JOBS=""
PYTHON_BIN="${OMNIINFER_MNN_PYTHON:-python3}"
CLEAN_BUILD=0
SMOKE_TEST=0

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
  _dep "${PYTHON_BIN}" "Python 3 interpreter"  "sudo apt install python3"  python3
  return ${rc}
}

usage() {
  cat <<'EOF'
Usage: build-mnn-linux.sh [options]

Options:
  --build-type <type>   Build type for MNN core, default: Release
  --python <path>       Python interpreter used to create the local venv
  --jobs <n>            Parallel build jobs, default: nproc
  --opencl              Build MNN with OpenCL enabled
  --clean               Remove previous MNN build products and recreate the venv
  --no-bootstrap        Do not auto-initialize the MNN git submodule
  --smoke-test          Verify the installed Python package imports `MNN.llm`
  --dry-run             Print actions without executing them
  -h, --help            Show this help text
EOF
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
    --jobs)
      JOBS="${2:?missing value for --jobs}"
      shift 2
      ;;
    --opencl)
      WITH_OPENCL=1
      shift
      ;;
    --clean)
      CLEAN_BUILD=1
      shift
      ;;
    --no-bootstrap)
      BOOTSTRAP_SUBMODULE=0
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

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ROOT}/../../../.." && pwd)"
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/linux/mnn-linux"
VENV_ROOT="${PACKAGE_ROOT}/venv"
BIN_ROOT="${PACKAGE_ROOT}/bin"
LOG_ROOT="${PACKAGE_ROOT}/logs"
MODELS_ROOT="${PACKAGE_ROOT}/models"
MNN_ROOT="${REPO_ROOT}/framework/mnn"
PIP_PACKAGE_ROOT="${MNN_ROOT}/pymnn/pip_package"
BUILD_ROOT="${MNN_ROOT}/pymnn_build"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' was not found in PATH." >&2
    exit 1
  fi
}

detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return
  fi
  printf '1\n'
}

run_cmd() {
  echo "+ $*"
  if [[ ${DRY_RUN} -eq 0 ]]; then
    "$@"
  fi
}

ensure_mnn_root() {
  if [[ -f "${MNN_ROOT}/CMakeLists.txt" ]]; then
    return
  fi
  if [[ ${BOOTSTRAP_SUBMODULE} -eq 0 ]]; then
    echo "MNN source tree was not found at ${MNN_ROOT}" >&2
    echo "Run: git submodule update --init framework/mnn" >&2
    exit 1
  fi
  require_command git
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  git -C ${REPO_ROOT} submodule update --init framework/mnn"
    return
  fi
  git -C "${REPO_ROOT}" submodule update --init framework/mnn
}

prepare_runtime_dirs() {
  run_cmd mkdir -p "${PACKAGE_ROOT}" "${BIN_ROOT}" "${LOG_ROOT}" "${MODELS_ROOT}"
  if [[ ${CLEAN_BUILD} -eq 1 ]]; then
    run_cmd rm -rf "${VENV_ROOT}" "${BUILD_ROOT}"
  fi
}

if [[ -z "${JOBS}" ]]; then
  JOBS="$(detect_jobs)"
fi

ensure_mnn_root
require_command "${PYTHON_BIN}"
prepare_runtime_dirs

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "Will create/update venv under ${VENV_ROOT}"
  echo "Will build PyMNN with LLM support from ${PIP_PACKAGE_ROOT}"
  exit 0
fi

if [[ ! -x "${VENV_ROOT}/bin/python3" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_ROOT}"
fi

"${VENV_ROOT}/bin/python3" -m pip install --upgrade pip setuptools wheel numpy

pushd "${PIP_PACKAGE_ROOT}" >/dev/null
DEPS_ARGS=(llm)
if [[ ${WITH_OPENCL} -eq 1 ]]; then
  DEPS_ARGS+=(opencl)
fi
PROJECT_ROOT="${MNN_ROOT}" \
  "${VENV_ROOT}/bin/python3" build_deps.py "${DEPS_ARGS[@]}"

PROJECT_ROOT="${MNN_ROOT}" \
  MNN_BUILD_DIR="${BUILD_ROOT}" \
  MAX_JOBS="${JOBS}" \
  "${VENV_ROOT}/bin/python3" setup.py install --deps "$(IFS=,; echo "${DEPS_ARGS[*]}")"
popd >/dev/null

cat > "${BIN_ROOT}/python3" <<EOF
#!/usr/bin/env bash
exec "${VENV_ROOT}/bin/python3" "\$@"
EOF
chmod +x "${BIN_ROOT}/python3"

if [[ ${SMOKE_TEST} -eq 1 ]]; then
  "${VENV_ROOT}/bin/python3" - <<'PY'
import MNN
import MNN.cv
import MNN.llm
print("MNN Python runtime is available")
PY
fi

echo
echo "Linux MNN build complete."
echo "Python runtime: ${VENV_ROOT}/bin/python3"
echo "Models directory: ${MODELS_ROOT}"
echo "Next step:"
echo "  ./omniinfer select mnn-linux"
echo "  ./omniinfer model load -m /absolute/path/to/mnn-model-dir-or-config.json"
