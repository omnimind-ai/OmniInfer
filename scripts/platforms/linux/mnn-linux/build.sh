#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
DRY_RUN=0
BOOTSTRAP_SUBMODULE=1
WITH_OPENCL=0
WITH_CUDA=0
JOBS=""
PYTHON_BIN="${OMNIINFER_MNN_PYTHON:-python3}"
CLEAN_BUILD=0
SMOKE_TEST=0
BUILD_FROM_SOURCE=0

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
  --cuda                Build MNN with CUDA enabled
  --clean               Remove previous MNN build products and recreate the venv
  --no-bootstrap        Do not auto-initialize the MNN git submodule
  --from-source         Build from the checked-out MNN source submodule
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
    --cuda)
      WITH_CUDA=1
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
    --from-source)
      BUILD_FROM_SOURCE=1
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
MODELS_ROOT="${REPO_ROOT}/.local/models"
MNN_ROOT="${REPO_ROOT}/framework/mnn"
PIP_PACKAGE_ROOT="${MNN_ROOT}/pymnn/pip_package"
BUILD_ROOT="${MNN_ROOT}/pymnn_build"

if [[ ${BUILD_FROM_SOURCE} -eq 0 ]]; then
  echo "No prebuilt install path is configured for mnn-linux." >&2
  echo "Re-run with --from-source to build from framework/mnn." >&2
  exit 1
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' was not found in PATH." >&2
    exit 1
  fi
}

python_has_headers() {
  "$1" - <<'PY' >/dev/null 2>&1
import sysconfig
from pathlib import Path

include_dir = sysconfig.get_paths().get("include")
raise SystemExit(0 if include_dir and (Path(include_dir) / "Python.h").is_file() else 1)
PY
}

select_python_with_headers() {
  if python_has_headers "${PYTHON_BIN}"; then
    return
  fi

  if command -v uv >/dev/null 2>&1; then
    local candidate=""
    candidate="$(uv python find --managed-python --no-python-downloads 3.13 2>/dev/null || true)"
    if [[ -n "${candidate}" ]] && python_has_headers "${candidate}"; then
      echo "${PYTHON_BIN} does not provide Python.h; using uv-managed Python: ${candidate}"
      PYTHON_BIN="${candidate}"
      return
    fi
  fi

  echo "${PYTHON_BIN} does not provide Python.h, which is required to build PyMNN." >&2
  echo "Install python3-dev/python3-venv, pass --python to a Python with headers, or install a uv-managed Python." >&2
  exit 1
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

configure_cuda_env() {
  if [[ ${WITH_CUDA} -eq 0 ]]; then
    return
  fi

  local cuda_root="${CUDAToolkit_ROOT:-${CUDA_HOME:-${CUDA_PATH:-}}}"
  if [[ -z "${cuda_root}" ]]; then
    for candidate in "${HOME}/.local/cuda-12.5" /usr/local/cuda /usr/local/cuda-12.5; do
      if [[ -f "${candidate}/include/cuda_runtime.h" && -d "${candidate}/lib64" ]]; then
        cuda_root="${candidate}"
        break
      fi
    done
  fi

  if [[ -z "${cuda_root}" || ! -f "${cuda_root}/include/cuda_runtime.h" ]]; then
    echo "CUDA headers were not found. Set CUDAToolkit_ROOT or CUDA_HOME before --cuda." >&2
    exit 1
  fi
  if [[ ! -d "${cuda_root}/lib64" ]]; then
    echo "CUDA library directory was not found: ${cuda_root}/lib64" >&2
    exit 1
  fi

  export CUDAToolkit_ROOT="${cuda_root}"
  export CUDA_HOME="${cuda_root}"
  export CUDA_PATH="${cuda_root}"
  export PATH="${cuda_root}/bin:${PATH}"
  export CPATH="${cuda_root}/include${CPATH:+:${CPATH}}"
  export LIBRARY_PATH="${cuda_root}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${cuda_root}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

append_path_list() {
  local var_name="$1"
  local new_value="$2"
  local old_value="${!var_name:-}"
  if [[ -z "${new_value}" ]]; then
    return
  fi
  if [[ -n "${old_value}" ]]; then
    printf -v "${var_name}" '%s:%s' "${old_value}" "${new_value}"
  else
    printf -v "${var_name}" '%s' "${new_value}"
  fi
  export "${var_name}"
}

append_cuda_component_env() {
  if [[ ${WITH_CUDA} -eq 0 ]]; then
    return
  fi

  local extra_roots="${OMNIINFER_MNN_CUDA_EXTRA_ROOTS:-}"
  if [[ -x "${VENV_ROOT}/bin/python3" ]]; then
    local venv_roots=""
    venv_roots="$("${VENV_ROOT}/bin/python3" - <<'PY'
import site
from pathlib import Path

roots = []
for base in site.getsitepackages():
    nvidia_root = Path(base) / "nvidia"
    if not nvidia_root.is_dir():
        continue
    for child in nvidia_root.iterdir():
        if child.is_dir() and ((child / "include").is_dir() or (child / "lib").is_dir()):
            roots.append(str(child))
print(":".join(roots))
PY
)"
    if [[ -n "${venv_roots}" ]]; then
      extra_roots="${extra_roots:+${extra_roots}:}${venv_roots}"
    fi
  fi

  if [[ -z "${extra_roots}" ]]; then
    return
  fi

  local include_dirs="" lib_dirs="" root=""
  local old_ifs="${IFS}"
  IFS=':'
  for root in ${extra_roots}; do
    if [[ -d "${root}/include" ]]; then
      include_dirs="${include_dirs:+${include_dirs}:}${root}/include"
    fi
    if [[ -d "${root}/lib" ]]; then
      lib_dirs="${lib_dirs:+${lib_dirs}:}${root}/lib"
    fi
    if [[ -d "${root}/lib64" ]]; then
      lib_dirs="${lib_dirs:+${lib_dirs}:}${root}/lib64"
    fi
  done
  IFS="${old_ifs}"

  append_path_list CPATH "${include_dirs}"
  append_path_list CMAKE_INCLUDE_PATH "${include_dirs}"
  append_path_list LIBRARY_PATH "${lib_dirs}"
  append_path_list LD_LIBRARY_PATH "${lib_dirs}"
  append_path_list CMAKE_LIBRARY_PATH "${lib_dirs}"
}

cuda_component_lib_dirs() {
  if [[ ! -x "${VENV_ROOT}/bin/python3" ]]; then
    return
  fi
  "${VENV_ROOT}/bin/python3" - <<'PY'
import site
from pathlib import Path

dirs = []
for base in site.getsitepackages():
    nvidia_root = Path(base) / "nvidia"
    if not nvidia_root.is_dir():
        continue
    for child in nvidia_root.iterdir():
        for lib_dir in (child / "lib", child / "lib64"):
            if lib_dir.is_dir():
                dirs.append(str(lib_dir))
print(":".join(dirs))
PY
}

create_venv() {
  if "${PYTHON_BIN}" -m venv "${VENV_ROOT}"; then
    return
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to create venv with ${PYTHON_BIN}, and uv was not found for fallback." >&2
    echo "Install python3-venv or provide a Python with ensurepip via --python." >&2
    exit 1
  fi

  echo "Falling back to uv venv because ${PYTHON_BIN} could not create a venv."
  rm -rf "${VENV_ROOT}"
  local resolved_python="${PYTHON_BIN}"
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    resolved_python="$(command -v "${PYTHON_BIN}")"
  fi
  uv venv --python "${resolved_python}" "${VENV_ROOT}"
}

install_python_deps() {
  if "${VENV_ROOT}/bin/python3" -m pip --version >/dev/null 2>&1; then
    "${VENV_ROOT}/bin/python3" -m pip install --upgrade pip setuptools wheel numpy
    return
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "The MNN venv does not have pip, and uv was not found for dependency install." >&2
    exit 1
  fi

  uv pip install --python "${VENV_ROOT}/bin/python3" --upgrade pip setuptools wheel numpy
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
select_python_with_headers
configure_cuda_env
prepare_runtime_dirs

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "Will create/update venv under ${VENV_ROOT}"
  echo "Will build PyMNN with LLM support from ${PIP_PACKAGE_ROOT}"
  exit 0
fi

if [[ ! -x "${VENV_ROOT}/bin/python3" ]]; then
  create_venv
fi

install_python_deps
append_cuda_component_env

pushd "${PIP_PACKAGE_ROOT}" >/dev/null
DEPS_ARGS=(llm)
if [[ ${WITH_OPENCL} -eq 1 ]]; then
  DEPS_ARGS+=(opencl)
fi
if [[ ${WITH_CUDA} -eq 1 ]]; then
  DEPS_ARGS+=(cuda)
fi
DEPS_SPEC="$(IFS=,; echo "${DEPS_ARGS[*]}")"
PROJECT_ROOT="${MNN_ROOT}" \
  "${VENV_ROOT}/bin/python3" build_deps.py "${DEPS_SPEC}"

PROJECT_ROOT="${MNN_ROOT}" \
  MNN_BUILD_DIR="${BUILD_ROOT}" \
  MAX_JOBS="${JOBS}" \
  "${VENV_ROOT}/bin/python3" setup.py install --deps "${DEPS_SPEC}"
popd >/dev/null

WRAPPER_LIBRARY_PATH="${VENV_ROOT}/lib"
if [[ ${WITH_CUDA} -eq 1 ]]; then
  if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
    WRAPPER_LIBRARY_PATH="${WRAPPER_LIBRARY_PATH}:${CUDA_HOME}/lib64"
  fi
  CUDA_COMPONENT_LIB_DIRS="$(cuda_component_lib_dirs)"
  if [[ -n "${CUDA_COMPONENT_LIB_DIRS}" ]]; then
    WRAPPER_LIBRARY_PATH="${WRAPPER_LIBRARY_PATH}:${CUDA_COMPONENT_LIB_DIRS}"
  fi
fi

cat > "${BIN_ROOT}/python3" <<EOF
#!/usr/bin/env bash
export LD_LIBRARY_PATH="${WRAPPER_LIBRARY_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
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
echo "  ./omniinfer backend select mnn-linux"
echo "  ./omniinfer model load -m /absolute/path/to/mnn-model-dir-or-config.json"
