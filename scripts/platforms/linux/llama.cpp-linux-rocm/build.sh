#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
DRY_RUN=0
JOBS=""
GPU_TARGETS=""
ROCM_PATH_OVERRIDE="${ROCM_PATH:-}"
CLEAN_BUILD=1
BOOTSTRAP_SUBMODULE=1
SMOKE_TEST=0

usage() {
  cat <<'EOF'
Usage: build-llama-linux-rocm.sh [options]

Options:
  --build-type <type>  CMake build type, default: Release
  --jobs <n>           Parallel build jobs, default: nproc
  --gpu-targets <id>   Override HIP GPU targets, for example gfx1151
  --rocm-path <path>   Override ROCm installation root
  --no-clean           Reuse the previous build directory instead of reconfiguring from scratch
  --no-bootstrap       Do not auto-initialize the llama.cpp git submodule
  --smoke-test         Run `llama-server --list-devices` after the build completes
  --dry-run            Print actions without executing them
  -h, --help           Show this help message
EOF
}

while (($# > 0)); do
  case "$1" in
    --build-type)
      BUILD_TYPE="${2:?missing value for --build-type}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:?missing value for --jobs}"
      shift 2
      ;;
    --gpu-targets)
      GPU_TARGETS="${2:?missing value for --gpu-targets}"
      shift 2
      ;;
    --rocm-path)
      ROCM_PATH_OVERRIDE="${2:?missing value for --rocm-path}"
      shift 2
      ;;
    --no-clean)
      CLEAN_BUILD=0
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
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ROOT}/../../../.." && pwd)"
PACKAGE_ROOT="${REPO_ROOT}/platform/Linux/llama.cpp-linux-rocm"
LLAMA_ROOT="${REPO_ROOT}/framework/llama.cpp"
BUILD_ROOT="${PACKAGE_ROOT}/build/llama.cpp-linux-rocm"
BIN_ROOT="${PACKAGE_ROOT}/bin"
LOG_ROOT="${PACKAGE_ROOT}/logs"
MODELS_ROOT="${PACKAGE_ROOT}/models"

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

ensure_llama_root() {
  if [[ -f "${LLAMA_ROOT}/CMakeLists.txt" ]]; then
    return
  fi

  if [[ ${BOOTSTRAP_SUBMODULE} -eq 0 ]]; then
    echo "llama.cpp source tree was not found at ${LLAMA_ROOT}" >&2
    echo "Run: git submodule update --init --recursive framework/llama.cpp" >&2
    exit 1
  fi

  if [[ ! -d "${REPO_ROOT}/.git" && ! -f "${REPO_ROOT}/.git" ]]; then
    echo "llama.cpp source tree was not found at ${LLAMA_ROOT}" >&2
    exit 1
  fi

  require_command git
  echo "llama.cpp source tree is missing. Bootstrapping the submodule..."
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  git -C ${REPO_ROOT} submodule update --init --recursive framework/llama.cpp"
    return
  fi
  git -C "${REPO_ROOT}" submodule update --init --recursive framework/llama.cpp

  if [[ ! -f "${LLAMA_ROOT}/CMakeLists.txt" ]]; then
    echo "Failed to prepare llama.cpp at ${LLAMA_ROOT}" >&2
    exit 1
  fi
}

prepare_runtime_dirs() {
  mkdir -p "${BUILD_ROOT}" "${BIN_ROOT}" "${LOG_ROOT}" "${MODELS_ROOT}"
  touch "${BIN_ROOT}/.gitkeep" "${LOG_ROOT}/.gitkeep" "${MODELS_ROOT}/.gitkeep"
}

detect_rocm_root() {
  if [[ -n "${ROCM_PATH_OVERRIDE}" ]]; then
    printf '%s\n' "${ROCM_PATH_OVERRIDE}"
    return
  fi

  if command -v hipcc >/dev/null 2>&1; then
    local compiler_root
    compiler_root="$(cd "$(dirname "$(command -v hipcc)")/.." && pwd)"
    if [[ -d "${compiler_root}" ]]; then
      printf '%s\n' "${compiler_root}"
      return
    fi
  fi

  if [[ -d /opt/rocm ]]; then
    printf '%s\n' "/opt/rocm"
    return
  fi

  local latest=""
  for candidate in /opt/rocm-*; do
    if [[ -d "${candidate}" ]]; then
      latest="${candidate}"
    fi
  done

  if [[ -n "${latest}" ]]; then
    printf '%s\n' "${latest}"
    return
  fi

  printf '%s\n' "/usr"
}

detect_gpu_targets() {
  if [[ -n "${GPU_TARGETS}" ]]; then
    printf '%s\n' "${GPU_TARGETS}"
    return
  fi

  if [[ -x "${ROCM_ROOT}/bin/rocminfo" ]]; then
    local detected
    detected="$("${ROCM_ROOT}/bin/rocminfo" 2>/dev/null | awk '/gfx[0-9a-z]+/ {print $NF; exit}')"
    if [[ -n "${detected}" ]]; then
      printf '%s\n' "${detected}"
      return
    fi
  fi

  printf '%s\n' ""
}

warn_rocm_permissions() {
  local missing=()
  if command -v id >/dev/null 2>&1; then
    if ! id -nG 2>/dev/null | tr ' ' '\n' | grep -qx 'render'; then
      missing+=("render")
    fi
  fi
  if [[ -e /dev/kfd && ! -r /dev/kfd ]]; then
    echo "Warning: /dev/kfd is not readable for the current user." >&2
  fi
  if ((${#missing[@]} > 0)); then
    echo "Warning: current user is not in the required group(s): ${missing[*]}" >&2
    echo "         ROCm runtime may fail until a new login session picks up the updated groups." >&2
  fi
}

copy_rocm_runtime_deps() {
  local rocm_root="$1"
  local bin_root="$2"
  declare -A seen=()
  local queue=()
  local current dep dep_real dep_name

  while IFS= read -r -d '' current; do
    queue+=("${current}")
  done < <(find "${bin_root}" -maxdepth 1 -type f \( -name 'llama-server' -o -name 'lib*.so*' \) -print0)

  while ((${#queue[@]} > 0)); do
    current="${queue[0]}"
    queue=("${queue[@]:1}")
    while IFS= read -r dep; do
      [[ -z "${dep}" || "${dep}" != /* ]] && continue
      dep_real="$(readlink -f "${dep}")"
      [[ -f "${dep_real}" ]] || continue
      case "${dep_real}" in
        "${rocm_root}"/*)
          dep_name="$(basename "${dep_real}")"
          if [[ -n "${seen[${dep_name}]:-}" ]]; then
            continue
          fi
          cp -L "${dep_real}" "${bin_root}/${dep_name}"
          seen["${dep_name}"]=1
          queue+=("${bin_root}/${dep_name}")
          ;;
      esac
    done < <(ldd "${current}" | awk '
      /=>/ && $3 ~ /^\// { print $3 }
      $1 ~ /^\// { print $1 }
    ')
  done
}

ensure_llama_root

require_command cmake

if [[ -z "${JOBS}" ]]; then
  JOBS="$(detect_jobs)"
fi

ROCM_ROOT="$(detect_rocm_root)"
export ROCM_PATH="${ROCM_ROOT}"
export HIP_PATH="${ROCM_ROOT}"
export PATH="${ROCM_ROOT}/bin:${ROCM_ROOT}/lib/llvm/bin:${PATH}"

require_command hipcc
require_command hipconfig
warn_rocm_permissions

HIPCXX_DIR="$(hipconfig -l)"
export HIPCXX="${HIPCXX_DIR}/clang"

GPU_TARGETS_DETECTED="$(detect_gpu_targets)"

CONFIGURE_ARGS=(
  -S "${LLAMA_ROOT}"
  -B "${BUILD_ROOT}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DBUILD_SHARED_LIBS=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_OPENSSL=OFF
  -DGGML_HIP=ON
  -DGGML_NATIVE=OFF
  -DCMAKE_C_COMPILER="${ROCM_ROOT}/lib/llvm/bin/clang"
  -DCMAKE_CXX_COMPILER="${ROCM_ROOT}/lib/llvm/bin/clang++"
)

if command -v ninja >/dev/null 2>&1; then
  CONFIGURE_ARGS+=(-G Ninja)
fi

if [[ -n "${GPU_TARGETS_DETECTED}" ]]; then
  CONFIGURE_ARGS+=(-DGPU_TARGETS="${GPU_TARGETS_DETECTED}")
fi

echo "Configuring llama.cpp Linux ROCm build..."
echo "  ROCm root: ${ROCM_ROOT}"
if [[ -n "${GPU_TARGETS_DETECTED}" ]]; then
  echo "  GPU targets: ${GPU_TARGETS_DETECTED}"
else
  echo "  GPU targets: auto (llama.cpp/CMake decides)"
fi
echo "  cmake ${CONFIGURE_ARGS[*]}"
echo "Building llama-server..."
echo "  cmake --build ${BUILD_ROOT} --target llama-server --config ${BUILD_TYPE} -j ${JOBS}"
if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  echo "Cleaning previous build directory: ${BUILD_ROOT}"
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  exit 0
fi

prepare_runtime_dirs
if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  rm -rf "${BUILD_ROOT}"
fi
mkdir -p "${BUILD_ROOT}"

cmake "${CONFIGURE_ARGS[@]}"
cmake --build "${BUILD_ROOT}" --target llama-server --config "${BUILD_TYPE}" -j "${JOBS}"

find "${BIN_ROOT}" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} +
cp -a "${BUILD_ROOT}/bin/." "${BIN_ROOT}/"
copy_rocm_runtime_deps "${ROCM_ROOT}" "${BIN_ROOT}"
chmod +x "${BIN_ROOT}/llama-server"

if [[ ! -x "${BIN_ROOT}/llama-server" ]]; then
  echo "Build finished but llama-server was not copied into ${BIN_ROOT}." >&2
  exit 1
fi

if [[ ${SMOKE_TEST} -eq 1 ]]; then
  echo "Running ROCm smoke test..."
  env LD_LIBRARY_PATH="${BIN_ROOT}:${ROCM_ROOT}/lib:${LD_LIBRARY_PATH:-}" "${BIN_ROOT}/llama-server" --list-devices
fi

echo
echo "Linux ROCm build complete."
echo "Binary package location: ${BIN_ROOT}"
echo "Models directory: ${MODELS_ROOT}"
echo "Next step:"
echo "  ./omniinfer select llama.cpp-linux-rocm"
echo "  ./omniinfer model load -m /absolute/path/to/model.gguf"
