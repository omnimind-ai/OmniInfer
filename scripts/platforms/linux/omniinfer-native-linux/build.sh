#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
DRY_RUN=0
JOBS=""
CLEAN_BUILD=0
BOOTSTRAP_SUBMODULE=1
SMOKE_TEST=0
USE_NATIVE=0
ENABLE_LTO=0

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
  _dep cmake "CMake build system"  "sudo apt install cmake"  cmake
  return ${rc}
}

usage() {
  cat <<'EOF'
Usage: build-omniinfer-native.sh [options]

Options:
  --build-type <type>  CMake build type, default: Release
  --jobs <n>           Parallel build jobs, default: nproc
  --native             Optimize host-side kernels for the current CPU
  --portable           Disable host-specific CPU tuning (default)
  --lto                Enable link-time optimization
  --clean              Remove the previous build directory before configuring
  --no-bootstrap       Do not auto-initialize the omniinfer-native git submodule
  --smoke-test         Run `llama-server --version` after the build completes
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
    --native)
      USE_NATIVE=1
      shift
      ;;
    --portable)
      USE_NATIVE=0
      shift
      ;;
    --lto)
      ENABLE_LTO=1
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
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/linux/omniinfer-native-linux"
LLAMA_ROOT="${REPO_ROOT}/framework/omniinfer-native"
BUILD_ROOT="${PACKAGE_ROOT}/build/omniinfer-native-linux"
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
    echo "omniinfer-native source tree was not found at ${LLAMA_ROOT}" >&2
    echo "Run: git submodule update --init --recursive framework/omniinfer-native" >&2
    exit 1
  fi

  if [[ ! -d "${REPO_ROOT}/.git" && ! -f "${REPO_ROOT}/.git" ]]; then
    echo "omniinfer-native source tree was not found at ${LLAMA_ROOT}" >&2
    exit 1
  fi

  require_command git
  echo "omniinfer-native source tree is missing. Bootstrapping the submodule..."
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  git -C ${REPO_ROOT} submodule update --init --recursive framework/omniinfer-native"
    return
  fi
  git -C "${REPO_ROOT}" submodule update --init --recursive framework/omniinfer-native

  if [[ ! -f "${LLAMA_ROOT}/CMakeLists.txt" ]]; then
    echo "Failed to prepare omniinfer-native at ${LLAMA_ROOT}" >&2
    exit 1
  fi
}

prepare_runtime_dirs() {
  mkdir -p "${BUILD_ROOT}" "${BIN_ROOT}" "${LOG_ROOT}" "${MODELS_ROOT}"
  touch "${BIN_ROOT}/.gitkeep" "${LOG_ROOT}/.gitkeep" "${MODELS_ROOT}/.gitkeep"
}

ensure_llama_root
require_command cmake

if [[ -z "${JOBS}" ]]; then
  JOBS="$(detect_jobs)"
fi

CONFIGURE_ARGS=(
  -S "${LLAMA_ROOT}"
  -B "${BUILD_ROOT}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DBUILD_SHARED_LIBS=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=ON
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_OPENSSL=OFF
  -DGGML_NATIVE=$( [[ ${USE_NATIVE} -eq 1 ]] && printf 'ON' || printf 'OFF' )
  -DGGML_LTO=$( [[ ${ENABLE_LTO} -eq 1 ]] && printf 'ON' || printf 'OFF' )
  -DGGML_VULKAN=ON
)

if command -v ninja >/dev/null 2>&1; then
  CONFIGURE_ARGS+=(-G Ninja)
fi

echo "Configuring OmniInfer Native Linux (EAGLE3) build..."
echo "  cmake ${CONFIGURE_ARGS[*]}"
echo "Building llama-server + llama-speculative-simple..."
echo "  cmake --build ${BUILD_ROOT} --target llama-server llama-speculative-simple --config ${BUILD_TYPE} -j ${JOBS}"
echo "CPU tuning mode: $( [[ ${USE_NATIVE} -eq 1 ]] && printf 'native' || printf 'portable' )"
echo "Link-time optimization: $( [[ ${ENABLE_LTO} -eq 1 ]] && printf 'enabled' || printf 'disabled' )"
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
cmake --build "${BUILD_ROOT}" --target llama-server llama-speculative-simple --config "${BUILD_TYPE}" -j "${JOBS}"

find "${BIN_ROOT}" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} +
cp -a "${BUILD_ROOT}/bin/." "${BIN_ROOT}/"
chmod +x "${BIN_ROOT}/llama-server"
[[ -f "${BIN_ROOT}/llama-speculative-simple" ]] && chmod +x "${BIN_ROOT}/llama-speculative-simple"

if [[ ! -x "${BIN_ROOT}/llama-server" ]]; then
  echo "Build finished but llama-server was not copied into ${BIN_ROOT}." >&2
  exit 1
fi

if [[ ${SMOKE_TEST} -eq 1 ]]; then
  echo "Running smoke test..."
  env LD_LIBRARY_PATH="${BIN_ROOT}:${LD_LIBRARY_PATH:-}" "${BIN_ROOT}/llama-server" --version >/dev/null
fi

echo
echo "OmniInfer Native Linux (EAGLE3) build complete."
echo "Binary package location: ${BIN_ROOT}"
echo "Models directory: ${MODELS_ROOT}"
echo "Next step:"
echo "  ./omniinfer select omniinfer-native-linux"
echo "  ./omniinfer model load -m /absolute/path/to/model.gguf"
