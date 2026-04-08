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

usage() {
  cat <<'EOF'
Usage: build-llama-linux-vulkan.sh [options]

Options:
  --build-type <type>  CMake build type, default: Release
  --jobs <n>           Parallel build jobs, default: nproc
  --native             Optimize host-side kernels for the current CPU
  --portable           Disable host-specific CPU tuning (default)
  --lto                Enable link-time optimization
  --clean              Remove the previous build directory before configuring
  --no-bootstrap       Do not auto-initialize the llama.cpp git submodule
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
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ROOT}/../../../.." && pwd)"
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/linux/llama.cpp-linux-vulkan"
LLAMA_ROOT="${REPO_ROOT}/framework/llama.cpp"
BUILD_ROOT="${PACKAGE_ROOT}/build/llama.cpp-linux-vulkan"
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
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_OPENSSL=OFF
  -DGGML_NATIVE=$( [[ ${USE_NATIVE} -eq 1 ]] && printf 'ON' || printf 'OFF' )
  -DGGML_LTO=$( [[ ${ENABLE_LTO} -eq 1 ]] && printf 'ON' || printf 'OFF' )
  -DGGML_VULKAN=ON
)

if command -v ninja >/dev/null 2>&1; then
  CONFIGURE_ARGS+=(-G Ninja)
fi

echo "Configuring llama.cpp Linux Vulkan build..."
echo "  cmake ${CONFIGURE_ARGS[*]}"
echo "Building llama-server..."
echo "  cmake --build ${BUILD_ROOT} --target llama-server --config ${BUILD_TYPE} -j ${JOBS}"
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
cmake --build "${BUILD_ROOT}" --target llama-server --config "${BUILD_TYPE}" -j "${JOBS}"

find "${BIN_ROOT}" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} +
cp -a "${BUILD_ROOT}/bin/." "${BIN_ROOT}/"
chmod +x "${BIN_ROOT}/llama-server"

if [[ ! -x "${BIN_ROOT}/llama-server" ]]; then
  echo "Build finished but llama-server was not copied into ${BIN_ROOT}." >&2
  exit 1
fi

if [[ ${SMOKE_TEST} -eq 1 ]]; then
  echo "Running Vulkan smoke test..."
  env LD_LIBRARY_PATH="${BIN_ROOT}:${LD_LIBRARY_PATH:-}" "${BIN_ROOT}/llama-server" --version >/dev/null
fi

echo
echo "Linux Vulkan build complete."
echo "Binary package location: ${BIN_ROOT}"
echo "Models directory: ${MODELS_ROOT}"
echo "Next step:"
echo "  ./omniinfer select llama.cpp-linux-vulkan"
echo "  ./omniinfer model load -m /absolute/path/to/model.gguf"
