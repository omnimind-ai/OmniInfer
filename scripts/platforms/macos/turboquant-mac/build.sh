#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
DRY_RUN=0
JOBS=""
CLEAN_BUILD=0
BOOTSTRAP_SUBMODULE=1
SMOKE_TEST=0

usage() {
  cat <<'EOF'
Usage: build.sh [options]

Options:
  --build-type <type>  CMake build type, default: Release
  --jobs <n>           Parallel build jobs, default: sysctl hw.ncpu
  --clean              Remove the previous build directory before configuring
  --no-bootstrap       Do not auto-initialize the TurboQuant git submodule
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
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/macos/turboquant-mac"
TURBOQUANT_ROOT="${REPO_ROOT}/framework/llama-cpp-turboquant"
BUILD_ROOT="${PACKAGE_ROOT}/build/turboquant-mac"
BIN_ROOT="${PACKAGE_ROOT}/bin"
LOG_ROOT="${PACKAGE_ROOT}/logs"
MODELS_ROOT="${PACKAGE_ROOT}/models"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' was not found in PATH." >&2
    exit 1
  fi
}

ensure_cmake_on_path() {
  if command -v cmake >/dev/null 2>&1; then
    return
  fi

  for candidate in "${HOME}"/Library/Python/*/bin/cmake; do
    if [[ -x "${candidate}" ]]; then
      export PATH="$(dirname "${candidate}"):${PATH}"
      return
    fi
  done
}

detect_jobs() {
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
    return
  fi
  printf '1\n'
}

ensure_turboquant_root() {
  if [[ -f "${TURBOQUANT_ROOT}/CMakeLists.txt" ]]; then
    return
  fi

  if [[ ${BOOTSTRAP_SUBMODULE} -eq 0 ]]; then
    echo "TurboQuant source tree was not found at ${TURBOQUANT_ROOT}" >&2
    echo "Run: git submodule update --init --recursive framework/llama-cpp-turboquant" >&2
    exit 1
  fi

  if [[ ! -d "${REPO_ROOT}/.git" && ! -f "${REPO_ROOT}/.git" ]]; then
    echo "TurboQuant source tree was not found at ${TURBOQUANT_ROOT}" >&2
    exit 1
  fi

  require_command git
  echo "TurboQuant source tree is missing. Bootstrapping the submodule..."
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  git -C ${REPO_ROOT} submodule update --init --recursive framework/llama-cpp-turboquant"
    return
  fi
  git -C "${REPO_ROOT}" submodule update --init --recursive framework/llama-cpp-turboquant

  if [[ ! -f "${TURBOQUANT_ROOT}/CMakeLists.txt" ]]; then
    echo "Failed to prepare TurboQuant at ${TURBOQUANT_ROOT}" >&2
    exit 1
  fi
}

prepare_runtime_dirs() {
  mkdir -p "${BUILD_ROOT}" "${BIN_ROOT}" "${LOG_ROOT}" "${MODELS_ROOT}"
}

copy_optional_binary() {
  local source_path="$1"
  local target_name="$2"
  if [[ -x "${source_path}" ]]; then
    cp "${source_path}" "${BIN_ROOT}/${target_name}"
    chmod +x "${BIN_ROOT}/${target_name}"
  fi
}

build_optional_target() {
  local target_name="$1"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  cmake --build ${BUILD_ROOT} --target ${target_name} --config ${BUILD_TYPE} -j ${JOBS}"
    return
  fi

  if ! cmake --build "${BUILD_ROOT}" --target "${target_name}" --config "${BUILD_TYPE}" -j "${JOBS}"; then
    echo "Warning: optional target '${target_name}' is not available in this TurboQuant checkout." >&2
  fi
}

ensure_cmake_on_path
ensure_turboquant_root
require_command cmake

if [[ -z "${JOBS}" ]]; then
  JOBS="$(detect_jobs)"
fi

CONFIGURE_ARGS=(
  -S "${TURBOQUANT_ROOT}"
  -B "${BUILD_ROOT}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_OPENSSL=OFF
  -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON
)

if command -v ninja >/dev/null 2>&1; then
  CONFIGURE_ARGS+=(-G Ninja)
fi

echo "Configuring TurboQuant macOS Metal build..."
echo "  cmake ${CONFIGURE_ARGS[*]}"
echo "Building llama-server..."
echo "  cmake --build ${BUILD_ROOT} --target llama-server --config ${BUILD_TYPE} -j ${JOBS}"
echo "Optional utility targets:"
echo "  llama-cli"
echo "  llama-bench"
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
build_optional_target "llama-cli"
build_optional_target "llama-bench"

cp "${BUILD_ROOT}/bin/llama-server" "${BIN_ROOT}/llama-server"
chmod +x "${BIN_ROOT}/llama-server"
copy_optional_binary "${BUILD_ROOT}/bin/llama-cli" "llama-cli"
copy_optional_binary "${BUILD_ROOT}/bin/llama-bench" "llama-bench"

if [[ ! -x "${BIN_ROOT}/llama-server" ]]; then
  echo "Build finished but llama-server was not copied into ${BIN_ROOT}." >&2
  exit 1
fi

if [[ ${SMOKE_TEST} -eq 1 ]]; then
  echo "Running macOS TurboQuant smoke test..."
  "${BIN_ROOT}/llama-server" --version >/dev/null
fi

echo
echo "macOS TurboQuant build complete."
echo "Binary package location: ${BIN_ROOT}/llama-server"
echo "Models directory: ${MODELS_ROOT}"
echo "Default runtime flags:"
echo "  -fa on --cache-type-k turbo4 --cache-type-v turbo4"
echo "Next step:"
echo "  ./omniinfer select turboquant-mac"
echo "  ./omniinfer model load -m /absolute/path/to/model.gguf"
