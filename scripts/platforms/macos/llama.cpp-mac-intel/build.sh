#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
DRY_RUN=0
JOBS=""
CLEAN_BUILD=0
BOOTSTRAP_SUBMODULE=1
SMOKE_TEST=0
INSTALL_PREBUILT=0

usage() {
  cat <<'EOF'
Usage: build.sh [options]

Options:
  --build-type <type>  CMake build type, default: Release
  --jobs <n>           Parallel build jobs, default: sysctl hw.ncpu
  --clean              Remove the previous build directory before configuring
  --no-bootstrap       Do not auto-initialize the llama.cpp git submodule
  --prebuilt           Download and install the configured upstream prebuilt archive
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
    --prebuilt)
      INSTALL_PREBUILT=1
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
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/macos/llama.cpp-mac-intel"
LLAMA_ROOT="${REPO_ROOT}/framework/llama.cpp"
BUILD_ROOT="${PACKAGE_ROOT}/build/llama.cpp-mac-intel"
BIN_ROOT="${PACKAGE_ROOT}/bin"
LOG_ROOT="${PACKAGE_ROOT}/logs"
MODELS_ROOT="${REPO_ROOT}/.local/models"

if [[ ${INSTALL_PREBUILT} -eq 1 ]]; then
  PREBUILT_ARGS=(
    --catalog "${REPO_ROOT}/scripts/prebuilt_backends.json"
    --platform macos
    --backend llama.cpp-mac-intel
    --runtime-dir "${PACKAGE_ROOT}"
    --models-dir "${MODELS_ROOT}"
  )
  [[ ${DRY_RUN} -eq 1 ]] && PREBUILT_ARGS+=(--dry-run)
  exec python3 "${REPO_ROOT}/scripts/platforms/common/install-prebuilt.py" "${PREBUILT_ARGS[@]}"
fi

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
}

reset_stale_cmake_cache() {
  local cache="${BUILD_ROOT}/CMakeCache.txt"
  [[ -f "${cache}" ]] || return

  local existing_generator
  existing_generator="$(awk -F= '/^CMAKE_GENERATOR:INTERNAL=/{print $2}' "${cache}" | tail -n 1)"
  [[ -n "${existing_generator}" ]] || return
  if [[ "${existing_generator}" == "${REQUESTED_GENERATOR}" ]]; then
    return
  fi

  echo "CMake generator changed from ${existing_generator} to ${REQUESTED_GENERATOR}; cleaning ${BUILD_ROOT}"
  rm -rf "${BUILD_ROOT}"
}

ensure_cmake_on_path
ensure_llama_root
require_command cmake

if [[ -z "${JOBS}" ]]; then
  JOBS="$(detect_jobs)"
fi

CONFIGURE_ARGS=(
  -S "${LLAMA_ROOT}"
  -B "${BUILD_ROOT}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_OSX_ARCHITECTURES=x86_64
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_OPENSSL=OFF
  -DGGML_NATIVE=OFF
  -DGGML_METAL=OFF
)

REQUESTED_GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
  REQUESTED_GENERATOR="Ninja"
  CONFIGURE_ARGS+=(-G "${REQUESTED_GENERATOR}")
fi

echo "Configuring llama.cpp macOS Intel build..."
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
else
  reset_stale_cmake_cache
fi
mkdir -p "${BUILD_ROOT}"

cmake "${CONFIGURE_ARGS[@]}"
cmake --build "${BUILD_ROOT}" --target llama-server --config "${BUILD_TYPE}" -j "${JOBS}"

cp "${BUILD_ROOT}/bin/llama-server" "${BIN_ROOT}/llama-server"
chmod +x "${BIN_ROOT}/llama-server"

if [[ ! -x "${BIN_ROOT}/llama-server" ]]; then
  echo "Build finished but llama-server was not copied into ${BIN_ROOT}." >&2
  exit 1
fi

if [[ ${SMOKE_TEST} -eq 1 ]]; then
  echo "Running macOS Intel smoke test..."
  if [[ "$(uname -m)" == "x86_64" ]]; then
    "${BIN_ROOT}/llama-server" --version >/dev/null
  else
    arch -x86_64 "${BIN_ROOT}/llama-server" --version >/dev/null
  fi
fi

echo
echo "macOS Intel build complete."
echo "Binary package location: ${BIN_ROOT}/llama-server"
echo "Models directory: ${MODELS_ROOT}"
echo "Next step:"
echo "  ./omniinfer backend select llama.cpp-mac-intel"
echo "  ./omniinfer model load -m /absolute/path/to/model.gguf"
