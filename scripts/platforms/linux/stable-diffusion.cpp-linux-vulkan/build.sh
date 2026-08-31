#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
JOBS=""
CLEAN_BUILD=0
BOOTSTRAP_SUBMODULE=1
SMOKE_TEST=0
BUILD_FROM_SOURCE=0
USE_NATIVE=0
ENABLE_LTO=0
MIN_VULKAN_HEADER_VERSION=301

if [[ -n "${VULKAN_SDK:-}" && -d "${VULKAN_SDK}/bin" ]]; then
  PATH="${VULKAN_SDK}/bin:${PATH}"
fi

vulkan_header_path() {
  local prefix
  local -a _cmake_prefixes=()
  if [[ -n "${VULKAN_SDK:-}" && -f "${VULKAN_SDK}/include/vulkan/vulkan_core.h" ]]; then
    printf '%s\n' "${VULKAN_SDK}/include/vulkan/vulkan_core.h"
    return 0
  fi
  IFS=':' read -r -a _cmake_prefixes <<< "${CMAKE_PREFIX_PATH:-}"
  for prefix in "${_cmake_prefixes[@]}"; do
    if [[ -n "${prefix}" && -f "${prefix}/include/vulkan/vulkan_core.h" ]]; then
      printf '%s\n' "${prefix}/include/vulkan/vulkan_core.h"
      return 0
    fi
  done
  if [[ -f /usr/include/vulkan/vulkan_core.h ]]; then
    printf '%s\n' /usr/include/vulkan/vulkan_core.h
    return 0
  fi
  return 1
}

vulkan_header_version() {
  local header
  header="$(vulkan_header_path)" || return 1
  awk '/^#define VK_HEADER_VERSION / { print $3; exit }' "${header}"
}

cmake_prefix_with_vulkan_sdk() {
  if [[ -n "${VULKAN_SDK:-}" ]]; then
    printf '%s%s%s\n' "${VULKAN_SDK}" "${CMAKE_PREFIX_PATH:+:}" "${CMAKE_PREFIX_PATH:-}"
  else
    printf '%s\n' "${CMAKE_PREFIX_PATH:-}"
  fi
}

spirv_headers_available() {
  local probe_dir probe_rc=0
  probe_dir="$(mktemp -d "${TMPDIR:-/tmp}/omniinfer-sdcpp-cmake.XXXXXX")" || return 1
  (
    cd "${probe_dir}"
    CMAKE_PREFIX_PATH="$(cmake_prefix_with_vulkan_sdk)" \
      cmake --find-package -DNAME=SPIRV-Headers -DCOMPILER_ID=GNU \
        -DLANGUAGE=CXX -DMODE=EXIST >/dev/null 2>&1
  ) || probe_rc=$?
  rm -rf -- "${probe_dir}"
  return "${probe_rc}"
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
  _dep cmake "CMake build system" "sudo apt install cmake" cmake
  _dep glslc "Vulkan shader compiler" "sudo apt install glslc" glslc
  local header_version=""
  header_version="$(vulkan_header_version 2>/dev/null || true)"
  if [[ "${header_version}" =~ ^[0-9]+$ ]] && ((header_version >= MIN_VULKAN_HEADER_VERSION)); then
    printf 'ok|vulkan-headers|recent Vulkan headers|VULKAN_SDK or CMAKE_PREFIX_PATH|\n'
  else
    printf 'missing|vulkan-headers|recent Vulkan headers|Install a current LunarG Vulkan SDK and set VULKAN_SDK (verified: 1.4.357.0)|\n'
    rc=1
  fi
  if spirv_headers_available; then
    printf 'ok|spirv-headers|SPIRV-Headers CMake package|VULKAN_SDK or CMAKE_PREFIX_PATH|\n'
  else
    printf 'missing|spirv-headers|SPIRV-Headers CMake package|Install SPIRV-Headers beside the Vulkan SDK and set VULKAN_SDK or CMAKE_PREFIX_PATH|\n'
    rc=1
  fi
  return ${rc}
}

usage() {
  cat <<'EOF'
Usage: build.sh [options]

Options:
  --build-type <type>  CMake build type, default: Release
  --jobs <n>           Parallel build jobs, default: nproc
  --native             Optimize host-side kernels for the current CPU
  --portable           Disable host-specific CPU tuning (default)
  --lto                Enable link-time optimization
  --clean              Remove this backend's previous build directory
  --no-bootstrap       Do not initialize the stable-diffusion.cpp submodule
  --from-source        Build from the pinned source submodule
  --smoke-test         Run sd-server and sd-cli help checks after building
  --dry-run            Print configure and build commands without executing them
  --check-deps         Report build dependencies in installer-readable format
  -h, --help           Show this help message
EOF
}

DRY_RUN=0
while (($# > 0)); do
  case "$1" in
    --build-type) BUILD_TYPE="${2:?missing value for --build-type}"; shift 2 ;;
    --jobs) JOBS="${2:?missing value for --jobs}"; shift 2 ;;
    --native) USE_NATIVE=1; shift ;;
    --portable) USE_NATIVE=0; shift ;;
    --lto) ENABLE_LTO=1; shift ;;
    --clean) CLEAN_BUILD=1; shift ;;
    --no-bootstrap) BOOTSTRAP_SUBMODULE=0; shift ;;
    --from-source) BUILD_FROM_SOURCE=1; shift ;;
    --smoke-test) SMOKE_TEST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --check-deps) check_deps; exit $? ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ROOT}/../../../.." && pwd)"
BACKEND_ID="stable-diffusion.cpp-linux-vulkan"
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/linux/${BACKEND_ID}"
SD_ROOT="${REPO_ROOT}/framework/stable-diffusion.cpp"
BUILD_ROOT="${PACKAGE_ROOT}/build/${BACKEND_ID}"
BIN_ROOT="${PACKAGE_ROOT}/bin"
LOG_ROOT="${PACKAGE_ROOT}/logs"

if [[ ${BUILD_FROM_SOURCE} -eq 0 ]]; then
  echo "No prebuilt install path is configured for ${BACKEND_ID}." >&2
  echo "Re-run with --from-source to build the pinned stable-diffusion.cpp submodule." >&2
  exit 1
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' was not found in PATH." >&2
    exit 1
  fi
}

detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  else
    printf '1\n'
  fi
}

ensure_source() {
  if [[ -f "${SD_ROOT}/CMakeLists.txt" && -f "${SD_ROOT}/ggml/CMakeLists.txt" && -f "${SD_ROOT}/thirdparty/libwebm/CMakeLists.txt" ]]; then
    return
  fi
  if [[ ${BOOTSTRAP_SUBMODULE} -eq 0 ]]; then
    echo "stable-diffusion.cpp source tree or nested dependencies are incomplete at ${SD_ROOT}" >&2
    exit 1
  fi
  require_command git
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "git -C ${REPO_ROOT} submodule update --init --recursive framework/stable-diffusion.cpp"
    return
  fi
  git -C "${REPO_ROOT}" submodule update --init --recursive framework/stable-diffusion.cpp
  if [[ ! -f "${SD_ROOT}/CMakeLists.txt" || ! -f "${SD_ROOT}/ggml/CMakeLists.txt" || ! -f "${SD_ROOT}/thirdparty/libwebm/CMakeLists.txt" ]]; then
    echo "Failed to prepare stable-diffusion.cpp and its pinned nested dependencies." >&2
    exit 1
  fi
}

require_command cmake
require_command glslc
header_version="$(vulkan_header_version 2>/dev/null || true)"
if [[ ! "${header_version}" =~ ^[0-9]+$ ]] || ((header_version < MIN_VULKAN_HEADER_VERSION)); then
  echo "Recent Vulkan headers are required (VK_HEADER_VERSION >= ${MIN_VULKAN_HEADER_VERSION})." >&2
  echo "Install a current LunarG Vulkan SDK and set VULKAN_SDK; Ubuntu 22.04's stock headers are too old for this pinned source." >&2
  exit 1
fi
if ! spirv_headers_available; then
  echo "SPIRV-Headers was not found by CMake." >&2
  echo "Install it beside the Vulkan SDK and set VULKAN_SDK or CMAKE_PREFIX_PATH." >&2
  exit 1
fi
ensure_source
[[ -n "${JOBS}" ]] || JOBS="$(detect_jobs)"

CONFIGURE_ARGS=(
  -S "${SD_ROOT}"
  -B "${BUILD_ROOT}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=$( [[ ${ENABLE_LTO} -eq 1 ]] && printf 'ON' || printf 'OFF' )
  -DSD_VULKAN=ON
  -DSD_BUILD_EXAMPLES=ON
  -DSD_SERVER_BUILD_FRONTEND=OFF
  -DSD_BUILD_SHARED_LIBS=OFF
  -DSD_BUILD_SHARED_GGML_LIB=OFF
  -DSD_WEBM=ON
  -DSD_WEBP=ON
  -DGGML_NATIVE=$( [[ ${USE_NATIVE} -eq 1 ]] && printf 'ON' || printf 'OFF' )
)
if command -v ninja >/dev/null 2>&1; then
  CONFIGURE_ARGS+=(-G Ninja)
fi
if [[ -n "${VULKAN_SDK:-}" ]]; then
  existing_prefix_path="${CMAKE_PREFIX_PATH:-}"
  existing_prefix_list="${existing_prefix_path//:/;}"
  CONFIGURE_ARGS+=(-DCMAKE_PREFIX_PATH="${VULKAN_SDK}${existing_prefix_list:+;}${existing_prefix_list}")
fi

echo "Configuring ${BACKEND_ID}..."
printf '  cmake'; printf ' %q' "${CONFIGURE_ARGS[@]}"; printf '\n'
echo "Building sd-server and sd-cli..."
echo "  cmake --build ${BUILD_ROOT} --target sd-server sd-cli --config ${BUILD_TYPE} -j ${JOBS}"
echo "CPU tuning: $( [[ ${USE_NATIVE} -eq 1 ]] && printf 'native' || printf 'portable' ); LTO: $( [[ ${ENABLE_LTO} -eq 1 ]] && printf 'on' || printf 'off' )"

if [[ ${DRY_RUN} -eq 1 ]]; then
  exit 0
fi
if [[ ${CLEAN_BUILD} -eq 1 && -d "${BUILD_ROOT}" ]]; then
  rm -rf -- "${BUILD_ROOT}"
fi
mkdir -p "${BUILD_ROOT}" "${BIN_ROOT}" "${LOG_ROOT}"
cmake "${CONFIGURE_ARGS[@]}"
cmake --build "${BUILD_ROOT}" --target sd-server sd-cli --config "${BUILD_TYPE}" -j "${JOBS}"

find "${BIN_ROOT}" -mindepth 1 -maxdepth 1 -type f -delete
cp -a "${BUILD_ROOT}/bin/." "${BIN_ROOT}/"
chmod +x "${BIN_ROOT}/sd-server" "${BIN_ROOT}/sd-cli"
for binary in sd-server sd-cli; do
  if [[ ! -x "${BIN_ROOT}/${binary}" ]]; then
    echo "Build finished but ${binary} was not copied into ${BIN_ROOT}." >&2
    exit 1
  fi
done

if [[ ${SMOKE_TEST} -eq 1 ]]; then
  "${BIN_ROOT}/sd-server" --help >/dev/null
  "${BIN_ROOT}/sd-cli" --help >/dev/null
fi

echo "${BACKEND_ID} build complete: ${BIN_ROOT}"
