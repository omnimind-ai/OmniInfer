#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
CLEAN_BUILD=0
DRY_RUN=0
JOBS=""

usage() {
  cat <<'EOF'
Usage: build-llama-zo.sh [options]

Build and package the native Android llama-zo CLI with CPU and Hexagon HTP
dynamic backends.

Options:
  --build-type <type>  CMake build type, default: Release
  --jobs <n>           Parallel build jobs, default: nproc
  --clean              Remove the existing build directory before configuring
  --dry-run            Validate inputs and print commands without building
  -h, --help           Show this help message

Environment overrides:
  ANDROID_NDK_ROOT     Android NDK root (required; ANDROID_NDK_HOME is also accepted)
  HEXAGON_SDK_ROOT     Hexagon SDK root (required)
  HEXAGON_TOOLS_ROOT   Hexagon Tools root (default: SDK Tools 19.0.04)
  CMAKE_BIN            CMake executable (default: cmake from PATH)
  CMAKE_GENERATOR      CMake generator (default: Ninja)
  LLAMA_ROOT           llama.cpp source root (default: framework/llama.cpp)
  BUILD_ROOT           Build directory (default: .local/build/android/llama-zo)
  PACKAGE_ROOT         Package directory (default: .local/runtime/android/llama-zo)
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_LLAMA_ROOT="${REPO_ROOT}/framework/llama.cpp"
DEFAULT_BUILD_ROOT="${REPO_ROOT}/.local/build/android/llama-zo"
DEFAULT_PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/android/llama-zo"

LLAMA_ROOT="${LLAMA_ROOT:-${DEFAULT_LLAMA_ROOT}}"
BUILD_ROOT="${BUILD_ROOT:-${DEFAULT_BUILD_ROOT}}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${DEFAULT_PACKAGE_ROOT}}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}"
HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-}"
HEXAGON_TOOLS_ROOT="${HEXAGON_TOOLS_ROOT:-${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/19.0.04}"
CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
REALPATH_BIN="$(command -v realpath || true)"

ANDROID_ABI="arm64-v8a"
ANDROID_API="26"
MINIMUM_CMAKE_VERSION="3.22.2"
HTP_ARCHES=(v68 v69 v73 v75 v79 v81)
OUTPUT_MARKER=".omniinfer-llama-zo-output"
OUTPUT_MARKER_MAGIC="omniinfer-llama-zo-output-v1"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file was not found: $1" >&2
    exit 1
  fi
}

require_executable() {
  if [[ ! -x "$1" ]]; then
    echo "Required executable was not found: $1" >&2
    exit 1
  fi
}

require_value() {
  if [[ -z "$2" ]]; then
    echo "$1 must be set." >&2
    exit 1
  fi
}

path_is_within() {
  [[ "$1" == "$2/"* ]]
}

directory_is_empty() {
  local first_entry
  first_entry="$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" || return 1
  [[ -z "${first_entry}" ]]
}

output_marker_is_valid() {
  local path="$1"
  local marker_magic
  local marker_path
  local extra_line

  [[ -f "${path}/${OUTPUT_MARKER}" && ! -L "${path}/${OUTPUT_MARKER}" ]] || return 1
  {
    IFS= read -r marker_magic || return 1
    IFS= read -r marker_path || return 1
    if IFS= read -r extra_line; then
      return 1
    fi
  } < "${path}/${OUTPUT_MARKER}"
  [[ "${marker_magic}" == "${OUTPUT_MARKER_MAGIC}" && "${marker_path}" == "${path}" ]]
}

validate_output_root() {
  local name="$1"
  local path="$2"
  local allow_unmarked="$3"

  if [[ "${path}" == "/" || "${path}" == "${REPO_ROOT}" || "${path}" == "${LOCAL_OUTPUT_ROOT}" ]]; then
    echo "${name} is not a safe output directory: ${path}" >&2
    exit 1
  fi
  if path_is_within "${REPO_ROOT}" "${path}"; then
    echo "${name} must not contain the OmniInfer checkout: ${path}" >&2
    exit 1
  fi
  if [[ "${path}" == "${LLAMA_ROOT}" ]] \
      || path_is_within "${LLAMA_ROOT}" "${path}" \
      || path_is_within "${path}" "${LLAMA_ROOT}"; then
    echo "${name} must not overlap the llama.cpp source tree: ${path}" >&2
    exit 1
  fi
  if [[ -e "${path}" && ! -d "${path}" ]]; then
    echo "${name} is not a directory: ${path}" >&2
    exit 1
  fi
  if [[ -d "${path}" && "${allow_unmarked}" -eq 0 ]] \
      && ! output_marker_is_valid "${path}" \
      && ! directory_is_empty "${path}"; then
    echo "Refusing to use an existing non-empty custom ${name}: ${path}" >&2
    echo "Use a new or empty directory, or one previously created by this build script." >&2
    exit 1
  fi
}

mark_output_root() {
  mkdir -p "$1"
  printf '%s\n%s\n' "${OUTPUT_MARKER_MAGIC}" "$1" > "$1/${OUTPUT_MARKER}"
}

remove_output_root() {
  validate_output_root "$1" "$2" "$3"
  rm -rf -- "$2"
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

version_at_least() {
  local actual="$1"
  local required="$2"
  [[ "$(printf '%s\n%s\n' "${required}" "${actual}" | sort -V | head -n 1)" == "${required}" ]]
}

require_value "ANDROID_NDK_ROOT (or ANDROID_NDK_HOME)" "${ANDROID_NDK_ROOT}"
require_value "HEXAGON_SDK_ROOT" "${HEXAGON_SDK_ROOT}"
require_value "CMAKE_BIN or cmake in PATH" "${CMAKE_BIN}"
require_executable "${REALPATH_BIN}"

LLAMA_ROOT="$(${REALPATH_BIN} -m -- "${LLAMA_ROOT}")"
BUILD_ROOT_LOGICAL="$(${REALPATH_BIN} -ms -- "${BUILD_ROOT}")"
PACKAGE_ROOT_LOGICAL="$(${REALPATH_BIN} -ms -- "${PACKAGE_ROOT}")"
BUILD_ROOT="$(${REALPATH_BIN} -m -- "${BUILD_ROOT}")"
PACKAGE_ROOT="$(${REALPATH_BIN} -m -- "${PACKAGE_ROOT}")"
DEFAULT_BUILD_ROOT="$(${REALPATH_BIN} -ms -- "${DEFAULT_BUILD_ROOT}")"
DEFAULT_PACKAGE_ROOT="$(${REALPATH_BIN} -ms -- "${DEFAULT_PACKAGE_ROOT}")"
LOCAL_OUTPUT_ROOT="$(${REALPATH_BIN} -ms -- "${REPO_ROOT}/.local")"

if [[ "${BUILD_ROOT}" != "${BUILD_ROOT_LOGICAL}" ]]; then
  echo "BUILD_ROOT must not contain symlink components: ${BUILD_ROOT_LOGICAL}" >&2
  exit 1
fi
if [[ "${PACKAGE_ROOT}" != "${PACKAGE_ROOT_LOGICAL}" ]]; then
  echo "PACKAGE_ROOT must not contain symlink components: ${PACKAGE_ROOT_LOGICAL}" >&2
  exit 1
fi

BUILD_ROOT_IS_DEFAULT=0
PACKAGE_ROOT_IS_DEFAULT=0
if [[ "${BUILD_ROOT}" == "${DEFAULT_BUILD_ROOT}" ]] \
    && path_is_within "${BUILD_ROOT}" "${LOCAL_OUTPUT_ROOT}"; then
  BUILD_ROOT_IS_DEFAULT=1
fi
if [[ "${PACKAGE_ROOT}" == "${DEFAULT_PACKAGE_ROOT}" ]] \
    && path_is_within "${PACKAGE_ROOT}" "${LOCAL_OUTPUT_ROOT}"; then
  PACKAGE_ROOT_IS_DEFAULT=1
fi

validate_output_root "BUILD_ROOT" "${BUILD_ROOT}" "${BUILD_ROOT_IS_DEFAULT}"
validate_output_root "PACKAGE_ROOT" "${PACKAGE_ROOT}" "${PACKAGE_ROOT_IS_DEFAULT}"
if [[ "${BUILD_ROOT}" == "${PACKAGE_ROOT}" ]] \
    || path_is_within "${BUILD_ROOT}" "${PACKAGE_ROOT}" \
    || path_is_within "${PACKAGE_ROOT}" "${BUILD_ROOT}"; then
  echo "BUILD_ROOT and PACKAGE_ROOT must be separate directories." >&2
  exit 1
fi

require_file "${LLAMA_ROOT}/CMakeLists.txt"
require_file "${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake"
require_file "${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake"
require_executable "${HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-clang"
require_executable "${CMAKE_BIN}"

if [[ "${CMAKE_GENERATOR}" == "Ninja" ]] && ! command -v ninja >/dev/null 2>&1; then
  echo "The Ninja generator was selected, but ninja was not found in PATH." >&2
  exit 1
fi

CMAKE_VERSION="$(${CMAKE_BIN} --version | awk 'NR == 1 { print $3 }')"
if ! version_at_least "${CMAKE_VERSION}" "${MINIMUM_CMAKE_VERSION}"; then
  echo "CMake ${MINIMUM_CMAKE_VERSION} or newer is required; found ${CMAKE_VERSION}." >&2
  exit 1
fi

if [[ -z "${JOBS}" ]]; then
  JOBS="$(detect_jobs)"
fi
if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer: ${JOBS}" >&2
  exit 1
fi

CONFIGURE_ARGS=(
  -S "${LLAMA_ROOT}"
  -B "${BUILD_ROOT}"
  -G "${CMAKE_GENERATOR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake"
  -DANDROID_ABI="${ANDROID_ABI}"
  -DANDROID_PLATFORM="android-${ANDROID_API}"
  -DANDROID_STL="c++_shared"
  -DBUILD_SHARED_LIBS=ON
  -DGGML_BACKEND_DL=ON
  -DGGML_CPU_ALL_VARIANTS=ON
  -DGGML_NATIVE=OFF
  -DGGML_OPENMP=OFF
  -DGGML_LLAMAFILE=OFF
  -DGGML_HEXAGON=ON
  -DGGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE=128
  -DHEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT}"
  -DHEXAGON_TOOLS_ROOT="${HEXAGON_TOOLS_ROOT}"
  -DPREBUILT_LIB_DIR="android_aarch64"
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_TOOLS=OFF
  -DLLAMA_BUILD_EXAMPLES=ON
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_BUILD_APP=OFF
  -DLLAMA_BUILD_UI=OFF
  -DLLAMA_USE_PREBUILT_UI=OFF
  -DLLAMA_OPENSSL=OFF
  -DLLAMA_TOOLS_INSTALL=OFF
)

BUILD_TARGETS=(llama-zo)
for arch in "${HTP_ARCHES[@]}"; do
  BUILD_TARGETS+=("htp-${arch}")
done

echo "Android llama-zo build configuration:"
echo "  source:         ${LLAMA_ROOT}"
echo "  build:          ${BUILD_ROOT}"
echo "  package:        ${PACKAGE_ROOT}"
echo "  NDK:            ${ANDROID_NDK_ROOT}"
echo "  Hexagon SDK:    ${HEXAGON_SDK_ROOT}"
echo "  Hexagon Tools:  ${HEXAGON_TOOLS_ROOT}"
echo "  CMake:          ${CMAKE_BIN} (${CMAKE_VERSION})"
echo "  target:         ${ANDROID_ABI}, API ${ANDROID_API}"
echo "  HTP skels:      ${HTP_ARCHES[*]}"
echo
printf '  configure:'
printf ' %q' "${CMAKE_BIN}" "${CONFIGURE_ARGS[@]}"
printf '\n'
printf '  build:'
printf ' %q' "${CMAKE_BIN}" --build "${BUILD_ROOT}" --target "${BUILD_TARGETS[@]}" --config "${BUILD_TYPE}" -j "${JOBS}"
printf '\n'

if [[ ${DRY_RUN} -eq 1 ]]; then
  exit 0
fi

if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  remove_output_root "BUILD_ROOT" "${BUILD_ROOT}" "${BUILD_ROOT_IS_DEFAULT}"
fi

mark_output_root "${BUILD_ROOT}"
"${CMAKE_BIN}" "${CONFIGURE_ARGS[@]}"
"${CMAKE_BIN}" --build "${BUILD_ROOT}" \
  --target "${BUILD_TARGETS[@]}" \
  --config "${BUILD_TYPE}" \
  -j "${JOBS}"

BUILD_BIN="${BUILD_ROOT}/bin"
LLAMA_ZO_BIN="${BUILD_BIN}/llama-zo"
LIBCXX_SHARED="${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"

require_executable "${LLAMA_ZO_BIN}"
require_file "${LIBCXX_SHARED}"

PACKAGE_PARENT="$(dirname "${PACKAGE_ROOT}")"
PACKAGE_NAME="$(basename "${PACKAGE_ROOT}")"
mkdir -p "${PACKAGE_PARENT}"
PACKAGE_STAGING="$(mktemp -d "${PACKAGE_PARENT}/.${PACKAGE_NAME}.staging.XXXXXX")"
PACKAGE_BACKUP=""
cleanup_package_publish() {
  if [[ -n "${PACKAGE_STAGING}" && -d "${PACKAGE_STAGING}" ]]; then
    rm -rf -- "${PACKAGE_STAGING}"
  fi
  if [[ -n "${PACKAGE_BACKUP}" && -d "${PACKAGE_BACKUP}" && ! -e "${PACKAGE_ROOT}" ]]; then
    mv -- "${PACKAGE_BACKUP}" "${PACKAGE_ROOT}" || true
    if [[ -d "${PACKAGE_ROOT}" ]]; then
      mark_output_root "${PACKAGE_ROOT}" || true
    fi
  fi
}
trap cleanup_package_publish EXIT

mark_output_root "${PACKAGE_STAGING}"
mkdir -p "${PACKAGE_STAGING}/bin" "${PACKAGE_STAGING}/lib"
cp -L "${LLAMA_ZO_BIN}" "${PACKAGE_STAGING}/bin/llama-zo"

shopt -s nullglob
BUILT_LIBRARIES=("${BUILD_BIN}"/lib*.so*)
if ((${#BUILT_LIBRARIES[@]} == 0)); then
  echo "No Android shared libraries were produced in ${BUILD_BIN}." >&2
  exit 1
fi

for library in "${BUILT_LIBRARIES[@]}"; do
  name="$(basename "${library}")"
  case "${name}" in
    libggml-cpu*.so|libggml-hexagon.so)
      cp -L "${library}" "${PACKAGE_STAGING}/bin/${name}"
      ;;
    libggml-cpu*.so.*|libggml-hexagon.so.*)
      ;;
    libggml-htp-*.so*)
      ;;
    *)
      cp -L "${library}" "${PACKAGE_STAGING}/lib/${name}"
      ;;
  esac
done
shopt -u nullglob

for arch in "${HTP_ARCHES[@]}"; do
  skel="${BUILD_ROOT}/ggml/src/ggml-hexagon/libggml-htp-${arch}.so"
  require_file "${skel}"
  cp -L "${skel}" "${PACKAGE_STAGING}/lib/"
done

cp -L "${LIBCXX_SHARED}" "${PACKAGE_STAGING}/lib/libc++_shared.so"
cp "${SCRIPT_DIR}/run-llama-zo.sh" "${PACKAGE_STAGING}/run-llama-zo.sh"
chmod +x "${PACKAGE_STAGING}/bin/llama-zo" "${PACKAGE_STAGING}/run-llama-zo.sh"

require_file "${PACKAGE_STAGING}/bin/libggml-hexagon.so"
shopt -s nullglob
CPU_BACKENDS=("${PACKAGE_STAGING}"/bin/libggml-cpu*.so)
shopt -u nullglob
if ((${#CPU_BACKENDS[@]} == 0)); then
  echo "No loadable Android CPU backend was packaged." >&2
  exit 1
fi
for arch in "${HTP_ARCHES[@]}"; do
  require_file "${PACKAGE_STAGING}/lib/libggml-htp-${arch}.so"
done

validate_output_root "PACKAGE_ROOT" "${PACKAGE_ROOT}" "${PACKAGE_ROOT_IS_DEFAULT}"
if [[ -d "${PACKAGE_ROOT}" ]]; then
  PACKAGE_BACKUP="$(mktemp -d "${PACKAGE_PARENT}/.${PACKAGE_NAME}.backup.XXXXXX")"
  rmdir -- "${PACKAGE_BACKUP}"
  mv -- "${PACKAGE_ROOT}" "${PACKAGE_BACKUP}"
  mark_output_root "${PACKAGE_BACKUP}"
fi
if ! mv -- "${PACKAGE_STAGING}" "${PACKAGE_ROOT}"; then
  if [[ -n "${PACKAGE_BACKUP}" && -d "${PACKAGE_BACKUP}" && ! -e "${PACKAGE_ROOT}" ]]; then
    mv -- "${PACKAGE_BACKUP}" "${PACKAGE_ROOT}"
    mark_output_root "${PACKAGE_ROOT}"
    PACKAGE_BACKUP=""
    echo "Failed to activate the new package; the previous package was restored." >&2
  elif [[ -n "${PACKAGE_BACKUP}" ]]; then
    echo "Failed to activate the new package; the previous package remains at ${PACKAGE_BACKUP}." >&2
  else
    echo "Failed to activate the new package." >&2
  fi
  exit 1
fi
PACKAGE_STAGING=""
if [[ -n "${PACKAGE_BACKUP}" ]]; then
  remove_output_root "PACKAGE_BACKUP" "${PACKAGE_BACKUP}" 0
  PACKAGE_BACKUP=""
fi
trap - EXIT

echo
echo "Android llama-zo package complete: ${PACKAGE_ROOT}"
echo "Push it with: adb push ${PACKAGE_ROOT} /data/local/tmp/"
