#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="Release"
DRY_RUN=0
JOBS=""
CLEAN_BUILD=0
BOOTSTRAP_SUBMODULE=1
DEPLOYMENT_TARGET="16.0"

usage() {
  cat <<'EOF'
Usage: build.sh [options]

Build llama.cpp as an xcframework for iOS (device + simulator).

Options:
  --build-type <type>  CMake build type, default: Release
  --jobs <n>           Parallel build jobs, default: sysctl hw.ncpu
  --clean              Remove previous build directories before configuring
  --no-bootstrap       Do not auto-initialize the llama.cpp git submodule
  --deployment-target  iOS deployment target, default: 16.0
  --dry-run            Print actions without executing them
  -h, --help           Show this help message
EOF
}

while (($# > 0)); do
  case "$1" in
    --build-type)       BUILD_TYPE="${2:?missing value}"; shift 2 ;;
    --jobs)             JOBS="${2:?missing value}"; shift 2 ;;
    --clean)            CLEAN_BUILD=1; shift ;;
    --no-bootstrap)     BOOTSTRAP_SUBMODULE=0; shift ;;
    --deployment-target) DEPLOYMENT_TARGET="${2:?missing value}"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ROOT}/../../../.." && pwd)"
LLAMA_ROOT="${REPO_ROOT}/framework/llama.cpp"
NATIVE_ROOT="${REPO_ROOT}/ios/native"
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/ios/llama.cpp-ios"
BUILD_ROOT_DEVICE="${PACKAGE_ROOT}/build/device"
BUILD_ROOT_SIM="${PACKAGE_ROOT}/build/simulator"
OUTPUT_DIR="${REPO_ROOT}/ios/OmniInferServer/Frameworks"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' was not found in PATH." >&2
    exit 1
  fi
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
  echo "Bootstrapping llama.cpp submodule..."
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  git -C ${REPO_ROOT} submodule update --init --recursive framework/llama.cpp"
    return
  fi
  git -C "${REPO_ROOT}" submodule update --init --recursive framework/llama.cpp
}

ensure_llama_root
require_command cmake
require_command xcodebuild
require_command libtool

if [[ -z "${JOBS}" ]]; then
  JOBS="$(detect_jobs)"
fi

# Common CMake arguments for both device and simulator.
COMMON_CMAKE_ARGS=(
  -S "${LLAMA_ROOT}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${DEPLOYMENT_TARGET}"
  -DCMAKE_OSX_ARCHITECTURES=arm64
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_BUILD_COMMON=ON
  -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON
)

echo "=== OmniInfer iOS llama.cpp xcframework builder ==="
echo "  Build type:        ${BUILD_TYPE}"
echo "  Deployment target: iOS ${DEPLOYMENT_TARGET}"
echo "  llama.cpp:         ${LLAMA_ROOT}"
echo "  Output:            ${OUTPUT_DIR}/llama.xcframework"
echo ""

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "[dry-run] Would build for device and simulator, then create xcframework."
  exit 0
fi

if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  echo "Cleaning previous builds..."
  rm -rf "${BUILD_ROOT_DEVICE}" "${BUILD_ROOT_SIM}"
fi

mkdir -p "${BUILD_ROOT_DEVICE}" "${BUILD_ROOT_SIM}" "${OUTPUT_DIR}"

# --- Build for iOS device (arm64) ---
echo "Configuring llama.cpp for iOS device (arm64)..."
cmake "${COMMON_CMAKE_ARGS[@]}" \
  -B "${BUILD_ROOT_DEVICE}" \
  -DCMAKE_SYSTEM_NAME=iOS

echo "Building for iOS device..."
cmake --build "${BUILD_ROOT_DEVICE}" --config "${BUILD_TYPE}" -j "${JOBS}"

# Compile the OmniInfer bridge for device.
echo "Compiling OmniInfer bridge (device)..."
BRIDGE_CXX_FLAGS="-std=c++17 -DOMNIINFER_BACKEND_LLAMA_CPP=1 -I${NATIVE_ROOT} -I${LLAMA_ROOT}/include -I${LLAMA_ROOT}/common -I${LLAMA_ROOT}/ggml/include -I${LLAMA_ROOT}/vendor"
xcrun --sdk iphoneos clang++ -c ${BRIDGE_CXX_FLAGS} -target arm64-apple-ios${DEPLOYMENT_TARGET} \
  -O2 -o "${BUILD_ROOT_DEVICE}/omniinfer_bridge.o" "${NATIVE_ROOT}/omniinfer_bridge.cpp"
ar rcs "${BUILD_ROOT_DEVICE}/libomniinfer-bridge.a" "${BUILD_ROOT_DEVICE}/omniinfer_bridge.o"

# --- Build for iOS simulator (arm64) ---
echo "Configuring llama.cpp for iOS simulator (arm64)..."
cmake "${COMMON_CMAKE_ARGS[@]}" \
  -B "${BUILD_ROOT_SIM}" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphonesimulator

echo "Building for iOS simulator..."
cmake --build "${BUILD_ROOT_SIM}" --config "${BUILD_TYPE}" -j "${JOBS}"

# Compile the OmniInfer bridge for simulator.
echo "Compiling OmniInfer bridge (simulator)..."
xcrun --sdk iphonesimulator clang++ -c ${BRIDGE_CXX_FLAGS} -target arm64-apple-ios${DEPLOYMENT_TARGET}-simulator \
  -O2 -o "${BUILD_ROOT_SIM}/omniinfer_bridge.o" "${NATIVE_ROOT}/omniinfer_bridge.cpp"
ar rcs "${BUILD_ROOT_SIM}/libomniinfer-bridge.a" "${BUILD_ROOT_SIM}/omniinfer_bridge.o"

# --- Combine static libraries per platform ---
combine_libs() {
  local build_dir="$1"
  local output="$2"

  # Collect all .a files produced by the build (llama, ggml, common, etc.).
  local libs=()
  while IFS= read -r -d '' lib; do
    libs+=("$lib")
  done < <(find "${build_dir}" -name '*.a' -print0)

  if [[ ${#libs[@]} -eq 0 ]]; then
    echo "No static libraries found in ${build_dir}" >&2
    exit 1
  fi

  echo "  Combining ${#libs[@]} libraries into $(basename "${output}")..."
  libtool -static -o "${output}" "${libs[@]}" 2>/dev/null
}

COMBINED_DEVICE="${BUILD_ROOT_DEVICE}/libomniinfer-llama.a"
COMBINED_SIM="${BUILD_ROOT_SIM}/libomniinfer-llama.a"

echo "Combining static libraries..."
combine_libs "${BUILD_ROOT_DEVICE}" "${COMBINED_DEVICE}"
combine_libs "${BUILD_ROOT_SIM}" "${COMBINED_SIM}"

# --- Collect headers ---
HEADER_DIR="${BUILD_ROOT_DEVICE}/combined-headers"
rm -rf "${HEADER_DIR}"
mkdir -p "${HEADER_DIR}"

# Only the OmniInfer bridge public header — this is what Swift imports.
cp "${NATIVE_ROOT}/omniinfer_bridge.h" "${HEADER_DIR}/"

# Create module map so Swift can import the xcframework.
cat > "${HEADER_DIR}/module.modulemap" <<'MODULEMAP'
module llama {
    header "omniinfer_bridge.h"
    export *
}
MODULEMAP

# --- Create xcframework ---
echo "Creating xcframework..."
rm -rf "${OUTPUT_DIR}/llama.xcframework"

xcodebuild -create-xcframework \
  -library "${COMBINED_DEVICE}" -headers "${HEADER_DIR}" \
  -library "${COMBINED_SIM}" -headers "${HEADER_DIR}" \
  -output "${OUTPUT_DIR}/llama.xcframework"

echo ""
echo "=== iOS xcframework build complete ==="
echo "  Output: ${OUTPUT_DIR}/llama.xcframework"
echo ""
echo "Next step: build the OmniInferServer Swift Package"
