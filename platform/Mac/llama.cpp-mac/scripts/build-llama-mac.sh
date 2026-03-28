#!/usr/bin/env bash

set -euo pipefail

BUILD_TYPE="${BUILD_TYPE:-Release}"
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/../../.." && pwd)"
LLAMA_ROOT="${REPO_ROOT}/framework/llama.cpp"
BUILD_ROOT="${PACKAGE_ROOT}/build/llama.cpp-mac"
BIN_ROOT="${PACKAGE_ROOT}/bin"

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

if [[ ! -d "${LLAMA_ROOT}" ]]; then
  echo "llama.cpp source tree was not found at ${LLAMA_ROOT}" >&2
  exit 1
fi

ensure_cmake_on_path
require_command cmake

mkdir -p "${BUILD_ROOT}" "${BIN_ROOT}"

echo "Configuring llama.cpp Metal build..."
cmake \
  -S "${LLAMA_ROOT}" \
  -B "${BUILD_ROOT}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_OPENSSL=OFF \
  -DGGML_METAL=ON

echo "Building llama-server..."
cmake --build "${BUILD_ROOT}" --target llama-server --config "${BUILD_TYPE}" -j "$(sysctl -n hw.ncpu)"

cp "${BUILD_ROOT}/bin/llama-server" "${BIN_ROOT}/llama-server"
chmod +x "${BIN_ROOT}/llama-server"

echo
echo "macOS Metal build complete."
echo "Binary package location: ${BIN_ROOT}/llama-server"
