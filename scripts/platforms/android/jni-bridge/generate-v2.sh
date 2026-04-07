#!/usr/bin/env bash

set -euo pipefail

APP_DIR=""
MODULE_NAME="app"
BRIDGE_PACKAGE=""
BRIDGE_CLASS="OmniInferBridge"
LIB_NAME="omniinfer-jni"
LLAMA_CPP_DIR=""
MNN_DIR=""

usage() {
  cat <<'EOF'
Usage: generate-v2.sh --app-dir <dir> --package <bridge.package> [options]

Generate a multi-backend OmniInfer JNI bridge (llama.cpp + MNN in one .so).
Built by the Android Gradle/CMake pipeline — no precompiled binaries needed.

Required:
  --app-dir <dir>          Android app project root
  --package <name>         Kotlin package for the generated bridge

Optional:
  --module <name>          Android module name (default: app)
  --class <name>           Kotlin bridge object name (default: OmniInferBridge)
  --lib-name <name>        Native library base name (default: omniinfer-jni)
  --llama-cpp-dir <dir>    Path to llama.cpp source tree (default: auto-detect)
  --mnn-dir <dir>          Path to MNN source tree (enables MNN backend)
  -h, --help               Show this help text
EOF
}

while (($# > 0)); do
  case "$1" in
    --app-dir)     APP_DIR="${2:?missing value}"; shift 2 ;;
    --module)      MODULE_NAME="${2:?missing value}"; shift 2 ;;
    --package)     BRIDGE_PACKAGE="${2:?missing value}"; shift 2 ;;
    --class)       BRIDGE_CLASS="${2:?missing value}"; shift 2 ;;
    --lib-name)    LIB_NAME="${2:?missing value}"; shift 2 ;;
    --llama-cpp-dir) LLAMA_CPP_DIR="${2:?missing value}"; shift 2 ;;
    --mnn-dir)     MNN_DIR="${2:?missing value}"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *)             echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${APP_DIR}" || -z "${BRIDGE_PACKAGE}" ]]; then
  usage >&2; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEMPLATE_ROOT="${SCRIPT_DIR}/templates"
APP_DIR="$(cd "${APP_DIR}" && pwd)"
MODULE_DIR="${APP_DIR}/${MODULE_NAME}"

[[ -d "${MODULE_DIR}" ]] || { echo "Module not found: ${MODULE_DIR}" >&2; exit 1; }

# Auto-detect llama.cpp.
if [[ -z "${LLAMA_CPP_DIR}" ]]; then
  if [[ -f "${REPO_ROOT}/framework/llama.cpp/CMakeLists.txt" ]]; then
    LLAMA_CPP_DIR="${REPO_ROOT}/framework/llama.cpp"
  else
    echo "llama.cpp not found. Use --llama-cpp-dir or: git submodule update --init framework/llama.cpp" >&2
    exit 1
  fi
fi
LLAMA_CPP_DIR="$(cd "${LLAMA_CPP_DIR}" && pwd)"

# Auto-detect MNN if not specified.
if [[ -z "${MNN_DIR}" && -f "${REPO_ROOT}/framework/mnn/CMakeLists.txt" ]]; then
  MNN_DIR="${REPO_ROOT}/framework/mnn"
fi
if [[ -n "${MNN_DIR}" ]]; then
  MNN_DIR="$(cd "${MNN_DIR}" && pwd)"
fi

# Convert MSYS/Cygwin paths to Windows paths for CMake.
to_native_path() {
  local p="$1"
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      if command -v cygpath >/dev/null 2>&1; then cygpath -m "$p"
      else echo "$p" | sed 's|^/\([a-zA-Z]\)/|\1:/|'; fi ;;
    *) echo "$p" ;;
  esac
}
LLAMA_CPP_DIR="$(to_native_path "${LLAMA_CPP_DIR}")"
[[ -n "${MNN_DIR}" ]] && MNN_DIR="$(to_native_path "${MNN_DIR}")"

PACKAGE_PATH="${BRIDGE_PACKAGE//./\/}"
JAVA_OUT_DIR="${MODULE_DIR}/src/main/java/${PACKAGE_PATH}"
CPP_OUT_DIR="${MODULE_DIR}/src/main/cpp/omniinfer-jni"

mkdir -p "${JAVA_OUT_DIR}" "${CPP_OUT_DIR}"

render_template() {
  local src="$1" dst="$2"
  sed \
    -e "s|__BRIDGE_PACKAGE__|${BRIDGE_PACKAGE}|g" \
    -e "s|__BRIDGE_CLASS__|${BRIDGE_CLASS}|g" \
    -e "s|__LIB_NAME__|${LIB_NAME}|g" \
    -e "s|__BRIDGE_CLASS_SLASH__|${BRIDGE_PACKAGE//./\/}/${BRIDGE_CLASS}|g" \
    -e "s|__LLAMA_CPP_DIR__|${LLAMA_CPP_DIR}|g" \
    -e "s|__MNN_DIR__|${MNN_DIR}|g" \
    "${src}" > "${dst}"
}

# Kotlin bridge.
render_template "${TEMPLATE_ROOT}/OmniInferBridge.kt.template" \
  "${JAVA_OUT_DIR}/${BRIDGE_CLASS}.kt"

# C++ sources — copy all templates (no package-specific substitution needed for headers).
render_template "${TEMPLATE_ROOT}/omniinfer_jni.cpp.template" \
  "${CPP_OUT_DIR}/omniinfer_jni.cpp"

cp "${TEMPLATE_ROOT}/inference_backend.h.template" "${CPP_OUT_DIR}/inference_backend.h"
cp "${TEMPLATE_ROOT}/backend_llama_cpp.h.template" "${CPP_OUT_DIR}/backend_llama_cpp.h"
cp "${TEMPLATE_ROOT}/backend_mnn.h.template" "${CPP_OUT_DIR}/backend_mnn.h"

# CMakeLists.txt.
render_template "${TEMPLATE_ROOT}/CMakeLists.txt.template" \
  "${CPP_OUT_DIR}/CMakeLists.txt"

# Summary.
echo ""
echo "OmniInfer JNI bridge generated successfully."
echo ""
echo "Generated files:"
echo "  Kotlin:     ${JAVA_OUT_DIR}/${BRIDGE_CLASS}.kt"
echo "  JNI C++:    ${CPP_OUT_DIR}/omniinfer_jni.cpp"
echo "  Backends:   ${CPP_OUT_DIR}/backend_llama_cpp.h"
[[ -n "${MNN_DIR}" ]] && echo "              ${CPP_OUT_DIR}/backend_mnn.h"
echo "  CMake:      ${CPP_OUT_DIR}/CMakeLists.txt"
echo ""
echo "Backends:"
echo "  llama.cpp:  ${LLAMA_CPP_DIR}"
if [[ -n "${MNN_DIR}" ]]; then
  echo "  MNN:        ${MNN_DIR}"
else
  echo "  MNN:        (disabled — use --mnn-dir to enable)"
fi
echo ""
echo "Add to ${MODULE_NAME}/build.gradle.kts:"
echo ""
echo '  android {'
echo '      defaultConfig {'
echo '          ndk { abiFilters += "arm64-v8a" }'
echo '          externalNativeBuild {'
echo '              cmake {'
echo '                  arguments += "-DGGML_NATIVE=OFF"'
echo '                  arguments += "-DGGML_LLAMAFILE=OFF"'
echo '                  arguments += "-DLLAMA_BUILD_COMMON=ON"'
if [[ -n "${MNN_DIR}" ]]; then
  echo '                  arguments += "-DOMNIINFER_BACKEND_MNN=ON"'
fi
echo '              }'
echo '          }'
echo '      }'
echo '      externalNativeBuild {'
echo '          cmake {'
echo '              path = file("src/main/cpp/omniinfer-jni/CMakeLists.txt")'
echo '          }'
echo '      }'
echo '  }'
