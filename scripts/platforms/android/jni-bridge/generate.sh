#!/usr/bin/env bash

set -euo pipefail

APP_DIR=""
MODULE_NAME="app"
BRIDGE_PACKAGE=""
BRIDGE_CLASS="OmniInferNativeBridge"
LIB_NAME="omniinfer-native-jni"
ASSET_ROOT="omniinfer-native"
ABI="arm64-v8a"
MIN_SDK="26"
QNN_BUNDLE_DIR=""
NDK_DIR="${ANDROID_NDK_HOME:-}"
SKIP_BUILD=0

usage() {
  cat <<'EOF'
Usage: generate.sh --app-dir <dir> --package <bridge.package> [options]

Generate an OmniInfer Android JNI bridge, copy runtime assets into an Android app module,
and optionally build the JNI shared library.

Options:
  --app-dir <dir>          Android app project root
  --module <name>          Android module name, default: app
  --package <name>         Kotlin package for the generated bridge, for example com.example.app.omniinfer
  --class <name>           Kotlin bridge object name, default: OmniInferNativeBridge
  --lib-name <name>        Native library base name without lib/so, default: omniinfer-native-jni
  --asset-root <name>      Asset root under src/main/assets, default: omniinfer-native
  --abi <abi>              Android ABI, default: arm64-v8a
  --min-sdk <api>          Android min SDK for native build, default: 26
  --qnn-bundle-dir <dir>   Optional QNN runner/runtime directory to copy into app assets
  --ndk-dir <dir>          Explicit Android NDK directory
  --skip-build             Generate sources/assets only, do not build the JNI .so
  -h, --help               Show this help text
EOF
}

while (($# > 0)); do
  case "$1" in
    --app-dir)
      APP_DIR="${2:?missing value for --app-dir}"
      shift 2
      ;;
    --module)
      MODULE_NAME="${2:?missing value for --module}"
      shift 2
      ;;
    --package)
      BRIDGE_PACKAGE="${2:?missing value for --package}"
      shift 2
      ;;
    --class)
      BRIDGE_CLASS="${2:?missing value for --class}"
      shift 2
      ;;
    --lib-name)
      LIB_NAME="${2:?missing value for --lib-name}"
      shift 2
      ;;
    --asset-root)
      ASSET_ROOT="${2:?missing value for --asset-root}"
      shift 2
      ;;
    --abi)
      ABI="${2:?missing value for --abi}"
      shift 2
      ;;
    --min-sdk)
      MIN_SDK="${2:?missing value for --min-sdk}"
      shift 2
      ;;
    --qnn-bundle-dir)
      QNN_BUNDLE_DIR="${2:?missing value for --qnn-bundle-dir}"
      shift 2
      ;;
    --ndk-dir)
      NDK_DIR="${2:?missing value for --ndk-dir}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
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

if [[ -z "${APP_DIR}" || -z "${BRIDGE_PACKAGE}" ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEMPLATE_ROOT="${SCRIPT_DIR}/templates"
APP_DIR="$(cd "${APP_DIR}" && pwd)"
MODULE_DIR="${APP_DIR}/${MODULE_NAME}"

if [[ ! -d "${MODULE_DIR}" ]]; then
  echo "Android module directory was not found: ${MODULE_DIR}" >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command was not found: $1" >&2
    exit 1
  fi
}

require_cmd sed
require_cmd find
require_cmd cp
require_cmd chmod
require_cmd curl
require_cmd unzip

PACKAGE_PATH="${BRIDGE_PACKAGE//./\/}"
JAVA_OUT_DIR="${MODULE_DIR}/src/main/java/${PACKAGE_PATH}"
CPP_OUT_DIR="${MODULE_DIR}/src/main/cpp/omniinfer-native-jni"
ASSET_OUT_DIR="${MODULE_DIR}/src/main/assets/${ASSET_ROOT}"
JNICALL_OUT_DIR="${MODULE_DIR}/src/main/jniLibs/${ABI}"
RUNTIME_OUT_DIR="${ASSET_OUT_DIR}/runtime"
QNN_OUT_DIR="${RUNTIME_OUT_DIR}/qnn"
BUILD_ROOT="${APP_DIR}/tmp/omniinfer-native-jni-build"
STAMP_VALUE="$(date -u +%Y%m%d%H%M%S)"

mkdir -p \
  "${JAVA_OUT_DIR}" \
  "${CPP_OUT_DIR}" \
  "${RUNTIME_OUT_DIR}/bin" \
  "${RUNTIME_OUT_DIR}/support" \
  "${RUNTIME_OUT_DIR}/backends/llama_cpp" \
  "${RUNTIME_OUT_DIR}/backends/omniinfer_native" \
  "${QNN_OUT_DIR}" \
  "${JNICALL_OUT_DIR}"

render_template() {
  local template_path="$1"
  local target_path="$2"
  sed \
    -e "s|__BRIDGE_PACKAGE__|${BRIDGE_PACKAGE}|g" \
    -e "s|__BRIDGE_CLASS__|${BRIDGE_CLASS}|g" \
    -e "s|__LIB_NAME__|${LIB_NAME}|g" \
    -e "s|__ASSET_ROOT__|${ASSET_ROOT}|g" \
    -e "s|__BRIDGE_CLASS_SLASH__|${BRIDGE_PACKAGE//./\/}/${BRIDGE_CLASS}|g" \
    "${template_path}" > "${target_path}"
}

copy_runtime_layout() {
  cp "${REPO_ROOT}/scripts/platforms/android/runtime/omniinfer-android" \
    "${RUNTIME_OUT_DIR}/bin/omniinfer-android"
  cp "${REPO_ROOT}/scripts/platforms/android/runtime/support/common.sh" \
    "${RUNTIME_OUT_DIR}/support/common.sh"
  cp "${REPO_ROOT}/scripts/platforms/android/runtime/backends/llama_cpp/backend.sh" \
    "${RUNTIME_OUT_DIR}/backends/llama_cpp/backend.sh"
  cp "${REPO_ROOT}/scripts/platforms/android/runtime/backends/omniinfer_native/backend.sh" \
    "${RUNTIME_OUT_DIR}/backends/omniinfer_native/backend.sh"
  chmod +x \
    "${RUNTIME_OUT_DIR}/bin/omniinfer-android" \
    "${RUNTIME_OUT_DIR}/support/common.sh" \
    "${RUNTIME_OUT_DIR}/backends/llama_cpp/backend.sh" \
    "${RUNTIME_OUT_DIR}/backends/omniinfer_native/backend.sh"
}

copy_qnn_bundle() {
  if [[ -z "${QNN_BUNDLE_DIR}" ]]; then
    echo "Skipping QNN asset copy: no --qnn-bundle-dir was provided."
    return
  fi
  if [[ ! -d "${QNN_BUNDLE_DIR}" ]]; then
    echo "QNN bundle directory was not found: ${QNN_BUNDLE_DIR}" >&2
    exit 1
  fi

  local patterns=(
    "qnn_llama_runner"
    "qnn_multimodal_runner"
    "libQnn*.so"
    "libqnn_executorch_backend.so"
  )

  local copied=0
  for pattern in "${patterns[@]}"; do
    while IFS= read -r path; do
      copied=1
      cp "${path}" "${QNN_OUT_DIR}/$(basename "${path}")"
      case "$(basename "${path}")" in
        qnn_*_runner)
          chmod +x "${QNN_OUT_DIR}/$(basename "${path}")"
          ;;
      esac
    done < <(find "${QNN_BUNDLE_DIR}" -maxdepth 1 -type f -name "${pattern}" | sort)
  done

  if [[ ${copied} -eq 0 ]]; then
    echo "No supported QNN payload files were found in ${QNN_BUNDLE_DIR}" >&2
    exit 1
  fi
}

resolve_ndk_dir() {
  local candidate
  for candidate in \
    "${NDK_DIR}" \
    "${ANDROID_NDK_ROOT:-}" \
    "${HOME:-}/Android/Sdk/ndk/26.2.11394342" \
    "${HOME:-}/Android/Sdk/ndk/26.1.10909125" \
    "/usr/lib/android-sdk/ndk/26.2.11394342" \
    "/usr/lib/android-sdk/ndk/26.1.10909125"; do
    if [[ -n "${candidate}" && -x "${candidate}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${MIN_SDK}-clang++" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  local cached_ndk="${REPO_ROOT}/.local/toolchains/android-ndk-r26c"
  local ndk_zip="${REPO_ROOT}/.local/toolchains/android-ndk-r26c-linux.zip"
  mkdir -p "$(dirname "${cached_ndk}")"
  if [[ ! -x "${cached_ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${MIN_SDK}-clang++" ]]; then
    echo "Downloading Android NDK r26c to ${cached_ndk}" >&2
    rm -rf "${cached_ndk}"
    curl -L "https://dl.google.com/android/repository/android-ndk-r26c-linux.zip" -o "${ndk_zip}"
    unzip -q -o "${ndk_zip}" -d "$(dirname "${cached_ndk}")"
  fi
  if [[ -x "${cached_ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${MIN_SDK}-clang++" ]]; then
    printf '%s\n' "${cached_ndk}"
    return 0
  fi

  echo "Unable to locate a usable Android NDK." >&2
  return 1
}

build_native_library() {
  local ndk_root
  ndk_root="$(resolve_ndk_dir)"
  local toolchain_root="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/bin"
  local cxx="${toolchain_root}/aarch64-linux-android${MIN_SDK}-clang++"
  local output_path="${JNICALL_OUT_DIR}/lib${LIB_NAME}.so"

  mkdir -p "${BUILD_ROOT}"

  echo "Building ${output_path}"
  "${cxx}" \
    -std=c++20 \
    -O2 \
    -fPIC \
    -shared \
    -static-libstdc++ \
    -DANDROID \
    -o "${output_path}" \
    "${CPP_OUT_DIR}/omniinfer_native_jni.cpp" \
    -llog
}

render_template "${TEMPLATE_ROOT}/OmniInferNativeBridge.kt.template" \
  "${JAVA_OUT_DIR}/${BRIDGE_CLASS}.kt"
render_template "${TEMPLATE_ROOT}/omniinfer_native_jni.cpp.template" \
  "${CPP_OUT_DIR}/omniinfer_native_jni.cpp"

copy_runtime_layout
copy_qnn_bundle
printf '%s\n' "${STAMP_VALUE}" > "${ASSET_OUT_DIR}/version.txt"

if [[ ${SKIP_BUILD} -eq 0 ]]; then
  build_native_library
else
  echo "Skipping JNI build because --skip-build was provided."
fi

cat <<EOF

Android JNI bridge generation complete.

Generated Kotlin bridge:
  ${JAVA_OUT_DIR}/${BRIDGE_CLASS}.kt

Generated native source:
  ${CPP_OUT_DIR}/omniinfer_native_jni.cpp

Generated runtime assets:
  ${ASSET_OUT_DIR}/version.txt
  ${RUNTIME_OUT_DIR}/bin/omniinfer-android
  ${RUNTIME_OUT_DIR}/support/common.sh
  ${RUNTIME_OUT_DIR}/backends/llama_cpp/backend.sh
  ${RUNTIME_OUT_DIR}/backends/omniinfer_native/backend.sh
  ${QNN_OUT_DIR}

Generated JNI library:
  ${JNICALL_OUT_DIR}/lib${LIB_NAME}.so
EOF
