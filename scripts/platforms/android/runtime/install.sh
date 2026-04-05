#!/usr/bin/env bash

set -euo pipefail

DRY_RUN=0
ARTIFACT_DIR=""
LLAMA_CLI_PATH="${OMNIINFER_ANDROID_LLAMA_CLI:-}"
MTMD_CLI_PATH="${OMNIINFER_ANDROID_MTMD_CLI:-}"
QNN_BUNDLE_DIR="${OMNIINFER_ANDROID_QNN_BUNDLE_DIR:-}"
LAUNCHER_ONLY=0

usage() {
  cat <<'EOF'
Usage: build-runtime.sh [options]

Options:
  --artifact-dir <dir>  Directory that contains Android CLI binaries such as libllama-cli.so and libmtmd-cli.so
  --llama-cli <path>    Explicit path to libllama-cli.so
  --mtmd-cli <path>     Explicit path to libmtmd-cli.so
  --qnn-bundle-dir <dir>  Optional directory that contains qnn_llama_runner, qnn_multimodal_runner, and QNN runtime libraries
  --launcher-only       Only install the OmniInfer Android launcher and runtime layout
  --dry-run             Print actions without modifying files
  -h, --help            Show this help message

Environment overrides:
  OMNIINFER_ANDROID_LLAMA_CLI
  OMNIINFER_ANDROID_MTMD_CLI
  OMNIINFER_ANDROID_QNN_BUNDLE_DIR
EOF
}

while (($# > 0)); do
  case "$1" in
    --artifact-dir)
      ARTIFACT_DIR="${2:?missing value for --artifact-dir}"
      shift 2
      ;;
    --llama-cli)
      LLAMA_CLI_PATH="${2:?missing value for --llama-cli}"
      shift 2
      ;;
    --mtmd-cli)
      MTMD_CLI_PATH="${2:?missing value for --mtmd-cli}"
      shift 2
      ;;
    --qnn-bundle-dir)
      QNN_BUNDLE_DIR="${2:?missing value for --qnn-bundle-dir}"
      shift 2
      ;;
    --launcher-only)
      LAUNCHER_ONLY=1
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
RUNTIME_ROOT="${REPO_ROOT}/.local/runtime/android"
BIN_ROOT="${RUNTIME_ROOT}/bin"
LIB_ROOT="${RUNTIME_ROOT}/lib/arm64-v8a"
QNN_ROOT="${RUNTIME_ROOT}/qnn"
STATE_ROOT="${RUNTIME_ROOT}/state"
SUPPORT_ROOT="${RUNTIME_ROOT}/support"
BACKENDS_ROOT="${RUNTIME_ROOT}/backends"
LAUNCHER_TEMPLATE="${SCRIPT_ROOT}/omniinfer-android"
LAUNCHER_OUTPUT="${BIN_ROOT}/omniinfer-android"

resolve_candidate() {
  local current="$1"
  local filename="$2"
  if [[ -n "${current}" ]]; then
    printf '%s\n' "${current}"
    return
  fi
  if [[ -n "${ARTIFACT_DIR}" && -f "${ARTIFACT_DIR}/${filename}" ]]; then
    printf '%s\n' "${ARTIFACT_DIR}/${filename}"
    return
  fi
  printf '%s\n' ""
}

run_cmd() {
  echo "+ $*"
  if [[ ${DRY_RUN} -eq 0 ]]; then
    "$@"
  fi
}

prepare_layout() {
  run_cmd mkdir -p \
    "${BIN_ROOT}" \
    "${LIB_ROOT}" \
    "${QNN_ROOT}" \
    "${STATE_ROOT}" \
    "${SUPPORT_ROOT}" \
    "${BACKENDS_ROOT}/llama_cpp" \
    "${BACKENDS_ROOT}/omniinfer_native"
}

install_launcher() {
  run_cmd cp "${LAUNCHER_TEMPLATE}" "${LAUNCHER_OUTPUT}"
  run_cmd chmod +x "${LAUNCHER_OUTPUT}"
}

install_runtime_support() {
  run_cmd cp "${SCRIPT_ROOT}/support/common.sh" "${SUPPORT_ROOT}/common.sh"
  run_cmd chmod +x "${SUPPORT_ROOT}/common.sh"
}

install_runtime_backends() {
  run_cmd cp "${SCRIPT_ROOT}/backends/llama_cpp/backend.sh" "${BACKENDS_ROOT}/llama_cpp/backend.sh"
  run_cmd chmod +x "${BACKENDS_ROOT}/llama_cpp/backend.sh"
  run_cmd cp "${SCRIPT_ROOT}/backends/omniinfer_native/backend.sh" "${BACKENDS_ROOT}/omniinfer_native/backend.sh"
  run_cmd chmod +x "${BACKENDS_ROOT}/omniinfer_native/backend.sh"
}

copy_optional_binary() {
  local source_path="$1"
  local target_name="$2"
  if [[ -z "${source_path}" ]]; then
    echo "Skipping ${target_name}: no source was provided."
    return
  fi
  if [[ ! -f "${source_path}" ]]; then
    echo "Warning: ${target_name} source was not found: ${source_path}" >&2
    return
  fi
  run_cmd cp "${source_path}" "${LIB_ROOT}/${target_name}"
  run_cmd chmod +x "${LIB_ROOT}/${target_name}"
}

copy_qnn_bundle() {
  local source_dir="$1"
  if [[ -z "${source_dir}" ]]; then
    echo "Skipping QNN bundle: no source directory was provided."
    return
  fi
  if [[ ! -d "${source_dir}" ]]; then
    echo "Warning: QNN bundle directory was not found: ${source_dir}" >&2
    return
  fi

  local patterns=(
    "qnn_llama_runner"
    "qnn_multimodal_runner"
    "libQnn*.so"
    "libqnn_executorch_backend.so"
  )

  for pattern in "${patterns[@]}"; do
    local found=0
    while IFS= read -r path; do
      found=1
      run_cmd cp "${path}" "${QNN_ROOT}/$(basename "${path}")"
      run_cmd chmod +x "${QNN_ROOT}/$(basename "${path}")"
    done < <(find "${source_dir}" -maxdepth 1 -type f -name "${pattern}" | sort)
    if [[ ${found} -eq 0 ]]; then
      echo "Skipping ${pattern}: no matching file was found in ${source_dir}."
    fi
  done
}

LLAMA_CLI_PATH="$(resolve_candidate "${LLAMA_CLI_PATH}" "libllama-cli.so")"
MTMD_CLI_PATH="$(resolve_candidate "${MTMD_CLI_PATH}" "libmtmd-cli.so")"

echo "Preparing Android runtime under ${RUNTIME_ROOT}"
prepare_layout
install_launcher
install_runtime_support
install_runtime_backends

if [[ ${LAUNCHER_ONLY} -eq 0 ]]; then
  copy_optional_binary "${LLAMA_CLI_PATH}" "libllama-cli.so"
  copy_optional_binary "${MTMD_CLI_PATH}" "libmtmd-cli.so"
  copy_qnn_bundle "${QNN_BUNDLE_DIR}"
fi

cat <<EOF

Android runtime preparation complete.

Installed launcher:
  ${LAUNCHER_OUTPUT}

Installed Android runtime modules:
  ${SUPPORT_ROOT}/common.sh
  ${BACKENDS_ROOT}/llama_cpp/backend.sh
  ${BACKENDS_ROOT}/omniinfer_native/backend.sh

Optional Android backend binaries:
  ${LIB_ROOT}/libllama-cli.so
  ${LIB_ROOT}/libmtmd-cli.so

Optional OmniInfer Native QNN runtime:
  ${QNN_ROOT}/qnn_llama_runner
  ${QNN_ROOT}/qnn_multimodal_runner
  ${QNN_ROOT}/libQnn*.so
  ${QNN_ROOT}/libqnn_executorch_backend.so

Typical next steps:
  bash ./scripts/platforms/android/build-runtime.sh --artifact-dir /path/to/android/artifacts --qnn-bundle-dir /path/to/qnn-bundle
  ./omniinfer backend list
  ./omniinfer select llama.cpp-llama
EOF
