#!/usr/bin/env bash

set -euo pipefail

DRY_RUN=0
ARTIFACT_DIR=""
LLAMA_CLI_PATH="${OMNIINFER_ANDROID_LLAMA_CLI:-}"
MTMD_CLI_PATH="${OMNIINFER_ANDROID_MTMD_CLI:-}"
LAUNCHER_ONLY=0

usage() {
  cat <<'EOF'
Usage: build-runtime.sh [options]

Options:
  --artifact-dir <dir>  Directory that contains Android CLI binaries such as libllama-cli.so and libmtmd-cli.so
  --llama-cli <path>    Explicit path to libllama-cli.so
  --mtmd-cli <path>     Explicit path to libmtmd-cli.so
  --launcher-only       Only install the OmniInfer Android launcher and runtime layout
  --dry-run             Print actions without modifying files
  -h, --help            Show this help message

Environment overrides:
  OMNIINFER_ANDROID_LLAMA_CLI
  OMNIINFER_ANDROID_MTMD_CLI
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
STATE_ROOT="${RUNTIME_ROOT}/state"
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
  run_cmd mkdir -p "${BIN_ROOT}" "${LIB_ROOT}" "${STATE_ROOT}"
}

install_launcher() {
  run_cmd cp "${LAUNCHER_TEMPLATE}" "${LAUNCHER_OUTPUT}"
  run_cmd chmod +x "${LAUNCHER_OUTPUT}"
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

LLAMA_CLI_PATH="$(resolve_candidate "${LLAMA_CLI_PATH}" "libllama-cli.so")"
MTMD_CLI_PATH="$(resolve_candidate "${MTMD_CLI_PATH}" "libmtmd-cli.so")"

echo "Preparing Android runtime under ${RUNTIME_ROOT}"
prepare_layout
install_launcher

if [[ ${LAUNCHER_ONLY} -eq 0 ]]; then
  copy_optional_binary "${LLAMA_CLI_PATH}" "libllama-cli.so"
  copy_optional_binary "${MTMD_CLI_PATH}" "libmtmd-cli.so"
fi

cat <<EOF

Android runtime preparation complete.

Installed launcher:
  ${LAUNCHER_OUTPUT}

Optional Android backend binaries:
  ${LIB_ROOT}/libllama-cli.so
  ${LIB_ROOT}/libmtmd-cli.so

Typical next steps:
  bash ./scripts/platforms/android/build-runtime.sh --artifact-dir /path/to/android/artifacts
  ./omniinfer backend list
  ./omniinfer select llama.cpp-llama
EOF
