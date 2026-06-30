#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOCAL_RUNTIME_ROOT="${REPO_ROOT}/.local/runtime/linux"
CPU_SCRIPT="${SCRIPT_DIR}/build-llama-cpu.sh"
ROCM_SCRIPT="${SCRIPT_DIR}/build-llama-rocm.sh"
VULKAN_SCRIPT="${SCRIPT_DIR}/build-llama-vulkan.sh"
S390X_SCRIPT="${SCRIPT_DIR}/build-llama-s390x.sh"
OPENVINO_SCRIPT="${SCRIPT_DIR}/build-llama-openvino.sh"
RUNTIME_BACKENDS_HELPER="${SCRIPT_DIR}/release_runtime_backends.py"
RUST_CLI_PACKAGER="${REPO_ROOT}/scripts/platforms/common/package-rust-cli.py"

PACKAGE_NAME="OmniInfer"
PLATFORM_TAG="linux-x64"
BUILD_CPU_BACKEND=0
BUILD_ROCM_BACKEND=0
BUILD_VULKAN_BACKEND=0
BUILD_S390X_BACKEND=0
BUILD_OPENVINO_BACKEND=0
BUILD_TYPE="Release"
GPU_TARGETS=""
ROCM_PATH_OVERRIDE=""
OPENVINO_ROOT_OVERRIDE=""
DRY_RUN=0
INCLUDE_PYTHON_FALLBACK=0

usage() {
  cat <<'EOF'
Usage: build-release.sh [options]

Options:
  --package-name <name>     Release directory name (default: OmniInfer)
  --platform-tag <tag>      Platform tag for output dir (default: linux-x64)
  --build-cpu-backend       Build the Linux CPU backend before packaging
  --build-rocm-backend      Build the Linux ROCm backend before packaging
  --build-vulkan-backend    Build the Linux Vulkan backend before packaging
  --build-s390x-backend     Build the Linux s390x backend before packaging
  --build-openvino-backend  Build the Linux OpenVINO backend before packaging
  --build-type <type>       CMake build type, default: Release
  --gpu-targets <targets>   Override HIP GPU targets for the ROCm build
  --rocm-path <path>        Override ROCm installation root for the ROCm build
  --openvino-root <path>    Override OpenVINO installation root for the OpenVINO build
  --dry-run                 Print actions without executing packaging steps
  --include-python-fallback Copy omniinfer.py/service_core for legacy or embedded Python backend paths
  -h, --help                Show this help message
EOF
}

while (($# > 0)); do
  case "$1" in
    --package-name)
      PACKAGE_NAME="${2:?missing value for --package-name}"
      shift 2
      ;;
    --platform-tag)
      PLATFORM_TAG="${2:?missing value for --platform-tag}"
      shift 2
      ;;
    --build-cpu-backend)
      BUILD_CPU_BACKEND=1
      shift
      ;;
    --build-rocm-backend)
      BUILD_ROCM_BACKEND=1
      shift
      ;;
    --build-vulkan-backend)
      BUILD_VULKAN_BACKEND=1
      shift
      ;;
    --build-s390x-backend)
      BUILD_S390X_BACKEND=1
      shift
      ;;
    --build-openvino-backend)
      BUILD_OPENVINO_BACKEND=1
      shift
      ;;
    --build-type)
      BUILD_TYPE="${2:?missing value for --build-type}"
      shift 2
      ;;
    --gpu-targets)
      GPU_TARGETS="${2:?missing value for --gpu-targets}"
      shift 2
      ;;
    --rocm-path)
      ROCM_PATH_OVERRIDE="${2:?missing value for --rocm-path}"
      shift 2
      ;;
    --openvino-root)
      OPENVINO_ROOT_OVERRIDE="${2:?missing value for --openvino-root}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --include-python-fallback)
      INCLUDE_PYTHON_FALLBACK=1
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

RELEASE_ROOT="${REPO_ROOT}/release/portable/${PLATFORM_TAG}/${PACKAGE_NAME}"
BUILD_ROOT="${REPO_ROOT}/release/build"
RUNTIME_ROOT="${RELEASE_ROOT}/runtime"
CONFIG_ROOT="${RELEASE_ROOT}/config"

run_cmd() {
  echo "+ $*"
  if [[ ${DRY_RUN} -eq 0 ]]; then
    "$@"
  fi
}

# --- prerequisite checks ---

require_command() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: Required command '$1' not found.${2:+ $2}" >&2
    exit 1
  fi
}

require_command python3 "Install Python 3.10+ first."

# --- optional backend builds ---

if [[ ${BUILD_CPU_BACKEND} -eq 1 ]]; then
  CPU_ARGS=(--from-source --build-type "${BUILD_TYPE}")
  [[ ${DRY_RUN} -eq 1 ]] && CPU_ARGS+=(--dry-run)
  run_cmd bash "${CPU_SCRIPT}" "${CPU_ARGS[@]}"
fi

if [[ ${BUILD_ROCM_BACKEND} -eq 1 ]]; then
  ROCM_ARGS=(--from-source --build-type "${BUILD_TYPE}")
  [[ -n "${GPU_TARGETS}" ]] && ROCM_ARGS+=(--gpu-targets "${GPU_TARGETS}")
  [[ -n "${ROCM_PATH_OVERRIDE}" ]] && ROCM_ARGS+=(--rocm-path "${ROCM_PATH_OVERRIDE}")
  [[ ${DRY_RUN} -eq 1 ]] && ROCM_ARGS+=(--dry-run)
  run_cmd bash "${ROCM_SCRIPT}" "${ROCM_ARGS[@]}"
fi

if [[ ${BUILD_VULKAN_BACKEND} -eq 1 ]]; then
  VULKAN_ARGS=(--from-source --build-type "${BUILD_TYPE}")
  [[ ${DRY_RUN} -eq 1 ]] && VULKAN_ARGS+=(--dry-run)
  run_cmd bash "${VULKAN_SCRIPT}" "${VULKAN_ARGS[@]}"
fi

if [[ ${BUILD_S390X_BACKEND} -eq 1 ]]; then
  S390X_ARGS=(--from-source --build-type "${BUILD_TYPE}")
  [[ ${DRY_RUN} -eq 1 ]] && S390X_ARGS+=(--dry-run)
  run_cmd bash "${S390X_SCRIPT}" "${S390X_ARGS[@]}"
fi

if [[ ${BUILD_OPENVINO_BACKEND} -eq 1 ]]; then
  OPENVINO_ARGS=(--from-source --build-type "${BUILD_TYPE}")
  [[ -n "${OPENVINO_ROOT_OVERRIDE}" ]] && OPENVINO_ARGS+=(--openvino-root "${OPENVINO_ROOT_OVERRIDE}")
  [[ ${DRY_RUN} -eq 1 ]] && OPENVINO_ARGS+=(--dry-run)
  run_cmd bash "${OPENVINO_SCRIPT}" "${OPENVINO_ARGS[@]}"
fi

# --- discover built backends ---

[[ ! -f "${RUNTIME_BACKENDS_HELPER}" ]] && { echo "ERROR: Runtime backend helper not found: ${RUNTIME_BACKENDS_HELPER}" >&2; exit 1; }

RUNTIME_BACKENDS_JSON="$(python3 "${RUNTIME_BACKENDS_HELPER}" discover --runtime-root "${LOCAL_RUNTIME_ROOT}" --json)"
mapfile -t backends < <(printf '%s\n' "${RUNTIME_BACKENDS_JSON}" | python3 -c 'import json, sys; print(*[item["id"] for item in json.load(sys.stdin)], sep="\n")')

if [[ ${#backends[@]} -eq 0 ]]; then
  echo "ERROR: No built backends found under ${LOCAL_RUNTIME_ROOT}." >&2
  echo "Build or install at least one backend first (e.g. build-llama-cpu.sh, vllm-linux-cuda/build.sh, or mnn-linux/build.sh)." >&2
  exit 1
fi

DEFAULT_BACKEND="${backends[0]}"
for candidate in \
  "llama.cpp-linux" \
  "llama.cpp-linux-vulkan" \
  "llama.cpp-linux-openvino" \
  "llama.cpp-linux-rocm" \
  "llama.cpp-linux-s390x"; do
  for b in "${backends[@]}"; do
    if [[ "${b}" == "${candidate}" ]]; then
      DEFAULT_BACKEND="${candidate}"
      break 2
    fi
  done
done

echo "Discovered ${#backends[@]} backend(s): ${backends[*]}"
echo "Default backend: ${DEFAULT_BACKEND}"
printf '%s\n' "${RUNTIME_BACKENDS_JSON}" | python3 -c 'import json, sys; [print("  - {id}: {copy_mode} from {source_dir}".format(**item)) for item in json.load(sys.stdin)]'

embedded_backends="$(printf '%s\n' "${RUNTIME_BACKENDS_JSON}" | python3 -c 'import json, sys; print(",".join(item["id"] for item in json.load(sys.stdin) if item.get("runtime_mode") == "embedded"))')"
if [[ -n "${embedded_backends}" && ${INCLUDE_PYTHON_FALLBACK} -eq 0 ]]; then
  echo "ERROR: Embedded Python backend(s) require --include-python-fallback: ${embedded_backends}" >&2
  echo "Build a no-Python package with external-server backends only, or pass --include-python-fallback explicitly." >&2
  exit 1
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo ""
  echo "[dry-run] Would package to: ${RELEASE_ROOT}"
  exit 0
fi

# --- clean and prepare ---

rm -rf "${RELEASE_ROOT}" "${BUILD_ROOT}"
mkdir -p "${RUNTIME_ROOT}" "${CONFIG_ROOT}" "${BUILD_ROOT}"
RUNTIME_BACKENDS_MANIFEST="${BUILD_ROOT}/runtime-backends.json"
printf '%s\n' "${RUNTIME_BACKENDS_JSON}" > "${RUNTIME_BACKENDS_MANIFEST}"

[[ ! -f "${RUST_CLI_PACKAGER}" ]] && { echo "ERROR: Rust CLI packager not found: ${RUST_CLI_PACKAGER}" >&2; exit 1; }

echo ""
echo "Building Rust omniinfer CLI..."
PACKAGER_ARGS=(
  --repo-root "${REPO_ROOT}" \
  --portable-root "${RELEASE_ROOT}" \
  --platform linux
)
[[ ${INCLUDE_PYTHON_FALLBACK} -eq 1 ]] && PACKAGER_ARGS+=(--include-python-fallback)
python3 "${RUST_CLI_PACKAGER}" "${PACKAGER_ARGS[@]}"

# --- config ---

cat > "${CONFIG_ROOT}/omniinfer.json" <<EOF
{
  "host": "127.0.0.1",
  "port": 9000,
  "default_backend": "${DEFAULT_BACKEND}",
  "default_thinking": "off",
  "window_mode": "hidden",
  "startup_timeout": 60,
  "runtime_root": "runtime"
}
EOF

# --- copy runtime backends ---

python3 "${RUNTIME_BACKENDS_HELPER}" copy \
  --manifest "${RUNTIME_BACKENDS_MANIFEST}" \
  --target-root "${RUNTIME_ROOT}"

# --- optional: usage doc ---

USAGE_TEMPLATE="${REPO_ROOT}/tmp/usage.md"
if [[ -f "${USAGE_TEMPLATE}" ]]; then
  cp "${USAGE_TEMPLATE}" "${RELEASE_ROOT}/README.md"
fi

# --- summary ---

echo ""
echo "============================================"
echo "Portable release ready."
echo "  Location:  ${RELEASE_ROOT}"
echo "  Platform:  ${PLATFORM_TAG}"
echo "  Backends:  ${backends[*]}"
echo "  Default:   ${DEFAULT_BACKEND}"
echo "  Python fallback: $([[ ${INCLUDE_PYTHON_FALLBACK} -eq 1 ]] && echo included || echo excluded)"
echo "============================================"
echo ""
echo "Run with:"
echo "  ${RELEASE_ROOT}/omniinfer backend list"
echo "  ${RELEASE_ROOT}/omniinfer chat --message \"Hello\""
echo "  ${RELEASE_ROOT}/${PACKAGE_NAME}"
