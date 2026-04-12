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

ensure_pyinstaller() {
  if ! python3 -c "import PyInstaller" &>/dev/null; then
    echo "PyInstaller not found. Installing..."
    python3 -m pip install pyinstaller || { echo "ERROR: Failed to install PyInstaller." >&2; exit 1; }
  fi
}

require_command python3 "Install Python 3.10+ first."
ensure_pyinstaller

# --- optional backend builds ---

if [[ ${BUILD_CPU_BACKEND} -eq 1 ]]; then
  CPU_ARGS=(--build-type "${BUILD_TYPE}")
  [[ ${DRY_RUN} -eq 1 ]] && CPU_ARGS+=(--dry-run)
  run_cmd bash "${CPU_SCRIPT}" "${CPU_ARGS[@]}"
fi

if [[ ${BUILD_ROCM_BACKEND} -eq 1 ]]; then
  ROCM_ARGS=(--build-type "${BUILD_TYPE}")
  [[ -n "${GPU_TARGETS}" ]] && ROCM_ARGS+=(--gpu-targets "${GPU_TARGETS}")
  [[ -n "${ROCM_PATH_OVERRIDE}" ]] && ROCM_ARGS+=(--rocm-path "${ROCM_PATH_OVERRIDE}")
  [[ ${DRY_RUN} -eq 1 ]] && ROCM_ARGS+=(--dry-run)
  run_cmd bash "${ROCM_SCRIPT}" "${ROCM_ARGS[@]}"
fi

if [[ ${BUILD_VULKAN_BACKEND} -eq 1 ]]; then
  VULKAN_ARGS=(--build-type "${BUILD_TYPE}")
  [[ ${DRY_RUN} -eq 1 ]] && VULKAN_ARGS+=(--dry-run)
  run_cmd bash "${VULKAN_SCRIPT}" "${VULKAN_ARGS[@]}"
fi

if [[ ${BUILD_S390X_BACKEND} -eq 1 ]]; then
  S390X_ARGS=(--build-type "${BUILD_TYPE}")
  [[ ${DRY_RUN} -eq 1 ]] && S390X_ARGS+=(--dry-run)
  run_cmd bash "${S390X_SCRIPT}" "${S390X_ARGS[@]}"
fi

if [[ ${BUILD_OPENVINO_BACKEND} -eq 1 ]]; then
  OPENVINO_ARGS=(--build-type "${BUILD_TYPE}")
  [[ -n "${OPENVINO_ROOT_OVERRIDE}" ]] && OPENVINO_ARGS+=(--openvino-root "${OPENVINO_ROOT_OVERRIDE}")
  [[ ${DRY_RUN} -eq 1 ]] && OPENVINO_ARGS+=(--dry-run)
  run_cmd bash "${OPENVINO_SCRIPT}" "${OPENVINO_ARGS[@]}"
fi

# --- discover built backends ---

backends=()
if [[ -d "${LOCAL_RUNTIME_ROOT}" ]]; then
  for dir in "${LOCAL_RUNTIME_ROOT}"/*/; do
    [[ ! -d "${dir}" ]] && continue
    backend_name="$(basename "${dir}")"
    if [[ -x "${dir}bin/llama-server" ]]; then
      backends+=("${backend_name}")
    fi
  done
fi

if [[ ${#backends[@]} -eq 0 ]]; then
  echo "ERROR: No built backends found under ${LOCAL_RUNTIME_ROOT}." >&2
  echo "Build at least one backend first (e.g. build-llama-cpu.sh)." >&2
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

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo ""
  echo "[dry-run] Would package to: ${RELEASE_ROOT}"
  exit 0
fi

# --- clean and prepare ---

rm -rf "${RELEASE_ROOT}" "${BUILD_ROOT}"
mkdir -p "${RUNTIME_ROOT}" "${CONFIG_ROOT}" "${BUILD_ROOT}"

GATEWAY_ENTRY="${REPO_ROOT}/omniinfer_gateway.py"
CLI_ENTRY="${REPO_ROOT}/service_core/cli.py"

[[ ! -f "${GATEWAY_ENTRY}" ]] && { echo "ERROR: Gateway entry not found: ${GATEWAY_ENTRY}" >&2; exit 1; }
[[ ! -f "${CLI_ENTRY}" ]] && { echo "ERROR: CLI entry not found: ${CLI_ENTRY}" >&2; exit 1; }

# --- PyInstaller: gateway (--onedir) ---

echo ""
echo "Building ${PACKAGE_NAME} (gateway) with PyInstaller..."
python3 -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name "${PACKAGE_NAME}" \
  --distpath "${REPO_ROOT}/release/portable/${PLATFORM_TAG}" \
  --workpath "${BUILD_ROOT}/pyinstaller-work-gateway" \
  --specpath "${BUILD_ROOT}/pyinstaller-spec-gateway" \
  "${GATEWAY_ENTRY}"

GATEWAY_BIN="${RELEASE_ROOT}/${PACKAGE_NAME}"
if [[ ! -f "${GATEWAY_BIN}" ]]; then
  echo "ERROR: Gateway build succeeded but ${PACKAGE_NAME} not found at ${GATEWAY_BIN}" >&2
  exit 1
fi

# --- PyInstaller: CLI (--onefile) ---

CLI_DIST="${BUILD_ROOT}/cli-dist"
echo ""
echo "Building omniinfer-cli (CLI) with PyInstaller..."
python3 -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --console \
  --name "omniinfer-cli" \
  --distpath "${CLI_DIST}" \
  --workpath "${BUILD_ROOT}/pyinstaller-work-cli" \
  --specpath "${BUILD_ROOT}/pyinstaller-spec-cli" \
  "${CLI_ENTRY}"

CLI_BIN="${CLI_DIST}/omniinfer-cli"
if [[ ! -f "${CLI_BIN}" ]]; then
  echo "ERROR: CLI build succeeded but omniinfer-cli not found at ${CLI_BIN}" >&2
  exit 1
fi

cp "${CLI_BIN}" "${RELEASE_ROOT}/omniinfer-cli"
chmod +x "${RELEASE_ROOT}/omniinfer-cli"

# --- launcher wrapper ---

cat > "${RELEASE_ROOT}/omniinfer" <<'LAUNCHER_EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/omniinfer-cli" "$@"
LAUNCHER_EOF
chmod +x "${RELEASE_ROOT}/omniinfer"

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

for backend_name in "${backends[@]}"; do
  source_root="${LOCAL_RUNTIME_ROOT}/${backend_name}"
  target_root="${RUNTIME_ROOT}/${backend_name}"

  mkdir -p "${target_root}/bin" "${target_root}/logs"

  if [[ -d "${source_root}/bin" ]]; then
    find "${source_root}/bin" -maxdepth 1 -type f \( -executable -o -name '*.so' -o -name '*.so.*' \) \
      -exec cp {} "${target_root}/bin/" \;
  fi
done

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
echo "============================================"
echo ""
echo "Run with:"
echo "  ${RELEASE_ROOT}/omniinfer backend list"
echo "  ${RELEASE_ROOT}/omniinfer chat --message \"Hello\""
echo "  ${RELEASE_ROOT}/${PACKAGE_NAME}"
