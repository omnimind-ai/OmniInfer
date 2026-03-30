#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PLATFORM_ROOT="${REPO_ROOT}/.local/runtime/linux"
LEGACY_PLATFORM_ROOT="${REPO_ROOT}/platform/Linux"
CPU_SCRIPT="${SCRIPT_DIR}/build-llama-cpu.sh"
ROCM_SCRIPT="${SCRIPT_DIR}/build-llama-rocm.sh"

PACKAGE_NAME="OmniInfer"
BUILD_CPU_BACKEND=0
BUILD_ROCM_BACKEND=0
BUILD_TYPE="Release"
GPU_TARGETS=""
ROCM_PATH_OVERRIDE=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: build-release.sh [options]

Options:
  --package-name <name>     Release directory name under release/portable
  --build-cpu-backend       Build the Linux CPU backend before packaging
  --build-rocm-backend      Build the Linux ROCm backend before packaging
  --build-type <type>       CMake build type, default: Release
  --gpu-targets <targets>   Override HIP GPU targets for the ROCm build
  --rocm-path <path>        Override ROCm installation root for the ROCm build
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
    --build-cpu-backend)
      BUILD_CPU_BACKEND=1
      shift
      ;;
    --build-rocm-backend)
      BUILD_ROCM_BACKEND=1
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

RELEASE_ROOT="${REPO_ROOT}/release/portable/${PACKAGE_NAME}"
RUNTIME_ROOT="${RELEASE_ROOT}/runtime"
CONFIG_ROOT="${RELEASE_ROOT}/config"
TEST_ASSETS_ROOT="${RELEASE_ROOT}/tests/pictures"

run_cmd() {
  echo "+ $*"
  if [[ ${DRY_RUN} -eq 0 ]]; then
    "$@"
  fi
}

copy_tree_contents() {
  local src="$1"
  local dest="$2"
  mkdir -p "${dest}"
  cp -a "${src}/." "${dest}/"
}

if [[ ${BUILD_CPU_BACKEND} -eq 1 ]]; then
  CPU_ARGS=(--build-type "${BUILD_TYPE}")
  if [[ ${DRY_RUN} -eq 1 ]]; then
    CPU_ARGS+=(--dry-run)
  fi
  run_cmd bash "${CPU_SCRIPT}" "${CPU_ARGS[@]}"
fi

if [[ ${BUILD_ROCM_BACKEND} -eq 1 ]]; then
  ROCM_ARGS=(--build-type "${BUILD_TYPE}")
  if [[ -n "${GPU_TARGETS}" ]]; then
    ROCM_ARGS+=(--gpu-targets "${GPU_TARGETS}")
  fi
  if [[ -n "${ROCM_PATH_OVERRIDE}" ]]; then
    ROCM_ARGS+=(--rocm-path "${ROCM_PATH_OVERRIDE}")
  fi
  if [[ ${DRY_RUN} -eq 1 ]]; then
    ROCM_ARGS+=(--dry-run)
  fi
  run_cmd bash "${ROCM_SCRIPT}" "${ROCM_ARGS[@]}"
fi

echo "Preparing Linux portable release at ${RELEASE_ROOT}"

if [[ ${DRY_RUN} -eq 1 ]]; then
  exit 0
fi

rm -rf "${RELEASE_ROOT}"
mkdir -p "${RUNTIME_ROOT}" "${CONFIG_ROOT}" "${TEST_ASSETS_ROOT}"

cp -a "${REPO_ROOT}/service_core" "${RELEASE_ROOT}/service_core"
cp -a "${REPO_ROOT}/omniinfer" "${RELEASE_ROOT}/omniinfer"
cp -a "${REPO_ROOT}/omniinfer_gateway.py" "${RELEASE_ROOT}/omniinfer_gateway.py"
cp -a "${REPO_ROOT}/tmp/usage.md" "${RELEASE_ROOT}/usage.md"
cp -a "${REPO_ROOT}/tests/pictures/test1.png" "${TEST_ASSETS_ROOT}/test1.png"
find "${RELEASE_ROOT}/service_core" -type d -name '__pycache__' -prune -exec rm -rf {} +

for backend_dir in llama.cpp-linux llama.cpp-linux-rocm; do
  source_root="${PLATFORM_ROOT}/${backend_dir}"
  if [[ ! -d "${source_root}" && -d "${LEGACY_PLATFORM_ROOT}/${backend_dir}" ]]; then
    source_root="${LEGACY_PLATFORM_ROOT}/${backend_dir}"
  fi
  if [[ ! -d "${source_root}" ]]; then
    continue
  fi
  mkdir -p "${RUNTIME_ROOT}/${backend_dir}"
  for child in bin logs models; do
    if [[ -d "${source_root}/${child}" ]]; then
      copy_tree_contents "${source_root}/${child}" "${RUNTIME_ROOT}/${backend_dir}/${child}"
    fi
  done
done

DEFAULT_BACKEND="llama.cpp-linux"
if [[ ! -x "${RUNTIME_ROOT}/llama.cpp-linux/bin/llama-server" && -x "${RUNTIME_ROOT}/llama.cpp-linux-rocm/bin/llama-server" ]]; then
  DEFAULT_BACKEND="llama.cpp-linux-rocm"
fi

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

cat > "${RELEASE_ROOT}/OmniInfer" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/omniinfer_gateway.py" "$@"
EOF
chmod +x "${RELEASE_ROOT}/OmniInfer" "${RELEASE_ROOT}/omniinfer"

echo
echo "Linux portable release ready."
echo "Release directory: ${RELEASE_ROOT}"
