#!/usr/bin/env bash

set -euo pipefail

CUDA_ROOT="${HOME}/.local/cuda-12.5"
BASE_CUDA_ROOT="${CUDAToolkit_ROOT:-${CUDA_HOME:-}}"
CACHE_DIR="${HOME}/.cache/omniinfer-cuda-debs"
STAGING_ROOT="${HOME}/.local/cuda-12.5-deb-root"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: install-cuda-cublas-local.sh [options]

Downloads NVIDIA CUDA cuBLAS development packages without sudo, extracts them
into a user-writable CUDA toolkit root, and prints the CUDAToolkit_ROOT export.

Options:
  --cuda-root DIR       User-local CUDA root, default: ~/.local/cuda-12.5
  --base-cuda-root DIR  Existing partial CUDA toolkit with bin/nvcc and cudart
  --cache-dir DIR      Download cache, default: ~/.cache/omniinfer-cuda-debs
  --staging-root DIR   Extraction staging dir, default: ~/.local/cuda-12.5-deb-root
  --dry-run            Print commands without running them
  -h, --help           Show this help message

Requires an NVIDIA apt repository configured for libcublas-12-5 and
libcublas-dev-12-5. No sudo is used.
EOF
}

while (($# > 0)); do
  case "$1" in
    --cuda-root)
      CUDA_ROOT="${2:?missing value for --cuda-root}"
      shift 2
      ;;
    --base-cuda-root)
      BASE_CUDA_ROOT="${2:?missing value for --base-cuda-root}"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="${2:?missing value for --cache-dir}"
      shift 2
      ;;
    --staging-root)
      STAGING_ROOT="${2:?missing value for --staging-root}"
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

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

detect_base_cuda_root() {
  if [[ -n "${BASE_CUDA_ROOT}" && -x "${BASE_CUDA_ROOT}/bin/nvcc" ]]; then
    printf '%s\n' "${BASE_CUDA_ROOT}"
    return 0
  fi

  local nvcc_path
  nvcc_path="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "${nvcc_path}" ]]; then
    (cd "$(dirname "${nvcc_path}")/.." && pwd -P)
    return 0
  fi

  if [[ -x /usr/local/cuda-12.5/bin/nvcc ]]; then
    printf '%s\n' /usr/local/cuda-12.5
    return 0
  fi
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    printf '%s\n' /usr/local/cuda
    return 0
  fi

  return 1
}

BASE_CUDA_ROOT="$(detect_base_cuda_root)" || {
  echo "No base CUDA toolkit with bin/nvcc was found." >&2
  echo "Install CUDA toolkit first, or pass --base-cuda-root /path/to/cuda." >&2
  exit 1
}

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get was not found; this user-local installer currently supports NVIDIA apt packages." >&2
  exit 1
fi
if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "dpkg-deb was not found; install dpkg tooling or extract the .deb packages manually." >&2
  exit 1
fi

echo "Base CUDA toolkit: ${BASE_CUDA_ROOT}"
echo "User CUDA root:    ${CUDA_ROOT}"
echo "Download cache:    ${CACHE_DIR}"
echo "Staging root:      ${STAGING_ROOT}"

run_cmd mkdir -p "${CACHE_DIR}" "${CUDA_ROOT}" "${STAGING_ROOT}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN: cd ${CACHE_DIR} && apt-get download libcublas-12-5 libcublas-dev-12-5"
else
  (
    cd "${CACHE_DIR}"
    apt-get download libcublas-12-5 libcublas-dev-12-5
  )
fi

shopt -s nullglob
runtime_debs=("${CACHE_DIR}"/libcublas-12-5_*.deb)
dev_debs=("${CACHE_DIR}"/libcublas-dev-12-5_*.deb)
shopt -u nullglob

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ ${#runtime_debs[@]} -eq 0 || ${#dev_debs[@]} -eq 0 ]]; then
    echo "Downloaded cuBLAS packages were not found in ${CACHE_DIR}." >&2
    exit 1
  fi
fi

for deb in "${runtime_debs[@]:-${CACHE_DIR}/libcublas-12-5_<version>_amd64.deb}"; do
  run_cmd dpkg-deb -x "${deb}" "${STAGING_ROOT}"
done
for deb in "${dev_debs[@]:-${CACHE_DIR}/libcublas-dev-12-5_<version>_amd64.deb}"; do
  run_cmd dpkg-deb -x "${deb}" "${STAGING_ROOT}"
done

run_cmd cp -a "${BASE_CUDA_ROOT}/." "${CUDA_ROOT}/."
run_cmd cp -a "${STAGING_ROOT}/usr/local/cuda-12.5/." "${CUDA_ROOT}/."

echo
echo "User-local CUDA/cuBLAS toolkit is ready:"
echo "  export CUDAToolkit_ROOT=${CUDA_ROOT}"
echo "  export CUDA_HOME=${CUDA_ROOT}"
echo
echo "Verify:"
echo "  CUDAToolkit_ROOT=${CUDA_ROOT} bash scripts/platforms/linux/llama.cpp-linux-cuda/build.sh --check-deps"
