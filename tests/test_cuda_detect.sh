#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="${REPO_ROOT}/scripts/platforms/linux/cuda-detect.sh"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

run_detect() {
  local root="$1"
  local require_lt="$2"
  OMNI_CUDA_DETECT_SKIP_DEFAULT_ROOTS=1 CUDAToolkit_ROOT="${root}" bash -c \
    "source '${HELPER}'; omni_cuda_detect '${require_lt}' || exit \$?; printf '%s|%s|%s|%s\n' \"\${OMNI_CUDA_TOOLKIT_ROOT}\" \"\${OMNI_CUDA_NVCC}\" \"\${OMNI_CUDA_CUBLAS_LIB}\" \"\${OMNI_CUDA_CUBLASLT_LIB}\""
}

incomplete="${TMP_ROOT}/cuda-incomplete"
mkdir -p "${incomplete}/bin"
printf '#!/usr/bin/env bash\nexit 0\n' > "${incomplete}/bin/nvcc"
chmod +x "${incomplete}/bin/nvcc"

if run_detect "${incomplete}" 0 >"${TMP_ROOT}/cuda-detect-incomplete.out" 2>"${TMP_ROOT}/cuda-detect-incomplete.err"; then
  echo "expected incomplete CUDA root to fail" >&2
  exit 1
fi

split_root="${TMP_ROOT}/cuda-split-root"
split_path="${TMP_ROOT}/cuda-split-path"
mkdir -p "${split_root}/include" "${split_root}/lib64" "${split_path}/bin"
touch "${split_root}/include/cublas_v2.h"
touch "${split_root}/lib64/libcublas.so"
printf '#!/usr/bin/env bash\nexit 0\n' > "${split_path}/bin/nvcc"
chmod +x "${split_path}/bin/nvcc"

if OMNI_CUDA_DETECT_SKIP_DEFAULT_ROOTS=1 CUDAToolkit_ROOT="${split_root}" PATH="${split_path}/bin:${PATH}" bash -c \
  "source '${HELPER}'; omni_cuda_detect 0" >"${TMP_ROOT}/cuda-detect-split.out" 2>"${TMP_ROOT}/cuda-detect-split.err"; then
  echo "expected CUDA detection to reject split nvcc/header/lib roots" >&2
  exit 1
fi

complete="${TMP_ROOT}/cuda-complete"
mkdir -p "${complete}/bin" "${complete}/include" "${complete}/lib64"
printf '#!/usr/bin/env bash\nexit 0\n' > "${complete}/bin/nvcc"
chmod +x "${complete}/bin/nvcc"
touch "${complete}/include/cublas_v2.h"
touch "${complete}/lib64/libcublas.so"
touch "${complete}/lib64/libcublasLt.so"

detected="$(run_detect "${complete}" 1)"
IFS='|' read -r detected_root detected_nvcc detected_cublas detected_cublaslt <<< "${detected}"
[[ "${detected_root}" == "${complete}" ]]
[[ "${detected_nvcc}" == "${complete}/bin/nvcc" ]]
[[ "${detected_cublas}" == "${complete}/lib64/libcublas.so" ]]
[[ "${detected_cublaslt}" == "${complete}/lib64/libcublasLt.so" ]]

env_output="$(OMNI_CUDA_DETECT_SKIP_DEFAULT_ROOTS=1 CUDAToolkit_ROOT="${complete}" bash -c \
  "source '${HELPER}'; omni_cuda_require_toolkit 1; printf '%s|%s\n' \"\${LIBRARY_PATH%%:*}\" \"\${LD_LIBRARY_PATH%%:*}\"")"
IFS='|' read -r library_path_head ld_library_path_head <<< "${env_output}"
[[ "${library_path_head}" == "${complete}/lib64" ]]
[[ "${ld_library_path_head}" == "${complete}/lib64" ]]

echo "cuda-detect tests passed"
