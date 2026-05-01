#!/usr/bin/env bash

omni_cuda_arch_dir() {
  case "$(uname -m 2>/dev/null || printf unknown)" in
    x86_64|amd64) printf 'x86_64-linux\n' ;;
    aarch64|arm64) printf 'aarch64-linux\n' ;;
    ppc64le) printf 'ppc64le-linux\n' ;;
    *) printf 'x86_64-linux\n' ;;
  esac
}

omni_cuda_abs_path() {
  local path="$1"
  if command -v readlink >/dev/null 2>&1; then
    readlink -f "${path}" 2>/dev/null && return
  fi
  (cd "${path}" 2>/dev/null && pwd -P)
}

omni_cuda_add_root() {
  local root="$1"
  [[ -z "${root}" || ! -d "${root}" ]] && return
  root="$(omni_cuda_abs_path "${root}")"
  [[ -z "${root}" ]] && return
  case ":${_OMNI_CUDA_ROOTS_SEEN:-}:" in
    *":${root}:"*) return ;;
  esac
  _OMNI_CUDA_ROOTS_SEEN="${_OMNI_CUDA_ROOTS_SEEN:-}:${root}"
  OMNI_CUDA_CANDIDATE_ROOTS+=("${root}")
}

omni_cuda_collect_candidate_roots() {
  OMNI_CUDA_CANDIDATE_ROOTS=()
  _OMNI_CUDA_ROOTS_SEEN=""

  omni_cuda_add_root "${CUDAToolkit_ROOT:-}"
  omni_cuda_add_root "${CUDA_HOME:-}"
  omni_cuda_add_root "${CUDA_PATH:-}"

  local nvcc_path
  nvcc_path="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "${nvcc_path}" ]]; then
    omni_cuda_add_root "$(cd "$(dirname "${nvcc_path}")/.." && pwd -P)"
  fi

  if [[ "${OMNI_CUDA_DETECT_SKIP_DEFAULT_ROOTS:-0}" != "1" ]]; then
    omni_cuda_add_root /usr/local/cuda
    local root
    for root in /usr/local/cuda-* /opt/cuda; do
      [[ -d "${root}" ]] && omni_cuda_add_root "${root}"
    done
  fi
}

omni_cuda_find_in_root() {
  local root="$1"
  shift
  local rel
  for rel in "$@"; do
    if [[ -e "${root}/${rel}" ]]; then
      printf '%s\n' "${root}/${rel}"
      return 0
    fi
  done
  return 1
}

omni_cuda_find_header() {
  local root="$1"
  local arch_dir
  arch_dir="$(omni_cuda_arch_dir)"
  omni_cuda_find_in_root "${root}" \
    "include/cublas_v2.h" \
    "targets/${arch_dir}/include/cublas_v2.h"
}

omni_cuda_find_lib() {
  local root="$1"
  local name="$2"
  local arch_dir
  arch_dir="$(omni_cuda_arch_dir)"
  omni_cuda_find_in_root "${root}" \
    "lib64/${name}" \
    "lib/${name}" \
    "targets/${arch_dir}/lib/${name}" \
    "targets/x86_64-linux/lib/${name}" \
    "targets/aarch64-linux/lib/${name}"
}

omni_cuda_library_dirs() {
  local root="$1"
  local arch_dir dir output=""
  arch_dir="$(omni_cuda_arch_dir)"
  for dir in \
    "${root}/lib64" \
    "${root}/lib" \
    "${root}/targets/${arch_dir}/lib" \
    "${root}/targets/x86_64-linux/lib" \
    "${root}/targets/aarch64-linux/lib"; do
    [[ -d "${dir}" ]] || continue
    case ":${output}:" in
      *":${dir}:"*) ;;
      *) output="${output}${output:+:}${dir}" ;;
    esac
  done
  printf '%s\n' "${output}"
}

omni_cuda_find_nvcc() {
  local root="$1"
  if [[ -x "${root}/bin/nvcc" ]]; then
    printf '%s\n' "${root}/bin/nvcc"
    return 0
  fi
  return 1
}

omni_cuda_find_private_runtime_libs() {
  find /usr/local/lib/ollama /usr/local/lib /usr/lib -maxdepth 4 \
    \( -name 'libcublas.so.*' -o -name 'libcublasLt.so.*' \) \
    2>/dev/null | sort -u
}

omni_cuda_summarize_private_runtime_libs() {
  local path total=0 shown=0 output=""
  while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    total=$((total + 1))
    if [[ ${shown} -lt 4 ]]; then
      output="${output}${output:+ }${path}"
      shown=$((shown + 1))
    fi
  done < <(omni_cuda_find_private_runtime_libs)

  if [[ ${total} -gt ${shown} ]]; then
    output="${output} ... (${total} total)"
  fi
  printf '%s\n' "${output}"
}

omni_cuda_detect() {
  local require_cublaslt="${1:-0}"
  OMNI_CUDA_TOOLKIT_ROOT=""
  OMNI_CUDA_NVCC=""
  OMNI_CUDA_CUBLAS_HEADER=""
  OMNI_CUDA_CUBLAS_LIB=""
  OMNI_CUDA_CUBLASLT_LIB=""
  OMNI_CUDA_LIBRARY_DIRS=""
  OMNI_CUDA_DETECT_REASON=""

  omni_cuda_collect_candidate_roots
  local root nvcc header cublas cublaslt first_reason
  for root in "${OMNI_CUDA_CANDIDATE_ROOTS[@]}"; do
    nvcc="$(omni_cuda_find_nvcc "${root}" || true)"
    header="$(omni_cuda_find_header "${root}" || true)"
    cublas="$(omni_cuda_find_lib "${root}" "libcublas.so" || true)"
    cublaslt="$(omni_cuda_find_lib "${root}" "libcublasLt.so" || true)"

    if [[ -z "${nvcc}" ]]; then
      first_reason="${first_reason:-${root}: missing bin/nvcc}"
      continue
    fi
    if [[ -z "${header}" ]]; then
      first_reason="${first_reason:-${root}: missing cublas_v2.h}"
      continue
    fi
    if [[ -z "${cublas}" ]]; then
      first_reason="${first_reason:-${root}: missing libcublas.so development symlink}"
      continue
    fi
    if [[ "${require_cublaslt}" == "1" && -z "${cublaslt}" ]]; then
      first_reason="${first_reason:-${root}: missing libcublasLt.so development symlink}"
      continue
    fi

    OMNI_CUDA_TOOLKIT_ROOT="${root}"
    OMNI_CUDA_NVCC="${nvcc}"
    OMNI_CUDA_CUBLAS_HEADER="${header}"
    OMNI_CUDA_CUBLAS_LIB="${cublas}"
    OMNI_CUDA_CUBLASLT_LIB="${cublaslt}"
    OMNI_CUDA_LIBRARY_DIRS="$(omni_cuda_library_dirs "${root}")"
    return 0
  done

  OMNI_CUDA_DETECT_REASON="${first_reason:-no CUDA toolkit candidate roots found}"
  local private_libs
  private_libs="$(omni_cuda_summarize_private_runtime_libs)"
  if [[ -n "${private_libs}" ]]; then
    OMNI_CUDA_DETECT_REASON="${OMNI_CUDA_DETECT_REASON}; found private/runtime cuBLAS libraries only: ${private_libs}"
  fi
  return 1
}

omni_cuda_install_hint() {
  cat <<'EOF'
Install CUDA cuBLAS development files, not only runtime/private libraries. On Ubuntu with NVIDIA CUDA apt repo, try: sudo apt install libcublas-dev-12-5. Or install the full CUDA toolkit and set CUDAToolkit_ROOT=/path/to/cuda or CUDA_HOME=/path/to/cuda.
EOF
}

omni_cuda_print_dep_status() {
  local require_cublaslt="${1:-0}"
  if omni_cuda_detect "${require_cublaslt}"; then
    printf 'ok|cuda-toolkit|CUDA toolkit with cuBLAS dev files|%s|%s\n' \
      "Using ${OMNI_CUDA_TOOLKIT_ROOT}" ""
    return 0
  fi
  printf 'missing|cuda-toolkit|CUDA toolkit with cuBLAS dev files|%s %s|%s\n' \
    "${OMNI_CUDA_DETECT_REASON}" "$(omni_cuda_install_hint)" "libcublas-dev-12-5"
  return 1
}

omni_cuda_print_nvcc_dep_status() {
  omni_cuda_collect_candidate_roots
  local root nvcc
  for root in "${OMNI_CUDA_CANDIDATE_ROOTS[@]}"; do
    nvcc="$(omni_cuda_find_nvcc "${root}" || true)"
    if [[ -n "${nvcc}" ]]; then
      printf 'ok|nvcc|NVIDIA CUDA compiler|Using %s|%s\n' "${nvcc}" ""
      return 0
    fi
  done
  printf 'missing|nvcc|NVIDIA CUDA compiler|Install the CUDA toolkit or set CUDAToolkit_ROOT/CUDA_HOME to a toolkit containing bin/nvcc|%s\n' "cuda-toolkit-12-5"
  return 1
}

omni_cuda_require_toolkit() {
  local require_cublaslt="${1:-0}"
  if omni_cuda_detect "${require_cublaslt}"; then
    export CUDAToolkit_ROOT="${OMNI_CUDA_TOOLKIT_ROOT}"
    export CUDA_HOME="${CUDA_HOME:-${OMNI_CUDA_TOOLKIT_ROOT}}"
    export PATH="${OMNI_CUDA_TOOLKIT_ROOT}/bin:${PATH}"
    export LIBRARY_PATH="${OMNI_CUDA_LIBRARY_DIRS}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${OMNI_CUDA_LIBRARY_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    return 0
  fi

  {
    echo "CUDA toolkit with cuBLAS development files was not found."
    echo "Reason: ${OMNI_CUDA_DETECT_REASON}"
    echo
    echo "Checked roots:"
    local root
    for root in "${OMNI_CUDA_CANDIDATE_ROOTS[@]:-}"; do
      echo "  - ${root}"
    done
    echo
    echo "$(omni_cuda_install_hint)"
  } >&2
  return 1
}
