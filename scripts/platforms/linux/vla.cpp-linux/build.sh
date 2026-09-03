#!/usr/bin/env bash

set -euo pipefail

BACKEND_ID="${OMNIINFER_VLA_CPP_BACKEND_ID:-vla.cpp-linux}"
BACKEND_LABEL="${OMNIINFER_VLA_CPP_BACKEND_LABEL:-vla.cpp Linux CPU}"
ENABLE_CUDA="${OMNIINFER_VLA_CPP_ENABLE_CUDA:-0}"
BUILD_TYPE="Release"
DRY_RUN=0
JOBS=""
CLEAN_BUILD=0
BOOTSTRAP_SUBMODULE=1
SMOKE_TEST=0
CHECK_DEPS=0
BUILD_FROM_SOURCE=0
USE_NATIVE=0
ENABLE_LTO=0
DEPENDENCY_PREFIX=""
DEPENDENCY_LIBRARY_DIRS=()
DEPENDENCY_PKG_CONFIG_DIRS=()
CUDA_ARCHITECTURES=""
LLAMA_SOURCE=""

check_deps() {
  local rc=0
  _dep() {
    local cmd="$1" desc="$2" hint="$3" pkg="${4:-}"
    if command -v "${cmd}" >/dev/null 2>&1; then
      printf 'ok|%s|%s|%s|%s\n' "${cmd}" "${desc}" "${hint}" "${pkg}"
    else
      printf 'missing|%s|%s|%s|%s\n' "${cmd}" "${desc}" "${hint}" "${pkg}"
      rc=1
    fi
  }
  _dep cmake "CMake build system" "sudo apt install cmake" cmake
  _dep pkg-config "pkg-config for libzmq discovery" "sudo apt install pkg-config" pkg-config
  _dep protoc "Protocol Buffers compiler" "sudo apt install protobuf-compiler" protobuf-compiler
  return ${rc}
}

usage() {
  cat <<'EOF'
Usage: build-vla-linux.sh [options]

Options:
  --build-type <type>          CMake build type, default: Release
  --jobs <n>                   Parallel build jobs, default: nproc
  --native                     Optimize host-side kernels for the current CPU
  --portable                   Disable host-specific CPU tuning (default)
  --lto                        Enable link-time optimization
  --clean                      Remove the previous build directory before configuring
  --dependency-prefix <path>   Prefix containing protobuf/cppzmq/libzmq dependencies
  --cuda-architectures <list>  CMAKE_CUDA_ARCHITECTURES value for CUDA builds
  --llama-source <path>        Reuse an existing llama.cpp source tree instead of downloading it
  --no-bootstrap               Do not auto-initialize the vla.cpp git submodule
  --from-source                Build from the checked-out source submodule
  --smoke-test                 Run `vla-server --help` after the build completes
  --dry-run                    Print actions without executing them
  -h, --help                   Show this help message
EOF
}

while (($# > 0)); do
  case "$1" in
    --build-type)
      BUILD_TYPE="${2:?missing value for --build-type}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:?missing value for --jobs}"
      shift 2
      ;;
    --native)
      USE_NATIVE=1
      shift
      ;;
    --portable)
      USE_NATIVE=0
      shift
      ;;
    --lto)
      ENABLE_LTO=1
      shift
      ;;
    --clean)
      CLEAN_BUILD=1
      shift
      ;;
    --dependency-prefix)
      DEPENDENCY_PREFIX="${2:?missing value for --dependency-prefix}"
      shift 2
      ;;
    --cuda-architectures)
      CUDA_ARCHITECTURES="${2:?missing value for --cuda-architectures}"
      shift 2
      ;;
    --llama-source)
      LLAMA_SOURCE="${2:?missing value for --llama-source}"
      shift 2
      ;;
    --no-bootstrap)
      BOOTSTRAP_SUBMODULE=0
      shift
      ;;
    --from-source)
      BUILD_FROM_SOURCE=1
      shift
      ;;
    --smoke-test)
      SMOKE_TEST=1
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
    --check-deps)
      CHECK_DEPS=1
      shift
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
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/linux/${BACKEND_ID}"
VLA_ROOT="${REPO_ROOT}/framework/vla.cpp"
BUILD_ROOT="${PACKAGE_ROOT}/build/${BACKEND_ID}"
BIN_ROOT="${PACKAGE_ROOT}/bin"
LOG_ROOT="${PACKAGE_ROOT}/logs"
MODELS_ROOT="${REPO_ROOT}/.local/models"

if [[ -n "${LLAMA_SOURCE}" ]]; then
  if [[ ! -f "${LLAMA_SOURCE}/CMakeLists.txt" ]]; then
    echo "llama.cpp source tree is missing CMakeLists.txt: ${LLAMA_SOURCE}" >&2
    exit 1
  fi
  LLAMA_SOURCE="$(cd "${LLAMA_SOURCE}" && pwd -P)"
fi

if [[ -n "${DEPENDENCY_PREFIX}" ]]; then
  if [[ ! -d "${DEPENDENCY_PREFIX}" ]]; then
    echo "Dependency prefix is not a directory: ${DEPENDENCY_PREFIX}" >&2
    exit 1
  fi
  resolved_dependency_prefix="$(cd "${DEPENDENCY_PREFIX}" && pwd -P)"
  case "${resolved_dependency_prefix}" in
    /|/usr|/lib|/lib64|/usr/lib|/usr/lib64)
      echo "--dependency-prefix must be an isolated non-system prefix: ${DEPENDENCY_PREFIX}" >&2
      exit 1
      ;;
  esac
  export PATH="${DEPENDENCY_PREFIX}/bin:${PATH}"
  export CMAKE_PREFIX_PATH="${DEPENDENCY_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  for candidate in \
    "${DEPENDENCY_PREFIX}/lib" \
    "${DEPENDENCY_PREFIX}/lib64" \
    "${DEPENDENCY_PREFIX}"/lib/*-linux-gnu; do
    if [[ -d "${candidate}" ]]; then
      DEPENDENCY_LIBRARY_DIRS+=("${candidate}")
      if [[ -d "${candidate}/pkgconfig" ]]; then
        DEPENDENCY_PKG_CONFIG_DIRS+=("${candidate}/pkgconfig")
      fi
    fi
  done
  if [[ -d "${DEPENDENCY_PREFIX}/share/pkgconfig" ]]; then
    DEPENDENCY_PKG_CONFIG_DIRS+=("${DEPENDENCY_PREFIX}/share/pkgconfig")
  fi
  if ((${#DEPENDENCY_LIBRARY_DIRS[@]} > 0)); then
    dependency_path="$(IFS=:; printf '%s' "${DEPENDENCY_LIBRARY_DIRS[*]}")"
    export LD_LIBRARY_PATH="${dependency_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  if ((${#DEPENDENCY_PKG_CONFIG_DIRS[@]} > 0)); then
    pkg_config_path="$(IFS=:; printf '%s' "${DEPENDENCY_PKG_CONFIG_DIRS[*]}")"
    export PKG_CONFIG_PATH="${pkg_config_path}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
  fi
fi

if [[ ${CHECK_DEPS} -eq 1 ]]; then
  check_deps
  exit $?
fi

if [[ ${BUILD_FROM_SOURCE} -eq 0 ]]; then
  echo "No prebuilt install path is configured for ${BACKEND_ID}." >&2
  echo "Re-run with --from-source to build from framework/vla.cpp." >&2
  exit 1
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' was not found in PATH." >&2
    exit 1
  fi
}

detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return
  fi
  printf '1\n'
}

ensure_vla_root() {
  if [[ -f "${VLA_ROOT}/CMakeLists.txt" ]]; then
    return
  fi

  if [[ ${BOOTSTRAP_SUBMODULE} -eq 0 ]]; then
    echo "vla.cpp source tree was not found at ${VLA_ROOT}" >&2
    echo "Run: git submodule update --init --recursive framework/vla.cpp" >&2
    exit 1
  fi

  if [[ ! -d "${REPO_ROOT}/.git" && ! -f "${REPO_ROOT}/.git" ]]; then
    echo "vla.cpp source tree was not found at ${VLA_ROOT}" >&2
    exit 1
  fi

  require_command git
  echo "vla.cpp source tree is missing. Bootstrapping the submodule..."
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  git -C ${REPO_ROOT} submodule update --init --recursive framework/vla.cpp"
    return
  fi
  git -C "${REPO_ROOT}" submodule update --init --recursive framework/vla.cpp

  if [[ ! -f "${VLA_ROOT}/CMakeLists.txt" ]]; then
    echo "Failed to prepare vla.cpp at ${VLA_ROOT}" >&2
    exit 1
  fi
}

prepare_runtime_dirs() {
  mkdir -p "${BUILD_ROOT}" "${BIN_ROOT}" "${LOG_ROOT}" "${MODELS_ROOT}"
  touch "${BIN_ROOT}/.gitkeep" "${LOG_ROOT}/.gitkeep" "${MODELS_ROOT}/.gitkeep"
}

dependency_library_path() {
  local value="${BIN_ROOT}"
  local candidate
  for candidate in "${DEPENDENCY_LIBRARY_DIRS[@]}"; do
    value="${value}:${candidate}"
  done
  printf '%s\n' "${value}"
}

dependency_is_bundleable() {
  local name="$1" path="$2"
  case "${name}" in
    ld-linux*.so*|libc.so*|libm.so*|libpthread.so*|libdl.so*|librt.so*|libstdc++.so*|libgcc_s.so*|libgomp.so*|libatomic.so*|libresolv.so*|libutil.so*|libnsl.so*|libanl.so*|libthread_db.so*)
      return 1
      ;;
  esac
  if [[ -n "${DEPENDENCY_PREFIX}" ]]; then
    local prefix
    prefix="$(readlink -f "${DEPENDENCY_PREFIX}")"
    if [[ "${path}" == "${prefix}/"* ]]; then
      return 0
    fi
  fi
  case "${name}" in
    libzmq.so*|libprotobuf.so*|libabsl_*.so*|libsodium.so*|libpgm*.so*|libnorm.so*|libutf8_*.so*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

collect_runtime_dependency_closure() {
  local root_binary="$1"
  local library_path
  library_path="$(dependency_library_path)"
  local -a queue=("${root_binary}")
  local index=0
  declare -A copied=()
  while ((index < ${#queue[@]})); do
    local object="${queue[${index}]}"
    index=$((index + 1))
    local output
    if ! output="$(LD_LIBRARY_PATH="${library_path}" ldd "${object}" 2>&1)"; then
      echo "Failed to inspect runtime dependencies for ${object}:" >&2
      echo "${output}" >&2
      exit 1
    fi
    if grep -Fq '=> not found' <<<"${output}"; then
      echo "Unresolved runtime dependency for ${object}:" >&2
      echo "${output}" >&2
      exit 1
    fi
    while read -r name arrow path _rest; do
      if [[ "${arrow}" != "=>" || "${path}" != /* ]]; then
        continue
      fi
      if [[ ! "${name}" =~ ^[A-Za-z0-9._+-]+$ ]]; then
        echo "Unsafe dependency name reported by ldd: ${name}" >&2
        exit 1
      fi
      if ! dependency_is_bundleable "${name}" "${path}"; then
        continue
      fi
      if [[ -n "${copied[${name}]:-}" ]]; then
        continue
      fi
      local resolved
      resolved="$(readlink -f "${path}")"
      if [[ ! -f "${resolved}" ]]; then
        echo "Resolved dependency is not a regular file: ${path}" >&2
        exit 1
      fi
      cp -L "${resolved}" "${BIN_ROOT}/${name}"
      chmod 0644 "${BIN_ROOT}/${name}"
      copied["${name}"]=1
      queue+=("${BIN_ROOT}/${name}")
    done <<<"${output}"
  done
}

validate_runtime_dependency_closure() {
  local library_path="${BIN_ROOT}"
  local -a objects=("${BIN_ROOT}/vla-server.bin")
  shopt -s nullglob
  objects+=("${BIN_ROOT}"/*.so "${BIN_ROOT}"/*.so.*)
  shopt -u nullglob
  local object output
  for object in "${objects[@]}"; do
    if ! output="$(LD_LIBRARY_PATH="${library_path}" ldd "${object}" 2>&1)"; then
      echo "Packaged runtime dependency validation failed for ${object}:" >&2
      echo "${output}" >&2
      exit 1
    fi
    if grep -Fq '=> not found' <<<"${output}"; then
      echo "Packaged runtime has unresolved dependencies for ${object}:" >&2
      echo "${output}" >&2
      exit 1
    fi
  done
}

write_vla_launcher_wrapper() {
  cat >"${BIN_ROOT}/vla-server" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SCRIPT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${SCRIPT_DIR}/vla-server.bin" "$@"
EOF
  chmod +x "${BIN_ROOT}/vla-server"
}

install_vla_server_binary() {
  local source_binary=""
  for candidate in "${BUILD_ROOT}/vla-server" "${BUILD_ROOT}/bin/vla-server"; do
    if [[ -x "${candidate}" ]]; then
      source_binary="${candidate}"
      break
    fi
  done
  if [[ -z "${source_binary}" ]]; then
    echo "Build finished but vla-server was not found under ${BUILD_ROOT}." >&2
    exit 1
  fi
  cp -a "${source_binary}" "${BIN_ROOT}/vla-server.bin"
  chmod 0755 "${BIN_ROOT}/vla-server.bin"
  collect_runtime_dependency_closure "${BIN_ROOT}/vla-server.bin"
  write_vla_launcher_wrapper
  validate_runtime_dependency_closure
}

ensure_vla_root
require_command cmake
require_command pkg-config
require_command protoc
require_command ldd
require_command readlink

if [[ -z "${JOBS}" ]]; then
  JOBS="$(detect_jobs)"
fi

CONFIGURE_ARGS=(
  -S "${VLA_ROOT}"
  -B "${BUILD_ROOT}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DBUILD_SHARED_LIBS=OFF
  -DCMAKE_SKIP_RPATH=ON
  -DGGML_NATIVE=$( [[ ${USE_NATIVE} -eq 1 ]] && printf 'ON' || printf 'OFF' )
  -DGGML_LTO=$( [[ ${ENABLE_LTO} -eq 1 ]] && printf 'ON' || printf 'OFF' )
  -DGGML_CUDA=$( [[ ${ENABLE_CUDA} -eq 1 ]] && printf 'ON' || printf 'OFF' )
)

if [[ -n "${CUDA_ARCHITECTURES}" ]]; then
  CONFIGURE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}")
fi

if [[ -n "${LLAMA_SOURCE}" ]]; then
  CONFIGURE_ARGS+=(-DFETCHCONTENT_SOURCE_DIR_LLAMA="${LLAMA_SOURCE}")
fi

if command -v ninja >/dev/null 2>&1; then
  CONFIGURE_ARGS+=(-G Ninja)
fi

echo "Configuring ${BACKEND_LABEL} build..."
echo "  cmake ${CONFIGURE_ARGS[*]}"
echo "Building vla-server..."
echo "  cmake --build ${BUILD_ROOT} --target vla-server --config ${BUILD_TYPE} -j ${JOBS}"
echo "CPU tuning mode: $( [[ ${USE_NATIVE} -eq 1 ]] && printf 'native' || printf 'portable' )"
echo "CUDA: $( [[ ${ENABLE_CUDA} -eq 1 ]] && printf 'enabled' || printf 'disabled' )"
echo "Link-time optimization: $( [[ ${ENABLE_LTO} -eq 1 ]] && printf 'enabled' || printf 'disabled' )"
if [[ -n "${DEPENDENCY_PREFIX}" ]]; then
  echo "Dependency prefix: ${DEPENDENCY_PREFIX}"
fi
if [[ -n "${LLAMA_SOURCE}" ]]; then
  echo "llama.cpp source: ${LLAMA_SOURCE}"
fi
if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  echo "Cleaning previous build directory: ${BUILD_ROOT}"
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  exit 0
fi

prepare_runtime_dirs

if [[ ${CLEAN_BUILD} -eq 1 ]]; then
  rm -rf "${BUILD_ROOT}"
fi
mkdir -p "${BUILD_ROOT}"

cmake "${CONFIGURE_ARGS[@]}"
cmake --build "${BUILD_ROOT}" --target vla-server --config "${BUILD_TYPE}" -j "${JOBS}"

find "${BIN_ROOT}" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} +
install_vla_server_binary

if [[ ! -x "${BIN_ROOT}/vla-server" ]]; then
  echo "Build finished but vla-server launcher was not installed into ${BIN_ROOT}." >&2
  exit 1
fi

if [[ ${SMOKE_TEST} -eq 1 ]]; then
  echo "Running smoke test..."
  "${BIN_ROOT}/vla-server" --help >/dev/null
fi

echo
echo "${BACKEND_LABEL} build complete."
echo "Binary package location: ${BIN_ROOT}"
echo "Models directory: ${MODELS_ROOT}"
echo "Next step:"
echo "  ./omniinfer backend select ${BACKEND_ID}"
echo "  ./omniinfer model load -m /absolute/path/to/vla-checkpoint.gguf"
