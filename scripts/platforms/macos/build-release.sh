#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOCAL_RUNTIME_ROOT="${REPO_ROOT}/.local/runtime/macos"
MLX_REQUIREMENTS_FILE="${SCRIPT_DIR}/mlx-mac/requirements.txt"

PACKAGE_NAME="OmniInfer"
PLATFORM_TAG=""
BUILD_TYPE="Release"
JOBS=""
CLEAN_BUILD=0
BOOTSTRAP_SUBMODULE=1
SMOKE_TEST=0
BUILD_MISSING=1
ALL_SUPPORTED=0
MLX_PYTHON=""
MLX_ENV_MANAGER="auto"
PYTHON_INDEX_URL="${PYTHON_INDEX_URL:-}"
CONDA_OVERRIDE_CHANNELS=0
SLIM_RELEASE=1
DRY_RUN=0
REQUESTED_BACKENDS=()
CONDA_CHANNELS=()

SUPPORTED_BACKENDS=(
  "llama.cpp-mac"
  "llama.cpp-mac-intel"
  "turboquant-mac"
  "mlx-mac"
)

usage() {
  cat <<'EOF'
Usage: build-release.sh [options]

Options:
  --package-name <name>       Release directory name (default: OmniInfer)
  --platform-tag <tag>        Output platform tag (default: macos-arm64 or macos-x64)
  --backends <ids>            Comma-separated backend ids to package
  --backend <id>              Backend id to package; can be passed multiple times
  --all-supported             Package all supported macOS backends, building missing ones
  --no-build-missing          Fail if a requested backend is not already built
  --build-type <type>         CMake build type, default: Release
  --jobs <n>                  Parallel build jobs passed to compiled backend scripts
  --clean                     Remove previous backend build dirs before compiling
  --no-bootstrap              Do not auto-initialize backend submodules
  --smoke-test                Run backend smoke tests after compiling missing backends
  --mlx-python <path>         Python 3.10+ interpreter used to build mlx-mac
  --mlx-env-manager <manager> Build mlx-mac release Python env with auto, uv, venv, or conda-pack (default: auto)
  --python-index-url <url>    Python package index URL for MLX dependency installation
  --conda-channel <channel>   Conda channel used by conda-pack mode; can be passed multiple times
  --conda-override-channels   Use only channels passed with --conda-channel
  --no-slim                   Keep test files, bytecode, and build-only Python assets in the release
  --dry-run                   Print actions without executing packaging steps
  -h, --help                  Show this help message

Supported backends:
  llama.cpp-mac, llama.cpp-mac-intel, turboquant-mac, mlx-mac
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_cmd() {
  echo "+ $*"
  if [[ ${DRY_RUN} -eq 0 ]]; then
    "$@"
  fi
}

contains_backend() {
  local needle="$1"
  local item
  shift
  for item in "$@"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

append_backend() {
  local backend="$1"
  [[ -z "${backend}" ]] && return
  if [[ ${#REQUESTED_BACKENDS[@]} -gt 0 ]]; then
    if contains_backend "${backend}" "${REQUESTED_BACKENDS[@]}"; then
      return
    fi
  fi
  REQUESTED_BACKENDS+=("${backend}")
}

append_backend_list() {
  local raw="$1"
  local item
  local old_ifs="${IFS}"
  IFS=","
  for item in ${raw}; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    append_backend "${item}"
  done
  IFS="${old_ifs}"
}

detect_platform_tag() {
  local machine
  machine="$(uname -m 2>/dev/null || echo unknown)"
  case "${machine}" in
    arm64|aarch64) echo "macos-arm64" ;;
    x86_64|amd64) echo "macos-x64" ;;
    *) echo "macos-${machine}" ;;
  esac
}

host_default_backend() {
  local machine
  machine="$(uname -m 2>/dev/null || echo unknown)"
  case "${machine}" in
    x86_64|amd64) echo "llama.cpp-mac-intel" ;;
    *) echo "llama.cpp-mac" ;;
  esac
}

build_script_for_backend() {
  case "$1" in
    llama.cpp-mac) echo "${SCRIPT_DIR}/build-llama-mac.sh" ;;
    llama.cpp-mac-intel) echo "${SCRIPT_DIR}/build-llama-mac-intel.sh" ;;
    turboquant-mac) echo "${SCRIPT_DIR}/build-turboquant-mac.sh" ;;
    mlx-mac) echo "${SCRIPT_DIR}/build-mlx-mac.sh" ;;
    *) return 1 ;;
  esac
}

runtime_ready() {
  local backend="$1"
  local root="${LOCAL_RUNTIME_ROOT}/${backend}"
  case "${backend}" in
    mlx-mac)
      [[ -x "${root}/venv/bin/python3" ]]
      ;;
    *)
      [[ -x "${root}/bin/llama-server" ]]
      ;;
  esac
}

python_supports_mlx_release() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)
PY
}

pick_mlx_release_python() {
  local candidate
  if [[ -n "${MLX_PYTHON}" ]]; then
    [[ -x "${MLX_PYTHON}" ]] || die "Configured --mlx-python is not executable: ${MLX_PYTHON}"
    python_supports_mlx_release "${MLX_PYTHON}" || die "--mlx-python must be Python 3.10, 3.11, 3.12, or 3.13."
    echo "${MLX_PYTHON}"
    return
  fi

  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 && python_supports_mlx_release "${candidate}"; then
      command -v "${candidate}"
      return
    fi
  done

  die "mlx-mac release packaging requires Python 3.10 through 3.13. Pass --mlx-python <path> if needed."
}

create_mlx_release_venv() {
  local python_bin="$1"
  local venv_root="$2"

  case "${MLX_ENV_MANAGER}" in
    auto)
      if command -v uv >/dev/null 2>&1; then
        create_mlx_release_uv_venv "${python_bin}" "${venv_root}"
      else
        create_mlx_release_stdlib_venv "${python_bin}" "${venv_root}"
      fi
      ;;
    uv)
      create_mlx_release_uv_venv "${python_bin}" "${venv_root}"
      ;;
    venv)
      create_mlx_release_stdlib_venv "${python_bin}" "${venv_root}"
      ;;
    conda-pack)
      create_mlx_release_conda_pack "${python_bin}" "${venv_root}"
      ;;
    *)
      die "Unsupported --mlx-env-manager '${MLX_ENV_MANAGER}'. Supported: auto, uv, venv, conda-pack."
      ;;
  esac
}

create_mlx_release_uv_venv() {
  local python_bin="$1"
  local venv_root="$2"
  local uv_pip_args=()

  require_command uv "Install uv or pass --mlx-env-manager venv/conda-pack."
  "${python_bin}" -m venv --copies "${venv_root}"
  uv_pip_args=(--python "${venv_root}/bin/python")
  [[ -n "${PYTHON_INDEX_URL}" ]] && uv_pip_args+=(--default-index "${PYTHON_INDEX_URL}")
  uv pip install "${uv_pip_args[@]}" -r "${MLX_REQUIREMENTS_FILE}"
}

create_mlx_release_stdlib_venv() {
  local python_bin="$1"
  local venv_root="$2"
  local pip_args=()

  "${python_bin}" -m venv --copies "${venv_root}"
  "${venv_root}/bin/python" -m pip install --upgrade pip setuptools wheel
  [[ -n "${PYTHON_INDEX_URL}" ]] && pip_args+=(--index-url "${PYTHON_INDEX_URL}")
  "${venv_root}/bin/python" -m pip install "${pip_args[@]}" -r "${MLX_REQUIREMENTS_FILE}"
}

conda_pack_available() {
  command -v conda-pack >/dev/null 2>&1 && return 0
  conda run -n base python -c "import conda_pack" >/dev/null 2>&1
}

run_conda_pack() {
  if command -v conda-pack >/dev/null 2>&1; then
    conda-pack "$@"
    return
  fi
  conda run -n base python -m conda_pack.cli "$@"
}

create_mlx_release_conda_pack() {
  local python_bin="$1"
  local venv_root="$2"
  local pip_args=()
  local conda_create_args=()
  local python_version
  local conda_env_root="${BUILD_ROOT}/mlx-mac-conda-env"
  local conda_archive="${BUILD_ROOT}/mlx-mac-conda-env.tar.gz"

  require_command conda "Install Miniconda/Anaconda first."
  if ! conda_pack_available; then
    die "conda-pack is required for --mlx-env-manager conda-pack. Install it in base with: conda install -n base -c conda-forge conda-pack"
  fi

  python_version="$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  rm -rf "${conda_env_root}" "${conda_archive}" "${venv_root}"

  if [[ ${CONDA_OVERRIDE_CHANNELS} -eq 1 ]]; then
    conda_create_args+=(--override-channels)
  fi
  for channel in "${CONDA_CHANNELS[@]}"; do
    conda_create_args+=(-c "${channel}")
  done

  conda create -y -p "${conda_env_root}" "${conda_create_args[@]}" "python=${python_version}" pip
  [[ -n "${PYTHON_INDEX_URL}" ]] && pip_args+=(--index-url "${PYTHON_INDEX_URL}")
  conda run -p "${conda_env_root}" python -m pip install "${pip_args[@]}" -r "${MLX_REQUIREMENTS_FILE}"

  mkdir -p "${venv_root}"
  run_conda_pack -p "${conda_env_root}" -o "${conda_archive}" --force
  tar -xzf "${conda_archive}" -C "${venv_root}"
}

validate_mlx_release_venv() {
  local venv_root="$1"
  local python_path="${venv_root}/bin/python3"
  local python_real
  local venv_real

  [[ -x "${python_path}" ]] || die "mlx-mac release Python is not executable: ${python_path}"
  python_real="$(realpath "${python_path}")"
  venv_real="$(realpath "${venv_root}")"
  case "${python_real}" in
    "${venv_real}"/*) ;;
    *) die "mlx-mac release Python resolves outside the package: ${python_real}" ;;
  esac

  "${python_path}" - <<'PY'
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)
PY
}

remove_site_packages_globs() {
  local site_packages="$1"
  local pattern
  shift
  for pattern in "$@"; do
    find "${site_packages}" -maxdepth 1 -name "${pattern}" -exec rm -rf {} +
  done
}

slim_mlx_release_venv() {
  local venv_root="$1"
  local site_packages

  [[ ${SLIM_RELEASE} -eq 1 ]] || return

  echo "Slimming mlx-mac release Python environment..."
  find "${venv_root}" -type d -name "__pycache__" -prune -exec rm -rf {} +
  find "${venv_root}" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

  for site_packages in "${venv_root}"/lib/python*/site-packages; do
    [[ -d "${site_packages}" ]] || continue

    find "${site_packages}" -type d \( -name tests -o -name test \) -prune -exec rm -rf {} +
    rm -rf \
      "${site_packages}/numpy/_core/tests" \
      "${site_packages}/numpy/f2py/tests" \
      "${site_packages}/numpy/lib/tests" \
      "${site_packages}/numpy/ma/tests" \
      "${site_packages}/numpy/random/tests" \
      "${site_packages}/numpy/typing/tests" \
      "${site_packages}/pandas/tests" \
      "${site_packages}/pyarrow/tests" \
      "${site_packages}/torch/include" \
      "${site_packages}/torch/share/cmake" \
      "${site_packages}/torch/testing"

    remove_site_packages_globs "${site_packages}" \
      "pip" \
      "pip-*.dist-info" \
      "wheel" \
      "wheel-*.dist-info"
  done
}

discover_built_backends() {
  local backend
  for backend in "${SUPPORTED_BACKENDS[@]}"; do
    if runtime_ready "${backend}"; then
      echo "${backend}"
    fi
  done
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "Required command '$1' was not found.${2:+ $2}"
  fi
}

ensure_pyinstaller() {
  if ! python3 -c "import PyInstaller" >/dev/null 2>&1; then
    echo "PyInstaller not found. Installing..."
    python3 -m pip install pyinstaller || die "Failed to install PyInstaller."
  fi
}

selected_backends_include_mlx() {
  contains_backend "mlx-mac" "${SELECTED_BACKENDS[@]}"
}

build_backend_if_missing() {
  local backend="$1"
  local script_path
  local args=()

  if runtime_ready "${backend}"; then
    echo "Using existing backend runtime: ${backend}"
    return
  fi

  if [[ ${BUILD_MISSING} -eq 0 ]]; then
    die "Requested backend '${backend}' is not built under ${LOCAL_RUNTIME_ROOT}."
  fi

  script_path="$(build_script_for_backend "${backend}")" || die "Unsupported backend: ${backend}"
  [[ -f "${script_path}" ]] || die "Build script not found for ${backend}: ${script_path}"

  echo "Backend '${backend}' is missing. Building it before packaging..."
  if [[ "${backend}" == "mlx-mac" ]]; then
    [[ -n "${MLX_PYTHON}" ]] && args+=(--python "${MLX_PYTHON}")
  else
    args+=(--build-type "${BUILD_TYPE}")
    [[ -n "${JOBS}" ]] && args+=(--jobs "${JOBS}")
    [[ ${CLEAN_BUILD} -eq 1 ]] && args+=(--clean)
    [[ ${BOOTSTRAP_SUBMODULE} -eq 0 ]] && args+=(--no-bootstrap)
    [[ ${SMOKE_TEST} -eq 1 ]] && args+=(--smoke-test)
  fi
  [[ ${DRY_RUN} -eq 1 ]] && args+=(--dry-run)
  run_cmd bash "${script_path}" "${args[@]}"

  if [[ ${DRY_RUN} -eq 0 ]] && ! runtime_ready "${backend}"; then
    die "Build finished but runtime is still incomplete for ${backend}."
  fi
}

copy_backend_runtime() {
  local backend="$1"
  local source_root="${LOCAL_RUNTIME_ROOT}/${backend}"
  local target_root="${RUNTIME_ROOT}/${backend}"

  rm -rf "${target_root}"
  mkdir -p "${target_root}/logs"

  if [[ "${backend}" == "mlx-mac" ]]; then
    local python_bin
    python_bin="$(pick_mlx_release_python)"
    [[ -f "${MLX_REQUIREMENTS_FILE}" ]] || die "mlx-mac requirements file not found: ${MLX_REQUIREMENTS_FILE}"
    echo "Creating mlx-mac release venv with ${python_bin}..."
    create_mlx_release_venv "${python_bin}" "${target_root}/venv"
    slim_mlx_release_venv "${target_root}/venv"
    validate_mlx_release_venv "${target_root}/venv"
    return
  fi

  [[ -x "${source_root}/bin/llama-server" ]] || die "llama-server not found for ${backend}: ${source_root}/bin"
  mkdir -p "${target_root}/bin"
  cp -a "${source_root}/bin/." "${target_root}/bin/"
  chmod +x "${target_root}/bin/llama-server"
}

copy_source_entrypoint() {
  cp "${REPO_ROOT}/omniinfer.py" "${RELEASE_ROOT}/omniinfer.py"
  rm -rf "${RELEASE_ROOT}/service_core"
  mkdir -p "${RELEASE_ROOT}/service_core"
  cp -a "${REPO_ROOT}/service_core/." "${RELEASE_ROOT}/service_core/"
  find "${RELEASE_ROOT}/service_core" -type d -name "__pycache__" -prune -exec rm -rf {} +
  find "${RELEASE_ROOT}/service_core" -type f -name "*.pyc" -delete
}

write_launcher() {
  cat > "${RELEASE_ROOT}/omniinfer" <<'EOF'
#!/bin/sh
set -eu

SCRIPT_PATH="$0"
case "$SCRIPT_PATH" in
  /*) ;;
  *) SCRIPT_PATH="$(pwd)/$SCRIPT_PATH" ;;
esac

ROOT="$(CDPATH= cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
MLX_PYTHON="$ROOT/runtime/mlx-mac/venv/bin/python3"
CONDA_UNPACK="$ROOT/runtime/mlx-mac/venv/bin/conda-unpack"

if [ ! -x "$MLX_PYTHON" ]; then
  echo "mlx-mac Python runtime was not found: $MLX_PYTHON" >&2
  exit 1
fi

if [ -x "$CONDA_UNPACK" ]; then
  "$MLX_PYTHON" "$CONDA_UNPACK" >/dev/null 2>&1 || {
    echo "Failed to relocate mlx-mac conda-pack runtime: $CONDA_UNPACK" >&2
    exit 1
  }
fi

exec "$MLX_PYTHON" "$ROOT/omniinfer.py" "$@"
EOF
  chmod +x "${RELEASE_ROOT}/omniinfer"
}

build_pyinstaller_cli() {
  require_command python3 "Install Python 3.10+ first."
  ensure_pyinstaller

  local cli_entry="${REPO_ROOT}/omniinfer.py"
  [[ -f "${cli_entry}" ]] || die "CLI entry point not found: ${cli_entry}"

  local pyinstaller_excludes=(
    "cv2"
    "matplotlib"
    "mkl"
    "numpy"
    "pandas"
    "PIL"
    "scipy"
    "sklearn"
    "sympy"
    "torch"
    "torchvision"
  )

  local pyinstaller_args=(
    --noconfirm
    --clean
    --onedir
    --console
    --name "omniinfer"
    --distpath "${CLI_DIST}"
    --workpath "${BUILD_ROOT}/pyinstaller-work-cli"
    --specpath "${BUILD_ROOT}/pyinstaller-spec-cli"
    --add-data "${MODEL_CATALOG_ROOT}:service_core/model_catalogs"
  )
  local exclude
  for exclude in "${pyinstaller_excludes[@]}"; do
    pyinstaller_args+=(--exclude-module "${exclude}")
  done
  pyinstaller_args+=("${cli_entry}")

  echo
  echo "Building omniinfer (CLI) with PyInstaller..."
  python3 -m PyInstaller "${pyinstaller_args[@]}"

  local cli_bin="${CLI_DIST}/omniinfer/omniinfer"
  [[ -f "${cli_bin}" ]] || die "CLI build succeeded but omniinfer not found at ${cli_bin}"

  cp -a "${CLI_DIST}/omniinfer/." "${RELEASE_ROOT}/"
}

install_mlx_source_cli() {
  echo
  echo "Installing source CLI launcher for mlx-mac..."
  copy_source_entrypoint
  write_launcher
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
    --backends)
      append_backend_list "${2:?missing value for --backends}"
      shift 2
      ;;
    --backend)
      append_backend "${2:?missing value for --backend}"
      shift 2
      ;;
    --all-supported)
      ALL_SUPPORTED=1
      shift
      ;;
    --no-build-missing)
      BUILD_MISSING=0
      shift
      ;;
    --build-type)
      BUILD_TYPE="${2:?missing value for --build-type}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:?missing value for --jobs}"
      shift 2
      ;;
    --clean)
      CLEAN_BUILD=1
      shift
      ;;
    --no-bootstrap)
      BOOTSTRAP_SUBMODULE=0
      shift
      ;;
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    --mlx-python)
      MLX_PYTHON="${2:?missing value for --mlx-python}"
      shift 2
      ;;
    --mlx-env-manager)
      MLX_ENV_MANAGER="${2:?missing value for --mlx-env-manager}"
      shift 2
      ;;
    --python-index-url)
      PYTHON_INDEX_URL="${2:?missing value for --python-index-url}"
      shift 2
      ;;
    --conda-channel)
      CONDA_CHANNELS+=("${2:?missing value for --conda-channel}")
      shift 2
      ;;
    --conda-override-channels)
      CONDA_OVERRIDE_CHANNELS=1
      shift
      ;;
    --no-slim)
      SLIM_RELEASE=0
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

[[ -z "${PLATFORM_TAG}" ]] && PLATFORM_TAG="$(detect_platform_tag)"

RELEASE_ROOT="${REPO_ROOT}/release/portable/${PLATFORM_TAG}/${PACKAGE_NAME}"
BUILD_ROOT="${REPO_ROOT}/release/build"
CLI_DIST="${BUILD_ROOT}/macos-cli-dist"
RUNTIME_ROOT="${RELEASE_ROOT}/runtime"
CONFIG_ROOT="${RELEASE_ROOT}/config"
MODEL_CATALOG_ROOT="${REPO_ROOT}/service_core/model_catalogs"

if [[ ${ALL_SUPPORTED} -eq 1 ]]; then
  SELECTED_BACKENDS=("${SUPPORTED_BACKENDS[@]}")
elif [[ ${#REQUESTED_BACKENDS[@]} -gt 0 ]]; then
  SELECTED_BACKENDS=("${REQUESTED_BACKENDS[@]}")
else
  SELECTED_BACKENDS=()
  while IFS= read -r backend; do
    SELECTED_BACKENDS+=("${backend}")
  done < <(discover_built_backends)
  if [[ ${#SELECTED_BACKENDS[@]} -eq 0 ]]; then
    SELECTED_BACKENDS=("$(host_default_backend)")
  fi
fi

for backend in "${SELECTED_BACKENDS[@]}"; do
  if ! contains_backend "${backend}" "${SUPPORTED_BACKENDS[@]}"; then
    die "Unsupported macOS backend '${backend}'. Supported: ${SUPPORTED_BACKENDS[*]}"
  fi
done

for backend in "${SELECTED_BACKENDS[@]}"; do
  build_backend_if_missing "${backend}"
done

DEFAULT_BACKEND="${SELECTED_BACKENDS[0]}"
for candidate in "$(host_default_backend)" "turboquant-mac" "mlx-mac" "llama.cpp-mac" "llama.cpp-mac-intel"; do
  if contains_backend "${candidate}" "${SELECTED_BACKENDS[@]}"; then
    DEFAULT_BACKEND="${candidate}"
    break
  fi
done

echo "Packaged ${#SELECTED_BACKENDS[@]} backend(s): ${SELECTED_BACKENDS[*]}"
echo "Default backend: ${DEFAULT_BACKEND}"
echo "Package root: ${RELEASE_ROOT}"
if selected_backends_include_mlx; then
  echo "CLI mode: source launcher with mlx-mac venv"
  echo "MLX env manager: ${MLX_ENV_MANAGER}"
  echo "Slim release: $([[ ${SLIM_RELEASE} -eq 1 ]] && echo yes || echo no)"
else
  echo "CLI mode: PyInstaller binary"
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "Dry run enabled. Release packaging was not executed."
  exit 0
fi

rm -rf "${RELEASE_ROOT}" "${BUILD_ROOT}"
mkdir -p "${RELEASE_ROOT}" "${RUNTIME_ROOT}" "${CONFIG_ROOT}" "${BUILD_ROOT}"

if selected_backends_include_mlx; then
  install_mlx_source_cli
else
  build_pyinstaller_cli
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

COPIED_BACKENDS=()
for backend in "${SELECTED_BACKENDS[@]}"; do
  copy_backend_runtime "${backend}"
  COPIED_BACKENDS+=("${backend}")
done

USAGE_TEMPLATE="${REPO_ROOT}/tmp/usage.md"
if [[ -f "${USAGE_TEMPLATE}" ]]; then
  cp "${USAGE_TEMPLATE}" "${RELEASE_ROOT}/README.md"
fi

echo
echo "============================================"
echo "Portable release ready."
echo "  Location:  ${RELEASE_ROOT}"
echo "  Platform:  ${PLATFORM_TAG}"
echo "  Backends:  ${COPIED_BACKENDS[*]}"
echo "  Default:   ${DEFAULT_BACKEND}"
echo "============================================"
echo
echo "Run with:"
echo "  ${RELEASE_ROOT}/omniinfer backend list"
echo "  ${RELEASE_ROOT}/omniinfer chat \"Hello\""
