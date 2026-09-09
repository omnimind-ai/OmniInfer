#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUNTIME_HOME="${OMNIINFER_VLA_RUNTIME_HOME:-}"
VLA_PYTHON="${OMNIINFER_VLA_RUNTIME_PYTHON:-${RUNTIME_HOME}/.venv/bin/python}"
PACKAGE_ROOT="${REPO_ROOT}/.local/runtime/linux/omniinfer-vla-linux-cuda"
CHECK_ONLY=0

usage() {
    cat <<'EOF'
Usage: build.sh [--runtime-home <path>] [--python <path>] [--check]

Creates the managed OmniInfer VLA Runtime launcher. The runtime home must
contain the provisioned VLA Python environment, the server adapter, and vla.proto.

Environment:
  OMNIINFER_VLA_RUNTIME_HOME    Runtime source home
  OMNIINFER_VLA_RUNTIME_PYTHON  Python executable used by the runtime
EOF
}

discover_runtime_home() {
    local candidate
    for candidate in "${REPO_ROOT}"/framework/*; do
        if [[ -f "${candidate}/omniinfer_server.py" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

while (($# > 0)); do
    case "$1" in
        --runtime-home)
            RUNTIME_HOME="${2:?missing value for --runtime-home}"
            shift 2
            ;;
        --python)
            VLA_PYTHON="${2:?missing value for --python}"
            shift 2
            ;;
        --check)
            CHECK_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${RUNTIME_HOME}" ]]; then
    RUNTIME_HOME="$(discover_runtime_home)" || {
        printf 'Set OMNIINFER_VLA_RUNTIME_HOME to the provisioned VLA runtime home.\n' >&2
        exit 1
    }
fi
if [[ -z "${OMNIINFER_VLA_RUNTIME_PYTHON:-}" ]]; then
    VLA_PYTHON="${RUNTIME_HOME}/.venv/bin/python"
fi

[[ -f "${RUNTIME_HOME}/omniinfer_server.py" ]] || {
    printf 'Missing server adapter in runtime home: %s\n' "${RUNTIME_HOME}" >&2
    exit 1
}
[[ -f "${RUNTIME_HOME}/vla.proto" ]] || {
    printf 'Missing vla.proto in runtime home: %s\n' "${RUNTIME_HOME}" >&2
    exit 1
}
[[ -f "${RUNTIME_HOME}/vla_pb2.py" ]] || {
    printf 'Missing bundled protobuf module in runtime home: %s\n' "${RUNTIME_HOME}" >&2
    exit 1
}
[[ -x "${VLA_PYTHON}" ]] || {
    printf 'Runtime Python is not executable: %s\n' "${VLA_PYTHON}" >&2
    exit 1
}

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    printf 'OmniInfer VLA Runtime prerequisites are available.\n'
    exit 0
fi

install -d "${PACKAGE_ROOT}/bin" "${PACKAGE_ROOT}/logs" "${PACKAGE_ROOT}/THIRD_PARTY_LICENSES"
install -m 0755 "${SCRIPT_DIR}/omniinfer-vla-server" "${PACKAGE_ROOT}/bin/omniinfer-vla-server"
install -m 0644 "${RUNTIME_HOME}/omniinfer_server.py" "${PACKAGE_ROOT}/omniinfer_vla_server.py"
install -m 0644 "${RUNTIME_HOME}/vla.proto" "${PACKAGE_ROOT}/vla.proto"
install -m 0644 "${RUNTIME_HOME}/vla_pb2.py" "${PACKAGE_ROOT}/vla_pb2.py"

printf 'Built OmniInfer VLA Runtime at %s\n' "${PACKAGE_ROOT}"
