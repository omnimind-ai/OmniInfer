#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  OmniInfer source installer for macOS / Linux
#
#  Usage:
#    curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install-from-source.sh | bash
#    curl -fsSL ... | bash -s -- --install-dir ~/my-omniinfer
#    curl -fsSL ... | bash -s -- --model /path/to/model.gguf
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────

INSTALL_DIR="$(pwd)/OmniInfer"
MODEL_PATH=""
NO_MODEL=0
SKIP_BUILD=0
PREBUILT_MODE=1
BACKEND_OVERRIDE=""
NON_INTERACTIVE=0
INSTALL_SYSTEM_DEPS="ask"
REPO_SSH="git@github.com:omnimind-ai/OmniInfer.git"
REPO_HTTPS="https://github.com/omnimind-ai/OmniInfer.git"

# ── Parse args ──────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)    INSTALL_DIR="$2";       shift 2 ;;
        --model|-m)       MODEL_PATH="$2";        shift 2 ;;
        --no-model)       NO_MODEL=1;             shift   ;;
        --skip-build)     SKIP_BUILD=1;           shift   ;;
        --prebuilt)       PREBUILT_MODE=1;         shift   ;;
        --from-source)    PREBUILT_MODE=0;         shift   ;;
        --backend)        BACKEND_OVERRIDE="$2";  shift 2 ;;
        --non-interactive) NON_INTERACTIVE=1;      shift   ;;
        --install-system-deps) INSTALL_SYSTEM_DEPS="yes"; shift ;;
        --no-install-system-deps) INSTALL_SYSTEM_DEPS="no"; shift ;;
        --help|-h)
            cat <<'HELP'
OmniInfer Source Installer

Usage:
  curl -fsSL <url>/install-from-source.sh | bash
  bash install-from-source.sh [OPTIONS]

Options:
  --install-dir DIR     Installation directory (default: ~/OmniInfer)
  --model, -m PATH      Path to a local GGUF model file or directory
  --no-model            Skip model setup without prompting
  --skip-build          Skip the backend build step
  --prebuilt            Install the selected backend from a configured prebuilt archive
  --from-source         Build the selected backend from source instead of using prebuilt install
  --backend ID          Force a specific backend (e.g. llama.cpp-linux-vulkan)
  --non-interactive     Do not prompt; fail with instructions if dependencies are missing
  --install-system-deps Automatically install missing system packages with sudo
  --no-install-system-deps
                        Never install system packages automatically
  -h, --help            Show this help

CUDA backends:
  CUDA builds require cuBLAS development files such as cublas_v2.h and libcublas.so.
  Runtime-only/private libraries, including Ollama's bundled cuBLAS libraries, are not
  enough. You can install the recommended system package or set CUDAToolkit_ROOT/CUDA_HOME
  to a complete CUDA toolkit.
  Without sudo, try: bash scripts/install-cuda-cublas-local.sh
HELP
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "${NO_MODEL}" -eq 1 ]] && [[ -n "${MODEL_PATH}" ]]; then
    echo "Cannot use --model and --no-model together" >&2
    exit 1
fi

BUILD_LOG_PATH=""
BUILD_STATUS="not-run"
SUMMARY_PATH=""
CUDA_EFFECTIVE_ARCH=""

# ── Helpers ─────────────────────────────────────────────────

info()    { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok()      { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()    { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
err()     { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*"; }
fatal()   { err "$*"; exit 1; }
bold()    { printf '\033[1m%s\033[0m' "$*"; }

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        fatal "'$1' is required but not found. $2"
    fi
    ok "$1"
}

# Run omniinfer CLI with the correct port
# OMNI_PORT must be set before calling this function
omniinfer_cmd() {
    "${INSTALL_DIR}/omniinfer" --port "${OMNI_PORT}" "$@"
}

# Resolve the TTY file descriptor for interactive input.
# Works in both normal terminal and curl|bash piped mode.
INPUT_TTY=""
resolve_tty() {
    if [[ -n "${INPUT_TTY}" ]]; then return; fi
    if [[ -t 0 ]]; then
        INPUT_TTY="/dev/stdin"
    elif [[ -e /dev/tty ]]; then
        INPUT_TTY="/dev/tty"
    fi
}

# Arrow-key menu selector.
#   select_menu <default_index> <label1> <label2> ...
# Prints the selected 0-based index to stdout.
select_menu() {
    local default="$1"; shift
    local options=("$@")
    local count=${#options[@]}
    local cur=${default}

    resolve_tty
    if [[ -z "${INPUT_TTY}" ]] || [[ "${NON_INTERACTIVE}" -eq 1 ]]; then
        echo "${default}"
        return
    fi

    # Read escape sequence continuation byte.
    # Bash 3 (macOS default) doesn't support fractional -t, so we use a
    # simple non-blocking read: try -t 1 which is fine because escape
    # sequence bytes arrive together in the same buffer.
    _read_seq() {
        IFS= read -rsn1 -t 1 "$1" < "${INPUT_TTY}" 2>/dev/null || eval "$1="
    }

    # Hide cursor
    printf '\033[?25l' >&2

    # Ensure cursor is restored on exit (Ctrl-C, etc.)
    trap 'printf "\033[?25h" >&2' EXIT

    # Draw menu
    _draw_menu() {
        for i in "${!options[@]}"; do
            if [[ "$i" -eq "${cur}" ]]; then
                printf '\033[1;36m  > %s\033[0m\n' "${options[$i]}" >&2
            else
                printf '    %s\n' "${options[$i]}" >&2
            fi
        done
    }

    _draw_menu

    # Read keys
    while true; do
        local key
        IFS= read -rsn1 key < "${INPUT_TTY}" || break
        if [[ "${key}" == $'\x1b' ]]; then
            local seq1 seq2
            _read_seq seq1
            if [[ "${seq1}" == "[" ]]; then
                _read_seq seq2
                case "${seq2}" in
                    A) (( cur > 0 )) && (( cur-- )) ;;            # Up
                    B) (( cur < count - 1 )) && (( cur++ )) ;;    # Down
                esac
            fi
        elif [[ "${key}" == "" ]]; then
            # Enter
            break
        fi
        # Move cursor up and redraw
        printf '\033[%dA\r' "${count}" >&2
        _draw_menu
    done

    # Show cursor
    printf '\033[?25h' >&2
    trap - EXIT

    echo "${cur}"
}

# Simple text prompt. Falls back to default in non-interactive / piped mode.
prompt_input() {
    local prompt_text="$1" default="$2" result
    if [[ "${NON_INTERACTIVE}" -eq 1 ]]; then
        echo "${default}"
        return
    fi
    resolve_tty
    if [[ -z "${INPUT_TTY}" ]]; then
        echo "${default}"
        return
    fi
    printf '%s' "${prompt_text}" >&2
    IFS= read -r result < "${INPUT_TTY}" || result=""
    echo "${result:-${default}}"
}

run_python() {
    if [[ "${PYTHON_CMD:-}" == "uv run python3" ]]; then
        uv run python3 "$@"
    else
        "${PYTHON_CMD:-python3}" "$@"
    fi
}

backend_has_prebuilt() {
    local backend_id="$1"
    local catalog_path="${INSTALL_DIR}/scripts/prebuilt_backends.json"
    [[ -f "${catalog_path}" ]] || return 1
    run_python - "${catalog_path}" "${PLATFORM_DIR}" "${backend_id}" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

catalog = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
platform = sys.argv[2]
backend = sys.argv[3]
entry = catalog.get("platforms", {}).get(platform, {}).get(backend)
raise SystemExit(0 if isinstance(entry, dict) and entry.get("url") else 1)
PY
}

detect_cuda_effective_arch() {
    if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
        echo "${CMAKE_CUDA_ARCHITECTURES}"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local compute_cap
        compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:].')" || compute_cap=""
        if [[ -n "${compute_cap}" ]]; then
            echo "${compute_cap}"
            return
        fi
    fi
    echo ""
}

write_install_summary() {
    SUMMARY_PATH="${INSTALL_DIR}/.local/install-summary.json"
    mkdir -p "$(dirname "${SUMMARY_PATH}")"
    export INSTALL_DIR
    export SELECTED_BACKEND="${SELECTED_BACKEND:-}"
    export MODEL_PATH="${MODEL_PATH:-}"
    export MODEL_CONFIGURED="${MODEL_CONFIGURED:-0}"
    export OMNI_PORT="${OMNI_PORT:-}"
    export SKIP_BUILD
    export BUILD_STATUS
    export BUILD_LOG_PATH
    export CUDA_EFFECTIVE_ARCH
    export SUMMARY_PATH
    run_python - <<'PY'
import json
import os
from pathlib import Path

summary = {
    "install_dir": os.environ.get("INSTALL_DIR", ""),
    "backend": os.environ.get("SELECTED_BACKEND", ""),
    "model_configured": os.environ.get("MODEL_CONFIGURED") == "1",
    "model_path": os.environ.get("MODEL_PATH") or None,
    "port": int(os.environ["OMNI_PORT"]) if os.environ.get("OMNI_PORT", "").isdigit() else None,
    "skip_build": os.environ.get("SKIP_BUILD") == "1",
    "build_status": os.environ.get("BUILD_STATUS", "not-run"),
    "build_log": os.environ.get("BUILD_LOG_PATH") or None,
    "cuda_effective_arch": os.environ.get("CUDA_EFFECTIVE_ARCH") or None,
}
path = Path(os.environ["SUMMARY_PATH"])
path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
    ok "Install summary written: ${SUMMARY_PATH}"
}

# ── Banner ──────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              OmniInfer Source Installer                  ║"
echo "║       Local LLM/VLM inference on every device            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Detect Android early ────────────────────────────────────

IS_ANDROID=0
if command -v getprop >/dev/null 2>&1; then
    if getprop ro.build.version.release >/dev/null 2>&1; then
        IS_ANDROID=1
    fi
fi

if [[ "${IS_ANDROID}" -eq 1 ]]; then
    fatal "Android/Termux installation via scripts/install-from-source.sh is no longer supported. Use the root android/ Gradle module instead."
fi

# ── Step 1: Check prerequisites ─────────────────────────────

info "Step 1/6: Checking prerequisites ..."
need_cmd git    "Install from https://git-scm.com/"
need_cmd cmake  "macOS: brew install cmake  |  Linux: apt install cmake"
PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
    ok "python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
    ok "python"
elif command -v uv >/dev/null 2>&1 && uv run python3 --version >/dev/null 2>&1; then
    PYTHON_CMD="uv run python3"
    ok "python3 (via uv)"
else
    fatal "'python3' is required but not found. Install from https://python.org/ or use uv: https://docs.astral.sh/uv/"
fi
need_cmd curl   "Install curl for your platform"
# C/C++ compiler (needed for building backends)
if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || command -v clang >/dev/null 2>&1; then
    ok "C++ compiler"
else
    fatal "No C/C++ compiler found. Install one of:
  macOS:   xcode-select --install
  Ubuntu:  sudo apt install build-essential"
fi
echo ""

# ── Step 2: Clone or update repo ────────────────────────────

info "Step 2/6: Preparing repository ..."
if [ -d "${INSTALL_DIR}/.git" ]; then
    info "Found existing clone at ${INSTALL_DIR}, updating ..."
    _pull_ok=0
    if command -v timeout >/dev/null 2>&1; then
        timeout 15 git -C "${INSTALL_DIR}" pull --ff-only 2>/dev/null && _pull_ok=1
    else
        GIT_SSH_COMMAND="ssh -o ConnectTimeout=10" \
            git -C "${INSTALL_DIR}" pull --ff-only 2>/dev/null && _pull_ok=1
    fi
    if [[ "${_pull_ok}" -eq 0 ]]; then
        warn "Pull failed or timed out (network issue?), continuing with existing code"
    fi
else
    info "Cloning OmniInfer to ${INSTALL_DIR} ..."
    CLONED_VIA_HTTPS=0
    info "Trying SSH (timeout 15s) ..."
    _ssh_ok=0
    if command -v timeout >/dev/null 2>&1; then
        timeout 15 git clone --depth 1 "${REPO_SSH}" "${INSTALL_DIR}" 2>/dev/null && _ssh_ok=1
    else
        # macOS/BSD: no timeout command, use GIT_SSH_COMMAND with ConnectTimeout
        GIT_SSH_COMMAND="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no" \
            git clone --depth 1 "${REPO_SSH}" "${INSTALL_DIR}" 2>/dev/null && _ssh_ok=1
    fi
    if [[ "${_ssh_ok}" -eq 0 ]]; then
        warn "SSH clone failed, falling back to HTTPS ..."
        rm -rf "${INSTALL_DIR}" 2>/dev/null || true
        if ! git clone --depth 1 "${REPO_HTTPS}" "${INSTALL_DIR}"; then
            fatal "git clone failed via both SSH and HTTPS. Check your network connection and try again."
        fi
        CLONED_VIA_HTTPS=1
    fi
    # If cloned via HTTPS, rewrite SSH submodule URLs to HTTPS so submodule init works
    if [[ "${CLONED_VIA_HTTPS}" -eq 1 ]]; then
        git -C "${INSTALL_DIR}" config --local url."https://github.com/".insteadOf "git@github.com:"
    fi
fi
if [[ ! -f "${INSTALL_DIR}/omniinfer" ]]; then
    fatal "Repository clone appears incomplete — omniinfer launcher not found in ${INSTALL_DIR}"
fi
ok "Repository ready at ${INSTALL_DIR}"

INSTALL_DEPS_HELPER="${INSTALL_DIR}/scripts/install-deps.sh"
if [[ -f "${INSTALL_DEPS_HELPER}" ]]; then
    # shellcheck source=scripts/install-deps.sh
    source "${INSTALL_DEPS_HELPER}"
else
    fatal "Installer dependency helper not found: ${INSTALL_DEPS_HELPER}"
fi

# ── Ensure a usable port ────────────────────────────────────
# If default port 9000 is occupied, find a free one and write config.

OMNI_PORT=9000
port_in_use() {
    if command -v ss >/dev/null 2>&1; then
        ss -tlnH "sport = :$1" 2>/dev/null | grep -q .
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$1" -sTCP:LISTEN -t >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tln 2>/dev/null | grep -q ":$1 "
    else
        return 1
    fi
}

# ── Find an available port ────────────────────────────────────
# Try default port 9000 first, check if it's an OmniInfer gateway
# If not or occupied by others, try alternative ports

_ATTEMPT_PORTS=(9000 9001 9002 9003 9004 9005 9010 9020 9050 9100 8900 8800 19000)
_OMNI_PORT_FOUND=""

for _TRY_PORT in "${_ATTEMPT_PORTS[@]}"; do
    if ! port_in_use "${_TRY_PORT}"; then
        _OMNI_PORT="${_TRY_PORT}"
        _OMNI_PORT_FOUND="free"
        break
    fi

    # Port is occupied, check if it's an OmniInfer gateway we can shut down
    info "Port ${_TRY_PORT} is occupied, checking if it's an OmniInfer gateway..."
    _quick_check=$(curl -sS -w "%{http_code}" --connect-timeout 1 --max-time 1 "http://127.0.0.1:${_TRY_PORT}/health" 2>/dev/null || true)
    _is_http=$?

    if [[ ${_is_http} -eq 0 ]]; then
        # Port responds to HTTP, try OmniInfer shutdown API
        _shutdown_response=$(curl -sS -w "\n%{http_code}" --connect-timeout 3 --max-time 10 -X POST "http://127.0.0.1:${_TRY_PORT}/omni/shutdown" 2>/dev/null || true)
        _http_code=$(echo "$_shutdown_response" | tail -1)
        if [[ "$_http_code" == "200" ]] || [[ "$_http_code" == "204" ]]; then
            ok "OmniInfer gateway found on port ${_TRY_PORT}, shutdown requested"
            # Wait for port to be released, max 10 seconds
            _wait_count=0
            while port_in_use "${_TRY_PORT}" && [[ ${_wait_count} -lt 20 ]]; do
                sleep 0.5
                _wait_count=$((_wait_count + 1))
            done
            if ! port_in_use "${_TRY_PORT}"; then
                ok "Port ${_TRY_PORT} is now available"
                _OMNI_PORT="${_TRY_PORT}"
                _OMNI_PORT_FOUND="released"
                break
            fi
        fi
    fi

    warn "Port ${_TRY_PORT} is occupied by another service, trying next port..."
done

if [[ -z "${_OMNI_PORT}" ]]; then
    fatal "Could not find an available port. Tried: ${_ATTEMPT_PORTS[*]}"
fi

OMNI_PORT="${_OMNI_PORT}"

if [[ "${_OMNI_PORT_FOUND}" == "free" ]] && [[ "${OMNI_PORT}" != "9000" ]]; then
    warn "Default port 9000 is occupied, will use alternative port ${OMNI_PORT}"
    info "To start the service on port ${OMNI_PORT}, use: ./omniinfer serve --port ${OMNI_PORT}"
    info "To list all running services, use: ./omniinfer ps"
fi

echo ""

# ── Step 3: Detect platform & choose backend ────────────────

info "Step 3/6: Detecting platform and hardware ..."

OS="$(uname -s)"
ARCH="$(uname -m)"
CUDA_EFFECTIVE_ARCH="$(detect_cuda_effective_arch)"

info "Platform: ${OS} (${ARCH})"
if [[ -n "${CUDA_EFFECTIVE_ARCH}" ]]; then
    info "CUDA effective architecture: ${CUDA_EFFECTIVE_ARCH}"
fi
echo ""

# Get available backends
declare -a BACKEND_IDS=()
declare -a BACKEND_DESCS=()

# Query gateway API for compatible backends (hardware-matched).
# First ensure the service is running, then wait for it to be ready.
omniinfer_cmd status >/dev/null 2>&1 || true
for _i in $(seq 1 30); do
    _health=$(curl -s -m 2 "http://127.0.0.1:${OMNI_PORT}/health" 2>/dev/null) || _health=""
    if echo "${_health}" | grep -q '"status"'; then break; fi
    sleep 1
done

_backends_json=$(curl -sS --connect-timeout 3 --max-time 5 "http://127.0.0.1:${OMNI_PORT}/omni/backends?scope=compatible" 2>/dev/null) || _backends_json=""
_recommended=$(echo "${_backends_json}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('recommended',''))" 2>/dev/null) || _recommended=""

# Parse API response (use process substitution to avoid subshell variable loss).
_parsed=$(echo "${_backends_json}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for b in d.get('data', []):
    print(b['id'] + '|' + b.get('description', ''))
" 2>/dev/null) || _parsed=""

if [[ -n "${_parsed}" ]]; then
    while IFS='|' read -r bid bdesc; do
        [[ -z "${bid}" ]] && continue
        BACKEND_IDS+=("${bid}")
        if [[ -n "${bdesc}" ]]; then
            BACKEND_DESCS+=("${bid}  —  ${bdesc}")
        else
            BACKEND_DESCS+=("${bid}")
        fi
    done <<< "${_parsed}"

    # Move recommended backend to top.
    if [[ -n "${_recommended}" ]] && [[ ${#BACKEND_IDS[@]} -gt 0 ]]; then
        for i in "${!BACKEND_IDS[@]}"; do
            if [[ "${BACKEND_IDS[$i]}" == "${_recommended}" ]] && [[ "$i" -gt 0 ]]; then
                rec_id="${BACKEND_IDS[$i]}"; rec_desc="${BACKEND_DESCS[$i]}"
                unset 'BACKEND_IDS[$i]'; unset 'BACKEND_DESCS[$i]'
                BACKEND_IDS=("${rec_id}" "${BACKEND_IDS[@]}")
                BACKEND_DESCS=("${rec_desc}  (recommended)" "${BACKEND_DESCS[@]}")
                break
            elif [[ "${BACKEND_IDS[$i]}" == "${_recommended}" ]] && [[ "$i" -eq 0 ]]; then
                BACKEND_DESCS[0]="${BACKEND_DESCS[0]}  (recommended)"
                break
            fi
        done
    fi
fi

# Fallback: parse CLI text output if API returned nothing.
if [[ ${#BACKEND_IDS[@]} -eq 0 ]]; then
    while IFS= read -r line; do
        id=$(echo "${line}" | sed -n 's/^[* ]*\([a-zA-Z0-9._-]*\)$/\1/p')
        if [[ -n "${id}" ]]; then
            BACKEND_IDS+=("${id}")
            BACKEND_DESCS+=("${id}")
        fi
        desc=$(echo "${line}" | sed -n 's/^    Description: *//p')
        if [[ -n "${desc}" ]] && [[ ${#BACKEND_IDS[@]} -gt 0 ]]; then
            last_idx=$(( ${#BACKEND_IDS[@]} - 1 ))
            BACKEND_DESCS[$last_idx]="${BACKEND_IDS[$last_idx]}  —  ${desc}"
        fi
    done <<< "$(omniinfer_cmd backend list --scope compatible 2>/dev/null)"
fi

if [[ ${#BACKEND_IDS[@]} -eq 0 ]]; then
    fatal "No backends found. Check your platform support."
fi

# Map OS to platform directory name (needed for dependency check below)
case "${OS}" in
    Darwin) PLATFORM_DIR="macos" ;;
    Linux)  PLATFORM_DIR="linux" ;;
    *)      PLATFORM_DIR="linux" ;;
esac

# Backend selection loop: select → check build deps → re-select if missing
while true; do
    if [[ -n "${BACKEND_OVERRIDE}" ]]; then
        SELECTED_BACKEND="${BACKEND_OVERRIDE}"
    else
        PREBUILT_MODE=0
        _prebuilt_ids=()
        _prebuilt_descs=()
        for _backend_idx in "${!BACKEND_IDS[@]}"; do
            if backend_has_prebuilt "${BACKEND_IDS[$_backend_idx]}"; then
                _prebuilt_ids+=("${BACKEND_IDS[$_backend_idx]}")
                _prebuilt_descs+=("${BACKEND_DESCS[$_backend_idx]}  (prebuilt)")
            fi
        done

        _menu_descs=()
        if [[ ${#_prebuilt_ids[@]} -gt 0 ]]; then
            _menu_descs+=("${_prebuilt_descs[@]}")
        fi
        _source_descs=()
        for _backend_idx in "${!BACKEND_IDS[@]}"; do
            _source_descs+=("${BACKEND_DESCS[$_backend_idx]}  (build from source)")
        done
        _menu_descs+=("${_source_descs[@]}")
        _menu_descs+=("Skip for now  —  install backend manually later")

        echo "  Available backends (arrow keys to move, Enter to select):"
        echo ""

        idx=$(select_menu 0 "${_menu_descs[@]}")

        # Last option = skip
        if [[ "${idx}" -eq $(( ${#_menu_descs[@]} - 1 )) ]]; then
            info "Skipping backend selection. You can install a backend later with:"
            echo "    cd ${INSTALL_DIR} && ./omniinfer backend list --scope compatible"
            echo "    ./omniinfer backend select <backend-id>"
            echo "    bash scripts/platforms/${PLATFORM_DIR}/<backend-id>/build.sh"
            SKIP_BUILD=1
            break
        fi

        if [[ "${idx}" -lt "${#_prebuilt_ids[@]}" ]]; then
            PREBUILT_MODE=1
            SELECTED_BACKEND="${_prebuilt_ids[$idx]}"
        else
            SELECTED_BACKEND="${BACKEND_IDS[$(( idx - ${#_prebuilt_ids[@]} ))]}"
        fi
    fi

    ok "Selected: ${SELECTED_BACKEND}"
    if [[ "${PREBUILT_MODE}" -eq 1 ]]; then
        info "Install mode: prebuilt"
    fi
    echo ""

    # Select backend via CLI.
    omniinfer_cmd backend select "${SELECTED_BACKEND}"

    # ── Pre-build dependency check ──────────────────────────────
    # Verify required build tools BEFORE starting the build.
    # Skip dependency probing when the build step is disabled.
    if [[ "${SKIP_BUILD}" -eq 1 || "${PREBUILT_MODE}" -eq 1 ]]; then
        break
    fi

    _build_script="${INSTALL_DIR}/scripts/platforms/${PLATFORM_DIR}/${SELECTED_BACKEND}/build.sh"
    if [[ ! -f "${_build_script}" ]]; then
        break  # will be caught by Step 4
    fi

    # Detect system package manager (once, before inner loop)
    _pkg_mgr="$(omni_install_deps_detect_pkg_mgr)"

    # Inner loop: check deps → (install → re-check) for the SAME backend
    _deps_satisfied=0
    _system_deps_install_attempted=0
    while true; do
        _dep_output=""
        _dep_rc=0
        _dep_output=$(bash "${_build_script}" --check-deps 2>/dev/null) || _dep_rc=$?
        _missing_count=$(printf '%s' "${_dep_output}" | grep -c '^missing|' || true)

        # If all deps satisfied, or script doesn't support --check-deps, proceed
        if [[ ${_dep_rc} -eq 0 ]] || [[ ${_missing_count} -eq 0 ]]; then
            if [[ ${_dep_rc} -eq 0 ]]; then
                ok "All build dependencies satisfied"
            fi
            _deps_satisfied=1
            break
        fi

        # ── Show formatted missing-dependency report ────────────────
        echo ""
        err "Missing build dependencies for $(bold "${SELECTED_BACKEND}"):"
        echo ""
        printf '  ┌──────────────────────────────────────────────────────────────────┐\n'
        while IFS='|' read -r _ds _dc _dp _dh _dpkg; do
            if [[ "${_ds}" == "missing" ]]; then
                printf '  │  \033[1;31m✗\033[0m  %-16s %s\n' "${_dc}" "${_dp}"
                printf '  │     \033[2m→ %s\033[0m\n' "${_dh}"
            fi
        done <<< "${_dep_output}"
        printf '  └──────────────────────────────────────────────────────────────────┘\n'
        echo ""

        # Collect unique package names from the 5th field
        declare -A _pkgs_seen=()
        _pkg_list=()
        while IFS='|' read -r _ds _dc _dp _dh _dpkg; do
            if [[ "${_ds}" == "missing" ]] && [[ -n "${_dpkg}" ]] && [[ -z "${_pkgs_seen[${_dpkg}]+x}" ]]; then
                _pkg_list+=("${_dpkg}")
                _pkgs_seen["${_dpkg}"]=1
            fi
        done <<< "${_dep_output}"

        if [[ ${#_pkg_list[@]} -gt 0 ]]; then
            omni_install_deps_print_fix "${_pkg_mgr}" "${_pkg_list[@]}"
            echo ""
        fi

        resolve_tty
        _can_prompt=1
        if [[ "${NON_INTERACTIVE}" -eq 1 ]] || [[ -z "${INPUT_TTY}" ]]; then
            _can_prompt=0
        fi

        _deps_policy="$(omni_install_deps_policy "${INSTALL_SYSTEM_DEPS}" "${NON_INTERACTIVE}" "${_can_prompt}")"

        if [[ "${_deps_policy}" == "auto-install" ]]; then
            if [[ ${#_pkg_list[@]} -eq 0 ]] || [[ -z "${_pkg_mgr}" ]]; then
                fatal "Missing dependencies could not be mapped to installable system packages. See the messages above."
            fi
            if [[ "${_system_deps_install_attempted}" -eq 1 ]]; then
                fatal "Dependencies are still missing after attempting system package installation. Run the fix command above, or set CUDAToolkit_ROOT/CUDA_HOME to a complete CUDA toolkit."
            fi
            echo ""
            info "Installing missing system packages: ${_pkg_list[*]} ..."
            echo ""
            _system_deps_install_attempted=1
            if ! omni_install_deps_run "${_pkg_mgr}" "${_pkg_list[@]}"; then
                warn "Installation failed. The package repository may not be configured."
                warn "Follow the fix command above, then re-run the installer."
                if [[ "${NON_INTERACTIVE}" -eq 1 ]]; then
                    exit 1
                fi
            fi
            info "Re-checking dependencies ..."
            continue
        fi

        if [[ "${_deps_policy}" == "fail-disabled" ]]; then
            fatal "Missing dependencies and automatic system package installation is disabled."
        fi

        if [[ "${_deps_policy}" == "fail-noninteractive" ]]; then
            fatal "Missing dependencies in non-interactive mode. Run the fix command above, or set CUDAToolkit_ROOT/CUDA_HOME to a complete CUDA toolkit."
        fi

        # Build menu options — offer install only if we have package names + a package manager
        if [[ ${#_pkg_list[@]} -gt 0 ]] && [[ -n "${_pkg_mgr}" ]]; then
            echo "  What would you like to do?"
            echo ""
            if [[ -n "${BACKEND_OVERRIDE}" ]]; then
                _choice=$(select_menu 0 \
                    "Install missing dependencies now  (sudo ${_pkg_mgr})" \
                    "Exit and install dependencies first")
            else
                _choice=$(select_menu 0 \
                    "Install missing dependencies now  (sudo ${_pkg_mgr})" \
                    "Choose a different backend" \
                    "Exit and install dependencies first")
            fi

            if [[ "${_choice}" -eq 0 ]]; then
                echo ""
                info "Installing missing system packages: ${_pkg_list[*]} ..."
                echo ""
                _install_ok=1
                omni_install_deps_run "${_pkg_mgr}" "${_pkg_list[@]}" || _install_ok=0
                echo ""
                if [[ "${_install_ok}" -eq 0 ]]; then
                    warn "Installation failed. The package repository may not be configured."
                    warn "Follow the fix command above, then re-run the installer."
                fi
                info "Re-checking dependencies ..."
                continue  # inner loop: re-check deps for same backend
            elif [[ "${_choice}" -eq 1 ]] && [[ -z "${BACKEND_OVERRIDE}" ]]; then
                break  # break inner loop → outer loop re-selects backend
            else
                echo ""
                info "Install the missing tools, then re-run the installer."
                exit 1
            fi
        else
            echo "  What would you like to do?"
            echo ""
            if [[ -n "${BACKEND_OVERRIDE}" ]]; then
                _choice=$(select_menu 0 "Exit and install dependencies first")
            else
                _choice=$(select_menu 0 \
                    "Choose a different backend" \
                    "Exit and install dependencies first")
            fi

            if [[ "${_choice}" -eq 0 ]] && [[ -z "${BACKEND_OVERRIDE}" ]]; then
                break  # break inner loop → outer loop re-selects backend
            else
                echo ""
                info "Install the missing tools, then re-run the installer."
                exit 1
            fi
        fi
    done

    # If deps satisfied, break outer loop → proceed to Step 4
    if [[ "${_deps_satisfied}" -eq 1 ]]; then
        break
    fi

    echo ""
done

# ── Step 4: Build backend ───────────────────────────────────
# Build scripts auto-bootstrap their required submodules.
# Build script is discovered by convention: scripts/platforms/<platform_dir>/<backend_id>/build.sh

if [[ "${PREBUILT_MODE}" -eq 1 ]]; then
    info "Step 4/6: Installing prebuilt backend ..."
else
    info "Step 4/6: Building backend ..."
fi

if [[ "${SKIP_BUILD}" -eq 1 ]]; then
    info "Skipping build (--skip-build)"
    BUILD_STATUS="skipped"
else
    RUNTIME_AVAILABLE=$(omniinfer_cmd backend list --scope installed 2>/dev/null | grep -c "^${SELECTED_BACKEND}[[:space:]]" || true)
    if [[ "${RUNTIME_AVAILABLE}" -gt 0 ]]; then
        ok "Backend ${SELECTED_BACKEND} already installed, skipping"
        BUILD_STATUS="already-built"
    else
        if [[ "${PREBUILT_MODE}" -eq 1 ]]; then
            info "Installing prebuilt ${SELECTED_BACKEND} ..."
        else
            info "Building ${SELECTED_BACKEND} (this may take a few minutes) ..."
        fi
        BUILD_LOG_DIR="${INSTALL_DIR}/tmp/test_results/install"
        mkdir -p "${BUILD_LOG_DIR}"
        _build_log_kind="build"
        [[ "${PREBUILT_MODE}" -eq 1 ]] && _build_log_kind="prebuilt"
        BUILD_LOG_PATH="${BUILD_LOG_DIR}/${SELECTED_BACKEND}-${_build_log_kind}-$(date +%Y%m%d-%H%M%S).log"
        info "Build log: ${BUILD_LOG_PATH}"
        set +e
        if [[ "${PREBUILT_MODE}" -eq 1 ]]; then
            omniinfer_cmd backend install "${SELECTED_BACKEND}" --prebuilt 2>&1 | tee "${BUILD_LOG_PATH}"
            _build_rc=${PIPESTATUS[0]}
        else
            FULL_BUILD_SCRIPT="${INSTALL_DIR}/scripts/platforms/${PLATFORM_DIR}/${SELECTED_BACKEND}/build.sh"
            if [[ ! -f "${FULL_BUILD_SCRIPT}" ]]; then
                echo "Build script not found: ${FULL_BUILD_SCRIPT}" >&2
                _build_rc=1
            else
                bash "${FULL_BUILD_SCRIPT}" --from-source 2>&1 | tee "${BUILD_LOG_PATH}"
                _build_rc=${PIPESTATUS[0]}
            fi
        fi
        set -e
        if [[ "${_build_rc}" -ne 0 ]]; then
            echo ""
            BUILD_STATUS="failed"
            write_install_summary
            fatal "Backend install failed (exit code ${_build_rc}). See ${BUILD_LOG_PATH} for details."
        fi
        BUILD_STATUS="built"
        [[ "${PREBUILT_MODE}" -eq 1 ]] && BUILD_STATUS="prebuilt"
        ok "Backend install complete"
    fi
fi
echo ""

# ── Step 5: Model configuration ─────────────────────────────

info "Step 5/6: Model configuration"
echo ""
echo "  How would you like to set up a model?"
echo ""

MODEL_CONFIGURED=0

if [[ -n "${MODEL_PATH}" ]]; then
    # Model provided via --model flag
    info "Using provided model: ${MODEL_PATH}"
    MODEL_CONFIGURED=1
elif [[ "${NO_MODEL}" -eq 1 ]]; then
    info "Skipping model configuration (--no-model)"
else
    model_choice=$(select_menu 0 \
        "Download a recommended model" \
        "Use a local model file" \
        "Skip (configure later)")

    case "${model_choice}" in
        0)
            # ── Download recommended model ──────────────────────
            info "Reading bundled model catalog ..."

            if [[ "${OS}" == "Darwin" ]]; then
                CATALOG_SYSTEM="mac"
            else
                CATALOG_SYSTEM="linux"
            fi

            CATALOG_PATH="${INSTALL_DIR}/crates/omniinfer-core/model_catalogs/${CATALOG_SYSTEM}.json"

            # Use embedded Python to parse catalog and present choices
            MODEL_INFO=$(
python3 - "${CATALOG_PATH}" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path

raw = Path(sys.argv[1]).read_bytes()
data = json.loads(raw.decode('utf-8-sig'))

# Collect small models (< 6 GiB) across all backends
models = []
seen = set()
for backend, families in data.items():
    if not isinstance(families, dict):
        continue
    for fam_name, fam_models in families.items():
        if not isinstance(fam_models, dict):
            continue
        for model_name, model_info in fam_models.items():
            if not isinstance(model_info, dict):
                continue
            quants = model_info.get('quantization', {})
            if not isinstance(quants, dict):
                continue
            # Prefer Q4_K_M
            for qname in ['Q4_K_M', 'Q6_K', 'Q8_0']:
                q = quants.get(qname)
                if not q or not isinstance(q, dict):
                    continue
                dl = q.get('download', '')
                size_str = q.get('size', '0')
                try:
                    size_gib = float(size_str)
                except (ValueError, TypeError):
                    continue
                if size_gib > 6.0 or size_gib < 0.1 or not dl:
                    continue
                key = f'{model_name}|{qname}'
                if key in seen:
                    continue
                seen.add(key)
                models.append((model_name, qname, size_gib, dl))
                break  # one quant per model

# Sort by size
models.sort(key=lambda x: x[2])

# Take top 6
for i, (name, quant, size, url) in enumerate(models[:6]):
    print(f'{i+1}|{name}|{quant}|{size:.2f}|{url}')
PY
            ) || true

            if [[ -z "${MODEL_INFO}" ]]; then
                warn "Could not read bundled model catalog. You can configure a model manually later."
            else
                echo ""
                echo "  Recommended models:"
                echo ""

                # Build menu labels from catalog
                declare -a DL_LABELS=()
                declare -a DL_LINES=()
                while IFS='|' read -r num name quant size url; do
                    DL_LABELS+=("$(printf '%-32s %-10s %s GiB' "${name}" "${quant}" "${size}")")
                    DL_LINES+=("${num}|${name}|${quant}|${size}|${url}")
                done <<< "${MODEL_INFO}"

                dl_idx=$(select_menu 0 "${DL_LABELS[@]}")
                dl_line="${DL_LINES[$dl_idx]}"

                IFS='|' read -r _ dl_name dl_quant dl_size dl_url <<< "${dl_line}"

                # Download
                MODELS_DIR="${INSTALL_DIR}/.local/models"
                mkdir -p "${MODELS_DIR}"
                dl_filename=$(basename "${dl_url}")
                MODEL_PATH="${MODELS_DIR}/${dl_filename}"

                if [[ -f "${MODEL_PATH}" ]]; then
                    ok "Model already downloaded: ${MODEL_PATH}"
                else
                    info "Downloading ${dl_name} (${dl_quant}, ${dl_size} GiB) ..."
                    info "Saving to: ${MODEL_PATH}"
                    curl -L --progress-bar -o "${MODEL_PATH}" "${dl_url}"
                    ok "Download complete: ${MODEL_PATH}"
                fi
                MODEL_CONFIGURED=1
            fi
            ;;
        1)
            # ── Use local model ─────────────────────────────────
            echo ""
            local_path=$(prompt_input "  Enter model path: " "")
            # Strip surrounding quotes if user pasted a quoted path
            local_path="${local_path#\"}"
            local_path="${local_path%\"}"
            local_path="${local_path#\'}"
            local_path="${local_path%\'}"
            if [[ -n "${local_path}" ]] && [[ -e "${local_path}" ]]; then
                MODEL_PATH="${local_path}"
                MODEL_CONFIGURED=1
                ok "Model: ${MODEL_PATH}"
            else
                warn "Path not found: ${local_path}"
                warn "Skipping model configuration."
            fi
            ;;
        2|*)
            info "Skipping model configuration."
            ;;
    esac
fi

echo ""

# ── Step 6: Load model & finish ──────────────────────────────

info "Step 6/6: Finishing up ..."
echo ""

if [[ "${MODEL_CONFIGURED}" -eq 1 ]] && [[ -n "${MODEL_PATH}" ]]; then
    info "Loading model ..."
    if ! omniinfer_cmd model load -m "${MODEL_PATH}"; then
        err "Failed to load model. Make sure the backend is built and the model path is correct."
        write_install_summary
        echo ""
        echo "  Try building the backend first, then re-run:"
        echo "    cd ${INSTALL_DIR}"
        echo "    ./omniinfer model load -m ${MODEL_PATH}"
        echo ""
        exit 1
    fi
    ok "Model loaded"
    echo ""

    # ── Cleanup function (runs on exit or Ctrl+C) ────────
    print_finish() {
        echo ""
        omniinfer_cmd shutdown 2>/dev/null || true
        write_install_summary

        cat <<FINISH

╔══════════════════════════════════════════════════════════╗
║                  Setup Complete!                         ║
╚══════════════════════════════════════════════════════════╝

  Install:  ${INSTALL_DIR}
  Backend:  ${SELECTED_BACKEND}
  Model:    $(basename "${MODEL_PATH}")

  Your backend selection is saved. Next time just run:

    cd ${INSTALL_DIR}
    ./omniinfer model load -m ${MODEL_PATH}
    ./omniinfer chat --message "Hello"

  The model needs to be loaded each time after a restart.
  The CLI auto-starts the service if needed.

  Other useful commands:
    ./omniinfer backend list              # list available backends
    ./omniinfer backend select <backend>  # switch backend
    ./omniinfer model list                # browse supported models
    ./omniinfer status                    # check current state
    ./omniinfer serve                     # start API server (http://127.0.0.1:${OMNI_PORT})
    ./omniinfer shutdown                  # stop the service

  Full documentation:
    CLI guide:   ${INSTALL_DIR}/docs/CLI.md
    API guide:   ${INSTALL_DIR}/docs/API.md
    Build guide: ${INSTALL_DIR}/docs/build.md

FINISH
    }
    trap print_finish EXIT

    # ── Interactive chat loop ─────────────────────────────
    ok "Setup complete! Try chatting with the model (type 'exit' to quit)."
    echo ""
    resolve_tty
    while true; do
        printf '\033[1;36mYou:\033[0m ' >&2
        user_msg=""
        if [[ -n "${INPUT_TTY}" ]] && [[ "${INPUT_TTY}" != "/dev/stdin" ]]; then
            IFS= read -r user_msg < "${INPUT_TTY}" || break
        else
            IFS= read -r user_msg || break
        fi
        [[ -z "${user_msg}" ]] && continue
        [[ "${user_msg}" == "exit" || "${user_msg}" == "quit" ]] && break
        printf '\033[1;32mAI:\033[0m ' >&2
        omniinfer_cmd chat --message "${user_msg}"
        echo ""
    done

else
    # ── No model configured — print next steps ──────────
    write_install_summary
    cat <<EOF

╔══════════════════════════════════════════════════════════╗
║                  Install Complete!                       ║
╚══════════════════════════════════════════════════════════╝

  Install:  ${INSTALL_DIR}
  Backend:  ${SELECTED_BACKEND}

  To start chatting, load a model first:

    cd ${INSTALL_DIR}
    ./omniinfer model load -m /path/to/model.gguf
    ./omniinfer chat --message "Hello"

  The model needs to be loaded each time after a restart.

  Other useful commands:
    ./omniinfer backend list              # list available backends
    ./omniinfer backend select <backend>  # switch backend
    ./omniinfer model list                # browse supported models
    ./omniinfer status                    # check current state
    ./omniinfer serve                     # start API server (http://127.0.0.1:${OMNI_PORT})
    ./omniinfer shutdown                  # stop the service

  Full documentation:
    CLI guide:   ${INSTALL_DIR}/docs/CLI.md
    API guide:   ${INSTALL_DIR}/docs/API.md
    Build guide: ${INSTALL_DIR}/docs/build.md

EOF
fi
