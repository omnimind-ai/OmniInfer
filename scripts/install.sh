#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  OmniInfer interactive installer for macOS / Linux / Android (Termux)
#
#  Usage:
#    curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.sh | bash
#    curl -fsSL ... | bash -s -- --install-dir ~/my-omniinfer
#    curl -fsSL ... | bash -s -- --model /path/to/model.gguf
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────

INSTALL_DIR="$(pwd)/OmniInfer"
MODEL_PATH=""
SKIP_BUILD=0
BACKEND_OVERRIDE=""
NON_INTERACTIVE=0
REPO_SSH="git@github.com:omnimind-ai/OmniInfer.git"
REPO_HTTPS="https://github.com/omnimind-ai/OmniInfer.git"

# ── Parse args ──────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)    INSTALL_DIR="$2";       shift 2 ;;
        --model|-m)       MODEL_PATH="$2";        shift 2 ;;
        --skip-build)     SKIP_BUILD=1;           shift   ;;
        --backend)        BACKEND_OVERRIDE="$2";  shift 2 ;;
        --non-interactive) NON_INTERACTIVE=1;      shift   ;;
        --help|-h)
            cat <<'HELP'
OmniInfer Installer

Usage:
  curl -fsSL <url>/install.sh | bash
  bash install.sh [OPTIONS]

Options:
  --install-dir DIR     Installation directory (default: ~/OmniInfer)
  --model, -m PATH      Path to a local GGUF model file or directory
  --skip-build          Skip the backend build step
  --backend ID          Force a specific backend (e.g. llama.cpp-linux-vulkan)
  --non-interactive     Accept all defaults without prompting
  -h, --help            Show this help
HELP
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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

# ── Banner ──────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║             OmniInfer Interactive Installer              ║"
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

# ── Step 1: Check prerequisites ─────────────────────────────

info "Step 1/6: Checking prerequisites ..."
need_cmd git    "Install from https://git-scm.com/"
need_cmd cmake  "Termux: pkg install cmake  |  macOS: brew install cmake  |  Linux: apt install cmake"
if [[ "${IS_ANDROID}" -eq 0 ]]; then
    need_cmd python3 "Install Python 3 from https://python.org/"
fi
need_cmd curl   "Install curl for your platform"
# C/C++ compiler (needed for building backends)
if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || command -v clang >/dev/null 2>&1; then
    ok "C++ compiler"
else
    fatal "No C/C++ compiler found. Install one of:
  macOS:   xcode-select --install
  Ubuntu:  sudo apt install build-essential
  Termux:  pkg install clang"
fi
echo ""

# ── Step 2: Clone or update repo ────────────────────────────

info "Step 2/6: Preparing repository ..."
if [ -d "${INSTALL_DIR}/.git" ]; then
    info "Found existing clone at ${INSTALL_DIR}, updating ..."
    git -C "${INSTALL_DIR}" pull --ff-only 2>/dev/null || warn "Pull failed, continuing with existing code"
else
    info "Cloning OmniInfer to ${INSTALL_DIR} ..."
    CLONED_VIA_HTTPS=0
    info "Trying SSH ..."
    if ! git clone --depth 1 "${REPO_SSH}" "${INSTALL_DIR}" 2>/dev/null; then
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
if [[ ! -f "${INSTALL_DIR}/omniinfer.py" ]]; then
    fatal "Repository clone appears incomplete — omniinfer.py not found in ${INSTALL_DIR}"
fi
ok "Repository ready at ${INSTALL_DIR}"

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

if port_in_use "${OMNI_PORT}"; then
    warn "Port ${OMNI_PORT} is in use, looking for a free port ..."
    for try_port in 9001 9002 9003 9004 9005 9010 9020 9050 9100 8900 8800 19000; do
        if ! port_in_use "${try_port}"; then
            OMNI_PORT="${try_port}"
            break
        fi
    done
    if [[ "${OMNI_PORT}" -eq 9000 ]]; then
        fatal "Could not find a free port"
    fi
    info "Using port ${OMNI_PORT}"
    CONFIG_DIR="${INSTALL_DIR}/config"
    mkdir -p "${CONFIG_DIR}"
    cat > "${CONFIG_DIR}/omniinfer.json" <<PORTCFG
{
  "host": "127.0.0.1",
  "port": ${OMNI_PORT}
}
PORTCFG
    ok "Config written: ${CONFIG_DIR}/omniinfer.json (port ${OMNI_PORT})"
fi
echo ""

# ── Cleanup: shut down any gateway started by the CLI on exit ──
_cleanup_gateway() {
    curl -sS -X POST "http://127.0.0.1:${OMNI_PORT}/omni/shutdown" >/dev/null 2>&1 || true
}
trap _cleanup_gateway EXIT

# ── Step 3: Detect platform & choose backend ────────────────

info "Step 3/6: Detecting platform and hardware ..."

OS="$(uname -s)"
ARCH="$(uname -m)"

IS_ANDROID_PLATFORM=0
if [[ "${IS_ANDROID}" -eq 1 ]]; then
    IS_ANDROID_PLATFORM=1
    info "Platform: Android / Termux (${ARCH})"
else
    info "Platform: ${OS} (${ARCH})"
fi
echo ""

# Get available backends
declare -a BACKEND_IDS=()
declare -a BACKEND_DESCS=()

if [[ "${IS_ANDROID_PLATFORM}" -eq 1 ]]; then
    # Android: runtime not installed yet, backends are fixed
    BACKEND_IDS+=("llama.cpp-llama");  BACKEND_DESCS+=("llama.cpp-llama  —  Text chat")
    BACKEND_IDS+=("llama.cpp-mtmd");   BACKEND_DESCS+=("llama.cpp-mtmd  —  Multimodal (text + vision)")
else
    # Desktop: query gateway API for compatible backends (hardware-matched)
    # First ensure the service is running
    "${INSTALL_DIR}/omniinfer" status >/dev/null 2>&1 || true

    _backends_json=$(curl -sS "http://127.0.0.1:${OMNI_PORT}/omni/backends?scope=compatible" 2>/dev/null) || _backends_json=""
    _recommended=$(echo "${_backends_json}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('recommended',''))" 2>/dev/null) || _recommended=""

    if echo "${_backends_json}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for b in d.get('data', []):
    print(b['id'] + '|' + b.get('description', ''))
" 2>/dev/null | while IFS='|' read -r bid bdesc; do
        BACKEND_IDS+=("${bid}")
        if [[ -n "${bdesc}" ]]; then
            BACKEND_DESCS+=("${bid}  —  ${bdesc}")
        else
            BACKEND_DESCS+=("${bid}")
        fi
    done; [[ ${#BACKEND_IDS[@]} -gt 0 ]]; then
        # Move recommended backend to top
        if [[ -n "${_recommended}" ]]; then
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
    else
        # Fallback: parse CLI text output
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
        done <<< "$("${INSTALL_DIR}/omniinfer" backend list 2>/dev/null)"
    fi
fi

if [[ ${#BACKEND_IDS[@]} -eq 0 ]]; then
    fatal "No backends found. Check your platform support."
fi

# If backend override is set, use it directly
if [[ -n "${BACKEND_OVERRIDE}" ]]; then
    SELECTED_BACKEND="${BACKEND_OVERRIDE}"
else
    echo "  Available backends (arrow keys to move, Enter to select):"
    echo ""

    idx=$(select_menu 0 "${BACKEND_DESCS[@]}")
    SELECTED_BACKEND="${BACKEND_IDS[$idx]}"
fi

ok "Selected: ${SELECTED_BACKEND}"
echo ""

# Select backend via CLI (skip on Android — runtime not installed yet)
if [[ "${IS_ANDROID_PLATFORM}" -eq 0 ]]; then
    "${INSTALL_DIR}/omniinfer" select "${SELECTED_BACKEND}"
fi

# ── Step 4: Build backend ───────────────────────────────────
# Build scripts auto-bootstrap their required submodules.
# Build script is discovered by convention: scripts/platforms/<platform_dir>/<backend_id>/build.sh

info "Step 4/6: Building backend ..."

# Map OS to platform directory name
case "${OS}" in
    Darwin) PLATFORM_DIR="macos" ;;
    Linux)  PLATFORM_DIR="linux" ;;
    *)      PLATFORM_DIR="linux" ;;
esac

if [[ "${IS_ANDROID_PLATFORM}" -eq 1 ]]; then
    # ── Android / Termux: build llama.cpp natively, install via build-runtime.sh ──
    ANDROID_BUILD_DIR="${INSTALL_DIR}/framework/llama.cpp/build-termux"
    ANDROID_LIB_DIR="${INSTALL_DIR}/.local/runtime/android/lib/arm64-v8a"

    ALREADY_BUILT=0
    [[ -x "${ANDROID_LIB_DIR}/libllama-cli.so" ]] && ALREADY_BUILT=1

    if [[ "${SKIP_BUILD}" -eq 1 ]]; then
        info "Skipping build (--skip-build)"
    elif [[ "${ALREADY_BUILT}" -eq 1 ]]; then
        ok "Android backend already built, skipping"
    else
        # Android/Termux builds llama.cpp directly, need the submodule
        info "Initializing llama.cpp submodule ..."
        git -C "${INSTALL_DIR}" submodule update --init --recursive --depth 1 --progress framework/llama.cpp
        info "Building llama.cpp natively in Termux (this may take a few minutes) ..."

        NPROC=4
        command -v nproc >/dev/null 2>&1 && NPROC=$(nproc)

        cmake -B "${ANDROID_BUILD_DIR}" \
              -S "${INSTALL_DIR}/framework/llama.cpp" \
              -DCMAKE_BUILD_TYPE=Release \
              -DGGML_OPENMP=OFF 2>&1 | tail -3

        cmake --build "${ANDROID_BUILD_DIR}" --target llama-cli -j"${NPROC}" 2>&1 | tail -5

        # Also try to build mtmd-cli (may not exist in all llama.cpp versions)
        cmake --build "${ANDROID_BUILD_DIR}" --target llama-mtmd-cli -j"${NPROC}" 2>&1 | tail -3 || true

        ok "Build complete"

        # Find built binaries
        LLAMA_CLI_BIN=$(find "${ANDROID_BUILD_DIR}" -name "llama-cli" -type f -executable 2>/dev/null | head -1)
        MTMD_CLI_BIN=$(find "${ANDROID_BUILD_DIR}" -name "llama-mtmd-cli" -type f -executable 2>/dev/null | head -1)

        if [[ -z "${LLAMA_CLI_BIN}" ]]; then
            fatal "llama-cli binary not found after build"
        fi

        # Install Android runtime layout
        RUNTIME_ARGS=("--llama-cli" "${LLAMA_CLI_BIN}")
        if [[ -n "${MTMD_CLI_BIN}" ]]; then
            RUNTIME_ARGS+=("--mtmd-cli" "${MTMD_CLI_BIN}")
        fi

        info "Installing Android runtime layout ..."
        bash "${INSTALL_DIR}/scripts/platforms/android/build-runtime.sh" "${RUNTIME_ARGS[@]}"
        ok "Android runtime installed"
    fi

    # Now that runtime is installed, select the backend
    "${INSTALL_DIR}/omniinfer" select "${SELECTED_BACKEND}"

else
    # ── Desktop: discover and run build script by convention ──
    FULL_BUILD_SCRIPT="${INSTALL_DIR}/scripts/platforms/${PLATFORM_DIR}/${SELECTED_BACKEND}/build.sh"
    if [[ ! -f "${FULL_BUILD_SCRIPT}" ]]; then
        fatal "Build script not found: ${FULL_BUILD_SCRIPT}"
    fi

    if [[ "${SKIP_BUILD}" -eq 1 ]]; then
        info "Skipping build (--skip-build)"
    else
        # Check if runtime is already available via CLI
        RUNTIME_AVAILABLE=$("${INSTALL_DIR}/omniinfer" backend list 2>/dev/null | grep -A3 "[* ]*${SELECTED_BACKEND}$" | grep -c "Runtime available: yes" || true)
        if [[ "${RUNTIME_AVAILABLE}" -gt 0 ]]; then
            ok "Backend ${SELECTED_BACKEND} already built, skipping"
        else
            info "Building ${SELECTED_BACKEND} (this may take a few minutes) ..."
            if ! bash "${FULL_BUILD_SCRIPT}"; then
                echo ""
                fatal "Build failed (exit code $?). See the messages above for details."
            fi
            ok "Build complete"
        fi
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
else
    model_choice=$(select_menu 0 \
        "Download a recommended model" \
        "Use a local model file" \
        "Skip (configure later)")

    case "${model_choice}" in
        0)
            # ── Download recommended model ──────────────────────
            info "Fetching model catalog ..."

            if [[ "${IS_ANDROID_PLATFORM}" -eq 1 ]]; then
                CATALOG_SYSTEM="linux"
            elif [[ "${OS}" == "Darwin" ]]; then
                CATALOG_SYSTEM="mac"
            else
                CATALOG_SYSTEM="linux"
            fi

            CATALOG_URL="https://omnimind-model.oss-cn-beijing.aliyuncs.com/backend/${CATALOG_SYSTEM}/model_list.json"

            # Use embedded Python to parse catalog and present choices
            MODEL_INFO=$(curl -sS "${CATALOG_URL}" | python3 -c "
import json, sys

raw = sys.stdin.buffer.read()
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
" 2>/dev/null) || true

            if [[ -z "${MODEL_INFO}" ]]; then
                warn "Could not fetch model catalog. You can configure a model manually later."
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
                MODELS_DIR="${INSTALL_DIR}/models"
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
    if ! "${INSTALL_DIR}/omniinfer" model load -m "${MODEL_PATH}"; then
        err "Failed to load model. Make sure the backend is built and the model path is correct."
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
        "${INSTALL_DIR}/omniinfer" shutdown 2>/dev/null || true

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
    ./omniinfer select <backend>          # switch backend
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
        "${INSTALL_DIR}/omniinfer" chat --message "${user_msg}"
        echo ""
    done

else
    # ── No model configured — print next steps ──────────
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
    ./omniinfer select <backend>          # switch backend
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
