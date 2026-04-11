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
REPO_URL="https://github.com/omnimind-ai/OmniInfer.git"

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
echo ""

# ── Step 2: Clone or update repo ────────────────────────────

info "Step 2/6: Preparing repository ..."
if [ -d "${INSTALL_DIR}/.git" ]; then
    info "Found existing clone at ${INSTALL_DIR}, updating ..."
    git -C "${INSTALL_DIR}" pull --ff-only 2>/dev/null || warn "Pull failed, continuing with existing code"
else
    info "Cloning OmniInfer to ${INSTALL_DIR} ..."
    git clone --depth 1 "${REPO_URL}" "${INSTALL_DIR}"
fi
ok "Repository ready at ${INSTALL_DIR}"
echo ""

# ── Step 3: Detect platform & choose backend ────────────────

info "Step 3/6: Detecting platform and hardware ..."

OS="$(uname -s)"
ARCH="$(uname -m)"

# Build the backend menu based on platform + detected hardware
declare -a BACKEND_IDS=()
declare -a BACKEND_LABELS=()
declare -a BACKEND_SCRIPTS=()
RECOMMENDED_INDEX=0

IS_ANDROID_PLATFORM=0
if [[ "${IS_ANDROID}" -eq 1 ]]; then
    IS_ANDROID_PLATFORM=1
    BACKEND_IDS+=("llama.cpp-llama");  BACKEND_LABELS+=("llama.cpp Text (Android)");       BACKEND_SCRIPTS+=("__android__")
    BACKEND_IDS+=("llama.cpp-mtmd");   BACKEND_LABELS+=("llama.cpp Multimodal (Android)");  BACKEND_SCRIPTS+=("__android__")
    RECOMMENDED_INDEX=0
else
    case "${OS}" in
        Darwin)
            if [ "${ARCH}" = "arm64" ]; then
                BACKEND_IDS+=("llama.cpp-mac");       BACKEND_LABELS+=("llama.cpp Metal (Apple Silicon)");  BACKEND_SCRIPTS+=("macos/build-llama-mac.sh")
                BACKEND_IDS+=("turboquant-mac");      BACKEND_LABELS+=("TurboQuant (Apple Silicon)");       BACKEND_SCRIPTS+=("macos/build-turboquant-mac.sh")
                BACKEND_IDS+=("mlx-mac");             BACKEND_LABELS+=("MLX embedded (Apple Silicon)");     BACKEND_SCRIPTS+=("macos/build-mlx-mac.sh")
                RECOMMENDED_INDEX=0
            else
                BACKEND_IDS+=("llama.cpp-mac-intel");  BACKEND_LABELS+=("llama.cpp CPU (Intel Mac)");       BACKEND_SCRIPTS+=("macos/build-llama-mac-intel.sh")
                RECOMMENDED_INDEX=0
            fi
            ;;
        Linux)
            BACKEND_IDS+=("llama.cpp-linux");          BACKEND_LABELS+=("llama.cpp CPU");                    BACKEND_SCRIPTS+=("linux/build-llama-cpu.sh")
            RECOMMENDED_INDEX=0

            if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
                ok "NVIDIA GPU detected"
                BACKEND_IDS+=("llama.cpp-linux-vulkan"); BACKEND_LABELS+=("llama.cpp Vulkan (NVIDIA)");       BACKEND_SCRIPTS+=("linux/build-llama-vulkan.sh")
                RECOMMENDED_INDEX=$(( ${#BACKEND_IDS[@]} - 1 ))
            fi

            if command -v rocm-smi >/dev/null 2>&1; then
                ok "AMD ROCm detected"
                BACKEND_IDS+=("llama.cpp-linux-rocm");  BACKEND_LABELS+=("llama.cpp ROCm (AMD)");             BACKEND_SCRIPTS+=("linux/build-llama-rocm.sh")
                RECOMMENDED_INDEX=$(( ${#BACKEND_IDS[@]} - 1 ))
            fi

            # Vulkan without nvidia-smi (e.g. Intel, other Vulkan drivers)
            if command -v vulkaninfo >/dev/null 2>&1 && ! command -v nvidia-smi >/dev/null 2>&1; then
                BACKEND_IDS+=("llama.cpp-linux-vulkan"); BACKEND_LABELS+=("llama.cpp Vulkan");                BACKEND_SCRIPTS+=("linux/build-llama-vulkan.sh")
            fi
            ;;
        *)
            fatal "This script supports macOS, Linux, and Android (Termux). For Windows, use install.ps1."
            ;;
    esac
fi

if [[ "${IS_ANDROID_PLATFORM}" -eq 1 ]]; then
    info "Platform: Android / Termux (${ARCH})"
else
    info "Platform: ${OS} (${ARCH})"
fi
echo ""

# If backend override is set, use it directly
if [[ -n "${BACKEND_OVERRIDE}" ]]; then
    SELECTED_BACKEND="${BACKEND_OVERRIDE}"
    # Find matching script
    SELECTED_SCRIPT=""
    for i in "${!BACKEND_IDS[@]}"; do
        if [[ "${BACKEND_IDS[$i]}" == "${SELECTED_BACKEND}" ]]; then
            SELECTED_SCRIPT="${BACKEND_SCRIPTS[$i]}"
            break
        fi
    done
    [[ -z "${SELECTED_SCRIPT}" ]] && fatal "Unknown backend: ${SELECTED_BACKEND}"
else
    echo "  Available backends (arrow keys to move, Enter to select):"
    echo ""

    # Build display labels with recommended tag
    declare -a MENU_LABELS=()
    for i in "${!BACKEND_IDS[@]}"; do
        label="${BACKEND_LABELS[$i]}"
        if [[ "$i" -eq "${RECOMMENDED_INDEX}" ]]; then
            label="${label} (recommended)"
        fi
        MENU_LABELS+=("${label}")
    done

    idx=$(select_menu "${RECOMMENDED_INDEX}" "${MENU_LABELS[@]}")

    SELECTED_BACKEND="${BACKEND_IDS[$idx]}"
    SELECTED_SCRIPT="${BACKEND_SCRIPTS[$idx]}"
fi

ok "Selected: ${SELECTED_BACKEND}"
echo ""

# ── Initialize required submodules ──────────────────────────

case "${SELECTED_BACKEND}" in
    llama.cpp-*|turboquant-mac)
        SUBMODULE="framework/llama.cpp"
        [[ "${SELECTED_BACKEND}" == "turboquant-mac" ]] && SUBMODULE="framework/llama-cpp-turboquant"
        info "Initializing ${SUBMODULE} submodule ..."
        git -C "${INSTALL_DIR}" submodule update --init --recursive --depth 1 --progress "${SUBMODULE}"
        ok "Submodule ready"
        ;;
    mlx-mac)
        info "MLX backend uses pip packages, no submodule needed"
        ;;
    *)
        info "No submodule needed for ${SELECTED_BACKEND}"
        ;;
esac
echo ""

# ── Step 4: Build backend ───────────────────────────────────

info "Step 4/6: Building backend ..."

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

    "${INSTALL_DIR}/omniinfer" select "${SELECTED_BACKEND}" 2>/dev/null
    ok "Backend ${SELECTED_BACKEND} selected"

else
    # ── Desktop: use platform build scripts ──
    FULL_BUILD_SCRIPT="${INSTALL_DIR}/scripts/platforms/${SELECTED_SCRIPT}"

    RUNTIME_CHECK="${INSTALL_DIR}/.local/runtime"
    ALREADY_BUILT=0
    case "${SELECTED_BACKEND}" in
        llama.cpp-mac*)   [[ -d "${RUNTIME_CHECK}/macos/${SELECTED_BACKEND}/bin" ]]  && ALREADY_BUILT=1 ;;
        llama.cpp-linux*) [[ -d "${RUNTIME_CHECK}/linux/${SELECTED_BACKEND}/bin" ]]  && ALREADY_BUILT=1 ;;
        turboquant-mac)   [[ -d "${RUNTIME_CHECK}/macos/turboquant-mac/bin" ]]       && ALREADY_BUILT=1 ;;
        mlx-mac)          [[ -d "${RUNTIME_CHECK}/macos/mlx-mac/venv" ]]             && ALREADY_BUILT=1 ;;
    esac

    if [[ "${SKIP_BUILD}" -eq 1 ]]; then
        info "Skipping build (--skip-build)"
    elif [[ "${ALREADY_BUILT}" -eq 1 ]]; then
        ok "Backend ${SELECTED_BACKEND} already built, skipping"
    else
        info "Building ${SELECTED_BACKEND} (this may take a few minutes) ..."
        bash "${FULL_BUILD_SCRIPT}"
        ok "Build complete"
    fi

    "${INSTALL_DIR}/omniinfer" select "${SELECTED_BACKEND}" 2>/dev/null
    ok "Backend ${SELECTED_BACKEND} selected"
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
    "${INSTALL_DIR}/omniinfer" model load -m "${MODEL_PATH}"
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

  Full documentation:
    CLI guide:   ${INSTALL_DIR}/docs/CLI.md
    API guide:   ${INSTALL_DIR}/docs/API.md
    Build guide: ${INSTALL_DIR}/docs/build.md

EOF
fi
