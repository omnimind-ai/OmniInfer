PROJECT_ROOT="$(CDPATH= cd -- "${RUNTIME_ROOT}/../../.." && pwd)"
LIB_ROOT="${RUNTIME_ROOT}/lib/arm64-v8a"
LEGACY_LIB_ROOT="${PROJECT_ROOT}/platform/Android/lib/arm64-v8a"
QNN_ROOT="${RUNTIME_ROOT}/qnn"

BACKEND_LLAMA="llama.cpp-llama"
BACKEND_MTMD="llama.cpp-mtmd"
BACKEND_QNN="omniinfer-native"

DEFAULT_STATE_ROOT="${HOME:-}/.config/omniinfer"
if [ -n "${HOME:-}" ] && [ "${HOME}" != "/" ] && [ -d "${HOME}" ] && [ -w "${HOME}" ]; then
  STATE_ROOT="${DEFAULT_STATE_ROOT}"
else
  STATE_ROOT="${PROJECT_ROOT}/.omniinfer/android-cli"
fi

STATE_FILE="${STATE_ROOT}/android-state.env"

ensure_state_dir() {
  mkdir -p "${STATE_ROOT}"
}

init_state() {
  ensure_state_dir
  if [ ! -f "${STATE_FILE}" ]; then
    cat > "${STATE_FILE}" <<'EOF'
SELECTED_BACKEND=llama.cpp-llama
MODEL_PATH=
MMPROJ_PATH=
CTX_SIZE=
THINKING=off
TOKENIZER_PATH=
DECODER_MODEL_VERSION=
NATIVE_PACKAGE_ROOT=
NATIVE_RUNNER_KIND=
NATIVE_TEXT_DECODER_PATH=
NATIVE_VISION_ENCODER_PATH=
NATIVE_TOK_EMBEDDING_PATH=
NATIVE_ATTENTION_SINK_PATH=
NATIVE_EVAL_MODE=
EOF
  fi
}

load_state() {
  init_state
  # shellcheck disable=SC1090
  . "${STATE_FILE}"
}

save_state() {
  init_state
  cat > "${STATE_FILE}" <<EOF
SELECTED_BACKEND=${SELECTED_BACKEND:-${BACKEND_LLAMA}}
MODEL_PATH=${MODEL_PATH:-}
MMPROJ_PATH=${MMPROJ_PATH:-}
CTX_SIZE=${CTX_SIZE:-}
THINKING=${THINKING:-off}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
DECODER_MODEL_VERSION=${DECODER_MODEL_VERSION:-}
NATIVE_PACKAGE_ROOT=${NATIVE_PACKAGE_ROOT:-}
NATIVE_RUNNER_KIND=${NATIVE_RUNNER_KIND:-}
NATIVE_TEXT_DECODER_PATH=${NATIVE_TEXT_DECODER_PATH:-}
NATIVE_VISION_ENCODER_PATH=${NATIVE_VISION_ENCODER_PATH:-}
NATIVE_TOK_EMBEDDING_PATH=${NATIVE_TOK_EMBEDDING_PATH:-}
NATIVE_ATTENTION_SINK_PATH=${NATIVE_ATTENTION_SINK_PATH:-}
NATIVE_EVAL_MODE=${NATIVE_EVAL_MODE:-}
EOF
}

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$(pwd)" "$1" ;;
  esac
}

reset_backend_state() {
  MODEL_PATH=""
  MMPROJ_PATH=""
  CTX_SIZE=""
  TOKENIZER_PATH=""
  DECODER_MODEL_VERSION=""
  NATIVE_PACKAGE_ROOT=""
  NATIVE_RUNNER_KIND=""
  NATIVE_TEXT_DECODER_PATH=""
  NATIVE_VISION_ENCODER_PATH=""
  NATIVE_TOK_EMBEDDING_PATH=""
  NATIVE_ATTENTION_SINK_PATH=""
  NATIVE_EVAL_MODE=""
}

backend_binary() {
  case "$1" in
    "${BACKEND_LLAMA}"|"${BACKEND_MTMD}")
      llama_cpp_backend_binary "$1"
      ;;
    "${BACKEND_QNN}")
      native_backend_binary
      ;;
    *)
      return 1
      ;;
  esac
}

backend_exists() {
  binary_path="$(backend_binary "$1" 2>/dev/null || true)"
  [ -n "${binary_path:-}" ] && [ -x "${binary_path}" ]
}

print_backends() {
  load_state
  for backend in "${BACKEND_LLAMA}" "${BACKEND_MTMD}" "${BACKEND_QNN}"; do
    binary_path="$(backend_binary "${backend}" 2>/dev/null || true)"
    exists="false"
    if [ -n "${binary_path}" ] && [ -x "${binary_path}" ]; then
      exists="true"
    fi
    selected="false"
    if [ "${SELECTED_BACKEND}" = "${backend}" ]; then
      selected="true"
    fi
    printf '%s\tselected=%s\tbinary_exists=%s\tpath=%s\n' "${backend}" "${selected}" "${exists}" "${binary_path}"
  done
}

print_status() {
  load_state
  cat <<EOF
backend=${SELECTED_BACKEND}
model=${MODEL_PATH}
mmproj=${MMPROJ_PATH}
ctx_size=${CTX_SIZE}
thinking=${THINKING}
tokenizer=${TOKENIZER_PATH}
decoder_model_version=${DECODER_MODEL_VERSION}
native_package_root=${NATIVE_PACKAGE_ROOT:-}
native_runner=${NATIVE_RUNNER_KIND:-}
native_eval_mode=${NATIVE_EVAL_MODE:-}
state_file=${STATE_FILE}
EOF
}

require_loaded_model() {
  load_state
  if [ -z "${MODEL_PATH}" ]; then
    echo "No model is loaded. Run: ./omniinfer model load -m /path/to/model.gguf" >&2
    exit 1
  fi
}

handle_select() {
  if [ $# -ne 1 ]; then
    echo "Usage: ./omniinfer select <backend>" >&2
    exit 1
  fi
  case "$1" in
    "${BACKEND_LLAMA}"|"${BACKEND_MTMD}"|"${BACKEND_QNN}") ;;
    *)
      echo "Unsupported Android backend: $1" >&2
      exit 1
      ;;
  esac
  load_state
  SELECTED_BACKEND="$1"
  save_state
  echo "Selected Android backend: ${SELECTED_BACKEND}"
}

handle_thinking() {
  load_state
  action="${1:-show}"
  case "${action}" in
    show)
      echo "${THINKING}"
      ;;
    set)
      value="${2:-}"
      case "${value}" in
        on|off) ;;
        *)
          echo "Usage: ./omniinfer thinking set <on|off>" >&2
          exit 1
          ;;
      esac
      THINKING="${value}"
      save_state
      echo "thinking=${THINKING}"
      ;;
    *)
      echo "Usage: ./omniinfer thinking <show|set>" >&2
      exit 1
      ;;
  esac
}

handle_model_load() {
  load_state
  reset_backend_state

  while [ $# -gt 0 ]; do
    case "$1" in
      -m|--model)
        MODEL_PATH="$(resolve_path "${2:-}")"
        shift 2
        ;;
      -mm|--mmproj)
        MMPROJ_PATH="$(resolve_path "${2:-}")"
        shift 2
        ;;
      --ctx-size)
        CTX_SIZE="${2:-}"
        shift 2
        ;;
      --tokenizer-path)
        TOKENIZER_PATH="$(resolve_path "${2:-}")"
        shift 2
        ;;
      --decoder-model-version)
        DECODER_MODEL_VERSION="${2:-}"
        shift 2
        ;;
      *)
        echo "Unknown model load argument: $1" >&2
        exit 1
        ;;
    esac
  done

  if [ -z "${MODEL_PATH}" ]; then
    echo "Usage: ./omniinfer model load -m /path/to/model.gguf [-mm /path/to/mmproj.gguf] [--ctx-size N]" >&2
    exit 1
  fi

  if [ "${SELECTED_BACKEND}" = "${BACKEND_QNN}" ]; then
    native_prepare_model_load
  fi

  save_state
  echo "Loaded Android model:"
  echo "  backend=${SELECTED_BACKEND}"
  echo "  model=${MODEL_PATH}"
  if [ -n "${MMPROJ_PATH}" ]; then
    echo "  mmproj=${MMPROJ_PATH}"
  fi
  if [ -n "${CTX_SIZE}" ]; then
    echo "  ctx_size=${CTX_SIZE}"
  fi
  if [ -n "${TOKENIZER_PATH}" ]; then
    echo "  tokenizer=${TOKENIZER_PATH}"
  fi
  if [ -n "${DECODER_MODEL_VERSION}" ]; then
    echo "  decoder_model_version=${DECODER_MODEL_VERSION}"
  fi
  if [ -n "${NATIVE_PACKAGE_ROOT}" ]; then
    echo "  native_package_root=${NATIVE_PACKAGE_ROOT}"
  fi
  if [ -n "${NATIVE_RUNNER_KIND}" ]; then
    echo "  native_runner=${NATIVE_RUNNER_KIND}"
  fi
}

handle_chat() {
  require_loaded_model

  message=""
  image=""
  stream=1
  system_prompt=""

  while [ $# -gt 0 ]; do
    case "$1" in
      --message)
        message="${2:-}"
        shift 2
        ;;
      --image)
        image="${2:-}"
        shift 2
        ;;
      --no-stream)
        stream=0
        shift
        ;;
      --system-prompt|-sys)
        system_prompt="${2:-}"
        shift 2
        ;;
      *)
        echo "Unknown chat argument: $1" >&2
        exit 1
        ;;
    esac
  done

  if [ -z "${message}" ]; then
    echo "Usage: ./omniinfer chat --message \"...\" [--image /path/to/image] [--no-stream]" >&2
    exit 1
  fi

  load_state
  if [ "${SELECTED_BACKEND}" = "${BACKEND_QNN}" ]; then
    native_run_chat "${message}" "${image}" "${stream}" "${system_prompt}"
    exit 0
  fi

  binary_path="$(backend_binary "${SELECTED_BACKEND}")"
  if [ ! -x "${binary_path}" ]; then
    echo "Backend binary is not available: ${binary_path}" >&2
    echo "Prepare the Android runtime first with scripts/platforms/android/build-runtime.sh" >&2
    exit 1
  fi

  set -- "${binary_path}" -m "${MODEL_PATH}" -p "${message}" -cnv
  if [ -n "${CTX_SIZE}" ]; then
    set -- "$@" -c "${CTX_SIZE}"
  fi
  if [ -n "${MMPROJ_PATH}" ]; then
    set -- "$@" -mmproj "${MMPROJ_PATH}"
  fi
  if [ -n "${image}" ]; then
    set -- "$@" --image "${image}"
  fi
  if [ "${stream}" -eq 0 ]; then
    set -- "$@" --no-display-prompt
  fi

  exec "$@"
}

handle_backend() {
  subcommand="${1:-list}"
  shift || true
  case "${subcommand}" in
    list)
      print_backends
      ;;
    *)
      echo "Unsupported backend command: ${subcommand}" >&2
      exit 1
      ;;
  esac
}

main() {
  cmd="${1:-backend}"
  if [ $# -gt 0 ]; then
    shift
  fi

  case "${cmd}" in
    backend)
      handle_backend "$@"
      ;;
    select)
      handle_select "$@"
      ;;
    model)
      subcommand="${1:-}"
      shift || true
      case "${subcommand}" in
        load)
          handle_model_load "$@"
          ;;
        *)
          echo "Usage: ./omniinfer model load -m /path/to/model.gguf" >&2
          exit 1
          ;;
      esac
      ;;
    chat)
      handle_chat "$@"
      ;;
    status)
      print_status
      ;;
    thinking)
      handle_thinking "$@"
      ;;
    shutdown)
      echo "Android direct mode has no background gateway to stop."
      ;;
    --help|-h|help)
      cat <<'EOF'
Android OmniInfer direct mode

Common commands:
  ./omniinfer backend list
  ./omniinfer select llama.cpp-llama
  ./omniinfer select llama.cpp-mtmd
  ./omniinfer select omniinfer-native
  ./omniinfer model load -m /data/local/tmp/model.gguf
  ./omniinfer model load -m /data/local/tmp/model.gguf -mm /data/local/tmp/mmproj.gguf
  ./omniinfer model load -m /data/local/tmp/qnn-dir --decoder-model-version qwen3
  ./omniinfer chat --message "Hello"
  ./omniinfer chat --image /data/local/tmp/test.jpg --message "Describe this image."
  ./omniinfer status
  ./omniinfer thinking show
  ./omniinfer thinking set on
  ./omniinfer shutdown
EOF
      ;;
    *)
      echo "Unsupported Android command: ${cmd}" >&2
      exit 1
      ;;
  esac
}
