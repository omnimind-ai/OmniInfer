QNN_MANIFEST_NAME="omniinfer-native.env"

native_reset_state() {
  NATIVE_PACKAGE_ROOT=""
  NATIVE_RUNNER_KIND=""
  NATIVE_TEXT_DECODER_PATH=""
  NATIVE_VISION_ENCODER_PATH=""
  NATIVE_TOK_EMBEDDING_PATH=""
  NATIVE_ATTENTION_SINK_PATH=""
  NATIVE_EVAL_MODE=""
}

qnn_model_dir() {
  case "$1" in
    *.pte) dirname "$1" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

qnn_package_root() {
  if [ -n "${NATIVE_PACKAGE_ROOT:-}" ]; then
    printf '%s\n' "${NATIVE_PACKAGE_ROOT}"
    return 0
  fi
  qnn_model_dir "${MODEL_PATH:-}"
}

resolve_qnn_component_path() {
  package_root="$1"
  target="$2"
  if [ -z "${target}" ]; then
    return 1
  fi
  case "${target}" in
    /*) printf '%s\n' "${target}" ;;
    *) printf '%s/%s\n' "${package_root}" "${target}" ;;
  esac
}

qnn_manifest_path() {
  target="$1"
  if [ -f "${target}" ] && [ "$(basename "${target}")" = "${QNN_MANIFEST_NAME}" ]; then
    printf '%s\n' "${target}"
    return 0
  fi
  if [ -d "${target}" ] && [ -f "${target}/${QNN_MANIFEST_NAME}" ]; then
    printf '%s\n' "${target}/${QNN_MANIFEST_NAME}"
    return 0
  fi
  if [ -f "${target}" ] && [ -f "$(dirname "${target}")/${QNN_MANIFEST_NAME}" ]; then
    printf '%s\n' "$(dirname "${target}")/${QNN_MANIFEST_NAME}"
    return 0
  fi
  return 1
}

discover_qnn_model() {
  target="$1"
  if [ -d "${target}" ]; then
    preferred=""
    if [ -f "${target}/hybrid_llama_qnn.pte" ]; then
      preferred="${target}/hybrid_llama_qnn.pte"
    fi
    if [ -n "${preferred}" ]; then
      printf '%s\n' "${preferred}"
      return 0
    fi
    candidates="$(find "${target}" -maxdepth 1 -type f -name '*.pte' | sort || true)"
    count="$(printf '%s\n' "${candidates}" | sed '/^$/d' | wc -l | tr -d ' ')"
    if [ "${count}" = "1" ]; then
      printf '%s\n' "${candidates}"
      return 0
    fi
    if [ "${count}" = "0" ]; then
      echo "No .pte model was found under: ${target}" >&2
    else
      echo "Multiple .pte models were found under: ${target}" >&2
      printf '%s\n' "${candidates}" >&2
    fi
    return 1
  fi
  if [ -f "${target}" ]; then
    printf '%s\n' "${target}"
    return 0
  fi
  echo "QNN model path was not found: ${target}" >&2
  return 1
}

infer_decoder_model_version() {
  lower="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${lower}" in
    *hybrid_llama_qnn.pte) printf '%s\n' qwen3 ;;
    *qwen3*) printf '%s\n' qwen3 ;;
    *qwen2.5*|*qwen2_5*|*qwen25*) printf '%s\n' qwen2_5 ;;
    *phi*4*mini*) printf '%s\n' phi_4_mini ;;
    *llama3*) printf '%s\n' llama3 ;;
    *llama2*) printf '%s\n' llama2 ;;
    *gemma3*) printf '%s\n' gemma3 ;;
    *gemma2*) printf '%s\n' gemma2 ;;
    *gemma*) printf '%s\n' gemma ;;
    *granite*) printf '%s\n' granite ;;
    *smollm3*) printf '%s\n' smollm3 ;;
    *smollm2*) printf '%s\n' smollm2_135m ;;
    *codegen*) printf '%s\n' codegen ;;
    *glm*) printf '%s\n' glm ;;
    *) return 1 ;;
  esac
}

infer_qnn_eval_mode_from_model_path() {
  lower="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${lower}" in
    *lookahead*) printf '%s\n' 2 ;;
    *hybrid*) printf '%s\n' 1 ;;
    *kv*) printf '%s\n' 0 ;;
    *) return 1 ;;
  esac
}

load_qnn_package_manifest() {
  package_root="$1"
  manifest_path="$2"

  OMNIINFER_NATIVE_FORMAT=""
  OMNIINFER_NATIVE_RUNNER=""
  OMNIINFER_NATIVE_DECODER_MODEL_VERSION=""
  OMNIINFER_NATIVE_TOKENIZER=""
  OMNIINFER_NATIVE_TEXT_DECODER=""
  OMNIINFER_NATIVE_VISION_ENCODER=""
  OMNIINFER_NATIVE_TOK_EMBEDDING=""
  OMNIINFER_NATIVE_ATTENTION_SINK=""
  OMNIINFER_NATIVE_EVAL_MODE=""

  # shellcheck disable=SC1090
  . "${manifest_path}"

  native_runner="${OMNIINFER_NATIVE_RUNNER:-llama}"
  case "${native_runner}" in
    llama|multimodal) ;;
    *)
      echo "Unsupported omniinfer-native runner in ${manifest_path}: ${native_runner}" >&2
      exit 1
      ;;
  esac

  native_text_decoder="$(resolve_qnn_component_path "${package_root}" "${OMNIINFER_NATIVE_TEXT_DECODER:-}" || true)"
  if [ -z "${native_text_decoder}" ] || [ ! -f "${native_text_decoder}" ]; then
    echo "omniinfer-native package is missing a valid text decoder in ${manifest_path}" >&2
    exit 1
  fi

  native_tokenizer="$(resolve_qnn_component_path "${package_root}" "${OMNIINFER_NATIVE_TOKENIZER:-}" || true)"
  if [ -z "${native_tokenizer}" ] && [ -f "${package_root}/tokenizer.json" ]; then
    native_tokenizer="${package_root}/tokenizer.json"
  fi

  native_version="${OMNIINFER_NATIVE_DECODER_MODEL_VERSION:-}"
  if [ -z "${native_version}" ]; then
    native_version="$(infer_decoder_model_version "${native_text_decoder}" || infer_decoder_model_version "${package_root}" || true)"
  fi

  native_eval_mode="${OMNIINFER_NATIVE_EVAL_MODE:-}"
  if [ -z "${native_eval_mode}" ]; then
    native_eval_mode="$(infer_qnn_eval_mode_from_model_path "${native_text_decoder}" || printf '%s' 1)"
  fi

  native_vision_encoder="$(resolve_qnn_component_path "${package_root}" "${OMNIINFER_NATIVE_VISION_ENCODER:-}" || true)"
  native_tok_embedding="$(resolve_qnn_component_path "${package_root}" "${OMNIINFER_NATIVE_TOK_EMBEDDING:-}" || true)"
  native_attention_sink="$(resolve_qnn_component_path "${package_root}" "${OMNIINFER_NATIVE_ATTENTION_SINK:-}" || true)"

  if [ "${native_runner}" = "multimodal" ]; then
    if [ -z "${native_vision_encoder}" ] || [ ! -f "${native_vision_encoder}" ]; then
      echo "omniinfer-native multimodal package is missing a valid vision encoder in ${manifest_path}" >&2
      exit 1
    fi
    if [ -z "${native_tok_embedding}" ] || [ ! -f "${native_tok_embedding}" ]; then
      echo "omniinfer-native multimodal package is missing a valid token embedding model in ${manifest_path}" >&2
      exit 1
    fi
  fi

  NATIVE_PACKAGE_ROOT="${package_root}"
  NATIVE_RUNNER_KIND="${native_runner}"
  NATIVE_TEXT_DECODER_PATH="${native_text_decoder}"
  NATIVE_VISION_ENCODER_PATH="${native_vision_encoder}"
  NATIVE_TOK_EMBEDDING_PATH="${native_tok_embedding}"
  NATIVE_ATTENTION_SINK_PATH="${native_attention_sink}"
  NATIVE_EVAL_MODE="${native_eval_mode}"
  MODEL_PATH="${native_text_decoder}"

  if [ -z "${TOKENIZER_PATH}" ] && [ -n "${native_tokenizer}" ]; then
    TOKENIZER_PATH="${native_tokenizer}"
  fi
  if [ -z "${DECODER_MODEL_VERSION}" ] && [ -n "${native_version}" ]; then
    DECODER_MODEL_VERSION="${native_version}"
  fi
}

qnn_runner_candidates() {
  case "${NATIVE_RUNNER_KIND:-}" in
    llama) printf '%s\n' qnn_llama_runner ;;
    multimodal) printf '%s\n' qnn_multimodal_runner ;;
    *)
      printf '%s\n' qnn_llama_runner
      printf '%s\n' qnn_multimodal_runner
      ;;
  esac
}

find_qnn_runner_by_name() {
  runner_name="$1"
  package_root="$(qnn_package_root)"
  if [ -x "${QNN_ROOT}/${runner_name}" ]; then
    printf '%s\n' "${QNN_ROOT}/${runner_name}"
    return 0
  fi
  if [ -n "${package_root:-}" ] && [ -x "${package_root}/${runner_name}" ]; then
    printf '%s\n' "${package_root}/${runner_name}"
    return 0
  fi
  return 1
}

native_backend_binary() {
  for runner_name in $(qnn_runner_candidates); do
    runner_path="$(find_qnn_runner_by_name "${runner_name}" 2>/dev/null || true)"
    if [ -n "${runner_path}" ]; then
      printf '%s\n' "${runner_path}"
      return 0
    fi
  done
  return 1
}

native_prepare_model_load() {
  requested_model_path="${MODEL_PATH}"
  manifest_path="$(qnn_manifest_path "${requested_model_path}" 2>/dev/null || true)"
  if [ -n "${manifest_path}" ]; then
    load_qnn_package_manifest "$(dirname "${manifest_path}")" "${manifest_path}"
    model_dir="$(qnn_package_root)"
  else
    MODEL_PATH="$(discover_qnn_model "${requested_model_path}")"
    model_dir="$(qnn_model_dir "${MODEL_PATH}")"
    NATIVE_PACKAGE_ROOT="${model_dir}"
    NATIVE_RUNNER_KIND="llama"
    NATIVE_TEXT_DECODER_PATH="${MODEL_PATH}"
    NATIVE_EVAL_MODE="$(infer_qnn_eval_mode_from_model_path "${MODEL_PATH}" || printf '%s' 1)"
    if [ -z "${TOKENIZER_PATH}" ] && [ -f "${model_dir}/tokenizer.json" ]; then
      TOKENIZER_PATH="${model_dir}/tokenizer.json"
    fi
    if [ -z "${DECODER_MODEL_VERSION}" ]; then
      DECODER_MODEL_VERSION="$(infer_decoder_model_version "${MODEL_PATH}" || infer_decoder_model_version "${model_dir}" || true)"
    fi
  fi

  if [ -z "${TOKENIZER_PATH}" ]; then
    echo "QNN model load requires tokenizer.json. Use --tokenizer-path or place tokenizer.json next to the model." >&2
    exit 1
  fi
  if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "Tokenizer was not found: ${TOKENIZER_PATH}" >&2
    exit 1
  fi
  if [ -z "${DECODER_MODEL_VERSION}" ]; then
    echo "QNN model load requires --decoder-model-version because it could not be inferred from the model path." >&2
    exit 1
  fi
}

print_qnn_output() {
  raw_path="$1"
  think_mode="$2"
  tmp_path="${STATE_ROOT}/qnn-output.tmp"

  if grep -q '<|im_start|>assistant' "${raw_path}" 2>/dev/null; then
    awk '
      BEGIN { capture = 0 }
      {
        line = $0
        if (capture == 0) {
          marker = "<|im_start|>assistant"
          idx = index(line, marker)
          if (idx == 0) {
            next
          }
          capture = 1
          line = substr(line, idx + length(marker))
        }
        sub(/<\|im_end\|>.*/, "", line)
        print line
        if ($0 ~ /<\|im_end\|>/) {
          exit
        }
      }
    ' "${raw_path}" > "${tmp_path}"
  else
    cp "${raw_path}" "${tmp_path}"
  fi

  if [ "${think_mode}" = "off" ]; then
    awk '
      BEGIN { skip = 0 }
      /<think>/ { skip = 1; next }
      /<\/think>/ { skip = 0; next }
      skip == 0 { print }
    ' "${tmp_path}" > "${tmp_path}.stripped"
    mv "${tmp_path}.stripped" "${tmp_path}"
  fi

  sed '/./,$!d' "${tmp_path}"
}

trim_qnn_stdout() {
  input_path="$1"
  output_path="$2"
  awk '
    BEGIN {
      skip = 0
    }
    {
      line = $0
      if (line ~ /^PyTorchObserver /) {
        next
      }
      gsub(/<\|im_end\|>|<\|eot_id\|>|<\|end\|>|<end_of_turn>|<\|end_of_text\|>|<\|endoftext\|>/, "", line)
      if (skip == 1) {
        if (line ~ /<\/think>/) {
          skip = 0
        }
        next
      }
      if (line ~ /<think>/) {
        skip = 1
        next
      }
      print line
    }
  ' "${input_path}" > "${output_path}"
}

native_run_chat() {
  native_message="$1"
  native_image="$2"
  native_stream="$3"
  native_system_prompt="$4"

  if [ -z "${TOKENIZER_PATH}" ] || [ -z "${DECODER_MODEL_VERSION}" ]; then
    echo "QNN backend requires tokenizer and decoder model version. Run model load with --tokenizer-path and --decoder-model-version first." >&2
    exit 1
  fi

  runner_path="$(native_backend_binary 2>/dev/null || true)"
  if [ -z "${runner_path}" ] || [ ! -x "${runner_path}" ]; then
    echo "QNN runner was not found. Prepare .local/runtime/android/qnn or place the required runner next to the package." >&2
    exit 1
  fi

  runner_root="$(dirname "${runner_path}")"
  output_dir="${STATE_ROOT}/qnn-outputs"
  output_path="${output_dir}/outputs.txt"
  perf_path="${output_dir}/inference_speed.txt"
  stdout_path="${output_dir}/stdout.txt"
  stderr_path="${output_dir}/stderr.txt"
  effective_ctx="${CTX_SIZE:-2048}"
  effective_eval_mode="${NATIVE_EVAL_MODE:-1}"
  mkdir -p "${output_dir}"
  rm -f "${output_path}" "${perf_path}" "${stdout_path}" "${stderr_path}"

  if [ "${NATIVE_RUNNER_KIND:-llama}" = "multimodal" ]; then
    if [ -z "${native_image}" ]; then
      echo "This omniinfer-native package requires --image because it uses the multimodal ExecuTorch runner." >&2
      exit 1
    fi
    if [ -z "${NATIVE_VISION_ENCODER_PATH}" ] || [ -z "${NATIVE_TOK_EMBEDDING_PATH}" ]; then
      echo "Multimodal omniinfer-native package is missing encoder or token embedding paths." >&2
      exit 1
    fi
    if ! (
      cd "${runner_root}" &&
      export LD_LIBRARY_PATH="${runner_root}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" &&
      "${runner_path}" \
        --decoder_model_version "${DECODER_MODEL_VERSION}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --output_path "${output_path}" \
        --performance_output_path "${perf_path}" \
        --shared_buffer \
        --decoder_path "${NATIVE_TEXT_DECODER_PATH:-${MODEL_PATH}}" \
        --encoder_path "${NATIVE_VISION_ENCODER_PATH}" \
        --tok_embedding_path "${NATIVE_TOK_EMBEDDING_PATH}" \
        --eval_mode "${effective_eval_mode}" \
        --temperature 0.0 \
        --system_prompt "${native_system_prompt}" \
        --seq_len "${effective_ctx}" \
        --prompt "${native_message}" \
        --image_path "${native_image}" \
        > /dev/null 2> "${stderr_path}"
    ); then
      echo "QNN multimodal runner execution failed." >&2
      cat "${stderr_path}" >&2 || true
      exit 1
    fi

    if [ ! -f "${output_path}" ]; then
      echo "QNN multimodal runner completed without producing ${output_path}" >&2
      cat "${stderr_path}" >&2 || true
      exit 1
    fi

    print_qnn_output "${output_path}" "${THINKING}"
    return 0
  fi

  if [ -n "${native_image}" ]; then
    echo "The current text omniinfer-native package does not support --image. Use a multimodal package instead." >&2
    exit 1
  fi

  if (
    cd "${runner_root}" &&
    export LD_LIBRARY_PATH="${runner_root}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" &&
    "${runner_path}" \
      --decoder_model_version "${DECODER_MODEL_VERSION}" \
      --tokenizer_path "${TOKENIZER_PATH}" \
      --output_path "${output_path}" \
      --performance_output_path "${perf_path}" \
      --shared_buffer \
      --model_path "${NATIVE_TEXT_DECODER_PATH:-${MODEL_PATH}}" \
      $( [ -n "${NATIVE_ATTENTION_SINK_PATH:-}" ] && printf '%s %s' "--attention_sink_rope_path" "${NATIVE_ATTENTION_SINK_PATH}" ) \
      --eval_mode "${effective_eval_mode}" \
      --temperature 0.0 \
      --system_prompt "${native_system_prompt}" \
      --seq_len "${effective_ctx}" \
      --prompt "${native_message}" \
      --simple_io \
      $( [ "${THINKING}" = "off" ] && printf '%s' "--strip_thinking" ) \
      > "${stdout_path}" 2> "${stderr_path}"
  ); then
    if [ -s "${stdout_path}" ]; then
      trim_qnn_stdout "${stdout_path}" "${stdout_path}.clean"
      sed '/./,$!d' "${stdout_path}.clean"
      return 0
    fi
  elif grep -q "unknown command line flag 'simple_io'" "${stderr_path}" 2>/dev/null; then
    :
  else
    echo "QNN runner execution failed." >&2
    cat "${stderr_path}" >&2 || true
    exit 1
  fi

  rm -f "${stdout_path}" "${stderr_path}"
  if ! (
    cd "${runner_root}" &&
    export LD_LIBRARY_PATH="${runner_root}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" &&
    "${runner_path}" \
      --decoder_model_version "${DECODER_MODEL_VERSION}" \
      --tokenizer_path "${TOKENIZER_PATH}" \
      --output_path "${output_path}" \
      --performance_output_path "${perf_path}" \
      --shared_buffer \
      --model_path "${NATIVE_TEXT_DECODER_PATH:-${MODEL_PATH}}" \
      $( [ -n "${NATIVE_ATTENTION_SINK_PATH:-}" ] && printf '%s %s' "--attention_sink_rope_path" "${NATIVE_ATTENTION_SINK_PATH}" ) \
      --eval_mode "${effective_eval_mode}" \
      --temperature 0.0 \
      --system_prompt "${native_system_prompt}" \
      --seq_len "${effective_ctx}" \
      --prompt "${native_message}" \
      > /dev/null 2> "${stderr_path}"
  ); then
    echo "QNN runner execution failed." >&2
    cat "${stderr_path}" >&2 || true
    exit 1
  fi

  if [ ! -f "${output_path}" ]; then
    echo "QNN runner completed without producing ${output_path}" >&2
    cat "${stderr_path}" >&2 || true
    exit 1
  fi

  print_qnn_output "${output_path}" "${THINKING}"
}
