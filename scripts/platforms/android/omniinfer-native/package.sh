#!/usr/bin/env bash

set -euo pipefail

ARTIFACT_DIR=""
RUNNER_KIND="auto"
DECODER_MODEL=""
DECODER_MODEL_VERSION=""
TOKENIZER_PATH=""
TEXT_DECODER_PATH=""
VISION_ENCODER_PATH=""
TOK_EMBEDDING_PATH=""
ATTENTION_SINK_PATH=""
EVAL_MODE=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: package-omniinfer-native.sh --artifact-dir <dir> [options]

Generate an omniinfer-native package manifest for official ExecuTorch Qualcomm llama artifacts.

Options:
  --artifact-dir <dir>         Artifact directory produced by ExecuTorch Qualcomm llama export
  --runner-kind <kind>         One of: auto, llama, multimodal
  --decoder-model <name>       Official ExecuTorch decoder model name such as qwen3-0_6b
  --decoder-model-version <v>  Explicit runtime decoder model version such as qwen3
  --tokenizer-path <path>      Explicit tokenizer path inside or outside the artifact directory
  --text-decoder <path>        Explicit text decoder .pte path
  --vision-encoder <path>      Explicit multimodal vision encoder .pte path
  --tok-embedding <path>       Explicit multimodal token embedding .pte path
  --attention-sink <path>      Optional attention sink evictor .pte path
  --eval-mode <0|1|2>          Explicit ExecuTorch runner eval mode
  --dry-run                    Print the manifest without writing files
  -h, --help                   Show this help text
EOF
}

while (($# > 0)); do
  case "$1" in
    --artifact-dir)
      ARTIFACT_DIR="${2:?missing value for --artifact-dir}"
      shift 2
      ;;
    --runner-kind)
      RUNNER_KIND="${2:?missing value for --runner-kind}"
      shift 2
      ;;
    --decoder-model)
      DECODER_MODEL="${2:?missing value for --decoder-model}"
      shift 2
      ;;
    --decoder-model-version)
      DECODER_MODEL_VERSION="${2:?missing value for --decoder-model-version}"
      shift 2
      ;;
    --tokenizer-path)
      TOKENIZER_PATH="${2:?missing value for --tokenizer-path}"
      shift 2
      ;;
    --text-decoder)
      TEXT_DECODER_PATH="${2:?missing value for --text-decoder}"
      shift 2
      ;;
    --vision-encoder)
      VISION_ENCODER_PATH="${2:?missing value for --vision-encoder}"
      shift 2
      ;;
    --tok-embedding)
      TOK_EMBEDDING_PATH="${2:?missing value for --tok-embedding}"
      shift 2
      ;;
    --attention-sink)
      ATTENTION_SINK_PATH="${2:?missing value for --attention-sink}"
      shift 2
      ;;
    --eval-mode)
      EVAL_MODE="${2:?missing value for --eval-mode}"
      shift 2
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

if [[ -z "${ARTIFACT_DIR}" ]]; then
  echo "--artifact-dir is required." >&2
  usage >&2
  exit 1
fi

resolve_path() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    return 1
  fi
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${ARTIFACT_DIR}" "${path}"
  fi
}

normalize_abs_path() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    return 1
  fi
  local resolved
  resolved="$(resolve_path "${path}")"
  if [[ ! -e "${resolved}" ]]; then
    return 1
  fi
  (cd "$(dirname "${resolved}")" && printf '%s/%s\n' "$(pwd)" "$(basename "${resolved}")")
}

manifest_ref() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    return 1
  fi
  if [[ "${path}" == "${ARTIFACT_DIR}/"* ]]; then
    printf '%s\n' "${path#${ARTIFACT_DIR}/}"
  else
    printf '%s\n' "${path}"
  fi
}

pick_existing_file() {
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

infer_runner_kind() {
  if [[ -n "${VISION_ENCODER_PATH}" || -n "${TOK_EMBEDDING_PATH}" ]]; then
    printf '%s\n' multimodal
    return 0
  fi
  if [[ -f "${ARTIFACT_DIR}/vision_encoder_qnn.pte" || -f "${ARTIFACT_DIR}/tok_embedding_qnn.pte" ]]; then
    printf '%s\n' multimodal
    return 0
  fi
  printf '%s\n' llama
}

infer_text_decoder() {
  pick_existing_file \
    "${ARTIFACT_DIR}/hybrid_llama_qnn.pte" \
    "${ARTIFACT_DIR}/lookahead_llama_qnn.pte" \
    "${ARTIFACT_DIR}/kv_llama_qnn.pte" \
    "${ARTIFACT_DIR}/stories260k_hybrid_llama_qnn.pte" \
    "${ARTIFACT_DIR}/stories260k_kv_llama_qnn.pte"
}

infer_eval_mode() {
  local path="$1"
  local lower
  lower="$(printf '%s' "${path}" | tr '[:upper:]' '[:lower:]')"
  case "${lower}" in
    *lookahead*) printf '%s\n' 2 ;;
    *hybrid*) printf '%s\n' 1 ;;
    *kv*) printf '%s\n' 0 ;;
    *) return 1 ;;
  esac
}

infer_decoder_model_version_from_name() {
  case "$1" in
    stories260k|stories110m) printf '%s\n' llama2 ;;
    llama3_2-1b_instruct|llama3_2-3b_instruct) printf '%s\n' llama3 ;;
    codegen2_1b) printf '%s\n' codegen ;;
    gemma-2b) printf '%s\n' gemma ;;
    gemma2-2b) printf '%s\n' gemma2 ;;
    gemma3-1b) printf '%s\n' gemma3 ;;
    granite_3_3-2b_instruct) printf '%s\n' granite ;;
    phi_4_mini) printf '%s\n' phi_4_mini ;;
    qwen2_5-0_5b|qwen2_5-1_5b) printf '%s\n' qwen2_5 ;;
    qwen3-0_6b|qwen3-1_7b) printf '%s\n' qwen3 ;;
    smollm2_135m) printf '%s\n' smollm2_135m ;;
    smollm3-3b) printf '%s\n' smollm3 ;;
    glm-1_5b) printf '%s\n' glm ;;
    smolvlm_500m_instruct) printf '%s\n' smolvlm ;;
    internvl3_1b) printf '%s\n' internvl3 ;;
    *) return 1 ;;
  esac
}

ARTIFACT_DIR="$(cd "${ARTIFACT_DIR}" && pwd)"
MANIFEST_PATH="${ARTIFACT_DIR}/omniinfer-native.env"

if [[ "${RUNNER_KIND}" = "auto" ]]; then
  RUNNER_KIND="$(infer_runner_kind)"
fi
case "${RUNNER_KIND}" in
  llama|multimodal) ;;
  *)
    echo "Unsupported --runner-kind: ${RUNNER_KIND}" >&2
    exit 1
    ;;
esac

if [[ -n "${TEXT_DECODER_PATH}" ]]; then
  TEXT_DECODER_PATH="$(normalize_abs_path "${TEXT_DECODER_PATH}")"
else
  TEXT_DECODER_PATH="$(infer_text_decoder || true)"
fi
if [[ -z "${TEXT_DECODER_PATH}" || ! -f "${TEXT_DECODER_PATH}" ]]; then
  echo "Unable to locate the text decoder .pte under ${ARTIFACT_DIR}" >&2
  exit 1
fi

if [[ -n "${TOKENIZER_PATH}" ]]; then
  TOKENIZER_PATH="$(normalize_abs_path "${TOKENIZER_PATH}")"
else
  TOKENIZER_PATH="$(pick_existing_file \
    "${ARTIFACT_DIR}/tokenizer.json" \
    "${ARTIFACT_DIR}/tokenizer.bin" \
    "${ARTIFACT_DIR}/tokenizer.model" || true)"
fi
if [[ -z "${TOKENIZER_PATH}" || ! -f "${TOKENIZER_PATH}" ]]; then
  echo "Unable to locate tokenizer.json/tokenizer.bin/tokenizer.model under ${ARTIFACT_DIR}" >&2
  exit 1
fi

if [[ -n "${VISION_ENCODER_PATH}" ]]; then
  VISION_ENCODER_PATH="$(normalize_abs_path "${VISION_ENCODER_PATH}")"
fi
if [[ -n "${TOK_EMBEDDING_PATH}" ]]; then
  TOK_EMBEDDING_PATH="$(normalize_abs_path "${TOK_EMBEDDING_PATH}")"
fi
if [[ -n "${ATTENTION_SINK_PATH}" ]]; then
  ATTENTION_SINK_PATH="$(normalize_abs_path "${ATTENTION_SINK_PATH}")"
fi

if [[ "${RUNNER_KIND}" = "multimodal" ]]; then
  if [[ -z "${VISION_ENCODER_PATH}" ]]; then
    VISION_ENCODER_PATH="$(pick_existing_file "${ARTIFACT_DIR}/vision_encoder_qnn.pte" || true)"
  fi
  if [[ -z "${TOK_EMBEDDING_PATH}" ]]; then
    TOK_EMBEDDING_PATH="$(pick_existing_file "${ARTIFACT_DIR}/tok_embedding_qnn.pte" || true)"
  fi
  if [[ -z "${VISION_ENCODER_PATH}" || ! -f "${VISION_ENCODER_PATH}" ]]; then
    echo "Multimodal package requires vision_encoder_qnn.pte" >&2
    exit 1
  fi
  if [[ -z "${TOK_EMBEDDING_PATH}" || ! -f "${TOK_EMBEDDING_PATH}" ]]; then
    echo "Multimodal package requires tok_embedding_qnn.pte" >&2
    exit 1
  fi
else
  if [[ -z "${ATTENTION_SINK_PATH}" ]]; then
    ATTENTION_SINK_PATH="$(pick_existing_file "${ARTIFACT_DIR}/attention_sink_evictor.pte" || true)"
  fi
fi

if [[ -z "${DECODER_MODEL_VERSION}" && -n "${DECODER_MODEL}" ]]; then
  DECODER_MODEL_VERSION="$(infer_decoder_model_version_from_name "${DECODER_MODEL}" || true)"
fi
if [[ -z "${DECODER_MODEL_VERSION}" ]]; then
  echo "Unable to infer decoder model version. Pass --decoder-model or --decoder-model-version." >&2
  exit 1
fi

if [[ -z "${EVAL_MODE}" ]]; then
  EVAL_MODE="$(infer_eval_mode "${TEXT_DECODER_PATH}" || printf '%s' 1)"
fi
case "${EVAL_MODE}" in
  0|1|2) ;;
  *)
    echo "--eval-mode must be one of 0, 1, or 2" >&2
    exit 1
    ;;
esac

manifest_text=$(
  cat <<EOF
OMNIINFER_NATIVE_FORMAT=1
OMNIINFER_NATIVE_RUNNER=${RUNNER_KIND}
OMNIINFER_NATIVE_DECODER_MODEL_VERSION=${DECODER_MODEL_VERSION}
OMNIINFER_NATIVE_TOKENIZER=$(manifest_ref "${TOKENIZER_PATH}")
OMNIINFER_NATIVE_TEXT_DECODER=$(manifest_ref "${TEXT_DECODER_PATH}")
OMNIINFER_NATIVE_EVAL_MODE=${EVAL_MODE}
EOF
)

if [[ -n "${VISION_ENCODER_PATH}" ]]; then
  manifest_text="${manifest_text}
OMNIINFER_NATIVE_VISION_ENCODER=$(manifest_ref "${VISION_ENCODER_PATH}")"
fi
if [[ -n "${TOK_EMBEDDING_PATH}" ]]; then
  manifest_text="${manifest_text}
OMNIINFER_NATIVE_TOK_EMBEDDING=$(manifest_ref "${TOK_EMBEDDING_PATH}")"
fi
if [[ -n "${ATTENTION_SINK_PATH}" ]]; then
  manifest_text="${manifest_text}
OMNIINFER_NATIVE_ATTENTION_SINK=$(manifest_ref "${ATTENTION_SINK_PATH}")"
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  printf '%s\n' "${manifest_text}"
  exit 0
fi

printf '%s\n' "${manifest_text}" > "${MANIFEST_PATH}"

cat <<EOF
Created omniinfer-native package manifest:
  ${MANIFEST_PATH}

Resolved package settings:
  runner=${RUNNER_KIND}
  decoder_model_version=${DECODER_MODEL_VERSION}
  eval_mode=${EVAL_MODE}
  tokenizer=$(manifest_ref "${TOKENIZER_PATH}")
  text_decoder=$(manifest_ref "${TEXT_DECODER_PATH}")
EOF

if [[ -n "${VISION_ENCODER_PATH}" ]]; then
  printf '  vision_encoder=%s\n' "$(manifest_ref "${VISION_ENCODER_PATH}")"
fi
if [[ -n "${TOK_EMBEDDING_PATH}" ]]; then
  printf '  tok_embedding=%s\n' "$(manifest_ref "${TOK_EMBEDDING_PATH}")"
fi
if [[ -n "${ATTENTION_SINK_PATH}" ]]; then
  printf '  attention_sink=%s\n' "$(manifest_ref "${ATTENTION_SINK_PATH}")"
fi
