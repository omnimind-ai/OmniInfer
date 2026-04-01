llama_cpp_backend_binary() {
  case "$1" in
    "${BACKEND_LLAMA}")
      if [ -x "${LIB_ROOT}/libllama-cli.so" ]; then
        printf '%s\n' "${LIB_ROOT}/libllama-cli.so"
      else
        printf '%s\n' "${LEGACY_LIB_ROOT}/libllama-cli.so"
      fi
      ;;
    "${BACKEND_MTMD}")
      if [ -x "${LIB_ROOT}/libmtmd-cli.so" ]; then
        printf '%s\n' "${LIB_ROOT}/libmtmd-cli.so"
      elif [ -x "${LEGACY_LIB_ROOT}/libmtmd-cli.so" ]; then
        printf '%s\n' "${LEGACY_LIB_ROOT}/libmtmd-cli.so"
      elif [ -x "${LIB_ROOT}/libllama-cli.so" ]; then
        printf '%s\n' "${LIB_ROOT}/libllama-cli.so"
      else
        printf '%s\n' "${LEGACY_LIB_ROOT}/libllama-cli.so"
      fi
      ;;
    *)
      return 1
      ;;
  esac
}
