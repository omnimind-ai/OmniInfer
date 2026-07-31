#!/system/bin/sh

set -eu

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ ! -x "${PACKAGE_ROOT}/bin/llama-zo" ]; then
  echo "llama-zo was not found in ${PACKAGE_ROOT}/bin" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${PACKAGE_ROOT}/bin:${PACKAGE_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ADSP_LIBRARY_PATH="${PACKAGE_ROOT}/lib${ADSP_LIBRARY_PATH:+:${ADSP_LIBRARY_PATH}}"
export DSP_LIBRARY_PATH="${PACKAGE_ROOT}/lib${DSP_LIBRARY_PATH:+:${DSP_LIBRARY_PATH}}"

exec "${PACKAGE_ROOT}/bin/llama-zo" "$@"
