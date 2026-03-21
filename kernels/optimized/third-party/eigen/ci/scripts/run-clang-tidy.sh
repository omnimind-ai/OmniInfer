#!/bin/bash
#
# Run clang-tidy on files changed in the current MR.
#
# Usage: run-clang-tidy.sh <base_sha> <build_dir>
#
# <base_sha>   The merge-base commit to diff against.
# <build_dir>  Path to a CMake build directory containing compile_commands.json.
#
# For header files under Eigen/src/<Module>/, the script generates a minimal
# driver .cpp that includes the parent module header so that
# InternalHeaderCheck.h does not #error out.

set -euo pipefail

BASE_SHA="${1:?Usage: run-clang-tidy.sh <base_sha> <build_dir>}"
BUILD_DIR="${2:?Usage: run-clang-tidy.sh <base_sha> <build_dir>}"

if [ ! -f "${BUILD_DIR}/compile_commands.json" ]; then
  echo "ERROR: ${BUILD_DIR}/compile_commands.json not found."
  echo "Run cmake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON first."
  exit 1
fi

# External-dependency modules that require third-party headers we don't have,
# and utility-only directories with no standalone module header.
SKIP_MODULES="AccelerateSupport|CholmodSupport|KLUSupport|MetisSupport|PaStiXSupport|PardisoSupport|SPQRSupport|SuperLUSupport|UmfPackSupport|TensorUtil"

# Get changed files (Added, Modified, Renamed).
CHANGED_FILES=$(git diff --name-only --diff-filter=AMR "${BASE_SHA}" HEAD)

if [ -z "${CHANGED_FILES}" ]; then
  echo "No changed files to check."
  exit 0
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

ERRORS=0

# Map a header path under Eigen/src/<Module>/ to its module include.
# e.g. Eigen/src/SVD/BDCSVD.h -> Eigen/SVD
module_include_for_header() {
  local header="$1"
  local module

  # Handle Eigen/src/<Module>/... and unsupported/Eigen/src/<Module>/...
  # Extract just the bare module name first.
  if [[ "${header}" =~ ^Eigen/src/([^/]+)/ ]]; then
    module="${BASH_REMATCH[1]}"
  elif [[ "${header}" =~ ^unsupported/Eigen/src/([^/]+)/ ]]; then
    module="${BASH_REMATCH[1]}"
  else
    return 1
  fi

  # Skip external-dependency modules and utility-only directories.
  if [[ "${module}" =~ ^(${SKIP_MODULES})$ ]]; then
    return 1
  fi

  if [[ "${header}" =~ ^unsupported/ ]]; then
    echo "unsupported/Eigen/${module}"
  else
    echo "Eigen/${module}"
  fi
}

# Pick a compile command from compile_commands.json to use as a template.
# We just need a valid set of compiler flags.
TEMPLATE_CMD=$(python3 -c "
import json, sys
with open('${BUILD_DIR}/compile_commands.json') as f:
    cmds = json.load(f)
# Pick the first .cpp entry.
for c in cmds:
    if c['file'].endswith('.cpp'):
        print(c['directory'])
        break
")

echo "Checking changed files with clang-tidy..."
echo "Base SHA: ${BASE_SHA}"
echo ""

for file in ${CHANGED_FILES}; do
  # Only check C++ source and header files.
  case "${file}" in
    *.cpp|*.cc|*.cxx)
      # Source file: run clang-tidy directly if it's in the compilation database.
      if grep -q "\"${file}\"" "${BUILD_DIR}/compile_commands.json" 2>/dev/null; then
        echo "=== ${file} ==="
        if ! clang-tidy -p "${BUILD_DIR}" "${file}" 2>&1; then
          ERRORS=$((ERRORS + 1))
        fi
      fi
      ;;
    *.h|*.hpp)
      # Header file: generate a driver .cpp that includes the right module.
      MODULE_INCLUDE=$(module_include_for_header "${file}" || true)
      if [ -z "${MODULE_INCLUDE}" ]; then
        # Not a recognized module header or in skip list.
        continue
      fi

      DRIVER="${TMPDIR}/tidy_driver_$(echo "${file}" | tr '/' '_').cpp"
      cat > "${DRIVER}" <<EOF
#include <${MODULE_INCLUDE}>
EOF

      echo "=== ${file} (via ${MODULE_INCLUDE}) ==="
      if ! clang-tidy \
            -p "${BUILD_DIR}" \
            --header-filter="$(echo "${file}" | sed 's/[.[\*^$()+?{|]/\\&/g')" \
            "${DRIVER}" 2>&1; then
        ERRORS=$((ERRORS + 1))
      fi
      ;;
  esac
done

if [ ${ERRORS} -gt 0 ]; then
  echo ""
  echo "clang-tidy reported issues in ${ERRORS} file(s)."
  exit 1
else
  echo ""
  echo "clang-tidy: all clean."
  exit 0
fi
