#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

while IFS= read -r build_script; do
    grep -q 'LLAMA_BUILD_UI=' "${build_script}" ||
        fail "${build_script#"${REPO_ROOT}/"} does not define the llama.cpp UI policy"
    grep -q 'LLAMA_USE_PREBUILT_UI=' "${build_script}" ||
        fail "${build_script#"${REPO_ROOT}/"} does not define the prebuilt UI policy"
    if grep -q 'LLAMA_BUILD_WEBUI=' "${build_script}"; then
        fail "${build_script#"${REPO_ROOT}/"} uses the deprecated llama.cpp Web UI option"
    fi
done < <(find \
    "${REPO_ROOT}/scripts/platforms/linux" \
    "${REPO_ROOT}/scripts/platforms/macos" \
    "${REPO_ROOT}/scripts/platforms/windows" \
    -mindepth 2 -maxdepth 2 -type f \( -name build.sh -o -name build.ps1 \) \
    -path '*/llama.cpp-*/*' | sort)

while IFS= read -r build_script; do
    bash "${build_script}" --help | grep -q -- '--from-source' ||
        fail "${build_script#"${REPO_ROOT}/"} does not expose --from-source"
done < <(find "${REPO_ROOT}/scripts/platforms/linux" "${REPO_ROOT}/scripts/platforms/macos" \
    -mindepth 2 -maxdepth 2 -type f -name build.sh | sort)

for build_script in \
    "${REPO_ROOT}/scripts/platforms/macos/llama.cpp-mac/build.sh" \
    "${REPO_ROOT}/scripts/platforms/macos/llama.cpp-mac-intel/build.sh"; do
    grep -q 'submodule update .*--depth 1 framework/llama.cpp' "${build_script}" ||
        fail "${build_script#"${REPO_ROOT}/"} does not use a shallow submodule bootstrap"
    if grep -Eq '\[\[ -(f|n) .*\]\] \|\| return$' "${build_script}"; then
        fail "${build_script#"${REPO_ROOT}/"} can fail a normal cache fast path under set -e"
    fi
done

install_dir="${TEST_ROOT}/checkout"
fake_bin="${TEST_ROOT}/bin"
case "$(uname -s)" in
    Darwin)
        platform_dir="macos"
        backend_id="llama.cpp-mac"
        ;;
    *)
        platform_dir="linux"
        backend_id="llama.cpp-linux"
        ;;
esac
mkdir -p \
    "${install_dir}/.git" \
    "${install_dir}/scripts/lib" \
    "${install_dir}/scripts/platforms/${platform_dir}/${backend_id}" \
    "${fake_bin}"

cp "${REPO_ROOT}/scripts/lib/source-install-deps.sh" "${install_dir}/scripts/lib/source-install-deps.sh"

cat >"${install_dir}/omniinfer" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${FIXTURE_COMMANDS_FILE:-}" ]]; then
    printf '%s\n' "$*" >>"${FIXTURE_COMMANDS_FILE}"
fi
case "$*" in
    --version) echo "omniinfer fixture" ;;
    "backend list --scope compatible --json") printf '{"data":[{"id":"%s","selector":"public-cpu"}]}\n' "${FIXTURE_BACKEND_ID:?}" ;;
    "backend resolve public-cpu") echo "${FIXTURE_BACKEND_ID:?}" ;;
    "backend list --scope installed") exit 0 ;;
esac
EOF
chmod +x "${install_dir}/omniinfer"

cat >"${fake_bin}/cargo" <<'EOF'
#!/usr/bin/env bash
mkdir -p target/debug
cp omniinfer target/debug/omniinfer
chmod +x target/debug/omniinfer
EOF
cat >"${fake_bin}/git" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "${fake_bin}/cargo" "${fake_bin}/git"

cat >"${install_dir}/scripts/platforms/${platform_dir}/${backend_id}/build.sh" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--check-deps" ]]; then
    exit 0
fi
printf '%s\n' "$@" >"${BUILD_ARGS_FILE:?}"
EOF
chmod +x "${install_dir}/scripts/platforms/${platform_dir}/${backend_id}/build.sh"

BUILD_ARGS_FILE="${TEST_ROOT}/build-args.txt" \
FIXTURE_COMMANDS_FILE="${TEST_ROOT}/commands.txt" \
FIXTURE_BACKEND_ID="${backend_id}" \
PATH="${fake_bin}:${PATH}" \
bash "${REPO_ROOT}/scripts/install-from-source.sh" \
    --install-dir "${install_dir}" \
    --backend "public-cpu" \
    --from-source \
    --no-model \
    --non-interactive \
    --no-install-system-deps >"${TEST_ROOT}/installer-output.txt"

[[ "$(cat "${TEST_ROOT}/build-args.txt")" == "--from-source" ]] ||
    fail "source installer did not preserve the backend source-build contract"
grep -qF './omniinfer serve --detach' "${TEST_ROOT}/installer-output.txt" ||
    fail "source installer completion guidance does not start the gateway"
grep -Eq '^serve --detach --port [0-9]+ --no-restore-model$' "${TEST_ROOT}/commands.txt" ||
    fail "source installer backend activation can restore a previous model"
if grep -qF 'The CLI auto-starts the service if needed.' "${REPO_ROOT}/scripts/install-from-source.sh"; then
    fail "source installer claims that model commands auto-start the gateway"
fi
python3 - "${install_dir}/.local/install-summary.json" "${backend_id}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["backend"] == sys.argv[2], summary
assert summary["build_status"] == "built", summary
assert summary["model_configured"] is False, summary
PY

echo "source installer shell tests passed"
