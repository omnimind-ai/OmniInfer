#!/usr/bin/env bash
# Lightweight OmniInfer CLI installer for Linux.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.sh | bash -s -- --version v0.3.2
set -euo pipefail

REPO="omnimind-ai/OmniInfer"
API_URL="https://api.github.com"
RELEASE_BASE_URL="https://github.com/${REPO}/releases/download"
INSTALL_DIR="${OMNIINFER_INSTALL_DIR:-${HOME}/.local/bin}"
VERSION="${OMNIINFER_VERSION:-latest}"
TARGET=""
DRY_RUN=0

info() { printf '[INFO] %s\n' "$*"; }
ok() { printf '[ OK ] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
fatal() {
    printf '[ERR ] %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'HELP'
OmniInfer CLI Installer

Downloads the CLI-only archive from GitHub Releases and installs it into
a user-writable bin directory. It does not clone the repository, install
backend runtimes, download models, or use sudo.

Usage:
  curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.sh | bash
  bash scripts/install.sh [OPTIONS]

Options:
  --version VERSION     Release tag to install, for example v0.3.2.
                        Default: latest GitHub Release.
  --install-dir DIR     Directory for the omniinfer executable.
                        Default: ~/.local/bin
  --repo OWNER/REPO     GitHub repository. Default: omnimind-ai/OmniInfer
  --base-url URL        Release download base URL.
                        Default: https://github.com/<repo>/releases/download
  --api-url URL         GitHub API base URL. Default: https://api.github.com
  --target TARGET       Override target triplet for testing. Supported: linux-x64
  --dry-run             Print the resolved install plan without downloading.
  -h, --help            Show this help.

For source checkout, backend runtime setup, or model setup, use:
  scripts/install-from-source.sh

After installing the CLI, run:
  omniinfer backend list
  omniinfer backend install <backend>
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            [[ $# -ge 2 ]] || fatal "--version requires a value"
            VERSION="$2"
            shift 2
            ;;
        --install-dir)
            [[ $# -ge 2 ]] || fatal "--install-dir requires a value"
            INSTALL_DIR="$2"
            shift 2
            ;;
        --repo)
            [[ $# -ge 2 ]] || fatal "--repo requires a value"
            REPO="$2"
            RELEASE_BASE_URL="https://github.com/${REPO}/releases/download"
            shift 2
            ;;
        --base-url)
            [[ $# -ge 2 ]] || fatal "--base-url requires a value"
            RELEASE_BASE_URL="${2%/}"
            shift 2
            ;;
        --api-url)
            [[ $# -ge 2 ]] || fatal "--api-url requires a value"
            API_URL="${2%/}"
            shift 2
            ;;
        --target)
            [[ $# -ge 2 ]] || fatal "--target requires a value"
            TARGET="$2"
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
            fatal "unknown option: $1"
            ;;
    esac
done

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || fatal "'$1' is required but was not found"
}

resolve_latest_version() {
    local response tag
    response="$(curl -fsSL "${API_URL}/repos/${REPO}/releases/latest")" \
        || fatal "failed to query latest release from ${API_URL}/repos/${REPO}/releases/latest"
    tag="$(printf '%s\n' "${response}" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
    [[ -n "${tag}" ]] || fatal "latest release response did not contain tag_name"
    printf '%s\n' "${tag}"
}

normalize_version() {
    local value="$1"
    if [[ "${value}" == latest ]]; then
        resolve_latest_version
        return
    fi
    if [[ "${value}" == v* ]]; then
        printf '%s\n' "${value}"
    else
        printf 'v%s\n' "${value}"
    fi
}

detect_target() {
    local system machine
    system="$(uname -s)"
    machine="$(uname -m)"
    case "${system}:${machine}" in
        Linux:x86_64|Linux:amd64)
            printf 'linux-x64\n'
            ;;
        Linux:aarch64|Linux:arm64)
            fatal "Linux arm64 release assets are not available yet"
            ;;
        Darwin:*)
            fatal "macOS installer support is not implemented in scripts/install.sh yet; use the GitHub Release archive for now"
            ;;
        *)
            fatal "unsupported platform: ${system}/${machine}"
            ;;
    esac
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        fatal "'sha256sum' or 'shasum' is required but neither was found"
    fi
}

checksum_for_asset() {
    local checksums_file="$1"
    local asset_name="$2"
    awk -v asset="${asset_name}" '$2 == asset { print $1 }' "${checksums_file}" | head -n 1
}

ensure_path_hint() {
    case ":${PATH}:" in
        *":${INSTALL_DIR}:"*) ;;
        *)
            warn "${INSTALL_DIR} is not on PATH for this shell"
            warn "Add this to your shell profile: export PATH=\"${INSTALL_DIR}:\$PATH\""
            ;;
    esac
}

need_cmd curl
need_cmd tar
need_cmd awk
need_cmd sed
need_cmd uname

if [[ -z "${TARGET}" ]]; then
    TARGET="$(detect_target)"
fi
[[ "${TARGET}" == "linux-x64" ]] || fatal "unsupported target: ${TARGET}"

VERSION="$(normalize_version "${VERSION}")"
ASSET_NAME="omniinfer-${VERSION}-${TARGET}.tar.gz"
ASSET_URL="${RELEASE_BASE_URL}/${VERSION}/${ASSET_NAME}"
CHECKSUMS_URL="${RELEASE_BASE_URL}/${VERSION}/checksums.txt"
DEST="${INSTALL_DIR}/omniinfer"

info "Version: ${VERSION}"
info "Target: ${TARGET}"
info "Install dir: ${INSTALL_DIR}"
info "Asset: ${ASSET_URL}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    ok "Dry run complete"
    exit 0
fi

WORK_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

ARCHIVE_PATH="${WORK_DIR}/${ASSET_NAME}"
CHECKSUMS_PATH="${WORK_DIR}/checksums.txt"
EXTRACT_DIR="${WORK_DIR}/extract"

mkdir -p "${EXTRACT_DIR}"

info "Downloading CLI archive"
curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o "${ARCHIVE_PATH}" "${ASSET_URL}"

info "Downloading checksums"
curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o "${CHECKSUMS_PATH}" "${CHECKSUMS_URL}"

EXPECTED_SHA="$(checksum_for_asset "${CHECKSUMS_PATH}" "${ASSET_NAME}")"
[[ -n "${EXPECTED_SHA}" ]] || fatal "checksums.txt does not contain ${ASSET_NAME}"

ACTUAL_SHA="$(sha256_file "${ARCHIVE_PATH}")"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
    fatal "checksum mismatch for ${ASSET_NAME}: expected ${EXPECTED_SHA}, got ${ACTUAL_SHA}"
fi
ok "Checksum verified: ${ACTUAL_SHA}"

info "Extracting archive"
tar -xzf "${ARCHIVE_PATH}" -C "${EXTRACT_DIR}"

PACKAGE_DIR="${EXTRACT_DIR}/OmniInfer"
BINARY_PATH="${PACKAGE_DIR}/omniinfer"
RUNTIME_HELPER_PATH="${PACKAGE_DIR}/omniinfer-rs"
[[ -f "${BINARY_PATH}" ]] || fatal "archive did not contain OmniInfer/omniinfer"

mkdir -p "${INSTALL_DIR}"
TMP_DEST="${INSTALL_DIR}/.omniinfer.tmp.$$"
cp "${BINARY_PATH}" "${TMP_DEST}"
chmod 0755 "${TMP_DEST}"

if [[ -f "${RUNTIME_HELPER_PATH}" ]]; then
    TMP_HELPER="${INSTALL_DIR}/.omniinfer-rs.tmp.$$"
    cp "${RUNTIME_HELPER_PATH}" "${TMP_HELPER}"
    chmod 0755 "${TMP_HELPER}"
    mv "${TMP_HELPER}" "${INSTALL_DIR}/omniinfer-rs"
    ok "Installed ${INSTALL_DIR}/omniinfer-rs"
fi

mv "${TMP_DEST}" "${DEST}"

ok "Installed ${DEST}"
if ! VERIFY_OUTPUT="$("${DEST}" --version 2>&1)"; then
    fatal "installed binary failed to run: ${VERIFY_OUTPUT}"
fi
printf '%s\n' "${VERIFY_OUTPUT}"
ok "Next: run 'omniinfer backend list' and 'omniinfer backend install <backend>'"
ensure_path_hint
