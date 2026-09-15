#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="$(mktemp -d)"
trap 'rm -rf -- "${test_root}"' EXIT
swiftc "${repo_root}/ios/OmniInferServer/Sources/OmniInferServer/BackendSelector.swift" \
    "${repo_root}/tests/fixtures/ios-backend-selectors/main.swift" -o "${test_root}/test-selectors"
"${test_root}/test-selectors"
