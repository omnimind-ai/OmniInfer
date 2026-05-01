#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/install-deps.sh"

assert_eq() {
  local expected="$1"
  local actual="$2"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "expected: ${expected}" >&2
    echo "actual:   ${actual}" >&2
    exit 1
  fi
}

repair_cmd="$(omni_install_deps_repair_command apt libcublas-dev-12-5)"
assert_eq "sudo apt-get update && sudo apt-get install -y libcublas-dev-12-5" "${repair_cmd}"

assert_eq "fail-noninteractive" "$(omni_install_deps_policy ask 1 0)"
assert_eq "fail-disabled" "$(omni_install_deps_policy no 0 1)"
assert_eq "auto-install" "$(omni_install_deps_policy yes 1 0)"
assert_eq "prompt" "$(omni_install_deps_policy ask 0 1)"

fix_output="$(omni_install_deps_print_fix apt libcublas-dev-12-5)"
[[ "${fix_output}" == *"sudo apt-get update && sudo apt-get install -y libcublas-dev-12-5"* ]]
[[ "${fix_output}" == *"CUDAToolkit_ROOT=/path/to/cuda"* ]]
[[ "${fix_output}" == *"Ollama's bundled libraries"* ]]

dry_run_output="$(OMNI_INSTALL_DEPS_DRY_RUN=1 omni_install_deps_run apt libcublas-dev-12-5)"
assert_eq "DRY RUN: sudo apt-get update && sudo apt-get install -y libcublas-dev-12-5" "${dry_run_output}"

echo "install-deps tests passed"
