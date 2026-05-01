#!/usr/bin/env bash

omni_install_deps_detect_pkg_mgr() {
    if command -v apt-get >/dev/null 2>&1; then   printf 'apt\n'
    elif command -v dnf >/dev/null 2>&1; then     printf 'dnf\n'
    elif command -v pacman >/dev/null 2>&1; then  printf 'pacman\n'
    elif command -v zypper >/dev/null 2>&1; then  printf 'zypper\n'
    elif command -v yum >/dev/null 2>&1; then     printf 'yum\n'
    else                                          printf '\n'
    fi
}

omni_install_deps_repair_command() {
    local pkg_mgr="$1"
    shift
    if [[ $# -eq 0 || -z "${pkg_mgr}" ]]; then
        return 1
    fi

    case "${pkg_mgr}" in
        apt)    printf 'sudo apt-get update && sudo apt-get install -y %s\n' "$*" ;;
        dnf)    printf 'sudo dnf install -y %s\n' "$*" ;;
        pacman) printf 'sudo pacman -S --noconfirm %s\n' "$*" ;;
        zypper) printf 'sudo zypper install -y %s\n' "$*" ;;
        yum)    printf 'sudo yum install -y %s\n' "$*" ;;
        *)      return 1 ;;
    esac
}

omni_install_deps_cuda_alt_hint() {
    cat <<'EOF'
Alternative: install a complete CUDA toolkit and set CUDAToolkit_ROOT=/path/to/cuda or CUDA_HOME=/path/to/cuda. Runtime-only or private cuBLAS libraries, such as Ollama's bundled libraries, are not enough for building.
EOF
}

omni_install_deps_policy() {
    local install_system_deps="$1"
    local non_interactive="$2"
    local can_prompt="$3"

    if [[ "${install_system_deps}" == "yes" ]]; then
        printf 'auto-install\n'
    elif [[ "${install_system_deps}" == "no" ]]; then
        printf 'fail-disabled\n'
    elif [[ "${non_interactive}" == "1" || "${can_prompt}" == "0" ]]; then
        printf 'fail-noninteractive\n'
    else
        printf 'prompt\n'
    fi
}

omni_install_deps_run() {
    local pkg_mgr="$1"
    shift
    if [[ $# -eq 0 ]]; then
        return 0
    fi

    local repair_cmd
    repair_cmd="$(omni_install_deps_repair_command "${pkg_mgr}" "$@")" || return 1
    if [[ "${OMNI_INSTALL_DEPS_DRY_RUN:-0}" == "1" ]]; then
        printf 'DRY RUN: %s\n' "${repair_cmd}"
        return 0
    fi

    case "${pkg_mgr}" in
        apt)    sudo apt-get update -qq && sudo apt-get install -y "$@" ;;
        dnf)    sudo dnf install -y "$@" ;;
        pacman) sudo pacman -S --noconfirm "$@" ;;
        zypper) sudo zypper install -y "$@" ;;
        yum)    sudo yum install -y "$@" ;;
        *)      return 1 ;;
    esac
}

omni_install_deps_print_fix() {
    local pkg_mgr="$1"
    shift
    local repair_cmd=""
    if repair_cmd="$(omni_install_deps_repair_command "${pkg_mgr}" "$@" 2>/dev/null)"; then
        echo "  Recommended fix:"
        echo "    ${repair_cmd}"
    elif [[ $# -gt 0 ]]; then
        echo "  Missing system packages:"
        echo "    $*"
        echo "  Install these packages with your system package manager, then re-run the installer."
    fi
    echo "  $(omni_install_deps_cuda_alt_hint)"
}
