#!/usr/bin/env bash
# Reproducible OmniInfer-VLA-Fast Pi0.5 comparison on the current Jetson.
#
# The OIVF Python/Rust wheel must be built first (see README.md).  The Python
# benchmark runs BF16 inference with OIVF's validated Thor SM110 tactics against
# the same raw workload and checkpoint.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OMNIINFER_VLA_FAST_ROOT="${OMNIINFER_VLA_FAST_ROOT:-$ROOT/framework/OmniInfer-VLA-Fast}"
VLA_ROOT="${OMNIINFER_VLA_RUNTIME_HOME:-$ROOT/framework/OmniInfer-VLA}"
VLA_PYTHON="${OMNIINFER_VLA_RUNTIME_PYTHON:-$VLA_ROOT/.venv/bin/python}"
PI_CHECKPOINT="${PI_CHECKPOINT:-$HOME/models/pi05_libero_finetuned_v044}"
PI05_TOKENIZER="${PI05_TOKENIZER:-$HOME/models/paligemma-3b-pt-224}"
WARMUP="${WARMUP:-10}"
TIMED="${TIMED:-10}"
OUTPUT="${OUTPUT:-$ROOT/.local/benchmarks/omniinfer-vla-fast-pi05-$(date +%Y%m%d-%H%M%S).json}"

[[ -x "$VLA_PYTHON" ]] || {
    echo "missing Python: $VLA_PYTHON" >&2
    exit 1
}
[[ -d "$OMNIINFER_VLA_FAST_ROOT/oivf/python/oivf" ]] || {
    echo "OmniInfer-VLA-Fast submodule is not initialized: $OMNIINFER_VLA_FAST_ROOT" >&2
    echo "run: git submodule update --init --recursive framework/OmniInfer-VLA-Fast" >&2
    exit 1
}

export OMNIINFER_VLA_FAST_ROOT
"$VLA_PYTHON" "$ROOT/scripts/benchmark_omniinfer_vla_fast_pi05.py" \
    --oivf-root "$OMNIINFER_VLA_FAST_ROOT" \
    --model-dir "$PI_CHECKPOINT" \
    --tokenizer "$PI05_TOKENIZER" \
    --mode on \
    --warmup "$WARMUP" \
    --timed "$TIMED" \
    --output "$OUTPUT"
