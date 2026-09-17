#!/usr/bin/env bash
# Benchmark OIVF's actual one-step action-generation pruning path through the
# same OmniInfer protobuf/ZMQ client as the baseline comparison.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OMNIINFER_VLA_FAST_ROOT="${OMNIINFER_VLA_FAST_ROOT:-$ROOT/framework/OmniInfer-VLA-Fast}"
VLA_ROOT="${OMNIINFER_VLA_RUNTIME_HOME:-$ROOT/framework/OmniInfer-VLA}"
VLA_PYTHON="${OMNIINFER_VLA_RUNTIME_PYTHON:-$VLA_ROOT/.venv/bin/python}"
PI_CHECKPOINT="${PI_CHECKPOINT:-$HOME/models/pi05_libero_finetuned_v044}"
PI05_TOKENIZER="${PI05_TOKENIZER:-$HOME/models/paligemma-3b-pt-224}"
WARMUP="${WARMUP:-10}"
TIMED="${TIMED:-10}"
STATE_DIR="${STATE_DIR:-$ROOT/.local/benchmarks}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"

SERVER_PID=""
SERVER_LOG=""

pick_port() {
    "$VLA_PYTHON" - <<'PY'
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT INT TERM

mkdir -p "$STATE_DIR"
port="$(pick_port)"
addr="tcp://127.0.0.1:$port"
SERVER_LOG="$STATE_DIR/omniinfer-vla-fast-pi05-pruning-${RUN_ID}.server.log"
result="$STATE_DIR/omniinfer-vla-fast-pi05-pruning-${RUN_ID}.json"

echo "========== OmniInfer-VLA-Fast Pi0.5 one-step pruning =========="
"$VLA_PYTHON" "$ROOT/scripts/omniinfer_vla_fast_server.py" \
    --bind "$addr" \
    --oivf-root "$OMNIINFER_VLA_FAST_ROOT" \
    --checkpoint "$PI_CHECKPOINT" \
    --tokenizer "$PI05_TOKENIZER" \
    --tactics-mode on \
    --num-flow-steps 1 \
    --flow-start-time 0.5 \
    --warm-start \
    --warm-start-alpha 0.5 \
    --replan 5 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
for _ in $(seq 1 900); do
    if grep -q "omniinfer-vla-fast-server: bound to .* ready\." "$SERVER_LOG" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        tail -80 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done
grep -q "omniinfer-vla-fast-server: bound to .* ready\." "$SERVER_LOG" || {
    tail -80 "$SERVER_LOG" >&2
    exit 1
}

"$VLA_PYTHON" "$ROOT/scripts/benchmark_omniinfer_vla_zmq.py" \
    --addr "$addr" --arch pi05 --num-images 3 --image-size 224 \
    --lang-len 48 --warmup "$WARMUP" --timed "$TIMED" --native \
    --output "$result"

stop_server
echo "result: $result"
echo "server log: $SERVER_LOG"
