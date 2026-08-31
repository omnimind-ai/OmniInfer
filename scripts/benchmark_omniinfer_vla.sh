#!/usr/bin/env bash
# One-command OmniInfer VLA Runtime benchmark for the two fixed LIBERO models.
#
# Usage:
#   ./benchmark_omniinfer_vla.sh             # run Pi0.5 and GR00T
#   ./benchmark_omniinfer_vla.sh pi05
#   ./benchmark_omniinfer_vla.sh gr00t
#   ./benchmark_omniinfer_vla.sh           # native Processor by default
#   NATIVE=0 ./benchmark_omniinfer_vla.sh  # legacy prepared-token path
#
# The script starts the OmniInfer VLA Runtime server, waits for the model and CUDA Graph to
# finish loading, runs the corresponding Python benchmark, prints a summary,
# and terminates only the server process that it started.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${OMNIINFER_VLA_RUNTIME_HOME:-}" ]]; then
    RUNTIME_HOME="$OMNIINFER_VLA_RUNTIME_HOME"
else
    RUNTIME_HOME=""
    for candidate in "$ROOT"/framework/*; do
        if [[ -f "$candidate/omniinfer_server.py" ]]; then
            RUNTIME_HOME="$candidate"
            break
        fi
    done
fi
[[ -n "$RUNTIME_HOME" ]] || {
    echo "Set OMNIINFER_VLA_RUNTIME_HOME to the provisioned VLA runtime home." >&2
    exit 1
}
VLA_PYTHON="${OMNIINFER_VLA_RUNTIME_PYTHON:-$RUNTIME_HOME/.venv/bin/python}"
VLA_SERVER="${OMNIINFER_VLA_SERVER:-$RUNTIME_HOME/omniinfer_server.py}"
SCRIPTS="${SCRIPTS:-$ROOT/scripts}"
STATE_DIR="${STATE_DIR:-$ROOT/.local/benchmarks}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
WARMUP="${WARMUP:-2}"
TIMED="${TIMED:-3}"
MODE="${1:-both}"
# Benchmark deployment latency by default: raw task/state/images enter the
# model-specific native Processor on every request. Set NATIVE=0 only when an
# engine-only prepared-input comparison is explicitly desired.
NATIVE="${NATIVE:-1}"

PI_CHECKPOINT="${PI_CHECKPOINT:-$HOME/models/pi05_libero_finetuned_v044}"
GROOT_CHECKPOINT="${GROOT_CHECKPOINT:-$HOME/models/GR00T-N1.7-LIBERO/libero_object}"
PI05_TOKENIZER="${PI05_TOKENIZER:-$HOME/models/paligemma-3b-pt-224}"
GROOT_PROCESSOR="${GROOT_PROCESSOR:-$HOME/models/Cosmos-Reason2-2B}"
GROOT_EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-LIBERO_PANDA}"

SERVER_PID=""
SERVER_LOG=""

die() {
    echo "ERROR: $*" >&2
    exit 1
}

check_file() {
    [[ -e "$1" ]] || die "missing: $1"
}

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
        echo "Stopping OmniInfer VLA Runtime server pid=$SERVER_PID"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server did not stop after 20s; sending KILL" >&2
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

cleanup() {
    stop_server
}
trap cleanup EXIT INT TERM

wait_ready() {
    local limit=900
    for _ in $(seq 1 "$limit"); do
        # Jetson images may not include ripgrep; grep is sufficient here.
        if grep -qE "omniinfer-vla-server: bound to .* ready\." "$SERVER_LOG" 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            tail -80 "$SERVER_LOG" >&2 || true
            return 1
        fi
        sleep 1
    done
    tail -80 "$SERVER_LOG" >&2 || true
    return 1
}

run_model() {
    local name="$1"
    local arch="$2"
    local checkpoint="$3"
    local num_images="$4"
    local vision_args="$5"
    local benchmark="$6"
    local result="$STATE_DIR/${name,,}-${RUN_ID}.json"
    local port
    local addr
    local -a benchmark_args
    local -a processor_args

    check_file "$checkpoint"
    check_file "$benchmark"
    port="$(pick_port)"
    addr="tcp://127.0.0.1:$port"
    SERVER_LOG="$STATE_DIR/${name,,}-${RUN_ID}.server.log"
    mkdir -p "$STATE_DIR"

    echo
    echo "========== $name =========="
    echo "checkpoint: $checkpoint"
    echo "endpoint:   $addr"
    echo "starting server..."
    processor_args=(--processor-mode prepared)
    if [[ "$NATIVE" == "1" ]]; then
        case "$arch" in
            pi05)
                check_file "$PI05_TOKENIZER"
                processor_args=(
                    --processor-mode native
                    --pi05-tokenizer "$PI05_TOKENIZER"
                )
                ;;
            gr00t_n17)
                check_file "$GROOT_PROCESSOR"
                processor_args=(
                    --processor-mode native
                    --processor-model-name-or-path "$GROOT_PROCESSOR"
                    --embodiment-tag "$GROOT_EMBODIMENT_TAG"
                )
                ;;
            *) die "native Processor is unsupported for architecture: $arch" ;;
        esac
    fi
    # shellcheck disable=SC2086
    "$VLA_PYTHON" "$VLA_SERVER" \
        --bind "$addr" \
        --checkpoint "$checkpoint" \
        --arch "$arch" \
        --num-images "$num_images" \
        --params-dtype bfloat16 \
        "${processor_args[@]}" \
        $vision_args \
        >"$SERVER_LOG" 2>&1 &
    SERVER_PID="$!"

    if ! wait_ready; then
        stop_server
        die "$name server failed to become ready; log: $SERVER_LOG"
    fi
    echo "server ready (pid=$SERVER_PID)"

    benchmark_args=(
        --addr "$addr"
        --warmup "$WARMUP"
        --timed "$TIMED"
        --output "$result"
    )
    if [[ "$NATIVE" == "1" ]]; then
        benchmark_args+=(--native)
    fi
    "$VLA_PYTHON" "$benchmark" "${benchmark_args[@]}"
    echo "result: $result"
    echo "server log: $SERVER_LOG"
    stop_server
}

case "$MODE" in
    pi05)
        run_model "Pi0.5" "pi05" "$PI_CHECKPOINT" 3 \
            "--vision-dtype float32" \
            "$SCRIPTS/benchmark_omniinfer_vla_pi05.py"
        ;;
    gr00t|groot)
        run_model "GR00T" "gr00t_n17" "$GROOT_CHECKPOINT" 2 \
            "" \
            "$SCRIPTS/benchmark_omniinfer_vla_gr00t.py"
        ;;
    both)
        run_model "Pi0.5" "pi05" "$PI_CHECKPOINT" 3 \
            "--vision-dtype float32" \
            "$SCRIPTS/benchmark_omniinfer_vla_pi05.py"
        run_model "GR00T" "gr00t_n17" "$GROOT_CHECKPOINT" 2 \
            "" \
            "$SCRIPTS/benchmark_omniinfer_vla_gr00t.py"
        ;;
    *)
        die "usage: $0 [pi05|gr00t|both]"
        ;;
esac

echo
echo "========== summary =========="
"$VLA_PYTHON" - "$STATE_DIR" "$RUN_ID" <<'PY'
import json
import pathlib
import sys

state_dir = pathlib.Path(sys.argv[1])
run_id = sys.argv[2]
files = sorted(state_dir.glob(f"*-{run_id}.json"))
if not files:
    print("No result JSON found.")
    raise SystemExit(0)

print(
    f"{'model':<12} {'processor':>13} {'engine':>13} "
    f"{'postprocess':>13} {'server total':>13} {'ZMQ RTT':>13}"
)
print("-" * 92)
for path in files:
    data = json.loads(path.read_text())
    name = "Pi0.5" if data["num_images"] == 3 else "GR00T N1.7"
    print(
        f"{name:<12} "
        f"{data['processor_mean_ms']:>10.2f} ms "
        f"{data['engine_mean_ms']:>10.2f} ms "
        f"{data['postprocess_mean_ms']:>10.2f} ms "
        f"{data['server_mean_ms']:>10.2f} ms "
        f"{data['wall_mean_ms']:>10.2f} ms"
    )
print("processor = native model Processor call, CUDA-synchronized")
print("engine     = synchronized OmniInfer VLA Runtime engine step")
PY
