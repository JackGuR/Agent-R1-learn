#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-examples/run_qwen2.5-3b.sh}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MIN_FREE_MB="${MIN_FREE_MB:-70000}"
MAX_USED_MB="${MAX_USED_MB:-1024}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
LOCK_DIR="${LOCK_DIR:-$PROJECT_DIR/.gpu-watcher.lock}"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

cleanup() {
    rmdir "$LOCK_DIR" 2>/dev/null || true
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[$(timestamp)] Error: nvidia-smi was not found." >&2
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/$TRAIN_SCRIPT" ]]; then
    echo "[$(timestamp)] Error: training script does not exist: $PROJECT_DIR/$TRAIN_SCRIPT" >&2
    exit 1
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[$(timestamp)] Error: another GPU watcher appears to be running: $LOCK_DIR" >&2
    exit 1
fi
trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(timestamp)] Waiting for one idle GPU..."
echo "[$(timestamp)] Conditions: free>=${MIN_FREE_MB}MB, used<=${MAX_USED_MB}MB, utilization<=${MAX_UTILIZATION}%"
echo "[$(timestamp)] Training script: $TRAIN_SCRIPT"

while true; do
    selected_gpu=""
    gpu_summary=""

    while IFS=, read -r index used free utilization; do
        index="${index//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        free="${free//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"

        gpu_summary+="GPU ${index}: used=${used}MB free=${free}MB util=${utilization}%; "

        if [[ -z "$selected_gpu" ]] &&
            (( free >= MIN_FREE_MB )) &&
            (( used <= MAX_USED_MB )) &&
            (( utilization <= MAX_UTILIZATION )); then
            selected_gpu="$index"
        fi
    done < <(
        nvidia-smi \
            --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits
    )

    if [[ -n "$selected_gpu" ]]; then
        log_file="$LOG_DIR/$(basename "$TRAIN_SCRIPT" .sh)_gpu${selected_gpu}_$(date +%Y%m%d_%H%M%S).log"
        echo "[$(timestamp)] Selected physical GPU ${selected_gpu}."
        echo "[$(timestamp)] Output will also be saved to: $log_file"

        cd "$PROJECT_DIR"
        export CUDA_VISIBLE_DEVICES="$selected_gpu"

        set +e
        bash "$TRAIN_SCRIPT" 2>&1 | tee "$log_file"
        exit_code=${PIPESTATUS[0]}
        set -e

        echo "[$(timestamp)] Training exited with code ${exit_code}."
        exit "$exit_code"
    fi

    echo "[$(timestamp)] No GPU is idle. ${gpu_summary}"
    sleep "$POLL_SECONDS"
done
