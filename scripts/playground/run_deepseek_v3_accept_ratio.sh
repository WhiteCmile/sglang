#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DEEPSEEK_PATH=/path/to/deepseek-v3 \
#   bash scripts/playground/run_deepseek_v3_accept_ratio.sh

: "${DEEPSEEK_PATH:?Please set DEEPSEEK_PATH}"
PORT="${PORT:-30001}"
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-8}"

echo "DEEPSEEK_PATH=${DEEPSEEK_PATH}"
echo "HOST=${HOST} PORT=${PORT} TP_SIZE=${TP_SIZE}"

python3 -m sglang.launch_server \
  --model-path "${DEEPSEEK_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP_SIZE}" \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16 \
  --disable-cuda-graph \
  --enable-metrics
