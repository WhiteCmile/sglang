#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DEEPSEEK_PATH=/path/to/deepseek-v3 \
#   EXPERT_OUT_DIR=/tmp/deepseek_verify_expert_topk \
#   bash scripts/playground/run_deepseek_v3_verify_expert_topk.sh

: "${DEEPSEEK_PATH:?Please set DEEPSEEK_PATH}"
EXPERT_OUT_DIR="${EXPERT_OUT_DIR:-/tmp/deepseek_verify_expert_topk}"
PORT="${PORT:-30001}"
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-8}"

mkdir -p "${EXPERT_OUT_DIR}"
echo "DEEPSEEK_PATH=${DEEPSEEK_PATH}"
echo "EXPERT_OUT_DIR=${EXPERT_OUT_DIR}"
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
  --speculative-verify-expert-topk-output-dir "${EXPERT_OUT_DIR}"
