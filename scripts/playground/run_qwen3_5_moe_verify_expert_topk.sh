#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   QWEN_PATH=Qwen/Qwen3.5-397B-A17B \
#   EXPERT_OUT_DIR=/tmp/qwen3_5_moe_verify_expert_topk \
#   bash scripts/playground/run_qwen3_5_moe_verify_expert_topk.sh

: "${QWEN_PATH:?Please set QWEN_PATH}"
# EXPERT_OUT_DIR="${EXPERT_OUT_DIR:-/tmp/qwen3_5_moe_verify_expert_topk}"
EXPERT_OUT_DIR="experiments/data/verify_expert_topk/qwen3_5_moe"
PORT="${PORT:-30001}"
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-8}"

mkdir -p "${EXPERT_OUT_DIR}"
echo "QWEN_PATH=${QWEN_PATH}"
echo "EXPERT_OUT_DIR=${EXPERT_OUT_DIR}"
echo "HOST=${HOST} PORT=${PORT} TP_SIZE=${TP_SIZE}"

python3 -m sglang.launch_server \
  --model-path "${QWEN_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP_SIZE}" \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16 \
  --disable-cuda-graph \
  --speculative-verify-expert-topk-output-dir "${EXPERT_OUT_DIR}"
