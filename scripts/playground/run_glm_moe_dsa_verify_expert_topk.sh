#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GLM_PATH=zai-org/GLM-5 \
#   EXPERT_OUT_DIR=/tmp/glm_verify_expert_topk \
#   bash scripts/playground/run_glm_moe_dsa_verify_expert_topk.sh

: "${GLM_PATH:?Please set GLM_PATH}"
# EXPERT_OUT_DIR="${EXPERT_OUT_DIR:-/tmp/glm_verify_expert_topk}"
EXPERT_OUT_DIR="experiments/data/verify_expert_topk/glm_moe_dsa"
PORT="${PORT:-30001}"
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-8}"

mkdir -p "${EXPERT_OUT_DIR}"
echo "GLM_PATH=${GLM_PATH}"
echo "EXPERT_OUT_DIR=${EXPERT_OUT_DIR}"
echo "HOST=${HOST} PORT=${PORT} TP_SIZE=${TP_SIZE}"

python3 -m sglang.launch_server \
  --model-path "${GLM_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP_SIZE}" \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16 \
  --disable-cuda-graph \
  --attention-backend fa3 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --speculative-verify-expert-topk-output-dir "${EXPERT_OUT_DIR}"
