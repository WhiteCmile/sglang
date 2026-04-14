NCCL_DEBUG=WARN \
NCCL_DEBUG_SUBSYS=INIT,ENV,NET,COLL \
NCCL_TIMEOUT=3600 \
NCCL_INIT_TIMEOUT=3600 \
NCCL_ASYNC_TIMEOUT=3600 \
    /share/zhaotianlang/apps/anaconda3/envs/eurosys26-sglang/bin/python3 -m sglang.launch_server \
    --model-path $DEEPSEEK_V3_PATH \
    --mem-fraction-static 0.7 \
    --tp-size 16 \
    --dp-size 2 \
    --ep-size 16 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --trust-remote-code \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --nnodes 2 \
    --node-rank 1 \
    --watchdog-timeout 3600 \
    --port 31000 \
    --dist-init-addr 10.204.92.2:11000

# NCCL_IB_DISABLE=1 \