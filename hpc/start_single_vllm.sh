#!/bin/bash
# ==============================================================================
# Single Model vLLM Service (Serves both Defense Agent & Target Assistant)
# Uses 4 GPUs with tensor parallelism for DeepSeek-V4-Flash
# ==============================================================================

set -e

mkdir -p logs

MODEL_PATH=${MODEL_PATH:-"/data/checkpoints/deepseek-ai/DeepSeek-V4-Flash-0731"}
MODEL_NAME=${MODEL_NAME:-"deepseek-v4-flash"}
PORT=${PORT:-8000}
GPU_DEVICES=${GPU_DEVICES:-"0,1,2,3"}
TP_SIZE=${TP_SIZE:-4}

echo "=========================================================="
echo "Launching Single vLLM Model Server"
echo "Model Path:  ${MODEL_PATH}"
echo "Served Name: ${MODEL_NAME}"
echo "Port:        ${PORT}"
echo "GPUs:        ${GPU_DEVICES} (Tensor Parallel Size: ${TP_SIZE})"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nohup vllm serve \
    "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 > logs/vllm_server.log 2>&1 &

PID_SERVER=$!
echo "vLLM server started in background with PID: ${PID_SERVER}"
echo "Logs are streaming to: logs/vllm_server.log"
echo "Check readiness with: tail -f logs/vllm_server.log"
echo "=========================================================="
