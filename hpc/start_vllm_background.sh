#!/bin/bash
# ==============================================================================
# Standalone vLLM Services Launcher (No Slurm Required)
# ==============================================================================

set -e

mkdir -p logs

REASONING_MODEL=${REASONING_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
TARGET_MODEL=${TARGET_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}

REASONING_PORT=8000
TARGET_PORT=8001

echo "=========================================================="
echo "Starting vLLM servers on GPUs 0 and 1..."
echo "Reasoning Agent: ${REASONING_MODEL} (GPU 0, Port ${REASONING_PORT})"
echo "Target LLM:      ${TARGET_MODEL} (GPU 1, Port ${TARGET_PORT})"
echo "=========================================================="

# 1. Start Reasoning Agent on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${REASONING_MODEL}" \
    --port "${REASONING_PORT}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 > logs/vllm_reasoning.log 2>&1 &
PID_REASONING=$!
echo "Reasoning Agent PID: ${PID_REASONING} (logs: logs/vllm_reasoning.log)"

# 2. Start Target Model on GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${TARGET_MODEL}" \
    --port "${TARGET_PORT}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 > logs/vllm_target.log 2>&1 &
PID_TARGET=$!
echo "Target Model PID:    ${PID_TARGET} (logs: logs/vllm_target.log)"

echo "----------------------------------------------------------"
echo "Servers launched in background. Monitoring readiness..."
echo "You can check logs with: tail -f logs/vllm_reasoning.log"
echo "=========================================================="
