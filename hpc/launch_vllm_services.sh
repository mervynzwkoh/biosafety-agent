#!/bin/bash
# ==============================================================================
# Launch vLLM Services for Biosafety Defense Agent (HPC Multi-GPU Deployment)
# ==============================================================================
#SBATCH --job-name=biosafety-vllm
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vllm_%j.log

set -e

# Module load or conda activation
# source /path/to/conda/bin/activate biosafety-env

REASONING_MODEL=${REASONING_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
TARGET_MODEL=${TARGET_MODEL:-"meta-llama/Meta-Llama-3-8B-Instruct"}

REASONING_PORT=8000
TARGET_PORT=8001

echo "Starting Reasoning Agent vLLM on GPU 0, port ${REASONING_PORT}..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "${REASONING_MODEL}" \
    --port "${REASONING_PORT}" \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 &
PID_REASONING=$!

echo "Starting Target Model vLLM on GPU 1, port ${TARGET_PORT}..."
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "${TARGET_MODEL}" \
    --port "${TARGET_PORT}" \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 &
PID_TARGET=$!

# Wait for both endpoints to become ready
echo "Waiting for vLLM services to be ready..."
sleep 30

echo "vLLM services running (Reasoning PID: ${PID_REASONING}, Target PID: ${PID_TARGET})."
wait
