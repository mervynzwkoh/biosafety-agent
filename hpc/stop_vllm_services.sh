#!/bin/bash
# Force-stop all running vLLM servers and tensor parallel workers
echo "Force-stopping all vLLM server and worker processes..."
pkill -9 -f vllm || true
sleep 1
echo "Checking GPU status..."
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || echo "All GPUs clear."
echo "vLLM cleanup complete."
