#!/bin/bash
# Stop running vLLM servers
echo "Stopping any running vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "vLLM processes stopped."
