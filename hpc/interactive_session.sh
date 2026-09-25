#!/bin/bash
# ==============================================================================
# Interactive Biosafety Defense Agent Session Runner
# ==============================================================================

set -e

PORT=${PORT:-8000}
ENDPOINT="http://localhost:${PORT}/v1/models"

echo "=========================================================="
echo "Checking vLLM Server Readiness on port ${PORT}..."
echo "=========================================================="

# Check if vLLM server is responding
if ! curl -s -f "${ENDPOINT}" > /dev/null 2>&1; then
    echo "⚠️  vLLM server is not responding at ${ENDPOINT}"
    echo ""
    echo "To launch the server first, run:"
    echo "    bash hpc/start_single_vllm.sh"
    echo ""
    echo "You can monitor the server initialization logs with:"
    echo "    tail -f logs/vllm_server.log"
    echo "=========================================================="
    exit 1
fi

echo "✅ vLLM server is online and ready."
echo "Starting interactive Biosafety Defense Agent session..."
echo "=========================================================="

python app.py --config-dir configs
