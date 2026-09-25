#!/bin/bash
# ==============================================================================
# Robust Shutdown Script for vLLM Server & GPU Worker Processes
# Ensures all vLLM processes are killed and GPU memory is completely released.
# ==============================================================================

set -e

PORT=${PORT:-8000}
PID_FILE="logs/vllm_server.pid"

echo "=========================================================="
echo "Stopping vLLM Services & Releasing GPU Memory"
echo "=========================================================="

# 1. Kill via saved PID and process tree if pid file exists
if [ -f "${PID_FILE}" ]; then
    SERVER_PID=$(cat "${PID_FILE}" | tr -d ' ')
    if [ -n "${SERVER_PID}" ] && ps -p "${SERVER_PID}" > /dev/null 2>&1; then
        echo "Terminating vLLM process tree for PID: ${SERVER_PID}..."
        pkill -P "${SERVER_PID}" -9 2>/dev/null || true
        kill -9 "${SERVER_PID}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
fi

# 2. Kill any process listening on the vLLM port
echo "Releasing port ${PORT}..."
if command -v fuser > /dev/null 2>&1; then
    fuser -k -9 "${PORT}/tcp" > /dev/null 2>&1 || true
fi
if command -v lsof > /dev/null 2>&1; then
    PORT_PIDS=$(lsof -ti:"${PORT}" 2>/dev/null || true)
    if [ -n "${PORT_PIDS}" ]; then
        echo "Killing processes on port ${PORT}: ${PORT_PIDS}"
        echo "${PORT_PIDS}" | xargs kill -9 2>/dev/null || true
    fi
fi

# 3. Kill all remaining vLLM-related processes owned by current user
echo "Terminating any lingering vLLM worker processes for user ${USER}..."
pkill -u "${USER}" -9 -f "vllm" 2>/dev/null || true
pkill -u "${USER}" -9 -f "vllm.entrypoints" 2>/dev/null || true

# 4. Kill lingering GPU compute processes owned by current user
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "Inspecting active GPU compute processes..."
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)
    for gpid in ${GPU_PIDS}; do
        if [ -n "${gpid}" ] && ps -u "${USER}" -o pid= 2>/dev/null | grep -qw "${gpid}"; then
            echo "Terminating GPU compute process PID: ${gpid}..."
            kill -9 "${gpid}" 2>/dev/null || true
        fi
    done
fi

sleep 2

# 5. Verify GPU memory status
echo "=========================================================="
echo "Verification:"
if command -v nvidia-smi > /dev/null 2>&1; then
    ACTIVE_APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
    if [ -z "${ACTIVE_APPS}" ]; then
        echo "✅ All GPU compute processes terminated. GPUs are clear and idle."
    else
        echo "⚠️ Active GPU processes remaining:"
        echo "${ACTIVE_APPS}"
    fi
else
    echo "✅ Process cleanup finished."
fi
echo "=========================================================="
