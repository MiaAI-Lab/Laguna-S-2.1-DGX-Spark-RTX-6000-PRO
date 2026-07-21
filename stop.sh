#!/usr/bin/env bash
set -u

CONTAINER_NAME="laguna-s-2.1-nvfp4"
PID_FILE="$(cd "$(dirname "$0")" && pwd)/.vllm.pid"

if docker rm -f "${CONTAINER_NAME}" 2>&1; then
  echo "Stopped and removed container ${CONTAINER_NAME}"
else
  echo "Container ${CONTAINER_NAME} not found"
fi

rm -f "${PID_FILE}"
echo "Cleaned up."
