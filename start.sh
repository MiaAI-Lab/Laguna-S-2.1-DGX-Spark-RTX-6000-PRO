#!/usr/bin/env bash
# =============================================================================
#  start.sh — Launch poolside/Laguna-S-2.1-NVFP4 with DFlash speculative
#  decoding via Docker (vLLM 0.25.1).
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
MODEL_ID="poolside/Laguna-S-2.1-NVFP4"
DRAFT_MODEL_ID="poolside/Laguna-S-2.1-DFlash-NVFP4"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.1}"
CONTAINER_NAME="laguna-s-2.1-nvfp4"
HOST="0.0.0.0"
PORT="8888"
WORK_DIR="$(pwd)"
HF_HOME="${HOME}/.cache/huggingface"
PID_FILE="${WORK_DIR}/.vllm.pid"
LOG_FILE="${WORK_DIR}/.vllm.log"
BOOTSTRAP_SCRIPT="/tmp/laguna-bootstrap.sh"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${HOME}/.cache/flashinfer}"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

# ---- Argument parsing -------------------------------------------------------
DOWNLOAD_ONLY=false
case "${1:-}" in
  --download-only)
    DOWNLOAD_ONLY=true
    shift
    ;;
  -h|--help)
    echo "Usage: $0 [--download-only]"
    echo ""
    echo "  --download-only    Download both models to ~/.cache/huggingface/hub then exit"
    echo "                     without starting vLLM."
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Unknown argument: $1"
    echo "Usage: $0 [--download-only]"
    exit 1
    ;;
esac

# ---- Prerequisite checks ----------------------------------------------------
command -v docker >/dev/null 2>&1 || { echo "FATAL: docker is required"; exit 1; }
command -v curl   >/dev/null 2>&1 || { echo "FATAL: curl is required";   exit 1; }

# ---- Env exports for FP4 kernel JIT (also passed to container) --------------
export CUTE_DSL_ARCH="sm_121a"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"
export PATH="/usr/local/cuda/bin:${PATH}"
export HF_HOME
export HF_TOKEN="${HF_TOKEN:-}"
mkdir -p "${HF_HOME}" "${FLASHINFER_CACHE_DIR}"

is_hf_model_id() { [[ "${1}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; }

hf_cache_repo_dir() { echo "${HF_HOME}/hub/models--${1//\//--}"; }

# ---- Model caching helpers --------------------------------------------------
model_is_fully_cached() {
  local cache_dir snapshot
  cache_dir="$(hf_cache_repo_dir "${1}")"
  [[ -d "${cache_dir}/snapshots" ]] || return 1
  for snapshot in "${cache_dir}"/snapshots/*/; do
    [[ -d "${snapshot}" ]] || continue
    [[ -f "${snapshot}/config.json" ]] || continue
    if [[ -f "${snapshot}/model.safetensors" ]] \
      || [[ -f "${snapshot}/model.safetensors.index.json" ]] \
      || compgen -G "${snapshot}/model-"*.safetensors >/dev/null \
      || [[ -f "${snapshot}/consolidated.safetensors" ]]; then
      return 0
    fi
  done
  return 1
}

download_model() {
  local model_id="$1"
  echo ""
  echo "  >> Downloading ${model_id} …"
  echo "     (cache: ${HF_HOME})"
  echo "     This can take a while for large models."

  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" hf download "${model_id}" \
      ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" huggingface-cli download "${model_id}" \
      ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  # Fallback: download inside Docker (slow, but works)
  docker run --rm \
    --entrypoint python3 \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    "${IMAGE}" \
    -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('${model_id}', token=os.environ.get('HF_TOKEN') or None)
"
}

ensure_model() {
  local model_id="$1" label="$2"
  if model_is_fully_cached "${model_id}"; then
    echo "  [✓] ${label} (${model_id}) is cached"
  else
    echo "  [↓] ${label} not cached — downloading …"
    download_model "${model_id}"
    if model_is_fully_cached "${model_id}"; then
      echo "  [✓] ${label} download complete"
    else
      echo "  [✗] ${label} download appears incomplete — check logs above"
      exit 1
    fi
  fi
}

# ---- Download both models (idempotent) --------------------------------------
echo "=============================================================================="
echo "  Laguna-S-2.1-NVFP4  +  DFlash Speculator"
echo "  $(date)"
echo "=============================================================================="
echo ""
echo "Checking model caches …"

ensure_model "${MODEL_ID}"      "Target model"
ensure_model "${DRAFT_MODEL_ID}" "Draft model (DFlash)"
echo ""

# ---- Early exit for download-only mode --------------------------------------
if ${DOWNLOAD_ONLY}; then
  echo "=============================================================================="
  echo "  Both models are cached. Exiting (--download-only)."
  echo "=============================================================================="
  exit 0
fi

# ---- Bootstrap script -------------------------------------------------------
# Write a tiny entrypoint that installs FlashInfer nightlies (needed for FP4
# support in the stock vllm-openai image) then execs vllm serve.
cat > "${BOOTSTRAP_SCRIPT}" << 'BOOTSTRAP'
#!/bin/bash
set -e
FLASHINFER_VERSION="0.6.15.dev20260712"
INSTALLED="$(python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null || true)"
if [[ "${INSTALLED}" == "${FLASHINFER_VERSION}" ]]; then
  echo "[bootstrap] FlashInfer ${FLASHINFER_VERSION} already installed"
else
  echo "[bootstrap] Installing FlashInfer ${FLASHINFER_VERSION} for FP4 support (found: ${INSTALLED:-none}) ..."
  # flashinfer-python, flashinfer-cubin and flashinfer-jit-cache must all be
  # the SAME version or flashinfer refuses to import. The jit-cache wheel
  # carries a +cu130 local version and is published under the cu130/
  # sub-index. Without it, kernels JIT-compile from source, which fails on
  # the stock vllm-openai image (no cuBLAS dev headers: cublasLt.h missing).
  # --no-deps keeps the stock image's flashinfer-*=0.6.13 requirement from
  # pulling the trio back down. Install the small companions separately.
  pip install --upgrade --no-deps \
    "flashinfer-python==${FLASHINFER_VERSION}" \
    "flashinfer-cubin==${FLASHINFER_VERSION}" \
    "flashinfer-jit-cache==${FLASHINFER_VERSION}+cu130" \
    --extra-index-url https://flashinfer.ai/whl/nightly/ \
    --extra-index-url https://flashinfer.ai/whl/nightly/cu130/ \
    && pip install --upgrade "cuda-tile==1.5.0" "nccl4py==0.3.1" \
    && echo "[bootstrap] Installed flashinfer ${FLASHINFER_VERSION} (python + cubin + jit-cache)" \
    || { echo "[bootstrap] FATAL: flashinfer ${FLASHINFER_VERSION} install failed; refusing to start with mismatched FP4 kernels (draft acceptance collapses)"; exit 1; }
fi
echo "[bootstrap] Starting vLLM ..."
exec vllm serve "$@"
BOOTSTRAP
chmod +x "${BOOTSTRAP_SCRIPT}"

# ---- Container lifecycle ----------------------------------------------------
# Remove any existing container (running or stale) so we start fresh.
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Removing existing container ${CONTAINER_NAME} …"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting vLLM container for ${MODEL_ID}"
echo "Image: ${IMAGE}"
echo "Listening on ${HOST}:${PORT}"
echo "FlashInfer cache: ${FLASHINFER_CACHE_DIR}"
echo ""

echo "Pulling ${IMAGE} ..."
docker pull "${IMAGE}" 2>&1 || { echo "Failed to pull image"; exit 1; }
echo ""

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --user root \
  --network host \
  --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --cap-add=IPC_LOCK \
  --ipc host \
  --gpus all \
  --workdir /workspace \
  --entrypoint /bootstrap.sh \
  -e VLLM_TARGET_DEVICE=cuda \
  -e VLLM_TRUST_REMOTE_CODE=1 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e MAX_JOBS="${MAX_JOBS}" \
  -e NVCC_THREADS="${NVCC_THREADS}" \
  -e FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS}" \
  -e "PATH=/usr/local/cuda/bin:${PATH}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${BOOTSTRAP_SCRIPT}:/bootstrap.sh:ro" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${FLASHINFER_CACHE_DIR}:/root/.cache/flashinfer" \
  -v "${WORK_DIR}:/workspace" \
  "${IMAGE}" \
  "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --attention-backend FLASHINFER \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 262144 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser poolside_v1 \
    --reasoning-parser poolside_v1 \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20}' \
    --speculative-config '{"model":"poolside/Laguna-S-2.1-DFlash-NVFP4","num_speculative_tokens":15,"method":"dflash"}' \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id:0:12})"
echo "Log: ${LOG_FILE}"
echo ""

# ---- Wait for readiness -----------------------------------------------------
log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]]; then
    kill "${log_follow_pid}" 2>/dev/null || true
    wait "${log_follow_pid}" 2>/dev/null || true
    log_follow_pid=""
  fi
}
trap cleanup EXIT INT TERM

echo "Waiting for HTTP readiness at ${READY_URL}"
echo "--- container logs ---"

docker logs -f "${CONTAINER_NAME}" 2>&1 &
log_follow_pid=$!

while ! curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo ""
    echo "vLLM container exited before becoming ready"
    exit 1
  fi
  sleep 2
done

cleanup

echo ""
echo "=============================================================================="
echo "  vLLM is ready!"
echo "  OpenAI-compatible endpoint:  http://${HOST}:${PORT}/v1"
echo "=============================================================================="
