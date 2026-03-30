#!/bin/bash
set -e

# ===================================================================
# FinQA Model Evaluation Shell Script
#
# Sequentially deploys each model on GPUs 4-7 via vLLM,
# runs evaluation, and stops the server before the next model.
#
# Prerequisites:
#   - Judge model already deployed on GPUs 0-3 (port 8000)
#   - Data prepared: python -m projects.finqa.prepare_finqa_data
# ===================================================================

export PYTHONUNBUFFERED=1

# ---------- Configuration ----------

# Base paths (adjust to your environment)
RLLM_ROOT=${RLLM_ROOT:-/home/qinhengyi/rllm}
CKPT_BASE="${RLLM_ROOT}/checkpoints/finqa-grpo-curriculum"
BASE_MODEL_PATH="${RLLM_ROOT}/projects/finqa/qwen"  # Qwen3-4B base model (for tokenizer)
OFFICIAL_MODEL_PATH="${RLLM_ROOT}/finqa4b"           # Official rLLM-FinQA-4B
MERGED_CKPT_DIR="${RLLM_ROOT}/checkpoints/finqa-grpo-curriculum/merged"  # Merged HF-format checkpoints

# Judge model configuration (local vLLM on GPUs 0-3)
export FINQA_JUDGE_API_TYPE=chat_completions
export FINQA_JUDGE_BASE_URL=http://localhost:8000/v1
export FINQA_JUDGE_API_KEY=None
# Set FINQA_JUDGE_MODEL to the model path used by the judge vLLM server
export FINQA_JUDGE_MODEL=${FINQA_JUDGE_MODEL:-/merchant_reco_l3/wangmuxuan/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8}

# vLLM settings for test model (GPUs 4-7)
VLLM_PORT=30000
VLLM_GPUS="4,5,6,7"
VLLM_TP=4  # tensor parallel size = number of GPUs
VLLM_GPU_MEM=0.90
VLLM_DTYPE="bfloat16"

# Eval settings
OUTPUT_DIR="${RLLM_ROOT}/eval_results"
PROJECT_NAME="finqa-eval"
N_PARALLEL=50
MAX_STEPS=20
MAX_PROMPT_LENGTH=4096

# ---------- Model List ----------
# Format: "model_name|model_path|tokenizer_path"
# For checkpoints, we auto-find the latest global_step_xx directory.

declare -a MODELS=(
    "finqa-grpo-incur|${CKPT_BASE}/finqa-grpo-incur|${BASE_MODEL_PATH}"
    "finqa-grpo-neg|${CKPT_BASE}/finqa-grpo-neg|${BASE_MODEL_PATH}"
    "finqa-grpo-nopen|${CKPT_BASE}/finqa-grpo-nopen|${BASE_MODEL_PATH}"
    "finqa4b-official|${OFFICIAL_MODEL_PATH}|${OFFICIAL_MODEL_PATH}"
)

# ---------- Helper Functions ----------

find_latest_checkpoint() {
    local ckpt_dir="$1"
    # Find the latest global_step_XX directory
    local latest=$(ls -d "${ckpt_dir}"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
    if [ -z "$latest" ]; then
        echo "$ckpt_dir"  # fallback: use the directory itself
    else
        # Check if there's an actor subdirectory (verl format)
        if [ -d "${latest}/actor" ]; then
            echo "${latest}/actor"
        else
            echo "$latest"
        fi
    fi
}

merge_if_needed() {
    # If the checkpoint dir contains FSDP sharded files (model_world_size_*_rank_*.pt),
    # merge them into HuggingFace format so vLLM can load it.
    local model_path="$1"
    local model_name="$2"

    # Check if this is an FSDP sharded checkpoint
    local shard_count=$(ls "${model_path}"/model_world_size_*_rank_*.pt 2>/dev/null | wc -l)
    if [ "$shard_count" -eq 0 ]; then
        # Not a sharded checkpoint, use as-is
        echo "$model_path"
        return 0
    fi

    # Check if already merged
    local merged_path="${MERGED_CKPT_DIR}/${model_name}"
    if [ -f "${merged_path}/config.json" ] && [ -f "${merged_path}/model.safetensors" -o -n "$(ls ${merged_path}/model*.safetensors 2>/dev/null)" ]; then
        echo "[Merge] Already merged: ${merged_path}" >&2
        echo "$merged_path"
        return 0
    fi

    # Need to merge
    echo "[Merge] Detected FSDP sharded checkpoint (${shard_count} shards), merging..." >&2
    echo "[Merge] Output: ${merged_path}" >&2
    python3 -m projects.finqa.scripts.merge_checkpoint \
        --ckpt_dir "${model_path}" \
        --base_model_path "${BASE_MODEL_PATH}" \
        --output_dir "${merged_path}" >&2

    if [ $? -ne 0 ]; then
        echo "[Merge] ERROR: Failed to merge checkpoint" >&2
        return 1
    fi

    echo "[Merge] Merge complete: ${merged_path}" >&2
    echo "$merged_path"
    return 0
}

start_vllm_server() {
    local model_path="$1"
    echo "[vLLM] Starting server for: ${model_path}"
    echo "[vLLM] GPUs: ${VLLM_GPUS}, Port: ${VLLM_PORT}, TP: ${VLLM_TP}"

    CUDA_VISIBLE_DEVICES=${VLLM_GPUS} python -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" \
        --port ${VLLM_PORT} \
        --dtype ${VLLM_DTYPE} \
        --tensor-parallel-size ${VLLM_TP} \
        --gpu-memory-utilization ${VLLM_GPU_MEM} \
        --max-model-len 20480 \
        --disable-log-requests \
        &

    VLLM_PID=$!
    echo "[vLLM] Server PID: ${VLLM_PID}"

    # Wait for server to be ready
    echo "[vLLM] Waiting for server to be ready..."
    local max_wait=300  # max 5 minutes
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "[vLLM] Server is ready! (waited ${waited}s)"
            return 0
        fi
        # Check if process is still alive
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "[vLLM] ERROR: Server process died unexpectedly"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done

    echo "[vLLM] ERROR: Server failed to start within ${max_wait}s"
    kill $VLLM_PID 2>/dev/null
    return 1
}

stop_vllm_server() {
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo "[vLLM] Stopping server (PID: ${VLLM_PID})..."
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null || true
        echo "[vLLM] Server stopped."
    fi
    VLLM_PID=""
}

# ---------- Main Loop ----------

# Ensure data is prepared
echo "Preparing data..."
python3 -m projects.finqa.prepare_finqa_data

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Starting FinQA evaluation for ${#MODELS[@]} models"
echo "Judge: ${FINQA_JUDGE_MODEL} at ${FINQA_JUDGE_BASE_URL}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name model_path tokenizer_path <<< "$model_entry"

    echo ""
    echo "============================================================"
    echo "Model: ${model_name}"
    echo "============================================================"

    # Resolve checkpoint path (find latest global_step)
    resolved_model_path=$(find_latest_checkpoint "$model_path")
    echo "Resolved model path: ${resolved_model_path}"
    echo "Tokenizer path: ${tokenizer_path}"

    # Merge FSDP sharded checkpoint if needed
    vllm_model_path=$(merge_if_needed "${resolved_model_path}" "${model_name}")
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to merge checkpoint for ${model_name}, skipping..."
        continue
    fi
    echo "vLLM model path: ${vllm_model_path}"

    # Start vLLM server
    if ! start_vllm_server "${vllm_model_path}"; then
        echo "ERROR: Failed to start vLLM for ${model_name}, skipping..."
        stop_vllm_server
        continue
    fi

    # Run evaluation
    echo ""
    echo "Running evaluation for ${model_name}..."
    python3 -m projects.finqa.eval_finqa \
        --model_name "${model_name}" \
        --model_path "${vllm_model_path}" \
        --tokenizer_path "${tokenizer_path}" \
        --vllm_model_name "${vllm_model_path}" \
        --base_url "http://localhost:${VLLM_PORT}/v1" \
        --output_dir "${OUTPUT_DIR}" \
        --project_name "${PROJECT_NAME}" \
        --n_parallel ${N_PARALLEL} \
        --max_steps ${MAX_STEPS} \
        --max_prompt_length ${MAX_PROMPT_LENGTH} \
    || echo "WARNING: Evaluation failed for ${model_name}"

    # Stop vLLM server
    stop_vllm_server

    echo ""
    echo "Completed evaluation for ${model_name}"
    echo ""
done

echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
