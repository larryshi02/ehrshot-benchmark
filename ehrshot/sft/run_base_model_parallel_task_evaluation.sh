#!/bin/bash
# Bash script to evaluate base Qwen3-8B model on multiple tasks in parallel.
# Each task runs on a separate GPU with tensor_parallel_size=1.
#
# This script coordinates parallel evaluation where:
# - Each GPU (0, 1, 2, 3) loads the Qwen3-8B model independently
# - Each GPU evaluates one task in parallel
# - If there are more than 4 tasks, batches are processed sequentially
# - TENSOR_PARALLEL_SIZE=1 for each task (single GPU per task)

set -e  # Exit on error

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
BASE_MODEL="Qwen/Qwen3-8B"
SERIALIZED_DATA_DIR="${SERIALIZED_DATA_DIR:-${SCRIPT_DIR}/serialized_multi_task_data}"
OUTPUT_DIR="${SCRIPT_DIR}/base_model_multi_task_results"

# GPU configuration (each GPU runs one task with tensor_parallel_size=1)
NUM_GPUS=${NUM_GPUS:-4}
GPU_START_ID=${GPU_START_ID:-0}

# VLLM configuration (per-GPU settings)
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20000}

# Evaluation configuration
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8500}
TEMPERATURE=${TEMPERATURE:-0.7}
NUM_SAMPLES=${NUM_SAMPLES:-10}

# Quantization (set to true if needed)
USE_QUANTIZATION=${USE_QUANTIZATION:-false}

# Parser LLM configuration
PARSER_MODEL_NAME="${PARSER_MODEL_NAME:-Qwen/Qwen2-1.5B-Instruct}"
PARSER_GPU_MEMORY_UTILIZATION=${PARSER_GPU_MEMORY_UTILIZATION:-0.1}

# Tasks to evaluate (default: 4 tasks, but can be extended)
TASKS="${TASKS:-acute_mi hyperlipidemia hypertension pancreatic_cancer}"

echo "=========================================="
echo "Parallel Multi-Task Base Model Evaluation"
echo "=========================================="
echo "Base Model: $BASE_MODEL"
echo "Serialized Data: $SERIALIZED_DATA_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "GPU IDs: $(seq -s ' ' $GPU_START_ID $((GPU_START_ID + NUM_GPUS - 1)))"
echo "Tasks: $TASKS"
echo "Tensor Parallel Size: 1 (each task runs on one GPU)"
echo "=========================================="
echo ""

# Check if serialized data directory exists
if [ ! -d "$SERIALIZED_DATA_DIR" ]; then
    echo "ERROR: Serialized data directory not found: $SERIALIZED_DATA_DIR"
    echo "Please run serialize_multi_task_data.py first to create serialized data"
    exit 1
fi

if [ ! "$(ls -A $SERIALIZED_DATA_DIR/*_all_splits.json 2>/dev/null)" ]; then
    echo "ERROR: No serialized data files found in: $SERIALIZED_DATA_DIR"
    echo "Expected files: {task_name}_all_splits.json (e.g., acute_mi_all_splits.json)"
    echo "Please run serialize_multi_task_data.py first to create serialized data"
    exit 1
fi

echo "Using pre-serialized data from: $SERIALIZED_DATA_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build quantization flag
QUANTIZATION_FLAG=""
if [ "$USE_QUANTIZATION" = "true" ]; then
    QUANTIZATION_FLAG="--use_quantization"
fi

# Build python command for parallel coordinator
PYTHON_CMD="python \"$SCRIPT_DIR/evaluate_base_model_parallel_tasks.py\" \
    --tasks $TASKS \
    --num_gpus \"$NUM_GPUS\" \
    --gpu_start_id \"$GPU_START_ID\" \
    --base_model_name \"$BASE_MODEL\" \
    --path_to_serialized_data \"$SERIALIZED_DATA_DIR\" \
    --gpu_memory_utilization \"$GPU_MEMORY_UTILIZATION\" \
    --max_model_len \"$MAX_MODEL_LEN\" \
    --max_new_tokens \"$MAX_NEW_TOKENS\" \
    --temperature \"$TEMPERATURE\" \
    --num_samples \"$NUM_SAMPLES\" \
    --parser_model_name \"$PARSER_MODEL_NAME\" \
    --parser_gpu_memory_utilization \"$PARSER_GPU_MEMORY_UTILIZATION\" \
    --output_dir \"$OUTPUT_DIR\" \
    $QUANTIZATION_FLAG"

# Run parallel evaluation
echo "Starting parallel evaluation..."
echo ""
eval $PYTHON_CMD

# Check exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Parallel evaluation completed successfully!"
else
    echo "Parallel evaluation completed with errors (exit code: $EXIT_CODE)"
fi
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

exit $EXIT_CODE

