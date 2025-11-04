#!/bin/bash
# Bash script to evaluate base Qwen3-8B model on 4 tasks:
# - Acute MI (new_acutemi)
# - Hyperlipidemia (new_hyperlipidemia)
# - Hypertension (new_hypertension)
# - Pancreatic Cancer (new_pancan)

set -e  # Exit on error

# Default paths (adjust these to your environment)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
BASE_MODEL="Qwen/Qwen3-8B"
SERIALIZED_DATA_DIR="${SERIALIZED_DATA_DIR:-${SCRIPT_DIR}/serialized_multi_task_data}"
OUTPUT_DIR="${SCRIPT_DIR}/base_model_multi_task_results"

# VLLM configuration
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20000}

# Evaluation configuration
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8500}
TEMPERATURE=${TEMPERATURE:-0.7}
NUM_SAMPLES=${NUM_SAMPLES:-10}

# Quantization (set to true if needed)
USE_QUANTIZATION=${USE_QUANTIZATION:-false}

echo "=========================================="
echo "Multi-Task Base Model Evaluation"
echo "=========================================="
echo "Base Model: $BASE_MODEL"
echo "Serialized Data: $SERIALIZED_DATA_DIR"
echo "Output Dir: $OUTPUT_DIR"
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

# Build python command
PYTHON_CMD="python \"$SCRIPT_DIR/evaluate_base_model_multi_task.py\" \
    --base_model_name \"$BASE_MODEL\" \
    --path_to_serialized_data \"$SERIALIZED_DATA_DIR\" \
    --tensor_parallel_size \"$TENSOR_PARALLEL_SIZE\" \
    --gpu_memory_utilization \"$GPU_MEMORY_UTILIZATION\" \
    --max_model_len \"$MAX_MODEL_LEN\" \
    --max_new_tokens \"$MAX_NEW_TOKENS\" \
    --temperature \"$TEMPERATURE\" \
    --num_samples \"$NUM_SAMPLES\" \
    --output_dir \"$OUTPUT_DIR\" \
    $QUANTIZATION_FLAG"

# Run evaluation
echo "Starting evaluation..."
eval $PYTHON_CMD

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

