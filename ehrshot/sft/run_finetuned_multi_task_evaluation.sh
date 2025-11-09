#!/bin/bash
# Bash script to evaluate finetuned Qwen model on 4 tasks:
# - Acute MI (new_acutemi)
# - Hyperlipidemia (new_hyperlipidemia)
# - Hypertension (new_hypertension)
# - Pancreatic Cancer (new_pancan)

set -e  # Exit on error

# Default paths (adjust these to your environment)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
BASE_MODEL="${BASE_MODEL:-${SCRIPT_DIR}/qwen_sft_output}"
SERIALIZED_DATA_DIR="${SERIALIZED_DATA_DIR:-${SCRIPT_DIR}/serialized_multi_task_data}"
OUTPUT_DIR="${SCRIPT_DIR}/finetuned_multi_task_results"

# VLLM configuration
# Note: Each process will use TENSOR_PARALLEL_SIZE=1 (one GPU per task)
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20000}

# Evaluation configuration
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8500}
TEMPERATURE=${TEMPERATURE:-0.7}
NUM_SAMPLES=${NUM_SAMPLES:-10}

# Quantization (set to true if needed)
USE_QUANTIZATION=${USE_QUANTIZATION:-false}

echo "=========================================="
echo "Multi-Task Finetuned Model Evaluation"
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

# Define tasks and their corresponding GPU IDs
# Each task will run on a separate GPU in parallel
TASKS=("acute_mi" "hyperlipidemia" "hypertension" "pancreatic_cancer")
GPU_IDS=(0 1 2 3)

# Validate that we have 4 GPUs available
NUM_GPUS=${#GPU_IDS[@]}
NUM_TASKS=${#TASKS[@]}

if [ "$NUM_GPUS" -ne "$NUM_TASKS" ]; then
    echo "ERROR: Number of GPUs ($NUM_GPUS) must match number of tasks ($NUM_TASKS)"
    exit 1
fi

echo "=========================================="
echo "Parallel Evaluation Configuration"
echo "=========================================="
echo "Tasks: ${TASKS[@]}"
echo "GPU IDs: ${GPU_IDS[@]}"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE (per task)"
echo "=========================================="
echo ""

# Function to run a single task on a specific GPU
run_task_on_gpu() {
    local task_name=$1
    local gpu_id=$2
    local log_file="$OUTPUT_DIR/${task_name}_gpu${gpu_id}.log"
    
    echo "Starting task '$task_name' on GPU $gpu_id (log: $log_file)"
    
    python "$SCRIPT_DIR/evaluate_finetuned_multi_task.py" \
        --base_model_name "$BASE_MODEL" \
        --path_to_serialized_data "$SERIALIZED_DATA_DIR" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --max_model_len "$MAX_MODEL_LEN" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --gpu_id "$gpu_id" \
        --task_name "$task_name" \
        $QUANTIZATION_FLAG \
        > "$log_file" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Task '$task_name' on GPU $gpu_id completed successfully"
    else
        echo "ERROR: Task '$task_name' on GPU $gpu_id failed with exit code $exit_code"
        echo "Check log file: $log_file"
    fi
    return $exit_code
}

# Export function so it can be used in background processes
export -f run_task_on_gpu
export SCRIPT_DIR BASE_MODEL SERIALIZED_DATA_DIR OUTPUT_DIR
export TENSOR_PARALLEL_SIZE GPU_MEMORY_UTILIZATION MAX_MODEL_LEN
export MAX_NEW_TOKENS TEMPERATURE NUM_SAMPLES QUANTIZATION_FLAG

# Launch all tasks in parallel
echo "Launching $NUM_TASKS tasks in parallel..."
PIDS=()
for i in "${!TASKS[@]}"; do
    task_name="${TASKS[$i]}"
    gpu_id="${GPU_IDS[$i]}"
    
    # Run task in background
    run_task_on_gpu "$task_name" "$gpu_id" &
    PIDS+=($!)
    echo "Launched task '$task_name' on GPU $gpu_id (PID: ${PIDS[$i]})"
done

echo ""
echo "All tasks launched. Waiting for completion..."
echo ""

# Wait for all background processes to complete
FAILED_TASKS=()
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    task_name="${TASKS[$i]}"
    gpu_id="${GPU_IDS[$i]}"
    
    wait $pid
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        FAILED_TASKS+=("$task_name (GPU $gpu_id)")
    fi
done

echo ""
echo "=========================================="
if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "All tasks completed successfully!"
else
    echo "WARNING: Some tasks failed:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
fi
echo "Results saved to: $OUTPUT_DIR"
echo "Log files: $OUTPUT_DIR/*_gpu*.log"
echo "=========================================="

