#!/bin/bash
# Bash script to evaluate base models on 4 tasks in parallel using data parallelization.
# Each task runs on a separate GPU (one GPU per task).
#
# Tasks evaluated:
# - Acute MI (new_acutemi)
# - Hyperlipidemia (new_hyperlipidemia)
# - Hypertension (new_hypertension)
# - Pancreatic Cancer (new_pancan)
#
# Note: Quantization is not used - models are loaded in bfloat16 precision.
# The output directory will automatically include the model name.

set -e  # Exit on error

# ============================================================================
# Configuration - Modify these as needed
# ============================================================================

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model configuration
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"  # Model name/path (e.g., "Qwen/Qwen3-8B")

# Data configuration
SERIALIZED_DATA_DIR="${SERIALIZED_DATA_DIR:-${SCRIPT_DIR}/serialized_multi_task_data}"

# Output configuration (base directory - model name will be automatically appended)
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-${SCRIPT_DIR}/base_model_multi_task_results}"

# VLLM configuration
# Each task uses one GPU (data parallelization, not tensor parallelization)
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}  # GPUs per task (1 for data parallel)
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}  # GPU memory usage (0-1)
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20000}  # Maximum sequence length

# Evaluation configuration
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8500}  # Maximum tokens to generate per sample
TEMPERATURE=${TEMPERATURE:-0.7}  # Sampling temperature
NUM_SAMPLES=${NUM_SAMPLES:-10}  # Number of samples per patient
# Data fraction: 0.05 = 5% of test set (for pilot runs), 1.0 = full dataset
DATA_FRACTION=${DATA_FRACTION:-1.00}  # Default to 5% for pilot runs

# ============================================================================
# GPU Parallelization Configuration
# ============================================================================
# Define which tasks run on which GPUs (one GPU per task)
# Tasks and GPU IDs must be in the same order and have matching counts

TASKS=("acute_mi" "hyperlipidemia" "hypertension" "pancreatic_cancer")
GPU_IDS=(0 1 2 3)

# ============================================================================
# Validation and Setup
# ============================================================================

# Print configuration summary
echo "=========================================="
echo "Multi-Task Base Model Evaluation"
echo "=========================================="
echo "Base Model: $BASE_MODEL"
echo "Serialized Data: $SERIALIZED_DATA_DIR"
echo "Output Base Dir: $OUTPUT_DIR_BASE"
echo "=========================================="
echo ""

# Validate serialized data directory
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

# Validate GPU and task configuration
NUM_TASKS=${#TASKS[@]}
NUM_GPUS=${#GPU_IDS[@]}

if [ "$NUM_GPUS" -ne "$NUM_TASKS" ]; then
    echo "ERROR: Number of GPUs ($NUM_GPUS) must match number of tasks ($NUM_TASKS)"
    echo "Please ensure TASKS and GPU_IDS arrays have the same length"
    exit 1
fi

# Display GPU-to-task mapping
echo "=========================================="
echo "GPU Parallelization Configuration"
echo "=========================================="
echo "Task-to-GPU Mapping:"
for i in "${!TASKS[@]}"; do
    echo "  ${TASKS[$i]} -> GPU ${GPU_IDS[$i]}"
done
echo ""
echo "Configuration:"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE (per task - data parallel mode)"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  Max New Tokens: $MAX_NEW_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo "  Num Samples: $NUM_SAMPLES"
echo "  Data Fraction: $DATA_FRACTION ($(awk "BEGIN {printf \"%.1f%%\", $DATA_FRACTION*100}") of test set)"
echo "  Quantization: DISABLED (models use bfloat16 precision)"
echo "=========================================="
echo ""

# ============================================================================
# Helper Functions
# ============================================================================

# Function to run a single task on a specific GPU
# Args:
#   $1: Task name (e.g., "acute_mi")
#   $2: GPU ID (e.g., 0)
run_task_on_gpu() {
    local task_name=$1
    local gpu_id=$2
    # Extract model name from BASE_MODEL to match Python script's extract_model_name_for_dir()
    # Python: base_model_name.split("/")[-1].lower() -> re.sub(r'[_\s]+', '-', ...)
    # Shell equivalent: extract last part, lowercase, replace underscores/spaces with hyphens
    local model_name=$(echo "$BASE_MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[_ ]\+/-/g')
    
    # Log file location: save logs in the model directory (same level as task directories)
    # Python script creates: OUTPUT_DIR_BASE/model_name/task_name/ for results
    # So we save logs to: OUTPUT_DIR_BASE/model_name/ (same level as task subdirectories)
    local model_output_dir="$OUTPUT_DIR_BASE/$model_name"
    mkdir -p "$model_output_dir"
    local log_file="$model_output_dir/${task_name}_gpu${gpu_id}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting task '$task_name' on GPU $gpu_id"
    echo "  Log file: $log_file"
    
    # Run evaluation for this task on the specified GPU
    # Python script will create: OUTPUT_DIR_BASE/model_name/task_name/ for results
    python "$SCRIPT_DIR/evaluate_base_model_multi_task.py" \
        --base_model_name "$BASE_MODEL" \
        --path_to_serialized_data "$SERIALIZED_DATA_DIR" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --max_model_len "$MAX_MODEL_LEN" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --num_samples "$NUM_SAMPLES" \
        --data_fraction "$DATA_FRACTION" \
        --output_dir "$OUTPUT_DIR_BASE" \
        --gpu_id "$gpu_id" \
        --task_name "$task_name" \
        > "$log_file" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Task '$task_name' on GPU $gpu_id completed successfully"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ ERROR: Task '$task_name' on GPU $gpu_id failed (exit code: $exit_code)"
        echo "  Check log file: $log_file"
    fi
    return $exit_code
}

# ============================================================================
# Export variables for background processes
# ============================================================================
export -f run_task_on_gpu
export SCRIPT_DIR BASE_MODEL SERIALIZED_DATA_DIR OUTPUT_DIR_BASE
export TENSOR_PARALLEL_SIZE GPU_MEMORY_UTILIZATION MAX_MODEL_LEN
export MAX_NEW_TOKENS TEMPERATURE NUM_SAMPLES DATA_FRACTION

# ============================================================================
# Launch Parallel Evaluation
# ============================================================================

echo "Launching $NUM_TASKS tasks in parallel (one per GPU)..."
echo ""

# Launch all tasks in parallel (background processes)
PIDS=()
for i in "${!TASKS[@]}"; do
    task_name="${TASKS[$i]}"
    gpu_id="${GPU_IDS[$i]}"
    
    # Run task in background
    run_task_on_gpu "$task_name" "$gpu_id" &
    PIDS+=($!)
    echo "  → Launched task '$task_name' on GPU $gpu_id (PID: ${PIDS[$i]})"
done

echo ""
echo "All tasks launched. Waiting for completion..."
# Extract model name for log file message
MODEL_NAME_FOR_LOGS=$(echo "$BASE_MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[_ ]\+/-/g')
echo "Monitor progress in log files: $OUTPUT_DIR_BASE/$MODEL_NAME_FOR_LOGS/*_gpu*.log"
echo ""

# Wait for all background processes to complete and collect results
FAILED_TASKS=()
COMPLETED_TASKS=()
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    task_name="${TASKS[$i]}"
    gpu_id="${GPU_IDS[$i]}"
    
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        COMPLETED_TASKS+=("$task_name (GPU $gpu_id)")
    else
        FAILED_TASKS+=("$task_name (GPU $gpu_id)")
    fi
done

# ============================================================================
# Final Summary
# ============================================================================

echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "✓ All tasks completed successfully!"
    echo ""
    echo "Completed tasks:"
    for task in "${COMPLETED_TASKS[@]}"; do
        echo "  ✓ $task"
    done
else
    echo "⚠ WARNING: Some tasks failed"
    echo ""
    if [ ${#COMPLETED_TASKS[@]} -gt 0 ]; then
        echo "Completed tasks:"
        for task in "${COMPLETED_TASKS[@]}"; do
            echo "  ✓ $task"
        done
        echo ""
    fi
    echo "Failed tasks:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  ✗ $task"
    done
fi
# Extract model name for final message (matching Python script's extraction logic)
MODEL_NAME=$(echo "$BASE_MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[_ ]\+/-/g')
echo ""
echo "Results saved to: $OUTPUT_DIR_BASE/$MODEL_NAME/{task_name}/"
echo "  - Each task has its own subdirectory (e.g., $OUTPUT_DIR_BASE/$MODEL_NAME/acute_mi/)"
echo "  - Model name extracted from: $BASE_MODEL -> $MODEL_NAME"
echo "  - Combined summary: $OUTPUT_DIR_BASE/$MODEL_NAME/all_tasks_summary.csv"
echo "  - Log files: $OUTPUT_DIR_BASE/$MODEL_NAME/*_gpu*.log"
echo "=========================================="

