#!/bin/bash
# Bash script to evaluate fine-tuned models on multiple tasks in parallel using vLLM.
# Each task runs on a separate GPU with LoRA adapter loaded.
# Output format: yes/no (single word)
# Samples per query: 10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
FINETUNED_MODELS_BASE_DIR="${FINETUNED_MODELS_BASE_DIR:-${SCRIPT_DIR}/qwen3_sft_clean_output}"
SERIALIZED_DATA_DIR="${SERIALIZED_DATA_DIR:-${SCRIPT_DIR}/serialized_multi_task_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/finetuned_evaluation_vllm_results}"
CONDA_ENV="${CONDA_ENV:-ehrshot-env-new}"

# VLLM configuration
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-9000}

# Evaluation configuration
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1}
TEMPERATURE=${TEMPERATURE:-0.7}
NUM_SAMPLES=${NUM_SAMPLES:-10}

# Task configurations
TASKS=(
    "acute_mi"
    "pancreatic_cancer"
    "hypertension"
    "hyperlipidemia"
)

# GPU assignments (one GPU per task, must match TASKS array length)
GPUS=(
    "0"
    "1"
    "2"
    "3"
)

mkdir -p "$OUTPUT_DIR"

# Launch task
launch_task() {
    local task_name=$1
    local gpu_id=$2
    local log_file="$OUTPUT_DIR/${task_name}_gpu${gpu_id}.log"
    
    (
        source $HOME/miniconda3/etc/profile.d/conda.sh
        conda activate $CONDA_ENV
        python "$SCRIPT_DIR/evaluate_finetuned_single_task_vllm.py" \
            --task_name "$task_name" \
            --gpu_id "$gpu_id" \
            --base_model_name "$BASE_MODEL" \
            --peft_model_path "$FINETUNED_MODELS_BASE_DIR/$task_name" \
            --path_to_serialized_data "$SERIALIZED_DATA_DIR" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --max_model_len "$MAX_MODEL_LEN" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --temperature "$TEMPERATURE" \
            --num_samples "$NUM_SAMPLES" \
            --output_dir "$OUTPUT_DIR"
    ) > "$log_file" 2>&1 &
    
    echo $!
}

# Wait for PIDs using polling (guaranteed to work)
wait_for_pids() {
    local pids=("$@")
    
    while true; do
        local all_done=true
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                all_done=false
                break
            fi
        done
        
        if [ "$all_done" = true ]; then
            break
        fi
        sleep 30
    done
}

echo "=========================================="
echo "Parallel Fine-Tuned Model Evaluation (vLLM)"
echo "=========================================="
echo "Base Model: $BASE_MODEL"
echo "Fine-tuned Models Dir: $FINETUNED_MODELS_BASE_DIR"
echo "Serialized Data: $SERIALIZED_DATA_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Tasks: ${#TASKS[@]}"
echo "GPUs: ${GPUS[@]}"
echo "Samples per query: $NUM_SAMPLES"
echo "=========================================="
echo ""

# Check if serialized data directory exists
if [ ! -d "$SERIALIZED_DATA_DIR" ]; then
    echo "ERROR: Serialized data directory not found: $SERIALIZED_DATA_DIR"
    echo "Please run serialize_multi_task_data.py first"
    exit 1
fi

if [ ! "$(ls -A $SERIALIZED_DATA_DIR/*_all_splits.json 2>/dev/null)" ]; then
    echo "ERROR: No serialized data files found in: $SERIALIZED_DATA_DIR"
    echo "Expected files: {task_name}_all_splits.json"
    exit 1
fi

# Check if fine-tuned models directory exists
if [ ! -d "$FINETUNED_MODELS_BASE_DIR" ]; then
    echo "ERROR: Fine-tuned models base directory not found: $FINETUNED_MODELS_BASE_DIR"
    echo "Please train models first using run_clean_finetuning.sh"
    exit 1
fi

# Validate arrays have same length
if [ ${#TASKS[@]} -ne ${#GPUS[@]} ]; then
    echo "ERROR: TASKS array (${#TASKS[@]} items) and GPUS array (${#GPUS[@]} items) must have the same length"
    exit 1
fi

# Validate that model directories exist for each task
for task_name in "${TASKS[@]}"; do
    task_model_dir="$FINETUNED_MODELS_BASE_DIR/$task_name"
    if [ ! -d "$task_model_dir" ]; then
        echo "ERROR: Fine-tuned model directory not found for task '$task_name': $task_model_dir"
        exit 1
    fi
    echo "Found model for task '$task_name': $task_model_dir"
done

echo ""
echo "All model directories validated successfully"
echo ""

echo "=========================================="
echo "Launching evaluation tasks..."
echo "=========================================="

PIDS=()
for i in "${!TASKS[@]}"; do
    task_name="${TASKS[$i]}"
    gpu_id="${GPUS[$i]}"
    pid=$(launch_task "$task_name" "$gpu_id")
    PIDS+=($pid)
    echo "  Launched: $task_name on GPU $gpu_id (PID: $pid)"
done

echo ""
echo "Waiting for all tasks to complete..."
wait_for_pids "${PIDS[@]}"
echo "All tasks completed!"
echo ""

# Check which tasks completed successfully
echo "=========================================="
echo "Task completion status:"
echo "=========================================="
for task_name in "${TASKS[@]}"; do
    summary_file="$OUTPUT_DIR/${task_name}_summary.json"
    if [ -f "$summary_file" ]; then
        echo "  ✓ $task_name - completed"
    else
        echo "  ✗ $task_name - failed or incomplete"
    fi
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Logs: $OUTPUT_DIR/*_gpu*.log"
echo "=========================================="
