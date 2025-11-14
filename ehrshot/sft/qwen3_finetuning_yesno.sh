#!/bin/bash
# Clean Fine-Tuning Pipeline - Runs Batch 1, waits for completion, then runs Batch 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MODEL_NAME="Qwen/Qwen3-8B"
SERIALIZED_DATA_DIR="${SERIALIZED_DATA_DIR:-${SCRIPT_DIR}/serialized_multi_task_data}"
OUTPUT_BASE_DIR="${SCRIPT_DIR}/qwen3_sft_yesno_output"
NUM_EPOCHS=3
EVAL_STEPS=100
SEED=42
CONDA_ENV="${CONDA_ENV:-ehrshot-env-new}"

# Batch 1: Run first
BATCH1_TASKS=("acute_mi" "pancreatic_cancer")
BATCH1_GPU_IDS=("0,1" "2,3")

# Batch 2: Run after Batch 1 completes
BATCH2_TASKS=("hypertension" "hyperlipidemia")
BATCH2_GPU_IDS=("0,1" "2,3")

mkdir -p "$OUTPUT_BASE_DIR"

# Launch task
launch_task() {
    local task_name=$1
    local gpu_id=$2
    local output_dir="$OUTPUT_BASE_DIR/${task_name}"
    local log_file="$OUTPUT_BASE_DIR/${task_name}_gpu${gpu_id}.log"
    
    mkdir -p "$output_dir"
    
    (
        source $HOME/miniconda3/etc/profile.d/conda.sh
        conda activate $CONDA_ENV
        python "$SCRIPT_DIR/qwen3_sft_yesno.py" \
            --model_name "$MODEL_NAME" \
            --task_name "$task_name" \
            --serialized_data_dir "$SERIALIZED_DATA_DIR" \
            --output_dir "$output_dir" \
            --num_train_epochs "$NUM_EPOCHS" \
            --eval_steps "$EVAL_STEPS" \
            --gpu_id "$gpu_id" \
            --seed "$SEED"
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

# Run batch
run_batch() {
    local batch_num=$1
    local tasks_ref=$2[@]
    local gpu_ids_ref=$3[@]
    local -a tasks=("${!tasks_ref}")
    local -a gpu_ids=("${!gpu_ids_ref}")
    
    echo "=========================================="
    echo "Starting Batch $batch_num"
    echo "=========================================="
    
    local pids=()
    for i in "${!tasks[@]}"; do
        pid=$(launch_task "${tasks[$i]}" "${gpu_ids[$i]}")
        pids+=($pid)
        echo "  Launched: ${tasks[$i]} on GPU(s) ${gpu_ids[$i]} (PID: $pid)"
    done
    
    echo "Waiting for Batch $batch_num to complete..."
    wait_for_pids "${pids[@]}"
    echo "Batch $batch_num completed!"
    echo ""
}

# Run batches sequentially
run_batch 1 BATCH1_TASKS BATCH1_GPU_IDS
run_batch 2 BATCH2_TASKS BATCH2_GPU_IDS

echo "=========================================="
echo "All batches completed!"
echo "Logs: $OUTPUT_BASE_DIR/*.log"
echo "=========================================="
