#!/bin/bash
# SFT Training Pipeline with Balanced Datasets
# - Concatenates original dataset with balanced train dataset for training
# - Uses balanced val dataset for evaluation (no 0.9/0.1 split)
# - Runs Batch 1, waits for completion, then runs Batch 2
# - Uses 2 GPUs per task, runs 2 tasks in parallel per batch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MODEL_NAME="Qwen/Qwen3-8B"
# Note: SPLITS_FILE is not used in balanced mode, but kept for reference
SPLITS_FILE="/home/lrshi/llm/ehrshot-benchmark/EHRSHOT_ASSETS/splits/person_id_map.csv"
OUTPUT_BASE_DIR="${SCRIPT_DIR}/qwen_sft_output_gpt5"
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=2e-4
MAX_LENGTH=4096
SEED=42
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"

# Batch 1: Run first
# Original datasets (for concatenation with balanced train)
BATCH1_ORIGINAL_DATASETS=(
    "./data_gpt5/acute_mi_sft_dataset.json"
    "./data_gpt5/pancreatic_cancer_sft_dataset.json"
)
# Balanced train datasets
BATCH1_BALANCED_TRAIN_DATASETS=(
    "./data_gpt5_balanced/train/acute_mi_balanced_sft_dataset.json"
    "./data_gpt5_balanced/train/pancreatic_cancer_balanced_sft_dataset.json"
)
# Balanced val datasets (for eval)
BATCH1_BALANCED_VAL_DATASETS=(
    "./data_gpt5_balanced/val/acute_mi_balanced_sft_dataset.json"
    "./data_gpt5_balanced/val/pancreatic_cancer_balanced_sft_dataset.json"
)
BATCH1_OUTPUT_DIRS=(
    "${OUTPUT_BASE_DIR}/acute_mi"
    "${OUTPUT_BASE_DIR}/pancreatic_cancer"
)
BATCH1_GPU_IDS=("0,1" "2,3")

# Batch 2: Run after Batch 1 completes
# Original datasets (for concatenation with balanced train)
BATCH2_ORIGINAL_DATASETS=(
    "./data_gpt5/hypertension_sft_dataset.json"
    "./data_gpt5/hyperlipidemia_sft_dataset.json"
)
# Balanced train datasets
BATCH2_BALANCED_TRAIN_DATASETS=(
    "./data_gpt5_balanced/train/hypertension_balanced_sft_dataset.json"
    "./data_gpt5_balanced/train/hyperlipidemia_balanced_sft_dataset.json"
)
# Balanced val datasets (for eval)
BATCH2_BALANCED_VAL_DATASETS=(
    "./data_gpt5_balanced/val/hypertension_balanced_sft_dataset.json"
    "./data_gpt5_balanced/val/hyperlipidemia_balanced_sft_dataset.json"
)
BATCH2_OUTPUT_DIRS=(
    "${OUTPUT_BASE_DIR}/hypertension"
    "${OUTPUT_BASE_DIR}/hyperlipidemia"
)
BATCH2_GPU_IDS=("0,1" "2,3")

mkdir -p "$OUTPUT_BASE_DIR"

# Launch task
launch_task() {
    local original_dataset_file=$1
    local balanced_train_dataset_file=$2
    local balanced_val_dataset_file=$3
    local output_dir=$4
    local gpu_id=$5
    local task_name=$(basename "$balanced_train_dataset_file" | sed 's/_balanced_sft_dataset.json//')
    local log_file="$OUTPUT_BASE_DIR/${task_name}_gpu${gpu_id}.log"
    
    # Check if balanced dataset files exist
    if [ ! -f "$balanced_train_dataset_file" ]; then
        echo "Error: Balanced train dataset file not found: $balanced_train_dataset_file"
        echo "Skipping task: $task_name"
        return 1
    fi
    
    if [ ! -f "$balanced_val_dataset_file" ]; then
        echo "Error: Balanced val dataset file not found: $balanced_val_dataset_file"
        echo "Skipping task: $task_name"
        return 1
    fi
    
    # Original dataset is optional (will use only balanced train if not provided)
    if [ ! -f "$original_dataset_file" ]; then
        echo "Warning: Original dataset file not found: $original_dataset_file"
        echo "Will use only balanced train dataset for training"
    fi
    
    mkdir -p "$output_dir"
    
    (
        source $HOME/miniconda3/etc/profile.d/conda.sh
        conda activate $CONDA_ENV
        python "$SCRIPT_DIR/qwen_sft_pipeline.py" \
            --model_name "$MODEL_NAME" \
            --original_dataset_file "$original_dataset_file" \
            --balanced_train_dataset_file "$balanced_train_dataset_file" \
            --balanced_val_dataset_file "$balanced_val_dataset_file" \
            --output_dir "$output_dir" \
            --num_train_epochs $NUM_EPOCHS \
            --per_device_train_batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM \
            --learning_rate $LEARNING_RATE \
            --max_length $MAX_LENGTH \
            --use_lora \
            --lora_r 16 \
            --lora_alpha 32 \
            --lora_dropout 0.1 \
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
    local original_datasets_ref=$2[@]
    local balanced_train_datasets_ref=$3[@]
    local balanced_val_datasets_ref=$4[@]
    local output_dirs_ref=$5[@]
    local gpu_ids_ref=$6[@]
    local -a original_datasets=("${!original_datasets_ref}")
    local -a balanced_train_datasets=("${!balanced_train_datasets_ref}")
    local -a balanced_val_datasets=("${!balanced_val_datasets_ref}")
    local -a output_dirs=("${!output_dirs_ref}")
    local -a gpu_ids=("${!gpu_ids_ref}")
    
    echo "=========================================="
    echo "Starting Batch $batch_num"
    echo "=========================================="
    
    local pids=()
    for i in "${!balanced_train_datasets[@]}"; do
        pid=$(launch_task \
            "${original_datasets[$i]}" \
            "${balanced_train_datasets[$i]}" \
            "${balanced_val_datasets[$i]}" \
            "${output_dirs[$i]}" \
            "${gpu_ids[$i]}")
        if [ $? -eq 0 ] && [ -n "$pid" ]; then
            pids+=($pid)
            echo "  Launched: $(basename "${balanced_train_datasets[$i]}") on GPU(s) ${gpu_ids[$i]} (PID: $pid)"
        fi
    done
    
    if [ ${#pids[@]} -eq 0 ]; then
        echo "No tasks launched in Batch $batch_num (missing dataset files?)"
        return
    fi
    
    echo "Waiting for Batch $batch_num to complete..."
    wait_for_pids "${pids[@]}"
    echo "Batch $batch_num completed!"
    echo ""
}

# Run batches sequentially
run_batch 1 BATCH1_ORIGINAL_DATASETS BATCH1_BALANCED_TRAIN_DATASETS BATCH1_BALANCED_VAL_DATASETS BATCH1_OUTPUT_DIRS BATCH1_GPU_IDS
run_batch 2 BATCH2_ORIGINAL_DATASETS BATCH2_BALANCED_TRAIN_DATASETS BATCH2_BALANCED_VAL_DATASETS BATCH2_OUTPUT_DIRS BATCH2_GPU_IDS

echo "=========================================="
echo "All batches completed!"
echo "Logs: $OUTPUT_BASE_DIR/*.log"
echo "=========================================="
