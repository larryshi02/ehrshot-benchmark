#!/bin/bash
# SFT Training Pipeline - Parallel Execution
# Runs 4 independent training tasks on GPUs 0, 1, 2, 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE_DIR="${SCRIPT_DIR}/qwen_sft_output_gpt5"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"

# --- DATASET CONFIGURATION ---

# 1. Acute MI
TASK1_TRAIN="./data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/train/acute_mi_balanced_sft_dataset.json"
TASK1_VAL="./data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/val/acute_mi_balanced_sft_dataset.json"
TASK1_OUT="${OUTPUT_BASE_DIR}/acute_mi"

# 2. Pancreatic Cancer
TASK2_TRAIN="./data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/train/pancreatic_cancer_balanced_sft_dataset.json"
TASK2_VAL="./data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/val/pancreatic_cancer_balanced_sft_dataset.json"
TASK2_OUT="${OUTPUT_BASE_DIR}/pancreatic_cancer"

# 3. Hypertension
TASK3_TRAIN="./data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/train/hypertension_balanced_sft_dataset.json"
TASK3_VAL="./data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/val/hypertension_balanced_sft_dataset.json"
TASK3_OUT="${OUTPUT_BASE_DIR}/hypertension"

# 4. Hyperlipidemia
TASK4_TRAIN="./data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/train/hyperlipidemia_balanced_sft_dataset.json"
TASK4_VAL="./data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/val/hyperlipidemia_balanced_sft_dataset.json"
TASK4_OUT="${OUTPUT_BASE_DIR}/hyperlipidemia"

# Arrays
ALL_TRAIN=("$TASK1_TRAIN" "$TASK2_TRAIN" "$TASK3_TRAIN" "$TASK4_TRAIN")
ALL_VAL=("$TASK1_VAL" "$TASK2_VAL" "$TASK3_VAL" "$TASK4_VAL")
ALL_OUT=("$TASK1_OUT" "$TASK2_OUT" "$TASK3_OUT" "$TASK4_OUT")
ALL_GPUS=("0" "1" "2" "3")

mkdir -p "$OUTPUT_BASE_DIR"

# --- LAUNCH FUNCTION ---
launch_task() {
    local train_file=$1
    local eval_file=$2
    local output_dir=$3
    local gpu_id=$4
    
    local task_name=$(basename "$train_file" | sed 's/_balanced_sft_dataset.json//')
    local log_file="$OUTPUT_BASE_DIR/${task_name}.log"
    
    mkdir -p "$output_dir"
    
    echo "  [GPU $gpu_id] Launching $task_name..."
    
    (
        source $HOME/miniconda3/etc/profile.d/conda.sh
        conda activate $CONDA_ENV
        
        # Force this sub-shell to only see one GPU
        export CUDA_ VISIBLE_DEVICES=$gpu_id
        # --- WANDB CONFIGURATION ---
        export WANDB_PROJECT="ehrshot-qwen-sft"
        # Name the run based on the task (e.g., "acute_mi")
        export WANDB_NAME="${task_name}"
        
        # Simplified python call - Configs are defaults inside the script
        python "$SCRIPT_DIR/qwen_sft_pipeline.py" \
            --train_file "$train_file" \
            --eval_file "$eval_file" \
            --output_dir "$output_dir"
            
    ) > "$log_file" 2>&1 &
    
    echo $!
}

# --- EXECUTION ---

echo "=========================================="
echo "Starting Parallel Training on 4 GPUs"
echo "=========================================="

pids=()

# Launch all 4 tasks
for i in "${!ALL_TRAIN[@]}"; do
    pid=$(launch_task \
        "${ALL_TRAIN[$i]}" \
        "${ALL_VAL[$i]}" \
        "${ALL_OUT[$i]}" \
        "${ALL_GPUS[$i]}")
    pids+=($pid)
done

echo ""
echo "Tasks launched. PIDs: ${pids[*]}"
echo "Waiting for completion..."

wait

echo "=========================================="
echo "All training tasks completed!"
echo "=========================================="