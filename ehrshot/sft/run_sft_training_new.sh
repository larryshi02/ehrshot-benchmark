#!/bin/bash
# SFT Training Pipeline - Parallel Execution
# Runs 4 independent training tasks on GPUs 0, 1, 2, 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"

# ============================================================
# USER CONFIGURATION - Edit these values only
# ============================================================

# --- MODEL CONFIGURATION ---
MODEL_NAME="Qwen/Qwen3-1.7B"

# --- LORA CONFIGURATION ---
LORA_RANK=16
# Note: lora_alpha will be automatically set to 2*lora_rank in the Python script

# --- DATASET CONFIGURATION ---

# 1. Acute MI
TASK1_TRAIN="${SCRIPT_DIR}/data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/train/acute_mi_balanced_sft_dataset.json"
TASK1_VAL="${SCRIPT_DIR}/data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/val_small/acute_mi_balanced_sft_dataset.json"

# 2. Pancreatic Cancer
TASK2_TRAIN="${SCRIPT_DIR}/data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/train/pancreatic_cancer_balanced_sft_dataset.json"
TASK2_VAL="${SCRIPT_DIR}/data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/val_small/pancreatic_cancer_balanced_sft_dataset.json"

# 3. Hypertension
TASK3_TRAIN="${SCRIPT_DIR}/data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/train/hypertension_balanced_sft_dataset.json"
TASK3_VAL="${SCRIPT_DIR}/data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/val_small/hypertension_balanced_sft_dataset.json"

# 4. Hyperlipidemia
TASK4_TRAIN="${SCRIPT_DIR}/data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/train/hyperlipidemia_balanced_sft_dataset.json"
TASK4_VAL="${SCRIPT_DIR}/data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/val_small/hyperlipidemia_balanced_sft_dataset.json"

# ============================================================
# END OF USER CONFIGURATION
# ============================================================

# Arrays (output paths will be set after parsing)
ALL_TRAIN=("$TASK1_TRAIN" "$TASK2_TRAIN" "$TASK3_TRAIN" "$TASK4_TRAIN")
ALL_VAL=("$TASK1_VAL" "$TASK2_VAL" "$TASK3_VAL" "$TASK4_VAL")
ALL_GPUS=("0" "1" "2" "3")

# ============================================================
# AUTOMATIC PARSING - Do not edit
# ============================================================

# Parse MODEL_SHORT_NAME from MODEL_NAME
# Example: "Qwen/Qwen3-8B" -> "qwen3-8b" (take only model name after last /)
MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')

# Parse DATASET_NAME from task path
# Extract: data_dir/dataset_type from the path
# Example: ".../data_gpt-5-mini_balanced_new/sft_supervised_reasoning_traces/train/..." 
#          -> "gpt-5-mini-balanced-new-sft-supervised-reasoning-traces"
DATASET_PATH=$(dirname $(dirname "$TASK1_TRAIN"))  # Get parent of parent of train file
DATA_DIR=$(basename $(dirname "$DATASET_PATH"))
DATASET_TYPE=$(basename "$DATASET_PATH")
DATASET_NAME="${DATA_DIR}_${DATASET_TYPE}"
DATASET_NAME=$(echo "$DATASET_NAME" | sed 's/data_//' | tr '_' '-' | tr '/' '-')

# WandB Project Name: ehrshot-<model>-<dataset>-r<rank>
WANDB_PROJECT="ehrshot-${MODEL_SHORT_NAME}-${DATASET_NAME}-r${LORA_RANK}"

# Output directory based on parsed names
OUTPUT_BASE_DIR="${SCRIPT_DIR}/output_${MODEL_SHORT_NAME}_r${LORA_RANK}_$(basename $(dirname "$DATASET_PATH"))_$(basename "$DATASET_PATH")"

# Update output paths
TASK1_OUT="${OUTPUT_BASE_DIR}/acute_mi"
TASK2_OUT="${OUTPUT_BASE_DIR}/pancreatic_cancer"
TASK3_OUT="${OUTPUT_BASE_DIR}/hypertension"
TASK4_OUT="${OUTPUT_BASE_DIR}/hyperlipidemia"
ALL_OUT=("$TASK1_OUT" "$TASK2_OUT" "$TASK3_OUT" "$TASK4_OUT")

echo "Configuration:"
echo "  Model: $MODEL_NAME -> $MODEL_SHORT_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  LoRA Rank: $LORA_RANK"
echo "  WandB Project: $WANDB_PROJECT"
echo "  Output Dir: $OUTPUT_BASE_DIR"
echo ""

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
    
    # Launch in background and return immediately so parent can track the job
    (
        source $HOME/miniconda3/etc/profile.d/conda.sh
        conda activate $CONDA_ENV
        
        # Force this shell to only see one GPU
        export CUDA_VISIBLE_DEVICES=$gpu_id
        # --- WANDB CONFIGURATION ---
        export WANDB_PROJECT="$WANDB_PROJECT"
        # Name the run based on the task
        export WANDB_NAME="${task_name}"
        
        # Run python directly
        python "$SCRIPT_DIR/qwen_sft_pipeline_new.py" \
            --train_file "$train_file" \
            --eval_file "$eval_file" \
            --output_dir "$output_dir" \
            --model_name "$MODEL_NAME" \
            --lora_rank "$LORA_RANK"
    ) > "$log_file" 2>&1 &
}

# --- EXECUTION ---

echo "=========================================="
echo "Starting Parallel Training on 4 GPUs"
echo "=========================================="

pids=()

# Launch all 4 tasks
for i in "${!ALL_TRAIN[@]}"; do
    launch_task \
        "${ALL_TRAIN[$i]}" \
        "${ALL_VAL[$i]}" \
        "${ALL_OUT[$i]}" \
        "${ALL_GPUS[$i]}"
    pids+=($!)  # Capture PID of the background job
done

echo ""
echo "Tasks launched. PIDs: ${pids[*]}"
echo "Waiting for completion (this will take a while)..."
echo "Tip: You can safely close this terminal. Processes are running in background."
echo "To monitor: tail -f ${OUTPUT_BASE_DIR}/*.log"
echo ""

# Wait for all background jobs
wait

echo ""
echo "=========================================="
echo "All training tasks completed!"
echo "=========================================="