#!/bin/bash
# SFT Training Pipeline - Parallel Execution
# Runs 4 independent training tasks on GPUs 0, 1, 2, 3

# Limit the main process (Matrix Math) to 8 threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Limit the data loader workers to 1 thread each (prevents explosion)
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"
PYTHON_SCRIPT="qwen_sft_unsloth.py"  # Name of your python script

# ============================================================
# USER CONFIGURATION - Edit these values only
# ============================================================

# --- MODEL CONFIGURATION ---
MODEL_NAME="Qwen/Qwen3-8B"

# --- LORA CONFIGURATION ---
LORA_RANK=16

# --- DATASET CONFIGURATION ---
# Data is stored in /dev/shm/ehrshot-data for faster I/O
DATA_DIR="/dev/shm/ehrshot-data/data_gpt-5-mini/sft_rt_yn/"

# 1. Acute MI
TASK1_TRAIN="${DATA_DIR}train_all/acute_mi_sft_dataset.json"
TASK1_VAL="${DATA_DIR}val_small/acute_mi_sft_dataset.json"

# 2. Pancreatic Cancer
TASK2_TRAIN="${DATA_DIR}train_all/pancreatic_cancer_sft_dataset.json"
TASK2_VAL="${DATA_DIR}val_small/pancreatic_cancer_sft_dataset.json"

# 3. Hypertension
TASK3_TRAIN="${DATA_DIR}train_all/hypertension_sft_dataset.json"
TASK3_VAL="${DATA_DIR}val_small/hypertension_sft_dataset.json"

# 4. Hyperlipidemia
TASK4_TRAIN="${DATA_DIR}train_all/hyperlipidemia_sft_dataset.json"
TASK4_VAL="${DATA_DIR}val_small/hyperlipidemia_sft_dataset.json"

# ============================================================
# END OF USER CONFIGURATION
# ============================================================

# ============================================================
# AUTOMATIC PARSING & NAMING LOGIC (before path changes)
# ============================================================

# Parse MODEL_SHORT_NAME from MODEL_NAME
# Example: "Qwen/Qwen3-8B" -> "qwen3-8b"
MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')

# Parse DATASET_NAME from task path (use original path before /dev/shm copy)
ORIG_DATASET_PATH=$(dirname $(dirname "$TASK1_TRAIN"))
DATA_DIR_NAME=$(basename $(dirname "$ORIG_DATASET_PATH"))
DATASET_TYPE=$(basename "$ORIG_DATASET_PATH")
DATASET_NAME="${DATA_DIR_NAME}_${DATASET_TYPE}"
DATASET_NAME=$(echo "$DATASET_NAME" | sed 's/data_//' | tr '_' '-' | tr '/' '-')

# --- UPDATE: Added "unsloth-" prefix ---
WANDB_PROJECT="epochs3-unsloth-ehrshot-${MODEL_SHORT_NAME}-${DATASET_NAME}-r${LORA_RANK}"

# --- UPDATE: Added "unsloth_" prefix to output directory ---
OUTPUT_BASE_DIR="${SCRIPT_DIR}/epochs3_unsloth_output_${MODEL_SHORT_NAME}_r${LORA_RANK}_${DATA_DIR_NAME}_${DATASET_TYPE}"

# Update output paths for specific tasks
TASK1_OUT="${OUTPUT_BASE_DIR}/acute_mi"
TASK2_OUT="${OUTPUT_BASE_DIR}/pancreatic_cancer"
TASK3_OUT="${OUTPUT_BASE_DIR}/hypertension"
TASK4_OUT="${OUTPUT_BASE_DIR}/hyperlipidemia"

# --- OPTION A: RUN ALL (Comment this OUT for test) ---

ALL_OUT=("$TASK1_OUT" "$TASK2_OUT" "$TASK3_OUT" "$TASK4_OUT")
ALL_TRAIN=("$TASK1_TRAIN" "$TASK2_TRAIN" "$TASK3_TRAIN" "$TASK4_TRAIN")
ALL_VAL=("$TASK1_VAL" "$TASK2_VAL" "$TASK3_VAL" "$TASK4_VAL")
ALL_GPUS=("0" "1" "2" "3")


# --- OPTION B: TEST HYPERTENSION ONLY (Comment this IN for test) ---

# ALL_OUT=("$TASK3_OUT")
# ALL_TRAIN=("$TASK3_TRAIN")
# ALL_VAL=("$TASK3_VAL")
# ALL_GPUS=("3")


echo "Configuration:"
echo "  Model: $MODEL_NAME -> $MODEL_SHORT_NAME"
echo "  Python Script: $PYTHON_SCRIPT"
echo "  WandB Project: $WANDB_PROJECT"
echo "  Output Base Dir: $OUTPUT_BASE_DIR"
echo "  Data Location: $DATA_DIR (RAM - /dev/shm)"
echo ""

mkdir -p "$OUTPUT_BASE_DIR"

# --- LAUNCH FUNCTION ---
launch_task() {
    local train_file=$1
    local eval_file=$2
    local output_dir=$3
    local gpu_id=$4
    
    local task_name=$(basename "$train_file" | sed 's/_sft_dataset.json//')
    local log_file="$OUTPUT_BASE_DIR/${task_name}.log"
    
    mkdir -p "$output_dir"
    
    echo "  [GPU $gpu_id] Launching $task_name..."
    
    # Launch in background subshell
    (
        source $HOME/miniconda3/etc/profile.d/conda.sh
        conda activate $CONDA_ENV
        
        # Force this process to see only ONE specific GPU
        export CUDA_VISIBLE_DEVICES=$gpu_id
        
        # Run python script
        python "$SCRIPT_DIR/$PYTHON_SCRIPT" \
            --train_file "$train_file" \
            --eval_file "$eval_file" \
            --output_dir "$output_dir" \
            --model_name "$MODEL_NAME" \
            --lora_rank "$LORA_RANK" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "${task_name}"
    ) > "$log_file" 2>&1 &
}

# --- EXECUTION ---

echo "=========================================="
echo "Starting Parallel Training on 4 GPUs (Unsloth Optimized)"
echo "=========================================="

pids=()

# Launch all 4 tasks
for i in "${!ALL_TRAIN[@]}"; do
    launch_task \
        "${ALL_TRAIN[$i]}" \
        "${ALL_VAL[$i]}" \
        "${ALL_OUT[$i]}" \
        "${ALL_GPUS[$i]}"
    pids+=($!)  # Capture PID
done

echo ""
echo "Tasks launched. PIDs: ${pids[*]}"
echo "Output and logs are saving to: $OUTPUT_BASE_DIR"
echo "To monitor: tail -f ${OUTPUT_BASE_DIR}/*.log"
echo ""

# Wait for all background jobs to finish
wait

echo ""
echo "=========================================="
echo "All Unsloth training tasks completed!"
echo "=========================================="