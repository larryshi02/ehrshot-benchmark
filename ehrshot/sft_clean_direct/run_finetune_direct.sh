#!/bin/bash
# =============================================================================
# Modular Direct Y/N SFT Training Pipeline
# =============================================================================
# This script trains 4 task-specific models for direct binary classification
# (Positive/Negative) without reasoning.
#
# Usage:
#   ./run_finetune_direct.sh original              # Uses train_all (default)
#   ./run_finetune_direct.sh original train_all    # Uses train_all explicitly
#   ./run_finetune_direct.sh original train_orig   # Uses train_orig (smaller set)
#   ./run_finetune_direct.sh rubricified train_orig
#
# =============================================================================

set -e  # Exit on error

# --- Thread Limiting (prevents CPU oversubscription) ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"
PYTHON_SCRIPT="qwen_sft_direct.py"

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

DATA_TYPE="${1:-original}"      # Default to original if not specified
TRAIN_SPLIT="${2:-train_all}"   # Default to train_all if not specified

if [[ "$DATA_TYPE" != "original" && "$DATA_TYPE" != "rubricified" ]]; then
    echo "ERROR: Invalid DATA_TYPE '$DATA_TYPE'. Must be 'original' or 'rubricified'."
    echo "Usage: $0 [original|rubricified] [train_all|train_orig]"
    exit 1
fi

if [[ "$TRAIN_SPLIT" != "train_all" && "$TRAIN_SPLIT" != "train_orig" ]]; then
    echo "ERROR: Invalid TRAIN_SPLIT '$TRAIN_SPLIT'. Must be 'train_all' or 'train_orig'."
    echo "Usage: $0 [original|rubricified] [train_all|train_orig]"
    exit 1
fi

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_NAME="Qwen/Qwen3-8B"
LORA_RANK=16

# =============================================================================
# DATA CONFIGURATION - Based on DATA_TYPE
# =============================================================================

if [[ "$DATA_TYPE" == "original" ]]; then
    DATA_DIR="/dev/shm/ehrshot-data/data_gpt-5-mini_sft_direct_originalEHR"
    DATA_SUFFIX="originalEHR"
else
    DATA_DIR="/dev/shm/ehrshot-data/data_gpt-5-mini_sft_direct_rubricified"
    DATA_SUFFIX="rubricified"
fi

# Tasks
TASKS=("acute_mi" "pancreatic_cancer" "hypertension" "hyperlipidemia")
GPUS=("0" "1" "2" "3")

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')

# WandB project: One project per data type + train split
# Format: ehrshot-direct-{datatype}-{split}-{model}
WANDB_PROJECT="ehrshot-direct-${DATA_SUFFIX}-${TRAIN_SPLIT}-${MODEL_SHORT_NAME}"

# Output directory under sft_clean_direct/finetuned_models/{data_type}_{split}
OUTPUT_BASE_DIR="${SCRIPT_DIR}/finetuned_models/${DATA_SUFFIX}_${TRAIN_SPLIT}"

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "=============================================="
echo "Direct Y/N SFT Training Pipeline"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data Type: $DATA_TYPE"
echo "  Train Split: $TRAIN_SPLIT"
echo "  Data Directory: $DATA_DIR"
echo "  Model: $MODEL_NAME"
echo "  LoRA Rank: $LORA_RANK"
echo "  WandB Project: $WANDB_PROJECT"
echo "  Output Base Dir: $OUTPUT_BASE_DIR"
echo ""
echo "Tasks: ${TASKS[*]}"
echo "GPUs: ${GPUS[*]}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# =============================================================================
# VALIDATION
# =============================================================================

echo "Validating data files..."
for task in "${TASKS[@]}"; do
    train_file="${DATA_DIR}/${TRAIN_SPLIT}/${task}_sft_dataset.json"
    val_file="${DATA_DIR}/val_small/${task}_sft_dataset.json"
    
    if [ ! -f "$train_file" ]; then
        echo "ERROR: Training file not found: $train_file"
        exit 1
    fi
    if [ ! -f "$val_file" ]; then
        echo "ERROR: Validation file not found: $val_file"
        exit 1
    fi
    echo "  ✓ Found data for: $task"
done
echo ""

# =============================================================================
# LAUNCH FUNCTION
# =============================================================================

launch_task() {
    local task_name=$1
    local gpu_id=$2
    
    local train_file="${DATA_DIR}/${TRAIN_SPLIT}/${task_name}_sft_dataset.json"
    local eval_file="${DATA_DIR}/val_small/${task_name}_sft_dataset.json"
    local output_dir="${OUTPUT_BASE_DIR}/${task_name}"
    local log_file="${OUTPUT_BASE_DIR}/${task_name}.log"
    
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

# =============================================================================
# EXECUTION
# =============================================================================

echo "=============================================="
echo "Starting Parallel Training on ${#GPUS[@]} GPUs"
echo "=============================================="

pids=()

# Launch all tasks
for i in "${!TASKS[@]}"; do
    launch_task "${TASKS[$i]}" "${GPUS[$i]}"
    pids+=($!)
done

echo ""
echo "Tasks launched. PIDs: ${pids[*]}"
echo "Output and logs are saving to: $OUTPUT_BASE_DIR"
echo ""
echo "To monitor progress:"
echo "  tail -f ${OUTPUT_BASE_DIR}/*.log"
echo ""
echo "To check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

# Wait for all background jobs to finish
wait

echo ""
echo "=============================================="
echo "All training tasks completed!"
echo "=============================================="
echo ""
echo "Models saved to:"
for task in "${TASKS[@]}"; do
    echo "  $OUTPUT_BASE_DIR/$task/"
done
echo ""
echo "Next step - Evaluate finetuned models:"
echo "  python orchestrator_eval.py --mode finetuned --data_type $DATA_TYPE \\"
echo "    --lora_base_dir $OUTPUT_BASE_DIR \\"
echo "    --output_dir ${SCRIPT_DIR}/eval_results/${DATA_SUFFIX}_${TRAIN_SPLIT}_finetuned"
