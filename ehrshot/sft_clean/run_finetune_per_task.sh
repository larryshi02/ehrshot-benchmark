#!/bin/bash
# =============================================================================
# Step 4: Fine-tune 4 Task-Specific Models in Parallel (Reasoning)
# =============================================================================
# Trains 4 separate Qwen3-8B models, one per clinical prediction task,
# each on a separate GPU.
#
# Usage:
#   ./run_finetune_per_task.sh                           # originalEHR + train_orig (default)
#   ./run_finetune_per_task.sh originalEHR train_orig    # Explicit
#   ./run_finetune_per_task.sh rubricified train_orig    # Rubricified data
#   ./run_finetune_per_task.sh originalEHR train_all     # Use train_all split
#
# =============================================================================

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipeline fails

# --- Thread Limiting (prevents CPU oversubscription) ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"
PYTHON_SCRIPT="qwen_sft_reasoning.py"

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

DATA_REPR="${1:-originalEHR}"     # Default: originalEHR
TRAIN_SPLIT="${2:-train_orig}"    # Default: train_orig

if [[ "$DATA_REPR" != "originalEHR" && "$DATA_REPR" != "rubricified" ]]; then
    echo "ERROR: Invalid DATA_REPR '$DATA_REPR'. Must be 'originalEHR' or 'rubricified'."
    echo "Usage: $0 [originalEHR|rubricified] [train_orig|train_all]"
    exit 1
fi

if [[ "$TRAIN_SPLIT" != "train_orig" && "$TRAIN_SPLIT" != "train_all" ]]; then
    echo "ERROR: Invalid TRAIN_SPLIT '$TRAIN_SPLIT'. Must be 'train_orig' or 'train_all'."
    echo "Usage: $0 [originalEHR|rubricified] [train_orig|train_all]"
    exit 1
fi

# =============================================================================
# MODEL & DATA CONFIGURATION
# =============================================================================

MODEL_NAME="Qwen/Qwen3-8B"
LORA_RANK=16

# Set data directory based on representation type
if [[ "$DATA_REPR" == "originalEHR" ]]; then
    DATA_DIR="/dev/shm/ehrshot-data/data_gpt-5-mini_sft_reasoning_originalEHR"
else
    DATA_DIR="/dev/shm/ehrshot-data/data_gpt-5-mini_sft_reasoning_rubricified"
fi

# Tasks and GPUs
TASKS=("acute_mi" "pancreatic_cancer" "hypertension" "hyperlipidemia")
GPUS=("0" "1" "2" "3")

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')

# WandB project name
WANDB_PROJECT="ehrshot-reasoning-${DATA_REPR}-${TRAIN_SPLIT}-${MODEL_SHORT_NAME}"

# Output directory: sft_clean/finetuned_models/reasoning_{repr}_{split}/
OUTPUT_BASE_DIR="${SCRIPT_DIR}/finetuned_models/reasoning_${DATA_REPR}_${TRAIN_SPLIT}"

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "=============================================="
echo "Step 4: Per-Task Fine-tuning (Reasoning)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data Representation: $DATA_REPR"
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
echo "Step 4: Per-Task Fine-tuning Complete!"
echo "=============================================="
echo ""
echo "Models saved to:"
for task in "${TASKS[@]}"; do
    echo "  $OUTPUT_BASE_DIR/$task/"
done
echo ""
echo "Next step - Evaluate on diagonal (each model on its trained task):"
echo "  python orchestrator_eval.py --mode per_task_diagonal \\"
echo "    --data_repr $DATA_REPR \\"
echo "    --lora_base_dir $OUTPUT_BASE_DIR \\"
echo "    --output_dir ${SCRIPT_DIR}/eval_results/reasoning_${DATA_REPR}_${TRAIN_SPLIT}_per_task"
