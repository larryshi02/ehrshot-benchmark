#!/bin/bash
# =============================================================================
# Direct Y/N SFT Training Pipeline - Parallel Execution
# =============================================================================
# This script trains 4 task-specific models for direct binary classification
# (Positive/Negative) without reasoning.
#
# Key differences from reasoning-based SFT:
# 1. Uses simplified prompts (no <think> tag instructions)
# 2. Trains only on single-token responses ("Positive" or "Negative")
# 3. Thinking mode is disabled in the training script
#
# Runs 4 independent training tasks on GPUs 0, 1, 2, 3
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
# USER CONFIGURATION - Edit these values as needed
# =============================================================================

# --- MODEL CONFIGURATION ---
MODEL_NAME="Qwen/Qwen3-8B"

# --- LORA CONFIGURATION ---
LORA_RANK=16

# --- DATASET CONFIGURATION ---
# Direct Y/N data (no reasoning)
DATA_DIR="${SCRIPT_DIR}/data"

# Training data
TASK1_TRAIN="${DATA_DIR}/train_all/acute_mi_sft_dataset.json"
TASK2_TRAIN="${DATA_DIR}/train_all/pancreatic_cancer_sft_dataset.json"
TASK3_TRAIN="${DATA_DIR}/train_all/hypertension_sft_dataset.json"
TASK4_TRAIN="${DATA_DIR}/train_all/hyperlipidemia_sft_dataset.json"

# Validation data
TASK1_VAL="${DATA_DIR}/val_small/acute_mi_balanced_sft_dataset.json"
TASK2_VAL="${DATA_DIR}/val_small/pancreatic_cancer_balanced_sft_dataset.json"
TASK3_VAL="${DATA_DIR}/val_small/hypertension_balanced_sft_dataset.json"
TASK4_VAL="${DATA_DIR}/val_small/hyperlipidemia_balanced_sft_dataset.json"

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================

# --- AUTOMATIC NAMING LOGIC ---
MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')

# WandB project name with "direct-yn" identifier
WANDB_PROJECT="epochs3-direct-yn-ehrshot-${MODEL_SHORT_NAME}-r${LORA_RANK}"

# Output directory with "direct_yn" identifier
OUTPUT_BASE_DIR="${SCRIPT_DIR}/output_direct_yn_${MODEL_SHORT_NAME}_r${LORA_RANK}"

# Task-specific output directories
TASK1_OUT="${OUTPUT_BASE_DIR}/acute_mi"
TASK2_OUT="${OUTPUT_BASE_DIR}/pancreatic_cancer"
TASK3_OUT="${OUTPUT_BASE_DIR}/hypertension"
TASK4_OUT="${OUTPUT_BASE_DIR}/hyperlipidemia"

# --- OPTION A: RUN ALL 4 TASKS IN PARALLEL ---
ALL_OUT=("$TASK1_OUT" "$TASK2_OUT" "$TASK3_OUT" "$TASK4_OUT")
ALL_TRAIN=("$TASK1_TRAIN" "$TASK2_TRAIN" "$TASK3_TRAIN" "$TASK4_TRAIN")
ALL_VAL=("$TASK1_VAL" "$TASK2_VAL" "$TASK3_VAL" "$TASK4_VAL")
ALL_GPUS=("0" "1" "2" "3")

# --- OPTION B: RUN SINGLE TASK FOR TESTING (uncomment to use) ---
# ALL_OUT=("$TASK3_OUT")
# ALL_TRAIN=("$TASK3_TRAIN")
# ALL_VAL=("$TASK3_VAL")
# ALL_GPUS=("0")

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "=============================================="
echo "Direct Y/N SFT Training Pipeline"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME -> $MODEL_SHORT_NAME"
echo "  LoRA Rank: $LORA_RANK"
echo "  Python Script: $PYTHON_SCRIPT"
echo "  WandB Project: $WANDB_PROJECT"
echo "  Output Base Dir: $OUTPUT_BASE_DIR"
echo "  Data Directory: $DATA_DIR"
echo ""
echo "Training Mode: Direct Y/N (no reasoning)"
echo "  - Simplified system prompts"
echo "  - Single-token responses (Positive/Negative)"
echo "  - Thinking mode disabled"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# =============================================================================
# LAUNCH FUNCTION
# =============================================================================

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

# =============================================================================
# VALIDATION
# =============================================================================

echo "Validating data files..."
for i in "${!ALL_TRAIN[@]}"; do
    if [ ! -f "${ALL_TRAIN[$i]}" ]; then
        echo "ERROR: Training file not found: ${ALL_TRAIN[$i]}"
        echo "Please run 'python generate_direct_yn_data.py' first."
        exit 1
    fi
    if [ ! -f "${ALL_VAL[$i]}" ]; then
        echo "ERROR: Validation file not found: ${ALL_VAL[$i]}"
        echo "Please run 'python generate_direct_yn_data.py' first."
        exit 1
    fi
done
echo "All data files found."
echo ""

# =============================================================================
# EXECUTION
# =============================================================================

echo "=============================================="
echo "Starting Parallel Training on ${#ALL_GPUS[@]} GPUs"
echo "=============================================="

pids=()

# Launch all tasks
for i in "${!ALL_TRAIN[@]}"; do
    launch_task \
        "${ALL_TRAIN[$i]}" \
        "${ALL_VAL[$i]}" \
        "${ALL_OUT[$i]}" \
        "${ALL_GPUS[$i]}"
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
echo "All Direct Y/N training tasks completed!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Evaluate models using eval_vllm_direct.py"
echo "2. Compute metrics using eval_vllm_compute_metrics.py"
