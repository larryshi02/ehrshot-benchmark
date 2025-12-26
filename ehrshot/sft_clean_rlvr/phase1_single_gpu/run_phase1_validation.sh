#!/bin/bash
# =============================================================================
# Phase 1: Single-GPU GRPO Validation
# =============================================================================
#
# PURPOSE:
#   Validate the RLVR pipeline on a single task before scaling to multi-GPU.
#   This catches configuration issues early without wasting 4 GPUs.
#
# USAGE:
#   ./run_phase1_validation.sh [TASK] [GPU_ID] [SFT_CHECKPOINT]
#
# ARGUMENTS:
#   TASK           - One of: acute_mi, hyperlipidemia, hypertension, pancreatic_cancer
#                    Default: acute_mi
#   GPU_ID         - GPU to use (0-3)
#                    Default: 0
#   SFT_CHECKPOINT - Path to SFT LoRA checkpoint
#                    Default: Auto-detected from sft_clean/finetuned_models/
#
# EXAMPLES:
#   # Quick validation with defaults (acute_mi on GPU 0)
#   ./run_phase1_validation.sh
#
#   # Specific task on specific GPU
#   ./run_phase1_validation.sh hypertension 2
#
#   # With custom SFT checkpoint
#   ./run_phase1_validation.sh acute_mi 0 /path/to/sft/checkpoint
#
# OUTPUTS:
#   - Model saved to: outputs/phase1_validation/{task}/
#   - Training stats: outputs/phase1_validation/{task}/training_stats.json
#   - Logs: outputs/phase1_validation/{task}.log
#
# =============================================================================

set -e  # Exit on error

# --- Parse Arguments ---
TASK="${1:-acute_mi}"
GPU_ID="${2:-0}"
SFT_CHECKPOINT="${3:-}"

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_BASE_DIR="$SCRIPT_DIR/../outputs/phase1_validation"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"

# Validate task
VALID_TASKS=("acute_mi" "hyperlipidemia" "hypertension" "pancreatic_cancer")
if [[ ! " ${VALID_TASKS[*]} " =~ " ${TASK} " ]]; then
    echo "ERROR: Invalid task '$TASK'"
    echo "Valid tasks: ${VALID_TASKS[*]}"
    exit 1
fi

# Auto-detect SFT checkpoint if not provided
if [ -z "$SFT_CHECKPOINT" ]; then
    # Try to find SFT checkpoint from sft_clean experiments
    SFT_BASE="$PROJECT_DIR/sft_clean/finetuned_models"
    
    # Check for aggregated model first (trained on all tasks)
    if [ -d "$SFT_BASE/reasoning_originalEHR_train_orig_aggregated" ]; then
        SFT_CHECKPOINT="$SFT_BASE/reasoning_originalEHR_train_orig_aggregated"
    # Then check for per-task model
    elif [ -d "$SFT_BASE/reasoning_originalEHR_train_orig/$TASK" ]; then
        SFT_CHECKPOINT="$SFT_BASE/reasoning_originalEHR_train_orig/$TASK"
    # Try train_all variants
    elif [ -d "$SFT_BASE/reasoning_originalEHR_train_all_aggregated" ]; then
        SFT_CHECKPOINT="$SFT_BASE/reasoning_originalEHR_train_all_aggregated"
    elif [ -d "$SFT_BASE/reasoning_originalEHR_train_all/$TASK" ]; then
        SFT_CHECKPOINT="$SFT_BASE/reasoning_originalEHR_train_all/$TASK"
    else
        echo "ERROR: Could not auto-detect SFT checkpoint"
        echo "Please provide path as third argument"
        echo "Checked: $SFT_BASE/"
        exit 1
    fi
fi

# Verify SFT checkpoint exists
if [ ! -d "$SFT_CHECKPOINT" ]; then
    echo "ERROR: SFT checkpoint not found: $SFT_CHECKPOINT"
    exit 1
fi

# --- Output paths ---
OUTPUT_DIR="$OUTPUT_BASE_DIR/$TASK"
LOG_FILE="$OUTPUT_BASE_DIR/${TASK}.log"

mkdir -p "$OUTPUT_DIR"

# --- Print Configuration ---
echo "=============================================="
echo "Phase 1: Single-GPU GRPO Validation"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Task:           $TASK"
echo "  GPU:            $GPU_ID"
echo "  SFT Checkpoint: $SFT_CHECKPOINT"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Log File:       $LOG_FILE"
echo ""
echo "GRPO Settings:"
echo "  Num Generations: 10"
echo "  Max Steps:       500"
echo "  Learning Rate:   5e-6"
echo "  Beta (KL):       0.05"
echo ""

# --- Thread Limiting ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1

# --- Run Training ---
echo "Starting training..."
echo "Logs will be written to: $LOG_FILE"
echo ""

(
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    python "$SCRIPT_DIR/train_grpo_single.py" \
        --task "$TASK" \
        --sft_checkpoint "$SFT_CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --num_generations 10 \
        --max_steps 500 \
        --learning_rate 5e-6 \
        --beta 0.05 \
        --wandb_project "ehrshot-rlvr-phase1"
        
) 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "Phase 1 Validation Complete!"
echo "=============================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo "Training stats: $OUTPUT_DIR/training_stats.json"
echo ""
echo "Next steps:"
echo "  1. Check training_stats.json for accuracy during training"
echo "  2. Run evaluation to compare with SFT baseline"
echo "  3. If successful, proceed to Phase 2 (multi-GPU training)"

