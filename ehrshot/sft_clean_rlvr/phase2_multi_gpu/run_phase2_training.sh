#!/bin/bash
# =============================================================================
# Phase 2: Multi-GPU GRPO Training with DDP
# =============================================================================
#
# PURPOSE:
#   Train GRPO model on aggregated data from all 4 clinical tasks using
#   Distributed Data Parallel (DDP) across all 4 GPUs.
#
# USAGE:
#   ./run_phase2_training.sh [SFT_CHECKPOINT] [NUM_GPUS] [MAX_STEPS]
#
# ARGUMENTS:
#   SFT_CHECKPOINT - Path to SFT LoRA checkpoint (aggregated model)
#                    Default: Auto-detected from sft_clean/finetuned_models/
#   NUM_GPUS       - Number of GPUs to use (1-4)
#                    Default: 4
#   MAX_STEPS      - Maximum training steps
#                    Default: 1000
#
# EXAMPLES:
#   # Full training with all defaults
#   ./run_phase2_training.sh
#
#   # Custom SFT checkpoint
#   ./run_phase2_training.sh /path/to/sft/checkpoint
#
#   # Use 2 GPUs with 500 steps
#   ./run_phase2_training.sh "" 2 500
#
# OUTPUTS:
#   - Model saved to: outputs/phase2_aggregated/
#   - Training stats: outputs/phase2_aggregated/training_stats.json
#   - Logs: outputs/phase2_aggregated.log
#
# RESOURCE REQUIREMENTS:
#   - Memory: ~50-70 GB per GPU
#   - Time: ~1-2 hours for 1000 steps
#
# =============================================================================

set -e  # Exit on error

# --- Parse Arguments ---
SFT_CHECKPOINT="${1:-}"
NUM_GPUS="${2:-4}"
MAX_STEPS="${3:-1000}"

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../outputs/phase2_aggregated"
LOG_FILE="$SCRIPT_DIR/../outputs/phase2_aggregated.log"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"

# GRPO Hyperparameters
NUM_GENERATIONS=10
LEARNING_RATE=5e-6
BETA=0.05
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=2

# Validate NUM_GPUS
if [[ ! "$NUM_GPUS" =~ ^[1-4]$ ]]; then
    echo "ERROR: NUM_GPUS must be 1-4, got: $NUM_GPUS"
    exit 1
fi

# Auto-detect SFT checkpoint if not provided
if [ -z "$SFT_CHECKPOINT" ]; then
    SFT_BASE="$PROJECT_DIR/sft_clean/finetuned_models"
    
    # Look for aggregated models (preferred for Phase 2)
    if [ -d "$SFT_BASE/reasoning_originalEHR_train_orig_aggregated" ]; then
        SFT_CHECKPOINT="$SFT_BASE/reasoning_originalEHR_train_orig_aggregated"
    elif [ -d "$SFT_BASE/reasoning_originalEHR_train_all_aggregated" ]; then
        SFT_CHECKPOINT="$SFT_BASE/reasoning_originalEHR_train_all_aggregated"
    else
        echo "ERROR: Could not auto-detect aggregated SFT checkpoint"
        echo "Please provide path as first argument"
        echo "Checked: $SFT_BASE/"
        exit 1
    fi
fi

# Verify SFT checkpoint exists
if [ ! -d "$SFT_CHECKPOINT" ]; then
    echo "ERROR: SFT checkpoint not found: $SFT_CHECKPOINT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# --- Print Configuration ---
echo "=============================================="
echo "Phase 2: Multi-GPU GRPO Training (DDP)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  SFT Checkpoint: $SFT_CHECKPOINT"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Log File:       $LOG_FILE"
echo "  Num GPUs:       $NUM_GPUS"
echo ""
echo "GRPO Settings:"
echo "  Num Generations:     $NUM_GENERATIONS"
echo "  Max Steps:           $MAX_STEPS"
echo "  Learning Rate:       $LEARNING_RATE"
echo "  Beta (KL):           $BETA"
echo "  Per-Device Batch:    $PER_DEVICE_BATCH_SIZE"
echo "  Gradient Accum:      $GRADIENT_ACCUMULATION_STEPS"
echo ""
echo "Effective Batch Size:  $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo "Completions per Step:  $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS * NUM_GENERATIONS))"
echo ""

# --- Thread Limiting ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Set visible GPUs ---
# For 4 GPUs: CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES=$GPU_LIST

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# --- Run Training ---
echo "Starting DDP training..."
echo "Logs will be written to: $LOG_FILE"
echo ""
echo "To monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

(
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        "$SCRIPT_DIR/train_grpo_ddp.py" \
        --sft_checkpoint "$SFT_CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --num_generations $NUM_GENERATIONS \
        --max_steps $MAX_STEPS \
        --learning_rate $LEARNING_RATE \
        --beta $BETA \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --wandb_project "ehrshot-rlvr-phase2"
        
) 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "Phase 2 Training Complete!"
echo "=============================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo "Training stats: $OUTPUT_DIR/training_stats.json"
echo ""
echo "Next steps:"
echo "  1. Evaluate RLVR model on all 4 tasks"
echo "  2. Compare AUROC/AUPRC with SFT baseline"
echo ""
echo "Evaluation command:"
echo "  cd ../evaluation"
echo "  ./run_eval_rlvr.sh $OUTPUT_DIR"

