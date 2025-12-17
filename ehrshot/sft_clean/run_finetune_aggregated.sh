#!/bin/bash
# =============================================================================
# Step 2: Fine-tune Single Aggregated Model on All 4 Tasks (Reasoning)
# =============================================================================
# Trains a single Qwen3-8B model on data aggregated from all 4 clinical
# prediction tasks. Uses a single GPU (simplest approach).
#
# Why single GPU?
#   - Qwen3-8B with LoRA (r=16) fits comfortably on one A100 80GB
#   - Data parallelism would require model copies on each GPU
#   - For LoRA fine-tuning, single GPU is sufficient and simpler
#   - Multi-GPU training (DeepSpeed/FSDP) adds complexity without significant benefit
#
# Usage:
#   ./run_finetune_aggregated.sh                           # originalEHR + train_orig (default)
#   ./run_finetune_aggregated.sh originalEHR train_orig    # Explicit
#   ./run_finetune_aggregated.sh rubricified train_orig    # Rubricified data
#   ./run_finetune_aggregated.sh originalEHR train_all     # Use train_all split
#
# =============================================================================

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipeline fails

# --- Thread Limiting ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

DATA_REPR="${1:-originalEHR}"     # Default: originalEHR
TRAIN_SPLIT="${2:-train_orig}"    # Default: train_orig
GPU_ID="${3:-0}"                  # Default: GPU 0

if [[ "$DATA_REPR" != "originalEHR" && "$DATA_REPR" != "rubricified" ]]; then
    echo "ERROR: Invalid DATA_REPR '$DATA_REPR'. Must be 'originalEHR' or 'rubricified'."
    echo "Usage: $0 [originalEHR|rubricified] [train_orig|train_all] [gpu_id]"
    exit 1
fi

if [[ "$TRAIN_SPLIT" != "train_orig" && "$TRAIN_SPLIT" != "train_all" ]]; then
    echo "ERROR: Invalid TRAIN_SPLIT '$TRAIN_SPLIT'. Must be 'train_orig' or 'train_all'."
    echo "Usage: $0 [originalEHR|rubricified] [train_orig|train_all] [gpu_id]"
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

# Merged dataset file path
MERGED_TRAIN_FILE="${DATA_DIR}/${TRAIN_SPLIT}/merged_sft_dataset.json"
MERGED_VAL_FILE="${DATA_DIR}/val_small/merged_sft_dataset.json"

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')

WANDB_PROJECT="ehrshot-reasoning-${DATA_REPR}-${TRAIN_SPLIT}-aggregated-${MODEL_SHORT_NAME}"
WANDB_RUN_NAME="aggregated_all_tasks"

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/finetuned_models/reasoning_${DATA_REPR}_${TRAIN_SPLIT}_aggregated"

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "=============================================="
echo "Step 2: Aggregated Fine-tuning (Reasoning)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data Representation: $DATA_REPR"
echo "  Train Split: $TRAIN_SPLIT"
echo "  Data Directory: $DATA_DIR"
echo "  Model: $MODEL_NAME"
echo "  LoRA Rank: $LORA_RANK"
echo "  GPU: $GPU_ID"
echo "  WandB Project: $WANDB_PROJECT"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# =============================================================================
# CREATE MERGED DATASET IF NEEDED
# =============================================================================

echo "Checking for merged dataset..."

if [ ! -f "$MERGED_TRAIN_FILE" ]; then
    echo "  Merged training dataset not found. Creating..."
    
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    
    python "$SCRIPT_DIR/merge_datasets.py" \
        --data_dir "$DATA_DIR" \
        --split "$TRAIN_SPLIT" \
        --output_file "$MERGED_TRAIN_FILE"
    
    echo "  ✓ Created: $MERGED_TRAIN_FILE"
else
    echo "  ✓ Found existing: $MERGED_TRAIN_FILE"
fi

if [ ! -f "$MERGED_VAL_FILE" ]; then
    echo "  Merged validation dataset not found. Creating..."
    
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    
    python "$SCRIPT_DIR/merge_datasets.py" \
        --data_dir "$DATA_DIR" \
        --split "val_small" \
        --output_file "$MERGED_VAL_FILE"
    
    echo "  ✓ Created: $MERGED_VAL_FILE"
else
    echo "  ✓ Found existing: $MERGED_VAL_FILE"
fi

echo ""

# =============================================================================
# VALIDATION
# =============================================================================

if [ ! -f "$MERGED_TRAIN_FILE" ]; then
    echo "ERROR: Training file not found: $MERGED_TRAIN_FILE"
    exit 1
fi

if [ ! -f "$MERGED_VAL_FILE" ]; then
    echo "ERROR: Validation file not found: $MERGED_VAL_FILE"
    exit 1
fi

# Show dataset stats
TRAIN_SIZE=$(python3 -c "import json; print(len(json.load(open('$MERGED_TRAIN_FILE'))))")
VAL_SIZE=$(python3 -c "import json; print(len(json.load(open('$MERGED_VAL_FILE'))))")

echo "Dataset Statistics:"
echo "  Training examples: $TRAIN_SIZE"
echo "  Validation examples: $VAL_SIZE"
echo ""

# =============================================================================
# EXECUTION
# =============================================================================

echo "=============================================="
echo "Starting Aggregated Training on GPU $GPU_ID"
echo "=============================================="

LOG_FILE="${OUTPUT_DIR}.log"
mkdir -p "$OUTPUT_DIR"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run with specific GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

python "$SCRIPT_DIR/qwen_sft_reasoning.py" \
    --train_file "$MERGED_TRAIN_FILE" \
    --eval_file "$MERGED_VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --lora_rank "$LORA_RANK" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "Step 2: Aggregated Fine-tuning Complete!"
echo "=============================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Next step - Evaluate aggregated model on all tasks:"
echo "  python orchestrator_eval.py --mode aggregated \\"
echo "    --data_repr $DATA_REPR \\"
echo "    --lora_path $OUTPUT_DIR \\"
echo "    --output_dir ${SCRIPT_DIR}/eval_results/reasoning_${DATA_REPR}_${TRAIN_SPLIT}_aggregated"
