#!/bin/bash
# =============================================================================
# Direct Y/N Evaluation Script
# =============================================================================
# This script evaluates models for direct binary classification on test data.
#
# Usage:
#   ./run_eval_direct.sh base original                  # Evaluate base model on original EHR
#   ./run_eval_direct.sh base rubricified               # Evaluate base model on rubricified EHR
#   ./run_eval_direct.sh finetuned original             # Finetuned on train_all (default)
#   ./run_eval_direct.sh finetuned original train_all   # Finetuned on train_all explicitly
#   ./run_eval_direct.sh finetuned original train_orig  # Finetuned on train_orig
#   ./run_eval_direct.sh finetuned rubricified train_orig
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

MODE="${1:-base}"           # base or finetuned
DATA_TYPE="${2:-original}"  # original or rubricified
TRAIN_SPLIT="${3:-train_all}"  # train_all or train_orig (only for finetuned mode)

if [[ "$MODE" != "base" && "$MODE" != "finetuned" ]]; then
    echo "ERROR: Invalid MODE '$MODE'. Must be 'base' or 'finetuned'."
    echo "Usage: $0 [base|finetuned] [original|rubricified] [train_all|train_orig]"
    exit 1
fi

if [[ "$DATA_TYPE" != "original" && "$DATA_TYPE" != "rubricified" ]]; then
    echo "ERROR: Invalid DATA_TYPE '$DATA_TYPE'. Must be 'original' or 'rubricified'."
    echo "Usage: $0 [base|finetuned] [original|rubricified] [train_all|train_orig]"
    exit 1
fi

if [[ "$TRAIN_SPLIT" != "train_all" && "$TRAIN_SPLIT" != "train_orig" ]]; then
    echo "ERROR: Invalid TRAIN_SPLIT '$TRAIN_SPLIT'. Must be 'train_all' or 'train_orig'."
    echo "Usage: $0 [base|finetuned] [original|rubricified] [train_all|train_orig]"
    exit 1
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data suffix for output directories
if [[ "$DATA_TYPE" == "original" ]]; then
    DATA_SUFFIX="originalEHR"
else
    DATA_SUFFIX="rubricified"
fi

# Set output directory based on mode, data type, and train split
if [[ "$MODE" == "base" ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/eval_results/${DATA_SUFFIX}_base"
else
    OUTPUT_DIR="${SCRIPT_DIR}/eval_results/${DATA_SUFFIX}_${TRAIN_SPLIT}_finetuned"
    LORA_BASE_DIR="${SCRIPT_DIR}/finetuned_models/${DATA_SUFFIX}_${TRAIN_SPLIT}"
fi

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "=============================================="
echo "Direct Y/N Evaluation"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Mode: $MODE"
echo "  Data Type: $DATA_TYPE"
if [[ "$MODE" == "finetuned" ]]; then
    echo "  Train Split: $TRAIN_SPLIT"
    echo "  LoRA Base Dir: $LORA_BASE_DIR"
fi
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# =============================================================================
# ACTIVATE CONDA AND RUN
# =============================================================================

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

if [[ "$MODE" == "base" ]]; then
    python "$SCRIPT_DIR/orchestrator_eval.py" \
        --mode base \
        --data_type "$DATA_TYPE" \
        --output_dir "$OUTPUT_DIR" \
        --gpus 0,1,2,3
else
    python "$SCRIPT_DIR/orchestrator_eval.py" \
        --mode finetuned \
        --data_type "$DATA_TYPE" \
        --lora_base_dir "$LORA_BASE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --gpus 0,1,2,3
fi

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
