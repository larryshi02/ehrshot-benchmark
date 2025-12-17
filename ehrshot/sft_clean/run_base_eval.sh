#!/bin/bash
# =============================================================================
# Step 1: Evaluate Base Qwen3-8B Model on All 4 Tasks (Reasoning)
# =============================================================================
# Runs the base model (no fine-tuning) on all 4 clinical prediction tasks
# in parallel across 4 GPUs.
#
# Usage:
#   ./run_base_eval.sh                    # Uses originalEHR (default)
#   ./run_base_eval.sh originalEHR        # Explicit originalEHR
#   ./run_base_eval.sh rubricified        # Use rubricified data
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

DATA_REPR="${1:-originalEHR}"  # Default: originalEHR

if [[ "$DATA_REPR" != "originalEHR" && "$DATA_REPR" != "rubricified" ]]; then
    echo "ERROR: Invalid DATA_REPR '$DATA_REPR'. Must be 'originalEHR' or 'rubricified'."
    echo "Usage: $0 [originalEHR|rubricified]"
    exit 1
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR="${SCRIPT_DIR}/eval_results/reasoning_${DATA_REPR}_base"
GPUS="0,1,2,3"

# =============================================================================
# PRINT CONFIG & RUN
# =============================================================================

echo "=============================================="
echo "Step 1: Base Model Evaluation (Reasoning)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data Representation: $DATA_REPR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  GPUs: $GPUS"
echo ""

# Activate conda
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run evaluation
python "$SCRIPT_DIR/orchestrator_eval.py" \
    --mode base \
    --data_repr "$DATA_REPR" \
    --output_dir "$OUTPUT_DIR" \
    --gpus "$GPUS"

echo ""
echo "=============================================="
echo "Step 1 Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
