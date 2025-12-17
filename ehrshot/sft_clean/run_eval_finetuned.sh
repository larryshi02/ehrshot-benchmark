#!/bin/bash
# =============================================================================
# Evaluate Fine-tuned Models (Reasoning)
# =============================================================================
# Usage:
#   ./run_eval_finetuned.sh aggregated originalEHR train_orig
#   ./run_eval_finetuned.sh per_task originalEHR train_orig
# =============================================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

EVAL_MODE="${1:-per_task}"
DATA_REPR="${2:-originalEHR}"
TRAIN_SPLIT="${3:-train_orig}"
GPUS="${4:-0,1,2,3}"

# Validate
if [[ "$EVAL_MODE" != "aggregated" && "$EVAL_MODE" != "per_task" ]]; then
    echo "ERROR: EVAL_MODE must be 'aggregated' or 'per_task'"
    echo "Usage: $0 [aggregated|per_task] [originalEHR|rubricified] [train_orig|train_all] [gpus]"
    exit 1
fi

if [[ "$DATA_REPR" != "originalEHR" && "$DATA_REPR" != "rubricified" ]]; then
    echo "ERROR: DATA_REPR must be 'originalEHR' or 'rubricified'"
    exit 1
fi

if [[ "$TRAIN_SPLIT" != "train_orig" && "$TRAIN_SPLIT" != "train_all" ]]; then
    echo "ERROR: TRAIN_SPLIT must be 'train_orig' or 'train_all'"
    exit 1
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

if [[ "$EVAL_MODE" == "aggregated" ]]; then
    LORA_PATH="${SCRIPT_DIR}/finetuned_models/reasoning_${DATA_REPR}_${TRAIN_SPLIT}_aggregated"
    OUTPUT_DIR="${SCRIPT_DIR}/eval_results/reasoning_${DATA_REPR}_${TRAIN_SPLIT}_aggregated"
    ORCH_EXTRA="--lora_path $LORA_PATH"
else
    LORA_BASE_DIR="${SCRIPT_DIR}/finetuned_models/reasoning_${DATA_REPR}_${TRAIN_SPLIT}"
    OUTPUT_DIR="${SCRIPT_DIR}/eval_results/reasoning_${DATA_REPR}_${TRAIN_SPLIT}_per_task"
    ORCH_EXTRA="--lora_base_dir $LORA_BASE_DIR"
fi

# =============================================================================
# RUN
# =============================================================================

echo "=============================================="
echo "Evaluate: $EVAL_MODE | $DATA_REPR | $TRAIN_SPLIT"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

python "$SCRIPT_DIR/orchestrator_eval.py" \
    --mode "$EVAL_MODE" \
    --data_repr "$DATA_REPR" \
    --output_dir "$OUTPUT_DIR" \
    --gpus "$GPUS" \
    $ORCH_EXTRA

echo "✅ Done: $OUTPUT_DIR"
