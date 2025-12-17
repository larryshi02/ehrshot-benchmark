#!/bin/bash
# =============================================================================
# Master Orchestration Script for All Direct Y/N Experiments
# =============================================================================
# This script runs all 6 experiments in sequence:
#
# 1. Evaluate base Qwen3-8B model using original EHR input (test set)
# 2. Evaluate base Qwen3-8B model using rubricified input (test set)
# 3. Fine-tune on original EHR (train_all) + Evaluate
# 4. Fine-tune on rubricified (train_all) + Evaluate
# 5. Fine-tune on original EHR (train_orig) + Evaluate
# 6. Fine-tune on rubricified (train_orig) + Evaluate
#
# Usage:
#   ./run_all_experiments.sh           # Skip base evals (default), run finetuning experiments
#   ./run_all_experiments.sh --run-base  # Include base model evaluations
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"
LOG_DIR="${SCRIPT_DIR}/experiment_logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Parse arguments
SKIP_BASE=true  # Default to skipping base evals (already done)
for arg in "$@"; do
    case $arg in
        --run-base)
            SKIP_BASE=false
            shift
            ;;
    esac
done

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_step() {
    echo ""
    echo "###############################################################################"
    echo "# $1"
    echo "###############################################################################"
    echo ""
}

run_with_logging() {
    local step_name=$1
    local log_file="${LOG_DIR}/${TIMESTAMP}_${step_name}.log"
    shift
    
    echo "Logging to: $log_file"
    "$@" 2>&1 | tee "$log_file"
    return ${PIPESTATUS[0]}
}

# =============================================================================
# PRINT EXPERIMENT PLAN
# =============================================================================

echo "=============================================="
echo "Direct Y/N Experiments - Master Pipeline"
echo "=============================================="
echo ""
echo "Timestamp: $TIMESTAMP"
echo "Log Directory: $LOG_DIR"
echo ""
echo "Experiment Plan:"
if [[ "$SKIP_BASE" == "false" ]]; then
    echo "  1. Base model evaluation on original EHR"
    echo "  2. Base model evaluation on rubricified EHR"
else
    echo "  1. [SKIPPED] Base model evaluation on original EHR"
    echo "  2. [SKIPPED] Base model evaluation on rubricified EHR"
fi
echo "  3. Fine-tune on original EHR (train_orig) + Evaluate"
echo "  4. Fine-tune on rubricified (train_orig) + Evaluate"
echo "  5. Fine-tune on original EHR (train_all) + Evaluate"
echo "  6. Fine-tune on rubricified (train_all) + Evaluate"
echo ""
echo "=============================================="
echo ""

# Activate conda
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# =============================================================================
# EXPERIMENT 1: Base Model on Original EHR
# =============================================================================

if [[ "$SKIP_BASE" == "false" ]]; then
    log_step "EXPERIMENT 1: Base Model Evaluation on Original EHR"
    
    python "$SCRIPT_DIR/orchestrator_eval.py" \
        --mode base \
        --data_type original \
        --output_dir "${SCRIPT_DIR}/eval_results/originalEHR_base" \
        --gpus 0,1,2,3
    
    echo "✅ Experiment 1 Complete"
fi

# =============================================================================
# EXPERIMENT 2: Base Model on Rubricified EHR
# =============================================================================

if [[ "$SKIP_BASE" == "false" ]]; then
    log_step "EXPERIMENT 2: Base Model Evaluation on Rubricified EHR"
    
    python "$SCRIPT_DIR/orchestrator_eval.py" \
        --mode base \
        --data_type rubricified \
        --output_dir "${SCRIPT_DIR}/eval_results/rubricified_base" \
        --gpus 0,1,2,3
    
    echo "✅ Experiment 2 Complete"
fi

# =============================================================================
# EXPERIMENT 3: Fine-tune on Original EHR (train_orig) + Evaluate
# =============================================================================

# log_step "EXPERIMENT 3: Fine-tune on Original EHR (train_orig)"

# echo "Step 3a: Fine-tuning on original EHR data (train_orig)..."
# bash "$SCRIPT_DIR/run_finetune_direct.sh" original train_orig

# echo "Step 3b: Evaluating finetuned models on original EHR test data..."
# python "$SCRIPT_DIR/orchestrator_eval.py" \
#     --mode finetuned \
#     --data_type original \
#     --lora_base_dir "${SCRIPT_DIR}/finetuned_models/originalEHR_train_orig" \
#     --output_dir "${SCRIPT_DIR}/eval_results/originalEHR_train_orig_finetuned" \
#     --gpus 0,1,2,3

# echo "✅ Experiment 3 Complete"

# =============================================================================
# EXPERIMENT 4: Fine-tune on Rubricified EHR (train_orig) + Evaluate
# =============================================================================

log_step "EXPERIMENT 4: Fine-tune on Rubricified EHR (train_orig)"

echo "Step 4a: Fine-tuning on rubricified EHR data (train_orig)..."
bash "$SCRIPT_DIR/run_finetune_direct.sh" rubricified train_orig

echo "Step 4b: Evaluating finetuned models on rubricified EHR test data..."
python "$SCRIPT_DIR/orchestrator_eval.py" \
    --mode finetuned \
    --data_type rubricified \
    --lora_base_dir "${SCRIPT_DIR}/finetuned_models/rubricified_train_orig" \
    --output_dir "${SCRIPT_DIR}/eval_results/rubricified_train_orig_finetuned" \
    --gpus 0,1,2,3

echo "✅ Experiment 4 Complete"

# =============================================================================
# EXPERIMENT 5: Fine-tune on Original EHR (train_all) + Evaluate
# =============================================================================

log_step "EXPERIMENT 5: Fine-tune on Original EHR (train_all)"

echo "Step 5a: Fine-tuning on original EHR data (train_all)..."
bash "$SCRIPT_DIR/run_finetune_direct.sh" original train_all

echo "Step 5b: Evaluating finetuned models on original EHR test data..."
python "$SCRIPT_DIR/orchestrator_eval.py" \
    --mode finetuned \
    --data_type original \
    --lora_base_dir "${SCRIPT_DIR}/finetuned_models/originalEHR_train_all" \
    --output_dir "${SCRIPT_DIR}/eval_results/originalEHR_train_all_finetuned" \
    --gpus 0,1,2,3

echo "✅ Experiment 5 Complete"

# =============================================================================
# EXPERIMENT 6: Fine-tune on Rubricified EHR (train_all) + Evaluate
# =============================================================================

log_step "EXPERIMENT 6: Fine-tune on Rubricified EHR (train_all)"

echo "Step 6a: Fine-tuning on rubricified EHR data (train_all)..."
bash "$SCRIPT_DIR/run_finetune_direct.sh" rubricified train_all

echo "Step 6b: Evaluating finetuned models on rubricified EHR test data..."
python "$SCRIPT_DIR/orchestrator_eval.py" \
    --mode finetuned \
    --data_type rubricified \
    --lora_base_dir "${SCRIPT_DIR}/finetuned_models/rubricified_train_all" \
    --output_dir "${SCRIPT_DIR}/eval_results/rubricified_train_all_finetuned" \
    --gpus 0,1,2,3

echo "✅ Experiment 6 Complete"

# =============================================================================
# SUMMARY
# =============================================================================

log_step "ALL EXPERIMENTS COMPLETE"

echo "Results saved to:"
if [[ "$SKIP_BASE" == "false" ]]; then
    echo "  1. ${SCRIPT_DIR}/eval_results/originalEHR_base/"
    echo "  2. ${SCRIPT_DIR}/eval_results/rubricified_base/"
fi
echo "  3. ${SCRIPT_DIR}/eval_results/originalEHR_train_orig_finetuned/"
echo "  4. ${SCRIPT_DIR}/eval_results/rubricified_train_orig_finetuned/"
echo "  5. ${SCRIPT_DIR}/eval_results/originalEHR_train_all_finetuned/"
echo "  6. ${SCRIPT_DIR}/eval_results/rubricified_train_all_finetuned/"
echo ""
echo "Finetuned models saved to:"
echo "  - ${SCRIPT_DIR}/finetuned_models/originalEHR_train_orig/"
echo "  - ${SCRIPT_DIR}/finetuned_models/rubricified_train_orig/"
echo "  - ${SCRIPT_DIR}/finetuned_models/originalEHR_train_all/"
echo "  - ${SCRIPT_DIR}/finetuned_models/rubricified_train_all/"
echo ""
echo "Logs saved to: $LOG_DIR"
