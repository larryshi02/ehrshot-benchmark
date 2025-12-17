#!/bin/bash
# =============================================================================
# Master Script: Run All Reasoning Experiments
# =============================================================================
# Executes all experiments in the correct sequence:
#
#   1. Evaluate base Qwen3-8B model on all 4 tasks
#   2. Fine-tune aggregated model (all tasks combined)
#   3. Evaluate aggregated model on all 4 tasks
#   4. Fine-tune 4 per-task models (in parallel)
#   5. Evaluate per-task models: full 4x4 matrix (diagonal prioritized, metrics computed incrementally)
#
# Usage:
#   ./run_all_experiments.sh                           # originalEHR + train_orig (default)
#   ./run_all_experiments.sh originalEHR train_orig    # Explicit
#   ./run_all_experiments.sh rubricified train_orig    # Rubricified data
#   ./run_all_experiments.sh originalEHR train_all     # Use train_all split
#
# Options:
#   --skip-base       Skip Step 1 (base model evaluation)
#   --skip-aggregated Skip Steps 2 & 3 (aggregated model training + evaluation)
#   --skip-pertask    Skip Steps 4 & 5 (per-task models)
#
# =============================================================================

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipeline fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"
LOG_DIR="${SCRIPT_DIR}/experiment_logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

# Defaults
DATA_REPR="originalEHR"
TRAIN_SPLIT="train_orig"
SKIP_BASE=false
SKIP_AGGREGATED=false
SKIP_PERTASK=false

# Parse positional and optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-base)
            SKIP_BASE=true
            shift
            ;;
        --skip-aggregated)
            SKIP_AGGREGATED=true
            shift
            ;;
        --skip-pertask)
            SKIP_PERTASK=true
            shift
            ;;
        originalEHR|rubricified)
            DATA_REPR="$1"
            shift
            ;;
        train_orig|train_all)
            TRAIN_SPLIT="$1"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [originalEHR|rubricified] [train_orig|train_all] [--skip-base] [--skip-aggregated] [--skip-pertask]"
            exit 1
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

# =============================================================================
# PRINT EXPERIMENT PLAN
# =============================================================================

echo "=============================================="
echo "Reasoning Experiments - Master Pipeline"
echo "=============================================="
echo ""
echo "Timestamp: $TIMESTAMP"
echo "Log Directory: $LOG_DIR"
echo ""
echo "Configuration:"
echo "  Data Representation: $DATA_REPR"
echo "  Train Split: $TRAIN_SPLIT"
echo ""
echo "Experiment Plan:"

if [[ "$SKIP_BASE" == "true" ]]; then
    echo "  1. [SKIPPED] Base model evaluation"
else
    echo "  1. Base model evaluation on all 4 tasks"
fi

if [[ "$SKIP_AGGREGATED" == "true" ]]; then
    echo "  2. [SKIPPED] Aggregated model (train + eval)"
else
    echo "  2. Aggregated model: Fine-tune on all tasks + Evaluate"
fi

if [[ "$SKIP_PERTASK" == "true" ]]; then
    echo "  4. [SKIPPED] Per-task models (train)"
    echo "  5. [SKIPPED] Per-task models (full 4x4 matrix eval)"
else
    echo "  4. Per-task models: Fine-tune 4 models in parallel"
    echo "  5. Per-task models: Evaluate full 4x4 matrix (diagonal first, metrics computed incrementally)"
fi

echo ""
echo "=============================================="
echo ""

# Activate conda
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# =============================================================================
# STEP 1: Base Model Evaluation
# =============================================================================

if [[ "$SKIP_BASE" == "false" ]]; then
    log_step "STEP 1: Base Model Evaluation"
    
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}_step1_base_eval.log"
    echo "Logging to: $LOG_FILE"
    
    bash "$SCRIPT_DIR/run_base_eval.sh" "$DATA_REPR" 2>&1 | tee "$LOG_FILE"
    
    echo "✅ Step 1 Complete"
fi

# =============================================================================
# STEP 2: Aggregated Model Training + Evaluation
# =============================================================================

if [[ "$SKIP_AGGREGATED" == "false" ]]; then
    log_step "STEP 2: Aggregated Model Training"
    
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}_step2_aggregated_train.log"
    echo "Logging to: $LOG_FILE"
    
    bash "$SCRIPT_DIR/run_finetune_aggregated.sh" "$DATA_REPR" "$TRAIN_SPLIT" 2>&1 | tee "$LOG_FILE"
    
    echo "✅ Step 2a (Training) Complete"
    
    # Step 3: Evaluate aggregated model
    log_step "STEP 3: Aggregated Model Evaluation"
    
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}_step3_aggregated_eval.log"
    echo "Logging to: $LOG_FILE"
    
    bash "$SCRIPT_DIR/run_eval_finetuned.sh" aggregated "$DATA_REPR" "$TRAIN_SPLIT" 2>&1 | tee "$LOG_FILE"
    
    echo "✅ Step 3 (Aggregated Eval) Complete"
fi

# =============================================================================
# STEP 4: Per-Task Model Training
# =============================================================================

if [[ "$SKIP_PERTASK" == "false" ]]; then
    log_step "STEP 4: Per-Task Model Training"
    
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}_step4_pertask_train.log"
    echo "Logging to: $LOG_FILE"
    
    bash "$SCRIPT_DIR/run_finetune_per_task.sh" "$DATA_REPR" "$TRAIN_SPLIT" 2>&1 | tee "$LOG_FILE"
    
    echo "✅ Step 4 (Training) Complete"
fi

# =============================================================================
# STEP 5: Per-Task Full Matrix Evaluation (diagonal prioritized, incremental metrics)
# =============================================================================

if [[ "$SKIP_PERTASK" == "false" ]]; then
    log_step "STEP 5: Per-Task Full Matrix Evaluation (16 jobs, metrics computed incrementally)"
    
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}_step5_pertask_matrix_eval.log"
    echo "Logging to: $LOG_FILE"
    
    bash "$SCRIPT_DIR/run_eval_finetuned.sh" per_task "$DATA_REPR" "$TRAIN_SPLIT" 2>&1 | tee "$LOG_FILE"
    
    echo "✅ Step 5 Complete"
fi

# =============================================================================
# SUMMARY
# =============================================================================

log_step "ALL EXPERIMENTS COMPLETE"

echo "Results saved to:"

if [[ "$SKIP_BASE" == "false" ]]; then
    echo "  Step 1 (Base):       ${SCRIPT_DIR}/eval_results/reasoning_${DATA_REPR}_base/"
fi

if [[ "$SKIP_AGGREGATED" == "false" ]]; then
    echo "  Step 2 (Aggregated): ${SCRIPT_DIR}/finetuned_models/reasoning_${DATA_REPR}_${TRAIN_SPLIT}_aggregated/"
    echo "  Step 3 (Agg Eval):   ${SCRIPT_DIR}/eval_results/reasoning_${DATA_REPR}_${TRAIN_SPLIT}_aggregated/"
fi

if [[ "$SKIP_PERTASK" == "false" ]]; then
    echo "  Step 4 (Per-Task):   ${SCRIPT_DIR}/finetuned_models/reasoning_${DATA_REPR}_${TRAIN_SPLIT}/"
    echo "  Step 5 (Eval):       ${SCRIPT_DIR}/eval_results/reasoning_${DATA_REPR}_${TRAIN_SPLIT}_per_task/"
fi

echo ""
echo "Logs saved to: $LOG_DIR"
echo ""
echo "=============================================="
echo "Pipeline Complete: $(date)"
echo "=============================================="
