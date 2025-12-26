#!/bin/bash
# =============================================================================
# Evaluate RLVR-Trained Model
# =============================================================================
#
# PURPOSE:
#   Evaluate an RLVR-trained model on all 4 clinical tasks using the same
#   evaluation protocol as SFT experiments (sampling-based, 10 samples per example).
#
# USAGE:
#   ./run_eval_rlvr.sh RLVR_CHECKPOINT [DATA_REPR] [OUTPUT_NAME]
#
# ARGUMENTS:
#   RLVR_CHECKPOINT - Path to RLVR model checkpoint (required)
#   DATA_REPR       - Data representation: "originalEHR" or "rubricified"
#                     Default: originalEHR
#   OUTPUT_NAME     - Name for output directory
#                     Default: Auto-generated from checkpoint path
#
# EXAMPLES:
#   # Evaluate Phase 1 model
#   ./run_eval_rlvr.sh ../outputs/phase1_validation/acute_mi
#
#   # Evaluate Phase 2 aggregated model
#   ./run_eval_rlvr.sh ../outputs/phase2_aggregated
#
#   # With custom output name
#   ./run_eval_rlvr.sh ../outputs/phase2_aggregated originalEHR rlvr_final
#
# OUTPUTS:
#   - Per-task results: eval_results/{output_name}/{task}/
#     - predictions.csv
#     - detailed_logs.json
#     - run_stats.json
#     - summary.json (AUROC, AUPRC)
#   - Summary: eval_results/{output_name}/metrics_summary.csv
#
# =============================================================================

set -e

# --- Parse Arguments ---
RLVR_CHECKPOINT="${1:-}"
DATA_REPR="${2:-originalEHR}"
OUTPUT_NAME="${3:-}"

if [ -z "$RLVR_CHECKPOINT" ]; then
    echo "ERROR: RLVR_CHECKPOINT is required"
    echo "Usage: $0 RLVR_CHECKPOINT [DATA_REPR] [OUTPUT_NAME]"
    exit 1
fi

# Verify checkpoint exists
if [ ! -d "$RLVR_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $RLVR_CHECKPOINT"
    exit 1
fi

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-EHRSHOT_ENV}"

# Base model (same as SFT)
MODEL_BASE="Qwen/Qwen3-8B"

# Data paths
if [ "$DATA_REPR" == "originalEHR" ]; then
    DATA_DIR="/dev/shm/ehrshot-data/serialized_multi_task_data"
elif [ "$DATA_REPR" == "rubricified" ]; then
    DATA_DIR="/dev/shm/ehrshot-data/serialized_multi_task_data_rubricified"
else
    echo "ERROR: Invalid DATA_REPR '$DATA_REPR'. Must be 'originalEHR' or 'rubricified'"
    exit 1
fi

# Auto-generate output name if not provided
if [ -z "$OUTPUT_NAME" ]; then
    OUTPUT_NAME="rlvr_$(basename "$RLVR_CHECKPOINT")"
fi

OUTPUT_BASE_DIR="$SCRIPT_DIR/../outputs/eval_results/$OUTPUT_NAME"

# Tasks
TASKS=("acute_mi" "hyperlipidemia" "hypertension" "pancreatic_cancer")
GPUS=("0" "1" "2" "3")

# --- Print Configuration ---
echo "=============================================="
echo "RLVR Model Evaluation"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  RLVR Checkpoint: $RLVR_CHECKPOINT"
echo "  Base Model:      $MODEL_BASE"
echo "  Data Repr:       $DATA_REPR"
echo "  Data Dir:        $DATA_DIR"
echo "  Output Dir:      $OUTPUT_BASE_DIR"
echo ""
echo "Tasks: ${TASKS[*]}"
echo "GPUs:  ${GPUS[*]}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# --- Evaluation Script (from sft_clean) ---
EVAL_SCRIPT="$PROJECT_DIR/sft_clean/eval_vllm_reasoning.py"
METRICS_SCRIPT="$PROJECT_DIR/sft_clean/eval_compute_metrics.py"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

# --- Launch Function ---
launch_eval() {
    local task_name=$1
    local gpu_id=$2
    
    local data_path="${DATA_DIR}/${task_name}_all_splits.json"
    local output_dir="${OUTPUT_BASE_DIR}/${task_name}"
    local log_file="${OUTPUT_BASE_DIR}/${task_name}.log"
    
    mkdir -p "$output_dir"
    
    echo "  [GPU $gpu_id] Evaluating $task_name..."
    
    (
        source $HOME/miniconda3/etc/profile.d/conda.sh
        conda activate $CONDA_ENV
        
        export CUDA_VISIBLE_DEVICES=$gpu_id
        
        python "$EVAL_SCRIPT" \
            --model_base "$MODEL_BASE" \
            --lora_path "$RLVR_CHECKPOINT" \
            --data_path "$data_path" \
            --task_name "$task_name" \
            --output_dir "$output_dir" \
            --num_samples 10
            
        # Compute metrics
        if [ -f "$METRICS_SCRIPT" ]; then
            python "$METRICS_SCRIPT" --predictions_path "$output_dir/predictions.csv"
        fi
        
    ) > "$log_file" 2>&1 &
}

# --- Run Evaluations in Parallel ---
echo "Starting parallel evaluation on ${#GPUS[@]} GPUs..."
echo ""

pids=()
for i in "${!TASKS[@]}"; do
    launch_eval "${TASKS[$i]}" "${GPUS[$i]}"
    pids+=($!)
done

echo "Evaluation jobs launched. PIDs: ${pids[*]}"
echo ""
echo "To monitor progress:"
echo "  tail -f ${OUTPUT_BASE_DIR}/*.log"
echo ""

# Wait for all jobs
wait

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="
echo ""

# --- Collect Results ---
echo "Results:"
echo ""

for task in "${TASKS[@]}"; do
    summary_file="${OUTPUT_BASE_DIR}/${task}/summary.json"
    if [ -f "$summary_file" ]; then
        auroc=$(python3 -c "import json; d=json.load(open('$summary_file')); print(f\"{d.get('auroc', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
        auprc=$(python3 -c "import json; d=json.load(open('$summary_file')); print(f\"{d.get('auprc', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
        echo "  $task: AUROC=$auroc, AUPRC=$auprc"
    else
        echo "  $task: Results not found"
    fi
done

echo ""
echo "Full results saved to: $OUTPUT_BASE_DIR"
