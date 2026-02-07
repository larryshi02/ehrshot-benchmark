#!/usr/bin/env bash
# =============================================================================
# 05_train/run.sh -- Fine-tune models for all representations.
#
# GPU Parallelization:
#   Jobs are dispatched in batches of NUM_GPUS (default: 4).
#   Each job is pinned to a single GPU via CUDA_VISIBLE_DEVICES.
#   If there are more jobs than GPUs, batches run sequentially
#   (e.g. 15 tasks -> 4 + 4 + 4 + 3).
#
# Configuration:
#   GPUS          -- space-separated GPU IDs (default: "0 1 2 3")
#   TASKS         -- override task list (default: all 15)
#   SEED          -- global random seed (default: 42)
#   WANDB_OFFLINE -- set to "1" to disable wandb
#
# Examples:
#   bash 05_train/run.sh                    # all tasks, GPUs 0-3
#   GPUS="0 1" bash 05_train/run.sh         # only 2 GPUs
#   TASKS="guo_icu guo_los" bash run.sh     # subset of tasks
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
MODEL_DIR="${DATA_DIR}/models"

# --- GPU config ---
GPUS=(${GPUS:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}

# --- Reproducibility ---
export PYTHONHASHSEED="${SEED:-42}"
SEED="${SEED:-42}"

# --- Tasks ---
DEFAULT_TASKS="guo_icu guo_los guo_readmission lab_thrombocytopenia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_anemia new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi chexpert"
TASKS=(${TASKS:-$DEFAULT_TASKS})

echo "GPUs: ${GPUS[*]} (${NUM_GPUS} total)"
echo "Tasks: ${TASKS[*]} (${#TASKS[@]} total)"
echo "Seed: ${SEED}"
echo ""

# --- Logging helper ---
LOG_DIR="${DATA_DIR}/logs/train"
mkdir -p "$LOG_DIR"

# =============================================================================
# Job functions -- each runs a single fine-tuning task on the assigned GPU
# =============================================================================

run_direct() {
    local repr="$1"
    local task="$2"
    local suffix="$3"

    local train_file="$DATA_DIR/sft/${repr}/train/${task}.json"
    local eval_file="$DATA_DIR/sft/${repr}/val/${task}.json"
    local out="$MODEL_DIR/direct_${repr}_${suffix}/${task}"
    local log="$LOG_DIR/direct_${repr}_${suffix}_${task}.log"

    if [ ! -f "$train_file" ]; then
        echo "  [skip] direct ${repr}/${task} (${suffix}): no train file"
        return 0
    fi

    local extra_args=""
    if [ "$suffix" = "n40" ]; then
        extra_args="--n_train 40 --cohort_file $DATA_DIR/rubric/${task}/patient_ids.json"
    fi

    python "$SCRIPT_DIR/finetune_direct.py" \
        --train_file "$train_file" \
        --eval_file "$eval_file" \
        --output_dir "$out" \
        --wandb_project "ehrshot-v2-direct-${repr}" \
        --wandb_run_name "${task}_${suffix}" \
        --seed "$SEED" \
        $extra_args \
        > "$log" 2>&1
}

run_reasoning() {
    local task="$1"
    local suffix="$2"

    local train_file="$DATA_DIR/sft/cot_supervised/train/${task}.json"
    local eval_file="$DATA_DIR/sft/cot_supervised/val/${task}.json"
    local out="$MODEL_DIR/reasoning_${suffix}/${task}"
    local log="$LOG_DIR/reasoning_${suffix}_${task}.log"

    if [ ! -f "$train_file" ]; then
        echo "  [skip] reasoning ${task} (${suffix}): no train file"
        return 0
    fi

    local extra_args=""
    if [ "$suffix" = "n40" ]; then
        extra_args="--n_train 40 --cohort_file $DATA_DIR/rubric/${task}/patient_ids.json"
    fi

    python "$SCRIPT_DIR/finetune_reasoning.py" \
        --train_file "$train_file" \
        --eval_file "$eval_file" \
        --output_dir "$out" \
        --wandb_project "ehrshot-v2-reasoning" \
        --wandb_run_name "${task}_${suffix}" \
        --seed "$SEED" \
        $extra_args \
        > "$log" 2>&1
}

# =============================================================================
# Generic GPU-parallel dispatcher
# =============================================================================
#
# Usage: dispatch_jobs JOB_ARRAY
#   Each element is a string: "function_name arg1 arg2 ..."
#   Jobs are dispatched in batches of NUM_GPUS.

dispatch_jobs() {
    local -n _jobs=$1
    local total=${#_jobs[@]}
    local batch_num=0

    echo "  Dispatching $total jobs across ${NUM_GPUS} GPUs"

    for ((i = 0; i < total; i += NUM_GPUS)); do
        batch_num=$((batch_num + 1))
        local pids=()
        local batch_end=$(( i + NUM_GPUS < total ? i + NUM_GPUS : total ))

        for ((j = 0; j < NUM_GPUS && i + j < total; j++)); do
            local gpu=${GPUS[$j]}
            local job_str="${_jobs[$((i + j))]}"
            # shellcheck disable=SC2086
            ( export CUDA_VISIBLE_DEVICES="$gpu"; $job_str ) &
            pids+=($!)
        done

        # Wait for this batch
        local fail=0
        for pid in "${pids[@]}"; do
            wait "$pid" || fail=1
        done

        if [ "$fail" -ne 0 ]; then
            echo "ERROR: One or more jobs in batch $batch_num failed. Check logs in $LOG_DIR"
            exit 1
        fi

        echo "  Batch ${batch_num} done (${batch_end}/${total})"
    done
}

# =============================================================================
# 1. Direct fine-tuning (n=full)
# =============================================================================
echo "=== Direct fine-tuning (n=full) ==="
DIRECT_FULL_JOBS=()
for repr in plaintext manualrubric llmrubric cot_unsupervised; do
    for task in "${TASKS[@]}"; do
        DIRECT_FULL_JOBS+=("run_direct $repr $task full")
    done
done
dispatch_jobs DIRECT_FULL_JOBS

# =============================================================================
# 2. Reasoning fine-tuning (n=full)
# =============================================================================
echo ""
echo "=== Reasoning fine-tuning (n=full) ==="
REASONING_FULL_JOBS=()
for task in "${TASKS[@]}"; do
    REASONING_FULL_JOBS+=("run_reasoning $task full")
done
dispatch_jobs REASONING_FULL_JOBS

# =============================================================================
# 3. Direct fine-tuning (n=40)
# =============================================================================
echo ""
echo "=== Direct fine-tuning (n=40) ==="
DIRECT_N40_JOBS=()
for repr in plaintext manualrubric llmrubric cot_unsupervised; do
    for task in "${TASKS[@]}"; do
        DIRECT_N40_JOBS+=("run_direct $repr $task n40")
    done
done
dispatch_jobs DIRECT_N40_JOBS

# =============================================================================
# 4. Reasoning fine-tuning (n=40)
# =============================================================================
echo ""
echo "=== Reasoning fine-tuning (n=40) ==="
REASONING_N40_JOBS=()
for task in "${TASKS[@]}"; do
    REASONING_N40_JOBS+=("run_reasoning $task n40")
done
dispatch_jobs REASONING_N40_JOBS

echo ""
echo "=== All training complete. Logs in $LOG_DIR ==="
