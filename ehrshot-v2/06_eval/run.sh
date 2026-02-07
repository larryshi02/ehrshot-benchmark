#!/usr/bin/env bash
# =============================================================================
# 06_eval/run.sh -- Evaluate all trained models and embedding-based approaches.
#
# GPU Parallelization:
#   GPU-bound jobs (eval_direct, eval_reasoning, generate_embeddings) are
#   dispatched in batches of NUM_GPUS (default: 4).
#   CPU-only jobs (eval_embeddings, compute_metrics) run sequentially.
#
# Configuration:
#   GPUS          -- space-separated GPU IDs (default: "0 1 2 3")
#   TASKS         -- override task list (default: all 15)
#   SEED          -- global random seed (default: 42)
#
# Examples:
#   bash 06_eval/run.sh
#   GPUS="0 1" bash 06_eval/run.sh
#   TASKS="guo_icu guo_los" bash 06_eval/run.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
MODEL_DIR="${DATA_DIR}/models"
RESULTS_DIR="${DATA_DIR}/results"

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
echo ""

# --- Logging ---
LOG_DIR="${DATA_DIR}/logs/eval"
mkdir -p "$LOG_DIR"

# =============================================================================
# Generic GPU-parallel dispatcher
# =============================================================================

dispatch_jobs() {
    local -n _jobs=$1
    local total=${#_jobs[@]}
    local batch_num=0

    if [ "$total" -eq 0 ]; then
        echo "  (no jobs to dispatch)"
        return 0
    fi

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
# A. Direct model evaluation (logprobs) -- GPU-parallelized
# =============================================================================

eval_direct_job() {
    local repr="$1" task="$2" suffix="$3"
    local test_file="$DATA_DIR/sft/${repr}/test/${task}.json"
    local lora_path="$MODEL_DIR/direct_${repr}_${suffix}/${task}"
    local out="$RESULTS_DIR/direct_${repr}_${suffix}/${task}"
    local log="$LOG_DIR/eval_direct_${repr}_${suffix}_${task}.log"

    if [ ! -f "$test_file" ] || [ ! -d "$lora_path" ]; then
        echo "  [skip] direct ${repr}/${task} (${suffix})" > "$log"
        return 0
    fi

    python "$SCRIPT_DIR/eval_direct.py" \
        --test_file "$test_file" \
        --lora_path "$lora_path" \
        --output_dir "$out" \
        > "$log" 2>&1
}

echo "=== Evaluating direct models (n=full + n=40) ==="
EVAL_DIRECT_JOBS=()
for suffix in full n40; do
    for repr in plaintext manualrubric llmrubric cot_unsupervised; do
        for task in "${TASKS[@]}"; do
            EVAL_DIRECT_JOBS+=("eval_direct_job $repr $task $suffix")
        done
    done
done
dispatch_jobs EVAL_DIRECT_JOBS

# =============================================================================
# B. Reasoning model evaluation (sampling) -- GPU-parallelized
# =============================================================================

eval_reasoning_job() {
    local task="$1" suffix="$2"
    local test_file="$DATA_DIR/sft/plaintext/test/${task}.json"
    local lora_path="$MODEL_DIR/reasoning_${suffix}/${task}"
    local out="$RESULTS_DIR/reasoning_${suffix}/${task}"
    local log="$LOG_DIR/eval_reasoning_${suffix}_${task}.log"

    if [ ! -f "$test_file" ] || [ ! -d "$lora_path" ]; then
        echo "  [skip] reasoning ${task} (${suffix})" > "$log"
        return 0
    fi

    python "$SCRIPT_DIR/eval_reasoning.py" \
        --test_file "$test_file" \
        --lora_path "$lora_path" \
        --output_dir "$out" \
        > "$log" 2>&1
}

echo ""
echo "=== Evaluating reasoning models (n=full + n=40) ==="
EVAL_REASONING_JOBS=()
for suffix in full n40; do
    for task in "${TASKS[@]}"; do
        EVAL_REASONING_JOBS+=("eval_reasoning_job $task $suffix")
    done
done
dispatch_jobs EVAL_REASONING_JOBS

# =============================================================================
# C. Embedding generation -- GPU-parallelized (1 repr per GPU)
# =============================================================================

gen_embed_job() {
    local repr="$1"
    local log="$LOG_DIR/embed_${repr}.log"

    python "$SCRIPT_DIR/generate_embeddings.py" \
        --sft_dir "$DATA_DIR/sft/${repr}" \
        --output_dir "$DATA_DIR/embeddings/${repr}" \
        > "$log" 2>&1
}

echo ""
echo "=== Generating embeddings (4 repr types -> 4 GPUs) ==="
EMBED_JOBS=()
for repr in plaintext manualrubric llmrubric cot_unsupervised; do
    EMBED_JOBS+=("gen_embed_job $repr")
done
dispatch_jobs EMBED_JOBS

# =============================================================================
# D. Embedding evaluation -- CPU-only, runs sequentially
# =============================================================================

echo ""
echo "=== Evaluating embeddings (n=full) -- CPU ==="
for repr in plaintext manualrubric llmrubric cot_unsupervised; do
    echo "  [eval_embeddings] ${repr}"
    python "$SCRIPT_DIR/eval_embeddings.py" \
        --embeddings_dir "$DATA_DIR/embeddings/${repr}" \
        --output_dir "$RESULTS_DIR/embedding_${repr}_full"
done

echo ""
echo "=== Evaluating embeddings (n=40) -- CPU ==="
for repr in plaintext manualrubric llmrubric cot_unsupervised; do
    for task in "${TASKS[@]}"; do
        python "$SCRIPT_DIR/eval_embeddings.py" \
            --embeddings_dir "$DATA_DIR/embeddings/${repr}" \
            --output_dir "$RESULTS_DIR/embedding_${repr}_n40" \
            --tasks "$task" \
            --n_train 40 \
            --cohort_file "$DATA_DIR/rubric/${task}/patient_ids.json"
    done
done

# =============================================================================
# E. Aggregate metrics -- CPU-only
# =============================================================================

echo ""
echo "=== Computing aggregate metrics ==="
for dir in "$RESULTS_DIR"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    # Skip the metrics subdir itself
    [ "$name" = "metrics" ] && continue
    preds=$(find "$dir" -name "predictions.csv" 2>/dev/null | tr '\n' ' ')
    if [ -n "$preds" ]; then
        echo "  [metrics] $name"
        # shellcheck disable=SC2086
        python "$SCRIPT_DIR/compute_metrics.py" \
            --predictions $preds \
            --output_dir "$RESULTS_DIR/metrics/$name"
    fi
done

echo ""
echo "=== All evaluation complete. Results in $RESULTS_DIR ==="
echo "=== Logs in $LOG_DIR ==="
