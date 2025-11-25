#!/bin/bash
# Parallel Evaluation Pipeline
# Usage: ./run_eval_pipeline.sh [EVAL_EXPERIMENT_NAME]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EVAL_NAME=${1:-"eval_experiment_yn"}

# make sure this is consistent with the EVAL_NAME
TRAINING_OUTPUT_DIR="${SCRIPT_DIR}/unsloth_output_qwen3-8b_r16_data_gpt-5-mini_sft_rt_yn" 

OUTPUT_ROOT="${SCRIPT_DIR}/${EVAL_NAME}"

# --- CONFIGURATION ---
MODEL_BASE="Qwen/Qwen3-8B" 
DATA_ROOT="/dev/shm/ehrshot-data/serialized_multi_task_data"



# Python Scripts Names
WORKER_SCRIPT="eval_vllm_worker.py"
AGGREGATOR_SCRIPT="eval_vllm_metric_calculator.py"

TASKS=("acute_mi" "pancreatic_cancer" "hypertension" "hyperlipidemia")

# --- SAFETY CHECKS ---
if [ ! -f "${SCRIPT_DIR}/${WORKER_SCRIPT}" ]; then
    echo "❌ CRITICAL ERROR: Could not find ${WORKER_SCRIPT} in ${SCRIPT_DIR}"
    echo "   Ensure the file exists and is named correctly."
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/${AGGREGATOR_SCRIPT}" ]; then
    echo "❌ CRITICAL ERROR: Could not find ${AGGREGATOR_SCRIPT} in ${SCRIPT_DIR}"
    echo "   Ensure the file exists and is named correctly."
    exit 1
fi

# --- HELPER FUNCTIONS ---
get_latest_checkpoint() {
    local base_dir=$1
    if [ ! -d "$base_dir" ]; then
        echo "Error: Directory $base_dir not found."
        return 1
    fi
    # Sort version numbers correctly
    local latest_ckpt=$(find "$base_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
    if [ -n "$latest_ckpt" ]; then
        echo "$latest_ckpt"
    else
        echo "$base_dir"
    fi
}

# --- MAIN EXECUTION ---

echo "Starting Evaluation Pipeline: $EVAL_NAME"
echo "Script Directory: $SCRIPT_DIR"
mkdir -p "$OUTPUT_ROOT"

# --- OUTER LOOP: Iterate through Source Models (SEQUENTIAL) ---
for SOURCE_TASK in "${TASKS[@]}"; do
    
    echo "---------------------------------------------------"
    echo "Processing Source Model: $SOURCE_TASK"
    
    TASK_BASE_DIR="${TRAINING_OUTPUT_DIR}/${SOURCE_TASK}"
    
    # Check if training dir exists
    if [ ! -d "$TASK_BASE_DIR" ]; then
         echo "⚠️  Training directory not found: $TASK_BASE_DIR. Skipping."
         continue
    fi

    LORA_PATH=$(get_latest_checkpoint "$TASK_BASE_DIR")
    echo "  > Adapter: $LORA_PATH"
    
    SOURCE_OUT_DIR="${OUTPUT_ROOT}/${SOURCE_TASK}"
    mkdir -p "$SOURCE_OUT_DIR"

    # --- INNER LOOP: Iterate through Target Tasks (PARALLEL) ---
    pids=()
    GPU_ID=0
    
    for TARGET_TASK in "${TASKS[@]}"; do
        DATA_PATH="${DATA_ROOT}/${TARGET_TASK}_all_splits.json"
        TARGET_OUT_DIR="${SOURCE_OUT_DIR}/${TARGET_TASK}"
        
        # Create dir explicitly before running
        mkdir -p "$TARGET_OUT_DIR"
        
        echo "  [GPU $GPU_ID] Launching evaluation on $TARGET_TASK..."
        
        # Background execution
        (
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            python "${SCRIPT_DIR}/${WORKER_SCRIPT}" \
                --model_base "$MODEL_BASE" \
                --lora_path "$LORA_PATH" \
                --data_path "$DATA_PATH" \
                --task_name "$TARGET_TASK" \
                --output_dir "$TARGET_OUT_DIR" > "${TARGET_OUT_DIR}/inference.log" 2>&1
            
            # Check exit code of python script inside the subshell
            if [ $? -ne 0 ]; then
                echo "  ❌ [GPU $GPU_ID] Failed! See logs: ${TARGET_OUT_DIR}/inference.log"
            fi
        ) &
        
        pids+=($!)
        ((GPU_ID++))
    done

    # --- BARRIER: WAIT FOR ALL GPUS ---
    echo "  > Waiting for 4 background tasks to finish..."
    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo "  > Finished Source Model: $SOURCE_TASK"

done

# --- PART 2: AGGREGATION ---

echo "---------------------------------------------------"
echo "Inference Complete. Running Aggregator..."

python "${SCRIPT_DIR}/${AGGREGATOR_SCRIPT}" \
    --eval_root "$OUTPUT_ROOT"

echo "Done! Results saved in $OUTPUT_ROOT"