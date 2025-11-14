#!/bin/bash

# Script to run backwards reasoning pipeline for multiple tasks
# Modify the paths below to match your setup

# Configuration
SERIALIZED_DATA_DIR="/home/lrshi/llm/ehrshot-benchmark/ehrshot/sft/serialized_multi_task_data"
TASK_INSTRUCTIONS="/home/lrshi/llm/ehrshot-benchmark/ehrshot/serialization/task_to_instructions.json"
OUTPUT_DIR="./data_gpt5"
SFT_DATASET_DIR="./data_gpt5"
MAX_EXAMPLES_PER_TASK=1000000
TEMPERATURE=1
MAX_COMPLETION_TOKENS=4096
MAX_WORKERS=20  # Number of parallel workers (1 = sequential, increase for faster processing)

# Tasks to process (default: all four tasks)
# Options: acute_mi hypertension hyperlipidemia pancreatic_cancer
TASKS=("acute_mi" "hypertension" "hyperlipidemia" "pancreatic_cancer")

# Optional: Filter by specific patient IDs (uncomment and modify if needed)
# PATIENT_IDS=(12345 67890)

echo "Starting multi-task backwards reasoning pipeline..."
echo "Tasks to process: ${TASKS[@]}"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SFT_DATASET_DIR"

# Build the command
CMD="python azure_backwards_reasoning_multi_task_pipeline.py \
    --serialized_data_dir \"$SERIALIZED_DATA_DIR\" \
    --task_to_instructions \"$TASK_INSTRUCTIONS\" \
    --tasks ${TASKS[@]} \
    --output_dir \"$OUTPUT_DIR\" \
    --sft_dataset_dir \"$SFT_DATASET_DIR\" \
    --max_examples_per_task $MAX_EXAMPLES_PER_TASK \
    --temperature $TEMPERATURE \
    --max_completion_tokens $MAX_COMPLETION_TOKENS \
    --max_workers $MAX_WORKERS"

# Add patient IDs filter if specified
if [ ! -z "${PATIENT_IDS[@]}" ]; then
    CMD="$CMD --patient_ids ${PATIENT_IDS[@]}"
fi

# Execute the command
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Multi-task backwards reasoning pipeline completed successfully!"
    echo "Output files:"
    echo "  - Backwards reasoning traces: $OUTPUT_DIR/*_backwards_reasoning.json"
    echo "  - SFT datasets: $SFT_DATASET_DIR/*_sft_dataset.json"
else
    echo "Multi-task backwards reasoning pipeline failed!"
    exit 1
fi

