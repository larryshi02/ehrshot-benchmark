#!/bin/bash

# Script to run balanced backwards reasoning pipeline for multiple tasks
# This pipeline generates additional reasoning traces to balance the training dataset
# and creates validation samples for evaluation

# Configuration
SERIALIZED_DATA_DIR="/home/lrshi/llm/ehrshot-benchmark/ehrshot/sft/serialized_multi_task_data"
TASK_INSTRUCTIONS="/home/lrshi/llm/ehrshot-benchmark/ehrshot/serialization/task_to_instructions.json"

# Output directories - validation examples only
VAL_OUTPUT_DIR="./data_gpt-5-mini"
VAL_SFT_DATASET_DIR="./data_gpt-5-mini"

TEMPERATURE=1
MAX_COMPLETION_TOKENS=4096
MAX_WORKERS=10  # Number of parallel workers (1 = sequential, increase for faster processing)
RANDOM_SEED=42

# Tasks to process (default: all four tasks)
# Options: acute_mi hypertension hyperlipidemia pancreatic_cancer
TASKS=("acute_mi" "hypertension" "hyperlipidemia" "pancreatic_cancer")

echo "Starting validation backwards reasoning pipeline..."
echo "Tasks to process: ${TASKS[@]}"
echo ""
echo "This pipeline will:"
echo "  - Sample validation examples (100 pos, 100 neg) and generate 3 traces each"
echo "  - For pancreatic_cancer: sample (50 pos, 50 neg) and generate 6 traces each"
echo "  - Use prefill in API calls to save costs"
echo ""

# Create output directories
mkdir -p "$VAL_OUTPUT_DIR"
mkdir -p "$VAL_SFT_DATASET_DIR"

# Build the command
CMD="python azure_backwards_reasoning_val_multi_task_pipeline.py \
    --serialized_data_dir \"$SERIALIZED_DATA_DIR\" \
    --task_to_instructions \"$TASK_INSTRUCTIONS\" \
    --tasks ${TASKS[@]} \
    --val_output_dir \"$VAL_OUTPUT_DIR\" \
    --val_sft_dataset_dir \"$VAL_SFT_DATASET_DIR\" \
    --temperature $TEMPERATURE \
    --max_completion_tokens $MAX_COMPLETION_TOKENS \
    --max_workers $MAX_WORKERS \
    --random_seed $RANDOM_SEED"

# Execute the command
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Validation backwards reasoning pipeline completed successfully!"
    echo ""
    echo "Output files:"
    echo "  Validation backwards reasoning traces: $VAL_OUTPUT_DIR/*_val_backwards_reasoning.json"
    echo "  Validation SFT datasets: $VAL_SFT_DATASET_DIR/*_val_sft.json"
else
    echo "Validation backwards reasoning pipeline failed!"
    exit 1
fi

