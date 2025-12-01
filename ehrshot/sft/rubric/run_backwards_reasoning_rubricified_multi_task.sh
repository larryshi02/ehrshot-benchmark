#!/bin/bash

# Script to run backwards reasoning pipeline for multiple tasks using rubricified patients
# Modify the paths below to match your setup

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Configuration
RUBRICIFIED_DATA_DIR="${SCRIPT_DIR}/rubricified_patients"
TASK_INSTRUCTIONS="${PROJECT_ROOT}/ehrshot/serialization/task_to_instructions.json"
OUTPUT_DIR="${SCRIPT_DIR}/data_gpt-5-mini_rubricified"
TEMPERATURE=1
MAX_COMPLETION_TOKENS=4096
MAX_WORKERS=15  # Number of parallel workers (1 = sequential, increase for faster processing)
RANDOM_SEED=42  # Random seed for train/val set sampling

# Tasks to process (default: all four tasks)
# Options: acute_mi hypertension hyperlipidemia pancreatic_cancer
TASKS=("acute_mi" "hypertension" "hyperlipidemia" "pancreatic_cancer")

echo "Starting multi-task backwards reasoning pipeline (rubricified)..."
echo "Tasks to process: ${TASKS[@]}"
echo "Rubricified data dir: $RUBRICIFIED_DATA_DIR"
echo "Output dir: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to script directory to run Python script
cd "$SCRIPT_DIR"

# Build the command
CMD="python azure_backwards_reasoning_rubricified_multi_task_pipeline.py \
    --rubricified_data_dir \"$RUBRICIFIED_DATA_DIR\" \
    --task_to_instructions \"$TASK_INSTRUCTIONS\" \
    --tasks ${TASKS[@]} \
    --output_dir \"$OUTPUT_DIR\" \
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
    echo "Multi-task backwards reasoning pipeline (rubricified) completed successfully!"
    echo "Output files:"
    echo "  - Training backwards reasoning traces: $OUTPUT_DIR/train_all/*_backwards_reasoning.json"
    echo "  - Training SFT datasets: $OUTPUT_DIR/train_all/*_backwards_sft_dataset.json"
    echo "  - Validation backwards reasoning traces: $OUTPUT_DIR/val/*_backwards_reasoning.json"
    echo "  - Validation SFT datasets: $OUTPUT_DIR/val/*_backwards_sft_dataset.json"
else
    echo "Multi-task backwards reasoning pipeline (rubricified) failed!"
    exit 1
fi
