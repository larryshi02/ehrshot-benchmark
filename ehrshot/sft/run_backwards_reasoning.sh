#!/bin/bash

# Script to run backwards reasoning pipeline
# Modify the paths below to match your setup

# Configuration
DATABASE_PATH="/home/lrshi/llm/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract"
LABELS_DIR="/home/lrshi/llm/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark/new_acutemi"
TASK_INSTRUCTIONS="/home/lrshi/llm/ehrshot-benchmark/ehrshot/serialization/task_to_instructions.json"
OUTPUT_DIR="./data"
MAX_EXAMPLES=1000000

# Alternative: Use pre-generated JSONL data
# DATA_DIR="/path/to/your/jsonl/data"

echo "Starting backwards reasoning pipeline..."

# Create output directory
mkdir -p $OUTPUT_DIR

# Option 1: Generate from database (Acute MI only)
echo "Generating reasoning traces from database for Acute MI task..."
python azure_backwards_reasoning_pipeline.py \
    --database "$DATABASE_PATH" \
    --path_to_labels_dir "$LABELS_DIR" \
    --task_to_instructions "$TASK_INSTRUCTIONS" \
    --max_examples $MAX_EXAMPLES \
    --output_file "$OUTPUT_DIR/acute_mi_sft_traces.json" \
    --sft_dataset_file "$OUTPUT_DIR/acute_mi_sft_dataset.json" \
    --temperature 0.3 \
    --max_tokens 4096

# Option 2: Use pre-generated JSONL (uncomment if needed)
# echo "Generating reasoning traces from JSONL data..."
# python azure_backwards_reasoning_pipeline.py \
#     --data_dir "$DATA_DIR" \
#     --max_examples $MAX_EXAMPLES \
#     --output_file "$OUTPUT_DIR/sft_traces.json" \
#     --sft_dataset_file "$OUTPUT_DIR/sft_dataset.json" \
#     --temperature 0.3 \
#     --max_tokens 4096

if [ $? -eq 0 ]; then
    echo "Backwards reasoning pipeline completed successfully!"
    echo "Output files:"
    echo "  - Acute MI SFT traces: $OUTPUT_DIR/acute_mi_sft_traces.json"
    echo "  - Acute MI SFT dataset: $OUTPUT_DIR/acute_mi_sft_dataset.json"
else
    echo "Backwards reasoning pipeline failed!"
    exit 1
fi
