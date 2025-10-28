#!/bin/bash

# Script to run multi-sample evaluation with KV caching
# This script evaluates the model with 10 samples per data point for better probability estimation

# Configuration
BASE_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PEFT_MODEL_PATH="./qwen_sft_output"
DATASET_FILE="../data/sft_dataset.json"
SPLITS_FILE="../data/splits.csv"
OUTPUT_DIR="./multi_sample_evaluation_results"
NUM_SAMPLES=10
TEMPERATURE=0.7

echo "Starting multi-sample evaluation with KV caching..."
echo "Configuration:"
echo "  Base Model: $BASE_MODEL_NAME"
echo "  PEFT Model: $PEFT_MODEL_PATH"
echo "  Dataset: $DATASET_FILE"
echo "  Splits: $SPLITS_FILE"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Samples per example: $NUM_SAMPLES"
echo "  Temperature: $TEMPERATURE"
echo ""

# Check if required files exist
if [ ! -f "$DATASET_FILE" ]; then
    echo "Error: Dataset file not found: $DATASET_FILE"
    exit 1
fi

if [ ! -f "$SPLITS_FILE" ]; then
    echo "Error: Splits file not found: $SPLITS_FILE"
    exit 1
fi

if [ ! -d "$PEFT_MODEL_PATH" ]; then
    echo "Error: PEFT model directory not found: $PEFT_MODEL_PATH"
    echo "Please run the fine-tuning pipeline first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running multi-sample evaluation..."
python evaluate_sft_model_multi_sample.py \
    --base_model_name "$BASE_MODEL_NAME" \
    --peft_model_path "$PEFT_MODEL_PATH" \
    --dataset_file "$DATASET_FILE" \
    --path_to_splits "$SPLITS_FILE" \
    --max_length 4096 \
    --max_new_tokens 512 \
    --temperature "$TEMPERATURE" \
    --num_samples "$NUM_SAMPLES" \
    --use_kv_cache \
    --cache_implementation torch \
    --output_dir "$OUTPUT_DIR" \
    --save_predictions \
    --save_plots

if [ $? -eq 0 ]; then
    echo "Multi-sample evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - multi_sample_evaluation_results.json: Detailed metrics and comparison results"
    echo "  - multi_sample_predictions_comparison.csv: Individual predictions for both models"
    echo "  - multi_sample_evaluation_plots.png: Comprehensive evaluation plots"
    echo "  - fine_tuned_multi_sample_outputs.json: Fine-tuned model detailed outputs"
    echo "  - baseline_multi_sample_outputs.json: Baseline model detailed outputs"
    echo ""
    echo "Key improvements with multi-sample evaluation:"
    echo "  - Better probability estimation through aggregation"
    echo "  - Confidence and consistency metrics"
    echo "  - KV caching for faster inference"
    echo "  - More robust evaluation with multiple samples"
else
    echo "Multi-sample evaluation failed!"
    exit 1
fi
