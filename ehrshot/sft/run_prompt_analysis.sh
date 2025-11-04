#!/bin/bash

# Script to analyze prompt lengths in the SFT dataset
# Usage: ./run_prompt_analysis.sh

# Set default paths
DATASET_FILE="/home/lrshi/llm/ehrshot-benchmark/ehrshot/sft/data/acute_mi_sft_dataset.json"
OUTPUT_DIR="./prompt_analysis"
TOKENIZER_NAME="Qwen/Qwen2.5-7B-Instruct"

echo "🔍 Analyzing prompt lengths in SFT dataset..."
echo "Dataset: $DATASET_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Tokenizer: $TOKENIZER_NAME"
echo ""

# Run the analysis
python analyze_prompt_lengths.py \
    --dataset_path "$DATASET_FILE" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis completed successfully!"
    echo "📊 Check the plots in: $OUTPUT_DIR"
    echo "   - prompt_length_distribution.png"
    echo "   - prompt_length_percentiles.png"
else
    echo ""
    echo "❌ Analysis failed!"
    exit 1
fi


