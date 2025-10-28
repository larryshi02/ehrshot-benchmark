#!/bin/bash

# Script to evaluate fine-tuned SFT model on test dataset
# Computes AUROC and other performance metrics

# Configuration
DATASET_FILE="./data_gpt-4o/acute_mi_sft_dataset.json"
SPLITS_FILE="/home/lrshi/llm/ehrshot-benchmark/EHRSHOT_ASSETS/splits/person_id_map.csv"
PEFT_MODEL_PATH="./qwen_sft_output_cleaned"
OUTPUT_DIR="./evaluation_results"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "Starting SFT model evaluation on test dataset..."

# Check if required files exist
if [ ! -f "$DATASET_FILE" ]; then
    echo "Error: Dataset file not found: $DATASET_FILE"
    echo "Please ensure the dataset file exists."
    exit 1
fi

if [ ! -f "$SPLITS_FILE" ]; then
    echo "Error: Splits file not found: $SPLITS_FILE"
    echo "Please ensure the splits file exists."
    exit 1
fi

if [ ! -d "$PEFT_MODEL_PATH" ]; then
    echo "Error: PEFT model directory not found: $PEFT_MODEL_PATH"
    echo "Please run run_sft_training.sh first to train the model."
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation (Fine-tuned vs Baseline comparison)
echo "Evaluating fine-tuned model vs baseline on test dataset..."
python evaluate_sft_model.py \
    --base_model_name "$BASE_MODEL" \
    --peft_model_path "$PEFT_MODEL_PATH" \
    --dataset_file "$DATASET_FILE" \
    --path_to_splits "$SPLITS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 32768 \
    --max_new_tokens 16384 \
    --temperature 0.1 \
    --use_quantization \
    --save_predictions \
    --save_plots

if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - evaluation_results.json: Detailed metrics and comparison results"
    echo "  - predictions_comparison.csv: Individual predictions for both models"
    echo "  - evaluation_comparison_plots.png: ROC curves and prediction distributions"
    echo "  - metrics_comparison.png: Bar chart comparing all metrics"
    echo ""
    echo "Key metrics comparison:"
    if [ -f "$OUTPUT_DIR/evaluation_results.json" ]; then
        echo "Fine-tuned Model:"
        echo "  - AUROC: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['fine_tuned']['auroc']:.4f}\")")"
        echo "  - Accuracy: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['fine_tuned']['accuracy']:.4f}\")")"
        echo "  - Precision: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['fine_tuned']['precision']:.4f}\")")"
        echo "  - Recall: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['fine_tuned']['recall']:.4f}\")")"
        echo ""
        echo "Baseline Model:"
        echo "  - AUROC: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['baseline']['auroc']:.4f}\")")"
        echo "  - Accuracy: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['baseline']['accuracy']:.4f}\")")"
        echo "  - Precision: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['baseline']['precision']:.4f}\")")"
        echo "  - Recall: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['baseline']['recall']:.4f}\")")"
        echo ""
        echo "Improvements:"
        echo "  - AUROC: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['comparison']['auroc_improvement']:+.4f}\")")"
        echo "  - Accuracy: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['comparison']['accuracy_improvement']:+.4f}\")")"
        echo "  - Precision: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['comparison']['precision_improvement']:+.4f}\")")"
        echo "  - Recall: $(python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(f\"{data['comparison']['recall_improvement']:+.4f}\")")"
    fi
else
    echo "Evaluation failed!"
    exit 1
fi
