#!/bin/bash

# Script to retrain the model with cleaned data
# This will fix the Unicode artifact issue

# Configuration
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
CLEANED_DATASET_FILE="./data_gpt-4o/acute_mi_sft_dataset_cleaned.json"
SPLITS_FILE="/home/lrshi/llm/ehrshot-benchmark/EHRSHOT_ASSETS/splits/person_id_map.csv"
OUTPUT_DIR="./qwen_sft_output_cleaned"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=2e-4

echo "Starting SFT retraining with cleaned data..."

# Check if cleaned dataset file exists
if [ ! -f "$CLEANED_DATASET_FILE" ]; then
    echo "Error: Cleaned dataset file not found: $CLEANED_DATASET_FILE"
    echo "Please run clean_training_data.py first to generate the cleaned dataset."
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run SFT training with cleaned data
echo "Retraining Qwen model with cleaned SFT data..."
python qwen_sft_pipeline.py \
    --model_name "$MODEL_NAME" \
    --dataset_file "$CLEANED_DATASET_FILE" \
    --path_to_splits "$SPLITS_FILE" \
    --split_type "train" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_length 4096 \
    --use_lora \
    --use_quantization \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1

if [ $? -eq 0 ]; then
    echo "SFT retraining with cleaned data completed successfully!"
    echo "Clean model saved to: $OUTPUT_DIR"
    echo ""
    echo "You can now use the clean fine-tuned model for inference."
    echo "Update your evaluation scripts to use: $OUTPUT_DIR"
else
    echo "SFT retraining failed!"
    exit 1
fi
