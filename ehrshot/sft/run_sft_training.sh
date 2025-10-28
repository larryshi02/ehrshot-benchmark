#!/bin/bash

# Script to run SFT training pipeline
# Modify the paths below to match your setup

# Configuration
DATASET_FILE="./data_gpt-4o/acute_mi_sft_dataset.json"
SPLITS_FILE="/home/lrshi/llm/ehrshot-benchmark/EHRSHOT_ASSETS/splits/person_id_map.csv"
OUTPUT_DIR="./qwen_sft_output"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=2e-4

echo "Starting SFT training pipeline..."

# Check if dataset file exists
if [ ! -f "$DATASET_FILE" ]; then
    echo "Error: Dataset file not found: $DATASET_FILE"
    echo "Please run run_backwards_reasoning.sh first to generate the dataset."
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run SFT training (TRAIN SPLIT ONLY)
echo "Training Qwen model with SFT (TRAIN SPLIT ONLY)..."
python qwen_sft_pipeline.py \
    --model_name "$MODEL_NAME" \
    --dataset_file "$DATASET_FILE" \
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
    echo "SFT training completed successfully! (TRAIN SPLIT ONLY)"
    echo "Model saved to: $OUTPUT_DIR"
    echo "Note: Model was trained only on train split patients"
    echo ""
    echo "You can now use the fine-tuned model for inference."
else
    echo "SFT training failed!"
    exit 1
fi
