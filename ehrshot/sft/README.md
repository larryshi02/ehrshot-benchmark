# Supervised Fine-Tuning (SFT) Pipeline for EHRshot

This directory contains a complete pipeline for supervised fine-tuning of the Qwen 8B model using backwards reasoning traces generated from ground truth labels.

## Overview

The pipeline consists of two main components:

1. **Azure Backwards Reasoning Pipeline**: Generates high-quality reasoning traces by providing the model with ground truth answers and asking it to explain the reasoning behind the correct prediction.

2. **Qwen SFT Pipeline**: Fine-tunes the Qwen 8B model using the generated reasoning traces in a supervised learning setup.

## Pipeline Architecture

```
EHR Data + Ground Truth Labels
           ↓
Azure Backwards Reasoning Pipeline
           ↓
Reasoning Traces (SFT Dataset)
           ↓
Qwen SFT Pipeline
           ↓
Fine-tuned Qwen Model
```

## Files

- `azure_backwards_reasoning_pipeline.py`: Generates backwards reasoning traces using Azure OpenAI
- `qwen_sft_pipeline.py`: Fine-tunes Qwen 8B model using the reasoning traces
- `azure_config.json`: Azure OpenAI configuration
- `sft_config.yaml`: SFT training configuration
- `requirements.txt`: Python dependencies
- `run_backwards_reasoning.sh`: Script to run backwards reasoning generation
- `run_sft_training.sh`: Script to run SFT training
- `setup.sh`: Environment setup script

## Setup

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or use the setup script
./setup.sh
```

### 2. Configure Azure OpenAI

Edit `azure_config.json` with your Azure OpenAI credentials:

```json
{
  "endpoint": "https://your-endpoint.openai.azure.com/",
  "api_key": "your-actual-api-key",
  "api_version": "2024-12-01-preview",
  "deployment": "gpt-4.1",
  "model": "gpt-4.1"
}
```

Alternatively, set environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
```

### 3. Configure SFT Training

Edit `sft_config.yaml` to adjust training parameters:

- Model name and settings
- LoRA configuration
- Training hyperparameters
- Logging and evaluation settings

## Usage

### Step 1: Generate Backwards Reasoning Traces

#### Option A: From Database
```bash
python azure_backwards_reasoning_pipeline.py \
    --database /path/to/femr/database \
    --path_to_labels_dir /path/to/labels \
    --task_to_instructions /path/to/task_instructions.json \
    --max_examples 1000 \
    --output_file sft_traces.json \
    --sft_dataset_file sft_dataset.json
```

#### Option B: From Pre-generated JSONL
```bash
python azure_backwards_reasoning_pipeline.py \
    --data_dir /path/to/jsonl/data \
    --max_examples 1000 \
    --output_file sft_traces.json \
    --sft_dataset_file sft_dataset.json
```

### Step 2: Fine-tune Qwen Model

```bash
python qwen_sft_pipeline.py \
    --dataset_file sft_dataset.json \
    --output_dir ./qwen_sft_output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --use_lora \
    --use_quantization
```

### Using Helper Scripts

```bash
# Generate reasoning traces
./run_backwards_reasoning.sh

# Train the model
./run_sft_training.sh
```

## Configuration Options

### Azure Backwards Reasoning Pipeline

- `--database`: Path to FEMR database
- `--path_to_labels_dir`: Path to labels directory
- `--task_to_instructions`: Path to task instructions JSON file
- `--data_dir`: Path to pre-generated JSONL files
- `--max_examples`: Maximum number of examples to process
- `--patient_ids`: Specific patient IDs to process
- `--temperature`: Sampling temperature (default: 0.3)
- `--max_tokens`: Maximum tokens to generate (default: 4096)

### Qwen SFT Pipeline

- `--model_name`: Model name or path (default: "Qwen/Qwen2.5-7B-Instruct")
- `--output_dir`: Output directory for trained model
- `--dataset_file`: Path to SFT dataset JSON file
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per device (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--use_lora`: Enable LoRA fine-tuning (default: True)
- `--use_quantization`: Enable quantization (default: True)

## Output Format

### SFT Dataset Format

The generated SFT dataset follows this format:

```json
[
  {
    "conversations": [
      {
        "role": "user",
        "content": "Given a patient's electronic healthcare record..."
      },
      {
        "role": "assistant",
        "content": "Based on the patient's medical history, I can identify several key risk factors...\n\nPatient Prediction: Positive"
      }
    ],
    "patient_id": 12345,
    "label_time": "2024-01-01T00:00:00",
    "label_value": true,
    "task_instruction": "predict acute MI"
  }
]
```

## Hardware Requirements

### For Backwards Reasoning Generation
- CPU: Any modern CPU
- Memory: 8GB+ RAM
- Network: Stable internet connection for Azure OpenAI API

### For SFT Training
- GPU: NVIDIA GPU with 24GB+ VRAM (A100, H100, RTX 4090, etc.)
- CPU: Multi-core CPU recommended
- Memory: 32GB+ RAM
- Storage: 100GB+ free space for model weights and datasets

## Tips for Best Results

1. **Quality over Quantity**: Focus on generating high-quality reasoning traces rather than large quantities.

2. **Temperature Settings**: Use lower temperature (0.1-0.3) for backwards reasoning to get more consistent, high-quality explanations.

3. **LoRA Configuration**: Adjust LoRA rank and alpha based on your dataset size and complexity.

4. **Batch Size**: Use gradient accumulation to simulate larger batch sizes while staying within GPU memory limits.

5. **Learning Rate**: Start with 2e-4 and adjust based on training dynamics.

6. **Evaluation**: Monitor both training and validation loss to avoid overfitting.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or increase gradient accumulation steps
2. **Azure API Errors**: Check API key and endpoint configuration
3. **Slow Training**: Enable gradient checkpointing and use quantization
4. **Poor Quality**: Adjust temperature and max_tokens for reasoning generation

### Debugging

Enable detailed logging by setting:
```bash
export LOGURU_LEVEL=DEBUG
```

## Monitoring Training

### Weights & Biases
Enable wandb logging:
```bash
python qwen_sft_pipeline.py --use_wandb --dataset_file sft_dataset.json
```

### Local Logging
Training logs are saved to the output directory with detailed metrics.

## Model Evaluation

After training, you can evaluate the model by:

1. Loading the fine-tuned model
2. Running inference on test data
3. Comparing predictions with ground truth
4. Analyzing reasoning quality

## Contributing

When making changes to the pipeline:

1. Update configuration files as needed
2. Test with small datasets first
3. Update documentation
4. Add appropriate logging and error handling

## License

This code is part of the EHRshot project and follows the same licensing terms.
