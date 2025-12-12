# Direct Y/N Classification Module

This module provides scripts for training and evaluating Qwen3-8B models for direct binary classification (Positive/Negative) on EHR prediction tasks **without reasoning**.

## Key Differences from Reasoning-Based SFT (`sft_clean/`)

| Aspect | Reasoning-Based (`sft_clean/`) | Direct Y/N (`sft_clean_direct_yn/`) |
|--------|-------------------------------|-------------------------------------|
| System Prompt | Includes `<think>` tag instructions | Simple "Respond with Positive or Negative" |
| Assistant Response | `<think>...</think>\n\nFinal Answer: Positive/Negative` | Just `Positive` or `Negative` |
| Training Focus | Learn to reason AND predict | Learn to predict directly |
| Evaluation Method | Parse "Final Answer" from generated text | Compute normalized logit probabilities |
| Thinking Mode | Enabled (model produces reasoning) | Disabled via `enable_thinking=False` |

## Directory Structure

```
sft_clean_direct_yn/
├── README.md                         # This file
├── data/                             # Training data (direct Y/N format)
│   ├── train_all/
│   │   ├── acute_mi_sft_dataset.json
│   │   ├── hyperlipidemia_sft_dataset.json
│   │   ├── hypertension_sft_dataset.json
│   │   └── pancreatic_cancer_sft_dataset.json
│   └── val_small/
│       ├── acute_mi_balanced_sft_dataset.json
│       └── ... (other validation files)
├── generate_direct_yn_data.py        # Script to generate training data
├── qwen_sft_direct.py                # Training script
├── run_qwen_sft_direct.sh            # Shell script for parallel training
├── eval_vllm_direct.py               # Evaluation script (probability-based)
├── eval_vllm_compute_metrics.py      # Metrics computation (AUROC, AUPRC)
└── orchestrator_eval.py              # Python orchestrator for parallel evaluation
```

## Quick Start

### 1. Generate Training Data (already done)

If you need to regenerate the data:

```bash
cd /path/to/ehrshot-benchmark/ehrshot/sft_clean_direct_yn
python generate_direct_yn_data.py
```

### 2. Train Models

Train task-specific models on 4 GPUs in parallel:

```bash
./run_qwen_sft_direct.sh
```

This will:
- Train 4 separate models (one per task) using LoRA
- Save models to `output_direct_yn_qwen3-8b_r16/`
- Log to WandB under project `epochs3-direct-yn-ehrshot-qwen3-8b-r16`

### 3. Evaluate Models

Evaluate base model on all 4 tasks (using 4 GPUs in parallel):

```bash
python orchestrator_eval.py --mode base --gpus 0,1,2,3
```

Evaluate finetuned models on their respective tasks (4 models on 4 GPUs):

```bash
python orchestrator_eval.py --mode finetuned --gpus 0,1,2,3
```

Custom settings:

```bash
# Use specific GPUs
python orchestrator_eval.py --mode base --gpus 2,3

# Custom LoRA directory
python orchestrator_eval.py --mode finetuned --lora_base_dir /path/to/lora/models

# Skip metrics computation
python orchestrator_eval.py --mode base --skip_metrics
```

Results are saved to `eval_results/`.

## Data Format

Each training example has the following structure:

```json
{
    "conversations": [
        {
            "role": "system",
            "content": "You are a medical expert. You will be provided with a patient's Electronic Healthcare Record (EHR).\nYour task is to predict: Will the patient develop an acute myocardial infarction in the next year?\n\nRespond with exactly one word: Positive or Negative.\n- Positive: The patient WILL develop the condition.\n- Negative: The patient will NOT develop the condition."
        },
        {
            "role": "user",
            "content": "# Electronic Healthcare Record\n..."
        },
        {
            "role": "assistant",
            "content": "Negative"
        }
    ],
    "patient_id": 115973340,
    "label_time": "2018-11-29T23:59:00",
    "label_value": false,
    "task_instruction": "Will the patient develop an acute myocardial infarction in the next year?"
}
```

## Evaluation Details

The evaluation script computes probabilities differently from the reasoning-based approach:

### Reasoning-Based (original)
1. Generate multiple samples with temperature > 0
2. Parse "Final Answer: Positive/Negative" from each sample
3. Count valid predictions, compute majority vote

### Direct Y/N (this module)
1. Single forward pass (temperature=0)
2. Extract logits for "Positive" and "Negative" tokens
3. Compute normalized probability: `P(Positive) = softmax([logit_pos, logit_neg])[0]`

This approach:
- Is **deterministic** (no sampling variance)
- **Always valid** (no parsing failures)
- Provides **calibrated probabilities** for AUROC/AUPRC

## Qwen3 Thinking Mode

Qwen3 models have an optional "thinking mode" where they produce internal reasoning before responding. This module **disables thinking mode** in two ways:

1. **Training**: The training data contains no `<think>` tags, so the model learns to respond directly
2. **Inference**: We pass `enable_thinking=False` to `apply_chat_template()`

## Scripts Reference

### `generate_direct_yn_data.py`
Transforms reasoning-based SFT data to direct Y/N format.

```bash
python generate_direct_yn_data.py \
    --input_dir /path/to/sft_rt_yn \
    --output_dir /path/to/output
```

### `qwen_sft_direct.py`
Trains a model using LoRA on direct Y/N data.

```bash
python qwen_sft_direct.py \
    --train_file data/train_all/acute_mi_sft_dataset.json \
    --eval_file data/val_small/acute_mi_balanced_sft_dataset.json \
    --output_dir output/acute_mi \
    --model_name Qwen/Qwen3-8B \
    --lora_rank 16 \
    --wandb_project my-project \
    --wandb_run_name acute_mi
```

### `eval_vllm_direct.py`
Evaluates a model by computing normalized probabilities.

```bash
# Evaluate finetuned model
python eval_vllm_direct.py \
    --model_base Qwen/Qwen3-8B \
    --lora_path output/acute_mi \
    --data_path test_data.json \
    --task_name acute_mi \
    --output_dir results/acute_mi

# Evaluate base model
python eval_vllm_direct.py \
    --model_base Qwen/Qwen3-8B \
    --data_path test_data.json \
    --task_name acute_mi \
    --output_dir results/base_model/acute_mi
```

### `eval_vllm_compute_metrics.py`
Computes AUROC, AUPRC, and other metrics from predictions.

```bash
python eval_vllm_compute_metrics.py --eval_root results/
```

## Output Format

The evaluation produces files compatible with the original metrics calculator:

- `predictions.csv`: Patient-level predictions with `probability_score`
- `detailed_logs.json`: Full details including logprobs
- `run_stats.json`: Run metadata
- `summary.json`: Computed metrics (after running `eval_vllm_compute_metrics.py`)

## Dependencies

Same as `sft_clean/`:
- PyTorch
- Transformers
- vLLM
- Unsloth
- scikit-learn
- pandas
- numpy
