# Direct Y/N Classification Module

This module provides scripts for training and evaluating Qwen3-8B models for **direct binary classification** (Positive/Negative) on EHR-based clinical outcome prediction tasks **without reasoning**.

## Tasks

The module supports 4 clinical prediction tasks:
- `acute_mi` - Acute myocardial infarction prediction
- `hyperlipidemia` - Hyperlipidemia prediction
- `hypertension` - Hypertension prediction
- `pancreatic_cancer` - Pancreatic cancer prediction

## Data Settings

Two input representations are supported:
1. **Original EHR** - Standard text-serialized EHR data
2. **Rubricified** - Structured rubric format of EHR data

## Training/Evaluation Data Alignment

**CRITICAL**: The training and evaluation scripts are designed to produce identical prompt formats:

### Message Format

| Component | Content |
|-----------|---------|
| **System** | `You are a medical expert specializing in clinical risk prediction.` |
| **User** | Task instruction + `--- Patient EHR ---` + EHR content + `--- End of EHR ---` + Response format instructions |
| **Assistant** | `Positive` or `Negative` (single token) |

### User Message Template

```
Based on the patient's Electronic Healthcare Record below, predict: {task_query}

--- Patient EHR ---

{EHR_CONTENT}

--- End of EHR ---

Respond with exactly one word: Positive or Negative.
- Positive: The patient WILL develop the condition.
- Negative: The patient will NOT develop the condition.
```

### Qwen3 Thinking Mode

Both training and evaluation scripts disable Qwen3's thinking mode via `enable_thinking=False` to ensure:
- Training data contains no `<think>` tags
- Model learns to respond directly without reasoning
- Inference produces single-token responses

## Directory Structure

```
sft_clean_direct/
├── qwen_sft_direct.py          # Training script (Unsloth + LoRA)
├── eval_vllm_direct.py         # Evaluation worker (vLLM, logit-based)
├── eval_vllm_compute_metrics.py # Metrics computation (AUROC, AUPRC)
├── orchestrator_eval.py        # Parallel evaluation orchestrator
├── run_finetune_direct.sh      # Fine-tuning wrapper (4 tasks in parallel)
├── run_eval_direct.sh          # Evaluation wrapper
├── run_all_experiments.sh      # Master script for all experiments
└── README.md                   # This file
```

### Output Directories (after running experiments)

```
sft_clean_direct/
├── finetuned_models/
│   ├── originalEHR_train_all/      # Models trained on original EHR (full training set)
│   │   ├── acute_mi/
│   │   ├── hyperlipidemia/
│   │   ├── hypertension/
│   │   └── pancreatic_cancer/
│   ├── originalEHR_train_orig/     # Models trained on original EHR (original split)
│   ├── rubricified_train_all/      # Models trained on rubricified EHR (full training set)
│   └── rubricified_train_orig/     # Models trained on rubricified EHR (original split)
└── eval_results/
    ├── originalEHR_base/                  # Base model on original EHR
    │   ├── acute_mi/
    │   │   ├── predictions.csv            # Per-patient predictions
    │   │   ├── detailed_logs.json         # Detailed inference logs
    │   │   ├── run_stats.json             # Run metadata
    │   │   └── summary.json               # Metrics (after compute_metrics)
    │   ├── hyperlipidemia/
    │   ├── hypertension/
    │   └── pancreatic_cancer/
    ├── rubricified_base/                  # Base model on rubricified
    ├── originalEHR_train_all_finetuned/   # Finetuned on original EHR (train_all)
    ├── originalEHR_train_orig_finetuned/  # Finetuned on original EHR (train_orig)
    ├── rubricified_train_all_finetuned/   # Finetuned on rubricified (train_all)
    └── rubricified_train_orig_finetuned/  # Finetuned on rubricified (train_orig)
```

**Note:** Each eval_results subdirectory contains `{task}/predictions.csv` files which are read by `eval_vllm_compute_metrics.py`.

## Data Locations

### Training Data
- **Original EHR**: `/dev/shm/ehrshot-data/data_gpt-5-mini_sft_direct_originalEHR/`
- **Rubricified**: `/dev/shm/ehrshot-data/data_gpt-5-mini_sft_direct_rubricified/`

Each contains:
- `train_all/{task}_sft_dataset.json` - Full training data (augmented)
- `train_orig/{task}_sft_dataset.json` - Original training split (smaller)
- `val_small/{task}_sft_dataset.json` - Validation data

### Test/Evaluation Data
- **Original EHR**: `/dev/shm/ehrshot-data/serialized_multi_task_data/`
- **Rubricified**: `/dev/shm/ehrshot-data/serialized_multi_task_data_rubricified/`

Each contains `{task}_all_splits.json` with `split` field (`train`/`val`/`test`).

## Quick Start

### Run All 6 Experiments

```bash
cd /path/to/ehrshot-benchmark/ehrshot/sft_clean_direct
./run_all_experiments.sh
```

This executes in order:
1. Evaluate base Qwen3-8B on original EHR (4 tasks, 4 GPUs)
2. Evaluate base Qwen3-8B on rubricified EHR (4 tasks, 4 GPUs)
3. Fine-tune on original EHR (train_all) → Evaluate (4 tasks, 4 GPUs)
4. Fine-tune on rubricified EHR (train_all) → Evaluate (4 tasks, 4 GPUs)
5. Fine-tune on original EHR (train_orig) → Evaluate (4 tasks, 4 GPUs)
6. Fine-tune on rubricified EHR (train_orig) → Evaluate (4 tasks, 4 GPUs)

### Run Individual Steps

#### Fine-tuning Only

```bash
# Using train_all (full training set, default)
./run_finetune_direct.sh original              # Fine-tune on original EHR (train_all)
./run_finetune_direct.sh original train_all    # Explicit train_all
./run_finetune_direct.sh rubricified train_all # Fine-tune on rubricified EHR (train_all)

# Using train_orig (original/smaller training split)
./run_finetune_direct.sh original train_orig   # Fine-tune on original EHR (train_orig)
./run_finetune_direct.sh rubricified train_orig # Fine-tune on rubricified EHR (train_orig)
```

#### Evaluation Only

```bash
# Base model evaluation
./run_eval_direct.sh base original        # Base model on original EHR
./run_eval_direct.sh base rubricified     # Base model on rubricified

# Finetuned model evaluation (specify train split for finetuned mode)
./run_eval_direct.sh finetuned original train_all   # Finetuned on train_all
./run_eval_direct.sh finetuned original train_orig  # Finetuned on train_orig
./run_eval_direct.sh finetuned rubricified train_all
./run_eval_direct.sh finetuned rubricified train_orig
```

### Python Scripts Directly

#### Training

```bash
CUDA_VISIBLE_DEVICES=0 python qwen_sft_direct.py \
    --train_file /path/to/train.json \
    --eval_file /path/to/val.json \
    --output_dir /path/to/output \
    --model_name Qwen/Qwen3-8B \
    --lora_rank 16 \
    --wandb_project my-project \
    --wandb_run_name acute_mi
```

#### Evaluation

```bash
# Evaluate finetuned model
CUDA_VISIBLE_DEVICES=0 python eval_vllm_direct.py \
    --model_base Qwen/Qwen3-8B \
    --lora_path /path/to/lora/adapter \
    --data_path /path/to/test_data.json \
    --task_name acute_mi \
    --output_dir /path/to/output

# Evaluate base model (omit --lora_path)
CUDA_VISIBLE_DEVICES=0 python eval_vllm_direct.py \
    --model_base Qwen/Qwen3-8B \
    --data_path /path/to/test_data.json \
    --task_name acute_mi \
    --output_dir /path/to/output
```

#### Compute Metrics

```bash
python eval_vllm_compute_metrics.py --eval_root /path/to/eval_results
```

## Evaluation Method

Unlike sampling-based evaluation, this module uses **logit-based probability computation**:

1. Single forward pass (temperature=0, deterministic)
2. Extract logits for "Positive" and "Negative" tokens
3. Compute normalized probability: `P(Positive) = softmax([logit_pos, logit_neg])[0]`

This approach:
- Is **deterministic** (no sampling variance)
- **Always valid** (no parsing failures)
- Provides **calibrated probabilities** for AUROC/AUPRC

## Output Files

Each evaluation produces:
- `predictions.csv` - Patient-level predictions with `probability_score`
- `detailed_logs.json` - Full details including logprobs
- `run_stats.json` - Run metadata
- `summary.json` - Computed metrics (after `eval_vllm_compute_metrics.py`)

## Dependencies

- PyTorch
- Transformers
- vLLM
- Unsloth
- scikit-learn
- pandas
- numpy
- loguru
- wandb

## Configuration

### Environment Variables

- `CONDA_ENV` - Conda environment name (default: `EHRSHOT_ENV`)

### Training Hyperparameters (qwen_sft_direct.py)

| Parameter | Default |
|-----------|---------|
| `max_length` | 10240 |
| `per_device_train_batch_size` | 16 |
| `learning_rate` | 1e-4 |
| `num_train_epochs` | 3 |
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `early_stopping_patience` | 5 |
