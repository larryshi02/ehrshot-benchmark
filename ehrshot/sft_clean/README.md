# Reasoning-Based Clinical Outcome Prediction Module

This module provides scripts for training and evaluating Qwen3-8B models for clinical outcome prediction tasks **with reasoning** (chain-of-thought). The model generates detailed clinical reasoning in `<think>` tags followed by a final answer.

## Tasks

The module supports 4 clinical prediction tasks:
- `acute_mi` - Acute myocardial infarction prediction
- `hyperlipidemia` - Hyperlipidemia prediction
- `hypertension` - Hypertension prediction
- `pancreatic_cancer` - Pancreatic cancer prediction

## Comparison with Direct Experiments

This module is designed for fair comparison with the `sft_clean_direct/` experiments:

| Setting | Direct (sft_clean_direct) | Reasoning (sft_clean) |
|---------|---------------------------|------------------------|
| Response Format | Single word: "Positive"/"Negative" | `<think>reasoning</think>` + "Final Answer: Positive/Negative" |
| Evaluation Method | Logit-based (extract log-probs) | Sampling-based (10 samples, parse answers) |
| Qwen3 Thinking | Disabled (`enable_thinking=False`) | Enabled (default) |
| **LoRA Rank** | **16** | **16** ✓ |
| **LoRA Alpha** | **32** | **32** ✓ |
| **Learning Rate** | **1e-4** | **1e-4** ✓ |
| **Epochs** | **3** | **3** ✓ |
| **Early Stopping Patience** | **3** | **3** ✓ |
| **Eval Steps** | **50** | **50** ✓ |
| **Batch Size (Train)** | **12** | **12** ✓ |

## Data Paths

### Training Data

| Representation | Path |
|----------------|------|
| originalEHR | `/dev/shm/ehrshot-data/data_gpt-5-mini_sft_reasoning_originalEHR/` |
| rubricified | `/dev/shm/ehrshot-data/data_gpt-5-mini_sft_reasoning_rubricified/` |

Each contains:
- `train_orig/{task}_sft_dataset.json` - Original training split
- `train_all/{task}_sft_dataset.json` - Full training data (augmented)
- `val_small/{task}_sft_dataset.json` - Validation data

### Test/Evaluation Data

| Representation | Path |
|----------------|------|
| originalEHR | `/dev/shm/ehrshot-data/serialized_multi_task_data/` |
| rubricified | `/dev/shm/ehrshot-data/serialized_multi_task_data_rubricified/` |

## Directory Structure

```
sft_clean/
├── qwen_sft_reasoning.py        # Training script (Unsloth + LoRA)
├── merge_datasets.py            # Merge task datasets for aggregated training
├── eval_vllm_reasoning.py       # Evaluation worker (vLLM, sampling-based)
├── eval_compute_metrics.py      # Metrics computation (AUROC, AUPRC)
├── orchestrator_eval.py         # Parallel evaluation orchestrator
├── run_base_eval.sh             # Step 1: Evaluate base model
├── run_finetune_aggregated.sh   # Step 2: Train aggregated model
├── run_finetune_per_task.sh     # Step 4: Train per-task models
├── run_eval_finetuned.sh        # Evaluate finetuned models
├── run_all_experiments.sh       # Master script for all experiments
└── README.md                    # This file
```

## Experiments

### Overview

| Step | Description | Script |
|------|-------------|--------|
| 1 | Evaluate base Qwen3-8B on all 4 tasks | `run_base_eval.sh` |
| 2 | Fine-tune aggregated model (all tasks combined) | `run_finetune_aggregated.sh` |
| 3 | Evaluate aggregated model on all 4 tasks | `run_eval_finetuned.sh aggregated` |
| 4 | Fine-tune 4 per-task models (parallel) | `run_finetune_per_task.sh` |
| 5 | Evaluate per-task models (diagonal + full matrix) | `run_eval_finetuned.sh per_task_*` |

### Run All Experiments

```bash
cd /path/to/ehrshot-benchmark/ehrshot/sft_clean

# Run all experiments with default settings (originalEHR + train_orig)
./run_all_experiments.sh

# Run with specific settings
./run_all_experiments.sh originalEHR train_orig    # Explicit default
./run_all_experiments.sh rubricified train_orig    # Rubricified representation
./run_all_experiments.sh originalEHR train_all     # Use larger training set

# Skip specific steps
./run_all_experiments.sh --skip-base               # Skip base model evaluation
./run_all_experiments.sh --skip-aggregated         # Skip aggregated model
./run_all_experiments.sh --skip-matrix             # Skip full 4x4 matrix evaluation
```

### Run Individual Steps

#### Step 1: Base Model Evaluation

```bash
./run_base_eval.sh                    # Default: originalEHR
./run_base_eval.sh originalEHR        # Explicit
./run_base_eval.sh rubricified        # Rubricified data
```

#### Step 2: Aggregated Model Training

```bash
./run_finetune_aggregated.sh                           # Default settings
./run_finetune_aggregated.sh originalEHR train_orig    # Explicit
./run_finetune_aggregated.sh rubricified train_orig    # Rubricified
```

#### Step 3: Aggregated Model Evaluation

```bash
./run_eval_finetuned.sh aggregated originalEHR train_orig
```

#### Step 4: Per-Task Model Training

```bash
./run_finetune_per_task.sh                           # Default settings
./run_finetune_per_task.sh originalEHR train_orig    # Explicit
./run_finetune_per_task.sh rubricified train_orig    # Rubricified
```

#### Step 4b: Per-Task Diagonal Evaluation

```bash
./run_eval_finetuned.sh per_task_diagonal originalEHR train_orig
```

#### Step 5: Per-Task Full Matrix Evaluation

```bash
./run_eval_finetuned.sh per_task_matrix originalEHR train_orig
```

## Output Structure

### Fine-tuned Models

```
finetuned_models/
├── reasoning_originalEHR_train_orig_aggregated/   # Step 2 output
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
└── reasoning_originalEHR_train_orig/              # Step 4 output
    ├── acute_mi/
    ├── hyperlipidemia/
    ├── hypertension/
    └── pancreatic_cancer/
```

### Evaluation Results

```
eval_results/
├── reasoning_originalEHR_base/                     # Step 1 output
│   ├── acute_mi/
│   │   ├── predictions.csv
│   │   ├── detailed_logs.json
│   │   ├── run_stats.json
│   │   └── summary.json
│   ├── hyperlipidemia/
│   ├── hypertension/
│   ├── pancreatic_cancer/
│   └── metrics_summary.csv
├── reasoning_originalEHR_train_orig_aggregated/    # Step 3 output
│   └── (same structure)
├── reasoning_originalEHR_train_orig_per_task/      # Step 4b output
│   └── (same structure)
└── reasoning_originalEHR_train_orig_per_task_matrix/  # Step 5 output
    ├── acute_mi/                    # Model trained on acute_mi
    │   ├── acute_mi/                # Evaluated on acute_mi (diagonal)
    │   ├── hyperlipidemia/          # Cross-task evaluation
    │   ├── hypertension/
    │   └── pancreatic_cancer/
    ├── hyperlipidemia/
    ├── hypertension/
    ├── pancreatic_cancer/
    ├── final_auroc_matrix.csv       # 4x4 AUROC table
    └── final_auprc_matrix.csv       # 4x4 AUPRC table
```

## Evaluation Method

Unlike the direct experiments (logit-based), reasoning experiments use **sampling-based evaluation**:

1. Generate N=10 responses per test example
2. Parse "Final Answer: Positive/Negative" from each response
3. Compute probability: `P(Positive) = count(Positive) / count(valid_responses)`

Sampling parameters:
- `temperature=0.7`
- `top_p=0.9`
- `max_tokens=4096`

## Multi-GPU Training Note

**Q: Why not use multi-GPU data parallelism for aggregated training?**

A: Qwen3-8B with LoRA (r=16) fits comfortably on a single A100 80GB GPU. Data parallelism would require:
- Loading model copies on each GPU
- Gradient synchronization overhead
- Added complexity for marginal benefit

For LoRA fine-tuning with moderate dataset sizes, single GPU is simpler and sufficient. Multi-GPU training (DeepSpeed ZeRO, FSDP) is more beneficial for:
- Full fine-tuning (all parameters)
- Larger models (70B+)
- Massive datasets

## Output Files

Each evaluation produces:
- `predictions.csv` - Patient-level predictions with `probability_score`
- `detailed_logs.json` - Full details including all sampled responses
- `run_stats.json` - Run metadata (invalid sample counts, etc.)
- `summary.json` - Computed metrics (AUROC, AUPRC with CIs)

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

## Environment Variables

- `CONDA_ENV` - Conda environment name (default: `EHRSHOT_ENV`)

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| `model_name` | Qwen/Qwen3-8B |
| `max_length` | 10240 |
| `per_device_train_batch_size` | 12 |
| `per_device_eval_batch_size` | 1 |
| `learning_rate` | 1e-4 |
| `num_train_epochs` | 3 |
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `early_stopping_patience` | 3 |
| `eval_steps` | 50 |
| `save_steps` | 50 |
