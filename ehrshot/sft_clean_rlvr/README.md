# RLVR: Reinforcement Learning with Verifiable Rewards for Clinical Reasoning

This module implements **GRPO (Group Relative Policy Optimization)** for clinical outcome prediction, building on SFT-trained Qwen3-8B models.

## Overview

**Research Question:** Does RLVR improve clinical outcome prediction accuracy beyond SFT alone, when SFT is trained on teacher-generated reasoning traces with correct answers?

### Method

```
Phase 1: SFT (Already Done)
├── Input: Patient EHR + GPT-5 reasoning traces (always correct)
├── Training: Next-token prediction on reasoning + answer
└── Output: Model that imitates GPT-5's reasoning style

Phase 2: RLVR (This Module)
├── Input: Patient EHR (no GPT-5 traces)
├── Training: GRPO with verifiable correctness rewards
├── Reward: +1 correct, -1 wrong, 0 malformed
└── Output: Model optimized for correctness, not imitation
```

### Key Differences from SFT

| Aspect | SFT | RLVR |
|--------|-----|------|
| Training signal | Imitate GPT-5 tokens | Correctness reward |
| Generation | Teacher-forced | Model's own samples |
| Exploration | None (single path) | Multiple paths (10 samples) |
| Objective | Cross-entropy loss | Policy gradient |

## Directory Structure

```
sft_clean_rlvr/
├── README.md                    # This file
├── common/                      # Shared utilities
│   ├── __init__.py
│   ├── constants.py             # Tasks, paths, hyperparameters
│   ├── reward_functions.py      # Verifiable reward computation
│   └── prompt_formatting.py     # Data loading and formatting
├── phase1_single_gpu/           # Validation (single GPU)
│   ├── train_grpo_single.py     # Training script
│   └── run_phase1_validation.sh # Launch script
├── phase2_multi_gpu/            # Production (4 GPUs, DDP)
│   ├── train_grpo_ddp.py        # DDP training script
│   └── run_phase2_training.sh   # Launch script
├── evaluation/                  # Model evaluation
│   └── run_eval_rlvr.sh         # Evaluate on all tasks
├── configs/                     # Configuration files
│   ├── grpo_phase1.yaml
│   └── grpo_phase2.yaml
└── outputs/                     # Created at runtime
    ├── phase1_validation/
    ├── phase2_aggregated/
    └── eval_results/
```

## Quick Start

### Prerequisites

1. **SFT checkpoint** - You need a trained SFT model to start RLVR from:
   ```
   sft_clean/finetuned_models/reasoning_originalEHR_train_orig_aggregated/
   ```

2. **Dependencies** - Add TRL if not already installed:
   ```bash
   pip install trl>=0.9.0
   ```

3. **Data** - Ensure data is available at `/dev/shm/ehrshot-data/`

### Phase 1: Validation (Recommended First Step)

Validate the pipeline works on a single task before scaling up:

```bash
cd phase1_single_gpu

# Quick test with defaults (acute_mi on GPU 0)
./run_phase1_validation.sh

# Specific task and GPU
./run_phase1_validation.sh hypertension 2

# With custom SFT checkpoint
./run_phase1_validation.sh acute_mi 0 /path/to/sft/checkpoint
```

**Expected output:**
- Model saved to `outputs/phase1_validation/acute_mi/`
- Training stats in `training_stats.json`
- Logs in `outputs/phase1_validation/acute_mi.log`

**Check if it worked:**
```bash
cat outputs/phase1_validation/acute_mi/training_stats.json
```

You should see accuracy > 50% (random baseline) if training is working.

### Phase 2: Full Training (4 GPUs)

After validating Phase 1, run full training:

```bash
cd phase2_multi_gpu

# Full training with all defaults
./run_phase2_training.sh

# Custom SFT checkpoint
./run_phase2_training.sh /path/to/sft/aggregated/checkpoint

# Adjust resources
./run_phase2_training.sh "" 2 500  # 2 GPUs, 500 steps
```

**Arguments:**
| Position | Name | Default | Description |
|----------|------|---------|-------------|
| 1 | SFT_CHECKPOINT | Auto-detect | Path to SFT LoRA adapter |
| 2 | NUM_GPUS | 4 | Number of GPUs (1-4) |
| 3 | MAX_STEPS | 1000 | Training steps |

**Expected output:**
- Model saved to `outputs/phase2_aggregated/`
- Training time: ~1-2 hours

### Evaluation

Evaluate the RLVR model and compare with SFT:

```bash
cd evaluation

# Evaluate RLVR model
./run_eval_rlvr.sh ../outputs/phase2_aggregated

# With custom output name
./run_eval_rlvr.sh ../outputs/phase2_aggregated originalEHR rlvr_v1
```

**Output:**
```
Results:
  acute_mi: AUROC=0.XXXX, AUPRC=0.XXXX
  hyperlipidemia: AUROC=0.XXXX, AUPRC=0.XXXX
  hypertension: AUROC=0.XXXX, AUPRC=0.XXXX
  pancreatic_cancer: AUROC=0.XXXX, AUPRC=0.XXXX
```

## Configuration Reference

### GRPO Hyperparameters

| Parameter | Phase 1 | Phase 2 | Description |
|-----------|---------|---------|-------------|
| `num_generations` | 10 | 10 | Samples per prompt |
| `max_completion_length` | 2048 | 2048 | Max response tokens |
| `temperature` | 0.7 | 0.7 | Sampling temperature |
| `learning_rate` | 5e-6 | 5e-6 | 20x lower than SFT |
| `beta` | 0.05 | 0.05 | KL penalty coefficient |
| `per_device_batch_size` | 1 | 2 | Prompts per GPU |
| `gradient_accumulation` | 4 | 2 | Steps to accumulate |
| `max_steps` | 500 | 1000 | Training steps |

### Effective Batch Sizes

| Phase | Batch × Accum × GPUs | Prompts/Step | Completions/Step |
|-------|----------------------|--------------|------------------|
| Phase 1 | 1 × 4 × 1 | 4 | 40 |
| Phase 2 | 2 × 2 × 4 | 16 | 160 |

### Memory Requirements

| Configuration | Memory per GPU | Fits A100 80GB? |
|---------------|----------------|-----------------|
| Phase 1 (batch=1, gen=10) | ~50-60 GB | Yes |
| Phase 2 (batch=2, gen=10) | ~60-70 GB | Yes |

## Reward Function

The reward is **verifiable** - computed by comparing the model's final answer to ground truth:

```python
def compute_reward(response: str, ground_truth: bool) -> float:
    parsed = parse_final_answer(response)  # 1=Positive, 0=Negative, -1=Invalid
    
    if parsed == -1:
        return 0.0   # Malformed - neutral reward
    
    correct = (parsed == 1) == ground_truth
    return 1.0 if correct else -1.0
```

**Parsing logic:**
- Looks for `"Final Answer: Positive"` or `"Final Answer: Negative"`
- Case-insensitive, handles bracket variations `[Positive]`
- Falls back to checking last few lines if explicit pattern not found

## Troubleshooting

### OOM (Out of Memory)

Reduce memory usage:
```bash
# In shell script, modify:
PER_DEVICE_BATCH_SIZE=1
NUM_GENERATIONS=8  # Instead of 10
```

Or enable more aggressive gradient checkpointing in the config.

### All Rewards are Zero

This means all responses are malformed (can't parse Final Answer):
1. Check that SFT model produces proper format
2. Inspect sample completions in logs
3. May need to increase `max_completion_length`

### Training Loss Not Decreasing

GRPO loss can be noisy. Check:
1. `training_stats.json` - Is accuracy improving?
2. Increase `beta` if model diverging from SFT
3. Decrease `learning_rate` for more stable training

### DDP Hangs

If multi-GPU training hangs:
```bash
# Kill any zombie processes
pkill -f train_grpo_ddp

# Check GPU status
nvidia-smi

# Try with fewer GPUs
./run_phase2_training.sh "" 2  # Use 2 GPUs
```

## Comparison with SFT

After training, compare results:

| Model | acute_mi | hyperlipidemia | hypertension | panc_cancer | Mean |
|-------|----------|----------------|--------------|-------------|------|
| SFT | X.XXX | X.XXX | X.XXX | X.XXX | X.XXX |
| SFT+RLVR | X.XXX | X.XXX | X.XXX | X.XXX | X.XXX |
| **Delta** | ±X.XXX | ±X.XXX | ±X.XXX | ±X.XXX | ±X.XXX |

For publication, include:
- Bootstrap confidence intervals (95%)
- Statistical significance test (paired t-test or Wilcoxon)

## Files Reference

### Common Utilities

| File | Purpose |
|------|---------|
| `common/constants.py` | Task definitions, paths, default configs |
| `common/reward_functions.py` | Verifiable reward computation |
| `common/prompt_formatting.py` | Data loading and GRPO formatting |

### Phase 1 Scripts

| File | Purpose |
|------|---------|
| `phase1_single_gpu/train_grpo_single.py` | Single-GPU GRPO training |
| `phase1_single_gpu/run_phase1_validation.sh` | Launch script with defaults |

### Phase 2 Scripts

| File | Purpose |
|------|---------|
| `phase2_multi_gpu/train_grpo_ddp.py` | Multi-GPU DDP training |
| `phase2_multi_gpu/run_phase2_training.sh` | Launch script with torchrun |

### Evaluation

| File | Purpose |
|------|---------|
| `evaluation/run_eval_rlvr.sh` | Evaluate on all 4 tasks in parallel |

## Citation

If you use this code, please cite:
- GRPO: [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2401.XXXXX)
- TRL Library: [Hugging Face TRL](https://github.com/huggingface/trl)

