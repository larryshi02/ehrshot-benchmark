#!/usr/bin/env python3
"""
Phase 1: Single-GPU GRPO Training for Validation

This script trains a single GRPO model on one task for validation purposes.
Use this to verify the RLVR pipeline works correctly before scaling to multi-GPU.

Usage:
    python train_grpo_single.py \\
        --task acute_mi \\
        --sft_checkpoint /path/to/sft/checkpoint \\
        --output_dir ./outputs/phase1_validation/acute_mi

Key Features:
    - Single GPU training (uses CUDA_VISIBLE_DEVICES)
    - Starts from SFT checkpoint
    - Balanced class sampling
    - Verifiable reward based on final answer correctness
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
import numpy as np
from datasets import Dataset
from transformers import TrainingArguments
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from loguru import logger

# Add parent directory to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.constants import TASKS, MODEL_CONFIG, GRPO_CONFIG, DATA_PATHS
from common.reward_functions import parse_final_answer, compute_reward
from common.prompt_formatting import load_sft_data_as_grpo_format, create_ground_truth_map


# =============================================================================
# REWARD FUNCTION FOR TRL
# =============================================================================

class RewardComputer:
    """
    Reward computer that maintains ground truth mapping.
    
    TRL's GRPOTrainer calls the reward function with completions and expects
    to get back rewards. We need to track which prompt each completion belongs
    to so we can look up the ground truth.
    """
    
    def __init__(self, dataset: List[Dict]):
        """
        Initialize with dataset containing ground truth labels.
        
        Args:
            dataset: List of GRPO-formatted examples with prompt_id and ground_truth
        """
        self.ground_truth_map = {ex["prompt_id"]: ex["ground_truth"] for ex in dataset}
        self.prompt_id_list = [ex["prompt_id"] for ex in dataset]
        
        # Stats tracking
        self.total_calls = 0
        self.correct_count = 0
        self.incorrect_count = 0
        self.malformed_count = 0
    
    def __call__(
        self,
        completions: List[str],
        prompts: List[str] = None,
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for a batch of completions.
        
        Note: TRL passes prompts as the formatted prompt strings. We need to
        match these back to our prompt_ids. For simplicity, we'll use the
        order from the dataset.
        """
        self.total_calls += 1
        
        rewards = []
        
        # Get metadata from kwargs if available
        prompt_ids = kwargs.get("prompt_id", None)
        ground_truths = kwargs.get("ground_truth", None)
        
        if ground_truths is not None:
            # Direct ground truth provided
            for completion, gt in zip(completions, ground_truths):
                reward = compute_reward(completion, gt)
                rewards.append(reward)
                self._update_stats(reward)
        elif prompt_ids is not None:
            # Look up ground truth by prompt_id
            for completion, pid in zip(completions, prompt_ids):
                gt = self.ground_truth_map.get(pid)
                if gt is None:
                    logger.warning(f"Unknown prompt_id: {pid}, using neutral reward")
                    reward = 0.0
                else:
                    reward = compute_reward(completion, gt)
                rewards.append(reward)
                self._update_stats(reward)
        else:
            # Fallback: can't compute meaningful rewards
            logger.warning("No ground_truth or prompt_id provided, using neutral rewards")
            rewards = [0.0] * len(completions)
        
        return rewards
    
    def _update_stats(self, reward: float):
        if reward > 0:
            self.correct_count += 1
        elif reward < 0:
            self.incorrect_count += 1
        else:
            self.malformed_count += 1
    
    def get_stats(self) -> Dict:
        total = self.correct_count + self.incorrect_count + self.malformed_count
        return {
            "total_completions": total,
            "correct": self.correct_count,
            "incorrect": self.incorrect_count,
            "malformed": self.malformed_count,
            "accuracy": self.correct_count / max(total, 1),
        }


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_grpo_dataset(examples: List[Dict], tokenizer) -> Dataset:
    """
    Convert GRPO examples to HuggingFace Dataset format.
    
    GRPO expects a dataset with 'prompt' field containing the formatted prompt string.
    """
    
    formatted_examples = []
    
    for ex in examples:
        # Apply chat template to get formatted prompt string
        prompt_str = tokenizer.apply_chat_template(
            ex["prompt"],
            tokenize=False,
            add_generation_prompt=True
        )
        
        formatted_examples.append({
            "prompt": prompt_str,
            "prompt_id": ex["prompt_id"],
            "ground_truth": ex["ground_truth"],
            "task": ex["task"],
        })
    
    return Dataset.from_list(formatted_examples)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Single-GPU GRPO Validation")
    
    # Required arguments
    parser.add_argument("--task", type=str, required=True, choices=TASKS,
                        help="Task to train on")
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="Path to SFT checkpoint (LoRA adapter)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for RLVR model")
    
    # Optional arguments
    parser.add_argument("--split", type=str, default="train_orig",
                        choices=["train_orig", "train_all"],
                        help="Training data split (default: train_orig)")
    parser.add_argument("--num_generations", type=int, default=GRPO_CONFIG.num_generations,
                        help=f"Samples per prompt (default: {GRPO_CONFIG.num_generations})")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum training steps (default: 500)")
    parser.add_argument("--learning_rate", type=float, default=GRPO_CONFIG.learning_rate,
                        help=f"Learning rate (default: {GRPO_CONFIG.learning_rate})")
    parser.add_argument("--beta", type=float, default=GRPO_CONFIG.beta,
                        help=f"KL penalty coefficient (default: {GRPO_CONFIG.beta})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--wandb_project", type=str, default="ehrshot-rlvr-phase1",
                        help="WandB project name")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Phase 1: Single-GPU GRPO Validation")
    logger.info("=" * 60)
    logger.info(f"Task: {args.task}")
    logger.info(f"SFT Checkpoint: {args.sft_checkpoint}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Num Generations: {args.num_generations}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Beta (KL): {args.beta}")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. Load Model from SFT Checkpoint
    # -------------------------------------------------------------------------
    logger.info("\n[1/4] Loading model from SFT checkpoint...")
    
    from unsloth import FastLanguageModel
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG.model_name,
        max_seq_length=MODEL_CONFIG.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map=None,  # Let trainer handle device
    )
    
    # Load LoRA adapter from SFT checkpoint
    logger.info(f"Loading LoRA adapter from: {args.sft_checkpoint}")
    model = PeftModel.from_pretrained(model, args.sft_checkpoint)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Model loaded: {MODEL_CONFIG.model_name}")
    logger.info(f"LoRA rank: {MODEL_CONFIG.lora_r}")
    
    # -------------------------------------------------------------------------
    # 2. Prepare Dataset
    # -------------------------------------------------------------------------
    logger.info("\n[2/4] Preparing dataset...")
    
    examples = load_sft_data_as_grpo_format(
        task_name=args.task,
        split=args.split,
        balance_classes=True,
        seed=args.seed
    )
    
    logger.info(f"Loaded {len(examples)} balanced examples")
    positives = sum(1 for ex in examples if ex["ground_truth"])
    logger.info(f"  Positives: {positives}, Negatives: {len(examples) - positives}")
    
    # Convert to HuggingFace Dataset
    dataset = prepare_grpo_dataset(examples, tokenizer)
    logger.info(f"Dataset prepared with {len(dataset)} examples")
    
    # -------------------------------------------------------------------------
    # 3. Setup GRPO Trainer
    # -------------------------------------------------------------------------
    logger.info("\n[3/4] Setting up GRPO trainer...")
    
    # Create reward computer
    reward_computer = RewardComputer(examples)
    
    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        
        # Generation settings
        num_generations=args.num_generations,
        max_completion_length=GRPO_CONFIG.max_completion_length,
        temperature=GRPO_CONFIG.temperature,
        top_p=GRPO_CONFIG.top_p,
        
        # Training settings
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_steps=args.max_steps,
        
        # Batch settings (single GPU)
        per_device_train_batch_size=GRPO_CONFIG.per_device_train_batch_size,
        gradient_accumulation_steps=GRPO_CONFIG.gradient_accumulation_steps,
        
        # Optimization
        warmup_ratio=GRPO_CONFIG.warmup_ratio,
        weight_decay=GRPO_CONFIG.weight_decay,
        bf16=GRPO_CONFIG.bf16,
        gradient_checkpointing=GRPO_CONFIG.gradient_checkpointing,
        
        # Logging
        logging_steps=GRPO_CONFIG.logging_steps,
        save_steps=GRPO_CONFIG.save_steps,
        report_to="wandb",
        run_name=f"rlvr_phase1_{args.task}",
        
        # Misc
        seed=args.seed,
        remove_unused_columns=False,  # Keep our custom columns
    )
    
    # Set WandB project
    os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_computer,
    )
    
    logger.info("GRPO Trainer configured")
    logger.info(f"  Effective batch size: {grpo_config.per_device_train_batch_size} × {grpo_config.gradient_accumulation_steps} = {grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps}")
    logger.info(f"  Total completions per step: {grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps * args.num_generations}")
    
    # -------------------------------------------------------------------------
    # 4. Train
    # -------------------------------------------------------------------------
    logger.info("\n[4/4] Starting GRPO training...")
    
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Save final model
    logger.info(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training stats
    stats = reward_computer.get_stats()
    stats_path = os.path.join(args.output_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Statistics")
    logger.info("=" * 60)
    logger.info(f"Total completions: {stats['total_completions']}")
    logger.info(f"Correct: {stats['correct']} ({stats['accuracy']*100:.1f}%)")
    logger.info(f"Incorrect: {stats['incorrect']}")
    logger.info(f"Malformed: {stats['malformed']}")
    logger.info("=" * 60)
    logger.info(f"\nModel saved to: {args.output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
