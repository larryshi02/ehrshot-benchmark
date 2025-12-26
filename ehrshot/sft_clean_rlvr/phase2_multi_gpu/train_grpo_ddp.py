#!/usr/bin/env python3
"""
Phase 2: Multi-GPU GRPO Training with DDP

This script trains a GRPO model on aggregated data from all 4 tasks using
Distributed Data Parallel (DDP) across multiple GPUs.

Usage:
    # Launch with torchrun for DDP
    torchrun --nproc_per_node=4 train_grpo_ddp.py \\
        --sft_checkpoint /path/to/sft/checkpoint \\
        --output_dir ./outputs/phase2_aggregated

Key Features:
    - DDP training across 4 GPUs
    - Aggregated data from all 4 clinical tasks
    - Balanced class sampling within each task
    - Verifiable reward based on final answer correctness
    - Starts from aggregated SFT checkpoint
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
import torch.distributed as dist
import numpy as np
from datasets import Dataset
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from loguru import logger

# Add parent directory to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.constants import TASKS, MODEL_CONFIG, GRPO_CONFIG, DATA_PATHS
from common.reward_functions import parse_final_answer, compute_reward
from common.prompt_formatting import load_aggregated_grpo_data


# =============================================================================
# REWARD FUNCTION FOR TRL (DDP-AWARE)
# =============================================================================

class RewardComputer:
    """
    Reward computer for DDP training.
    
    Each GPU process maintains its own stats, which are aggregated at the end.
    """
    
    def __init__(self):
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
        """
        self.total_calls += 1
        
        rewards = []
        ground_truths = kwargs.get("ground_truth", None)
        
        if ground_truths is not None:
            for completion, gt in zip(completions, ground_truths):
                reward = compute_reward(completion, gt)
                rewards.append(reward)
                self._update_stats(reward)
        else:
            # Fallback: can't compute meaningful rewards without ground truth
            logger.warning("No ground_truth provided, using neutral rewards")
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
    parser = argparse.ArgumentParser(description="Phase 2: Multi-GPU GRPO Training")
    
    # Required arguments
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="Path to SFT checkpoint (LoRA adapter)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for RLVR model")
    
    # Optional arguments
    parser.add_argument("--split", type=str, default="train_orig",
                        choices=["train_orig", "train_all"],
                        help="Training data split (default: train_orig)")
    parser.add_argument("--num_generations", type=int, default=10,
                        help="Samples per prompt (default: 10)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum training steps (default: 1000)")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="KL penalty coefficient (default: 0.05)")
    parser.add_argument("--per_device_batch_size", type=int, default=2,
                        help="Batch size per GPU (default: 2)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--wandb_project", type=str, default="ehrshot-rlvr-phase2",
                        help="WandB project name")
    
    # DDP arguments (set by torchrun)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for DDP (set by torchrun)")
    
    args = parser.parse_args()
    
    # Determine if we're in DDP mode
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = local_rank in [-1, 0]
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if is_main_process:
        logger.info("=" * 60)
        logger.info("Phase 2: Multi-GPU GRPO Training (DDP)")
        logger.info("=" * 60)
        logger.info(f"World Size (GPUs): {world_size}")
        logger.info(f"SFT Checkpoint: {args.sft_checkpoint}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"Num Generations: {args.num_generations}")
        logger.info(f"Per-Device Batch Size: {args.per_device_batch_size}")
        logger.info(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"Learning Rate: {args.learning_rate}")
        logger.info(f"Beta (KL): {args.beta}")
        logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. Load Model from SFT Checkpoint
    # -------------------------------------------------------------------------
    if is_main_process:
        logger.info("\n[1/4] Loading model from SFT checkpoint...")
    
    from unsloth import FastLanguageModel
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG.model_name,
        max_seq_length=MODEL_CONFIG.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map=None,  # Let DDP handle device mapping
    )
    
    # Load LoRA adapter from SFT checkpoint
    if is_main_process:
        logger.info(f"Loading LoRA adapter from: {args.sft_checkpoint}")
    model = PeftModel.from_pretrained(model, args.sft_checkpoint)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if is_main_process:
        logger.info(f"Model loaded: {MODEL_CONFIG.model_name}")
    
    # -------------------------------------------------------------------------
    # 2. Prepare Aggregated Dataset
    # -------------------------------------------------------------------------
    if is_main_process:
        logger.info("\n[2/4] Preparing aggregated dataset...")
    
    examples = load_aggregated_grpo_data(
        tasks=TASKS,
        split=args.split,
        balance_classes=True,
        seed=args.seed
    )
    
    if is_main_process:
        logger.info(f"Loaded {len(examples)} balanced examples from all tasks")
        
        # Count per task
        task_counts = {}
        for ex in examples:
            task = ex["task"]
            task_counts[task] = task_counts.get(task, 0) + 1
        for task, count in sorted(task_counts.items()):
            logger.info(f"  {task}: {count}")
    
    # Convert to HuggingFace Dataset
    dataset = prepare_grpo_dataset(examples, tokenizer)
    
    if is_main_process:
        logger.info(f"Dataset prepared with {len(dataset)} examples")
    
    # -------------------------------------------------------------------------
    # 3. Setup GRPO Trainer
    # -------------------------------------------------------------------------
    if is_main_process:
        logger.info("\n[3/4] Setting up GRPO trainer...")
    
    # Create reward computer
    reward_computer = RewardComputer()
    
    # Calculate effective batch size
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * world_size
    completions_per_step = effective_batch * args.num_generations
    
    if is_main_process:
        logger.info(f"Effective batch size: {args.per_device_batch_size} × {args.gradient_accumulation_steps} × {world_size} = {effective_batch}")
        logger.info(f"Total completions per step: {completions_per_step}")
    
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
        
        # Batch settings (DDP)
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Optimization
        warmup_ratio=GRPO_CONFIG.warmup_ratio,
        weight_decay=GRPO_CONFIG.weight_decay,
        bf16=GRPO_CONFIG.bf16,
        gradient_checkpointing=GRPO_CONFIG.gradient_checkpointing,
        
        # DDP settings
        ddp_find_unused_parameters=False,
        
        # Logging
        logging_steps=GRPO_CONFIG.logging_steps,
        save_steps=GRPO_CONFIG.save_steps,
        report_to="wandb" if is_main_process else "none",
        run_name="rlvr_phase2_aggregated",
        
        # Misc
        seed=args.seed,
        remove_unused_columns=False,
    )
    
    # Set WandB project (only on main process)
    if is_main_process:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_computer,
    )
    
    if is_main_process:
        logger.info("GRPO Trainer configured")
    
    # -------------------------------------------------------------------------
    # 4. Train
    # -------------------------------------------------------------------------
    if is_main_process:
        logger.info("\n[4/4] Starting GRPO training...")
    
    try:
        trainer.train()
        if is_main_process:
            logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Save final model (only on main process)
    if is_main_process:
        logger.info(f"\nSaving model to {args.output_dir}...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training stats
        stats = reward_computer.get_stats()
        stats["world_size"] = world_size
        stats["effective_batch_size"] = effective_batch
        
        stats_path = os.path.join(args.output_dir, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Statistics (Main Process)")
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

