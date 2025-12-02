#!/usr/bin/env python3
"""
Modularized SFT pipeline for Qwen 8B.
Optimized for A100 80GB (BF16) using Unsloth.
"""

import os
import re
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any

from unsloth import FastLanguageModel

import torch
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from loguru import logger
from transformers import (
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

import wandb

# --- CONFIGURATION ---

@dataclass
class SFTConfig:
    """
    Training Configuration with Defaults for A100 80GB.
    """
    # Model
    model_name: str = "Qwen/Qwen3-8B" # Will effectively load Qwen 2.5/3 via Unsloth
    max_length: int = 10240
    
    # Hardware / Precision
    bf16: bool = True
    
    # A100 80GB with Unsloth has huge headroom, so we use Batch Size 8 directly.
    per_device_train_batch_size: int = 12
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = False
    
    # Early Stopping
    early_stopping_patience: int = 10 
    early_stopping_threshold: float = 0.0 
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0  # To maximize unsloth optimization
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # System
    seed: int = 42

    dataloader_num_workers: int = 2
    dataloader_prefetch_factor: int = 1
    dataloader_pin_memory: bool = True 
    dataloader_persistent_workers: bool = True 

# --- DATA PROCESSING FUNCTIONS ---

def clean_text(content: str) -> str:
    """Removes 'Assistant:' artifacts from the end of user prompts."""
    return re.sub(r'\s*Assistant:\s*$', '', content)

def process_single_example(example: Dict, tokenizer: Any, max_length: int) -> Dict:
    """
    Tokenizes a single conversation, applying chat templates and masking the prompt.
    """
    clean_conversation = []
    
    # 1. Clean Data
    for turn in example["conversations"]:
        role = turn['role']
        content = turn['content']
        if role == 'user':
            content = clean_text(content)
        clean_conversation.append({"role": role, "content": content})

    # 2. Format & Tokenize Full Text (User + Assistant)
    try:
        full_text = tokenizer.apply_chat_template(
            clean_conversation, tokenize=False, add_generation_prompt=False
        )
    except Exception as e:
        logger.error(f"Template Error: {e}")
        return {"input_ids": [], "labels": []}

    # 3. Format & Tokenize Prompt Only (User) for masking
    prompt_messages = clean_conversation[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_tokens = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=False
    )["input_ids"]
    
    prompt_tokens = tokenizer(
        prompt_text, truncation=True, max_length=max_length, add_special_tokens=False
    )["input_ids"]

    # 4. Create Labels (Mask Prompt with -100)
    labels = list(full_tokens)
    prompt_len = len(prompt_tokens)
    
    if prompt_len < len(labels):
        labels[:prompt_len] = [-100] * prompt_len
    else:
        labels = [-100] * len(labels) # Truncation edge case

    return {
        "input_ids": full_tokens,
        "labels": labels,
        "attention_mask": [1] * len(full_tokens)
    }

def create_datasets(train_path: str, eval_path: str, tokenizer: Any, config: SFTConfig) -> DatasetDict:
    """Loads and preprocesses the datasets."""
    logger.info("Loading datasets...")
    
    raw_datasets = load_dataset("json", data_files={"train": train_path, "validation": eval_path})
    
    processed_datasets = raw_datasets.map(
        lambda x: process_single_example(x, tokenizer, config.max_length),
        batched=False,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing & Masking"
    )
    
    return processed_datasets


# --- MODEL & TRAINING FUNCTIONS ---

def load_model_and_tokenizer(config: SFTConfig):
    """Loads model and tokenizer via Unsloth for max efficiency."""
    logger.info(f"Loading Model & Tokenizer via Unsloth: {config.model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_length,
        dtype=torch.bfloat16 if config.bf16 else None,
        load_in_4bit=False,  # Keep False for BF16 finetuning
        # --- CRITICAL FIX: Disable auto device mapping ---
        # Since we use CUDA_VISIBLE_DEVICES in shell script, we must let 
        # Trainer handle the single visible device rather than Unsloth's auto-map.
        device_map=None, 
    )

    if config.use_lora:
        logger.info("Applying Unsloth optimized LoRA...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.lora_target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, 
            bias="none",
            use_gradient_checkpointing="unsloth" if config.gradient_checkpointing else False,
            random_state=config.seed,
        )
    
    # Qwen/Llama often default to None for pad_token, which crashes DataCollator
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure the ID is also synced to prevent config artifacts
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Enforce right padding for Training (Crucial for Causal LMs)
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def run_training(model, tokenizer, datasets, output_dir, wandb_project: str, wandb_run_name: str, config: SFTConfig):
    """Sets up Trainer and starts training."""
    
    logger.info(f"WandB Project: {wandb_project}")
    logger.info(f"WandB Run Name: {wandb_run_name}")
    
    os.environ["WANDB_PROJECT"] = wandb_project
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        fp16=not config.bf16,
        optim="adamw_torch_fused",
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_persistent_workers=config.dataloader_persistent_workers,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        report_to="wandb",
        run_name=wandb_run_name,
        remove_unused_columns=True,
        group_by_length=True, 
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        callbacks=[early_stopping]
    )

    logger.info("Starting Training...")
    trainer.train()
    
    logger.info(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir) # Unsloth save method
    tokenizer.save_pretrained(output_dir)


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Qwen SFT Pipeline")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSON")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to validation JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Model name from HuggingFace")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (alpha will be 2*rank)")
    parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="WandB run name")
    args = parser.parse_args()

    # 1. Load Config
    config = SFTConfig(
        model_name=args.model_name,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_rank * 2
    )
    
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  LoRA Rank: {config.lora_r}")
    
    # Set Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 2. Load Components (Merged Step)
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 3. Process Data
    datasets = create_datasets(args.train_file, args.eval_file, tokenizer, config)
    
    # 4. Verify Labels
    if all(l == -100 for l in datasets["train"][0]["labels"]):
        logger.error("⚠️  First training example has NO labels! Check dataset format.")
        
    # 5. Train
    run_training(model, tokenizer, datasets, args.output_dir, args.wandb_project, args.wandb_run_name, config)

if __name__ == "__main__":
    main()