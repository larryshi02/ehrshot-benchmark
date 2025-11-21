#!/usr/bin/env python3
"""
Modularized SFT pipeline for Qwen 8B.
Optimized for A100 80GB (Flash Attn 2, BF16).
"""

import os
import re
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any

import torch
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from loguru import logger
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb

# --- CONFIGURATION ---

@dataclass
class SFTConfig:
    """
    Training Configuration with Defaults for A100 80GB.
    """
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    max_length: int = 16384
    
    # Hardware / Precision
    bf16: bool = True
    attn_implementation: str = "sdpa"  # Use PyTorch SDPA (flash_attention_2 requires CUDA toolkit)
    
    # Optimization (Total Batch Size = 2 * 4 * Num_GPUs = 8 per GPU effective)
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    
    # Early Stopping
    early_stopping_patience: int = 3  # Stop if no improvement for 3 evaluations
    early_stopping_threshold: float = 0.0  # Minimum improvement required
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # System
    seed: int = 42
    dataloader_num_workers: int = 4


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
    
    # Process with batched=False to handle complex logic safely
    processed_datasets = raw_datasets.map(
        lambda x: process_single_example(x, tokenizer, config.max_length),
        batched=False,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing & Masking"
    )
    
    return processed_datasets


# --- MODEL & TRAINING FUNCTIONS ---

def load_tokenizer(config: SFTConfig) -> Any:
    """Loads tokenizer with RIGHT padding for SFT."""
    logger.info(f"Loading Tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right" 
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(config: SFTConfig) -> Any:
    """Loads model with Flash Attn 2 and LoRA adapters."""
    logger.info(f"Loading Model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
        attn_implementation=config.attn_implementation,
        device_map="auto" # Will default to cuda:0 if single GPU visible
    )
    
    if config.use_lora:
        logger.info("Applying LoRA...")
        # Enable gradient checkpointing before applying LoRA
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            # Enable input gradients for embeddings (required for gradient checkpointing)
            model.enable_input_require_grads()
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    return model

def run_training(model, tokenizer, datasets, output_dir, wandb_project: str, wandb_run_name: str, config: SFTConfig):
    """Sets up Trainer and starts training."""
    
    logger.info(f"WandB Project: {wandb_project}")
    logger.info(f"WandB Run Name: {wandb_run_name}")
    
    # Set WandB project as environment variable (TrainingArguments doesn't have a project parameter)
    # This ensures the project name from the argument is used
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
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower eval_loss is better
        dataloader_num_workers=config.dataloader_num_workers,
        report_to="wandb",
        run_name=wandb_run_name,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False if config.use_lora else None,
    )

    # DataCollatorForSeq2Seq handles dynamic right-padding automatically
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8
    )

    # Early stopping callback
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
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Qwen SFT Pipeline")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSON")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to validation JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Model name from HuggingFace")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank (alpha will be 2*rank)")
    parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="WandB run name")
    args = parser.parse_args()

    # 1. Load Config (Defaults are defined in the class)
    # Note: lora_alpha is automatically set to 2 * lora_rank
    config = SFTConfig(
        model_name=args.model_name,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_rank * 2
    )
    
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  LoRA Rank: {config.lora_r}, Alpha: {config.lora_alpha}")
    logger.info(f"  Batch Size: {config.per_device_train_batch_size}, Grad Accum: {config.gradient_accumulation_steps}")
    logger.info(f"  Early Stopping: patience={config.early_stopping_patience}, threshold={config.early_stopping_threshold}")
    
    # Set Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 2. Load Components
    tokenizer = load_tokenizer(config)
    model = load_model(config)
    
    # 3. Process Data
    datasets = create_datasets(args.train_file, args.eval_file, tokenizer, config)
    
    # 4. Verify Labels (Sanity Check)
    if all(l == -100 for l in datasets["train"][0]["labels"]):
        logger.error("⚠️  First training example has NO labels! Check dataset format.")
        
    # 5. Train
    run_training(model, tokenizer, datasets, args.output_dir, args.wandb_project, args.wandb_run_name, config)

if __name__ == "__main__":
    main()