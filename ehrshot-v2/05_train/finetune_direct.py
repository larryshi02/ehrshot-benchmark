#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B for direct Yes/No clinical outcome prediction.

The model is trained to output a single token ("Yes" or "No") given
an SFT prompt.  This script is used for four representation types:
  plaintext, manualrubric, llmrubric, cot_unsupervised.

Key design choices:
  - LoRA (r=16, alpha=32) via Unsloth for efficient training.
  - Prompt masking: loss only on the assistant response token(s).
  - Qwen3 thinking mode disabled (enable_thinking=False).
  - Validation-based early stopping (patience=20, eval_steps=50).
  - Supports --n_train 40 to filter training data to the 40 rubric
    cohort patients (for sample-efficiency experiments).

Inputs:
  --train_file    : Path to training SFT JSON.
  --eval_file     : Path to validation SFT JSON.
  --output_dir    : Where to save LoRA adapter weights.
  --n_train       : If set (e.g. 40), filter train data to this many
                    patients using the rubric cohort IDs.
  --cohort_file   : Path to patient_ids.json from 03_rubric (required
                    when --n_train is used).

Outputs:
  {output_dir}/adapter_config.json, adapter_model.safetensors, tokenizer files.

Connects to:
  - Upstream  : 02_create_sft, 03_rubric, 04_cot
  - Downstream: 06_eval/eval_direct.py
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from unsloth import FastLanguageModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import FINETUNE_MODEL


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    model_name: str = FINETUNE_MODEL
    max_length: int = 12288
    bf16: bool = True

    per_device_train_batch_size: int = 12
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = False

    early_stopping_patience: int = 20
    early_stopping_threshold: float = 0.0

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    seed: int = 42


# ---------------------------------------------------------------------------
# Tokenization with prompt masking
# ---------------------------------------------------------------------------

def tokenize_example(example: Dict, tokenizer: Any, max_length: int) -> Dict:
    convos = example["conversations"]

    full_text = tokenizer.apply_chat_template(
        convos, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        convos[:-1], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    full_ids = tokenizer(full_text, truncation=True, max_length=max_length,
                         add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length,
                           add_special_tokens=False)["input_ids"]

    labels = list(full_ids)
    plen = len(prompt_ids)
    if plen < len(labels):
        labels[:plen] = [-100] * plen
    else:
        labels = [-100] * len(labels)

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


# ---------------------------------------------------------------------------
# Data loading with optional n_train filtering
# ---------------------------------------------------------------------------

def load_and_prepare(train_path: str, eval_path: str,
                     tokenizer: Any, cfg: SFTConfig,
                     n_train: int | None = None,
                     cohort_ids: List[int] | None = None) -> DatasetDict:
    # Filter training data to cohort subset if n_train specified
    if n_train is not None and cohort_ids is not None:
        with open(train_path) as f:
            train_data = json.load(f)
        cohort_set = set(cohort_ids)
        train_data = [r for r in train_data if r["patient_id"] in cohort_set]
        logger.info(f"Filtered training data to {len(train_data)} records "
                    f"(n_train={n_train}, cohort IDs: {len(cohort_set)})")
        # Write filtered data to a temp file
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(train_data, tmp)
        tmp.close()
        train_path = tmp.name

    raw = load_dataset("json", data_files={
        "train": train_path, "validation": eval_path})
    processed = raw.map(
        lambda x: tokenize_example(x, tokenizer, cfg.max_length),
        batched=False, remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )
    return processed


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: SFTConfig):
    logger.info(f"Loading {cfg.model_name} via Unsloth")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_length,
        dtype=torch.bfloat16 if cfg.bf16 else None,
        load_in_4bit=False,
        device_map=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_r,
        target_modules=cfg.lora_target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if cfg.gradient_checkpointing else False,
        random_state=cfg.seed,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, tokenizer, datasets, output_dir: str,
          wandb_project: str, wandb_run: str, cfg: SFTConfig):
    os.environ["WANDB_PROJECT"] = wandb_project

    args = TrainingArguments(
        output_dir=output_dir,
        seed=cfg.seed,
        data_seed=cfg.seed,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        optim="adamw_torch_fused",
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=wandb_run,
        remove_unused_columns=True,
        group_by_length=True,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True,
        pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience,
            early_stopping_threshold=cfg.early_stopping_threshold,
        )],
    )

    logger.info("Starting direct Yes/No training...")
    trainer.train()

    logger.info(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--wandb_project", default="ehrshot-v2-direct")
    p.add_argument("--wandb_run_name", default="run")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--n_train", type=int, default=None,
                   help="If set, use only this many patients (from cohort)")
    p.add_argument("--cohort_file", default=None,
                   help="Path to patient_ids.json (required when --n_train is set)")
    args = p.parse_args()

    cfg = SFTConfig(learning_rate=args.learning_rate,
                    num_train_epochs=args.num_epochs,
                    seed=args.seed)

    # Full reproducibility: PYTHONHASHSEED + all RNG seeds
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    set_seed(cfg.seed)  # sets random, numpy, torch, cuda seeds

    model, tokenizer = load_model(cfg)

    cohort_ids = None
    if args.n_train is not None:
        if args.cohort_file is None:
            raise ValueError("--cohort_file required when --n_train is set")
        with open(args.cohort_file) as f:
            cohort_ids = json.load(f)

    datasets = load_and_prepare(
        args.train_file, args.eval_file, tokenizer, cfg,
        n_train=args.n_train, cohort_ids=cohort_ids,
    )

    train(model, tokenizer, datasets, args.output_dir,
          args.wandb_project, args.wandb_run_name, cfg)


if __name__ == "__main__":
    main()
