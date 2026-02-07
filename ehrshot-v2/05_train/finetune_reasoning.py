#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B on supervised CoT reasoning traces.

The model learns to produce:
  <think>
  1. ### PATIENT SNAPSHOT
  ...
  6. ### CONCLUSION

  Final Answer: Yes/No
  </think>

This is the "reasoning" fine-tuning variant -- the loss is on the full
assistant response (reasoning + answer), not just the final token.

Inputs:
  --train_file : Supervised CoT SFT JSON (data/sft/cot_supervised/train/{task}.json).
  --eval_file  : Supervised CoT SFT JSON for val split.
  --output_dir : Where to save LoRA adapter weights.

Same LoRA config as finetune_direct.py for fair comparison.
Supports --n_train 40 for sample-efficiency experiments.

Outputs:
  LoRA adapter weights in {output_dir}/.

Connects to:
  - Upstream  : 04_cot/generate_supervised_cot.py
  - Downstream: 06_eval/eval_reasoning.py
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
from datasets import load_dataset, DatasetDict
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


@dataclass
class SFTConfig:
    model_name: str = FINETUNE_MODEL
    max_length: int = 12288
    bf16: bool = True

    per_device_train_batch_size: int = 4  # smaller -- reasoning outputs are long
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 3
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True  # needed for long sequences

    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    seed: int = 42


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


def load_model(cfg: SFTConfig):
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--wandb_project", default="ehrshot-v2-reasoning")
    p.add_argument("--wandb_run_name", default="run")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--cohort_file", default=None)
    args = p.parse_args()

    cfg = SFTConfig(learning_rate=args.learning_rate,
                    num_train_epochs=args.num_epochs,
                    seed=args.seed)

    # Full reproducibility: PYTHONHASHSEED + all RNG seeds
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    set_seed(cfg.seed)  # sets random, numpy, torch, cuda seeds

    model, tokenizer = load_model(cfg)

    train_path = args.train_file
    if args.n_train is not None:
        if args.cohort_file is None:
            raise ValueError("--cohort_file required when --n_train is set")
        with open(args.cohort_file) as f:
            cohort_ids = set(json.load(f))
        with open(train_path) as f:
            data = json.load(f)
        data = [r for r in data if r["patient_id"] in cohort_ids]
        logger.info(f"Filtered to {len(data)} records (n_train={args.n_train})")
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, tmp); tmp.close()
        train_path = tmp.name

    raw = load_dataset("json", data_files={
        "train": train_path, "validation": args.eval_file})
    datasets = raw.map(
        lambda x: tokenize_example(x, tokenizer, cfg.max_length),
        batched=False, remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    os.environ["WANDB_PROJECT"] = args.wandb_project
    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
        run_name=args.wandb_run_name,
        remove_unused_columns=True,
        group_by_length=True,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True,
        pad_to_multiple_of=8, label_pad_token_id=-100,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience,
            early_stopping_threshold=cfg.early_stopping_threshold,
        )],
    )

    logger.info("Starting reasoning fine-tuning...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.success(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
