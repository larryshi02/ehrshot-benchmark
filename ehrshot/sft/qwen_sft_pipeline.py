#!/usr/bin/env python3
"""
Supervised Fine-Tuning pipeline for Qwen 8B model using reasoning traces.
Uses transformers, datasets, and other libraries for fine-tuning.
"""

import os
import json
import argparse
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from datasets import Dataset, DatasetDict
from loguru import logger
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import wandb
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning"""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    trust_remote_code: bool = True
    use_quantization: bool = False  # Disabled to match qwen3_sft_yesno.py
    quantization_config: Optional[Dict] = None
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Training configuration
    output_dir: str = "./qwen_sft_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # Generation configuration
    max_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50  # Must equal eval_steps for load_best_model_at_end to work correctly
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Other
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    use_wandb: bool = False
    wandb_project: str = "qwen-sft-ehrshot"


class SFTDatasetProcessor:
    """Processes datasets for supervised fine-tuning"""
    
    def __init__(self, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Add special tokens if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def format_conversation(self, conversation: List[Dict[str, str]]) -> tuple:
        """Format conversation for Qwen model, returning user and assistant content separately"""
        user_content = ""
        assistant_content = ""
        
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            
            if role == "user":
                user_content += content
            elif role == "assistant":
                assistant_content += content
        
        return user_content, assistant_content
    
    def tokenize_function(self, examples):
        """Tokenize examples for training, masking user content in loss"""
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        
        for conversation in examples["conversations"]:
            # Get user and assistant content separately
            user_content, assistant_content = self.format_conversation(conversation)
            
            # Tokenize user content
            user_tokens = self.tokenizer.encode(
                user_content,
                add_special_tokens=False,
                truncation=False
            )
            
            # Tokenize assistant content
            assistant_tokens = self.tokenizer.encode(
                assistant_content,
                add_special_tokens=False,
                truncation=False
            )
            
            # Check total length and truncate if needed
            total_length = len(user_tokens) + len(assistant_tokens)
            if total_length > self.max_length:
                # Truncate from the beginning if needed
                if len(user_tokens) > self.max_length - len(assistant_tokens):
                    user_tokens = user_tokens[-(self.max_length - len(assistant_tokens)):]
                
                # Also truncate assistant if still too long
                if len(user_tokens) + len(assistant_tokens) > self.max_length:
                    assistant_tokens = assistant_tokens[:self.max_length - len(user_tokens)]
            
            # Concatenate for input
            input_ids = user_tokens + assistant_tokens
            
            # Create labels: -100 for user tokens (masked), actual token ids for assistant tokens
            labels = [-100] * len(user_tokens) + assistant_tokens
            attention_mask = [1] * len(input_ids)
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)
        
        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask
        }
    
    def process_dataset(self, dataset: List[Dict]) -> Dataset:
        """Process dataset for training"""
        # Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_list(dataset)
        
        # Tokenize
        tokenized_dataset = hf_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=hf_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset


class QwenSFTTrainer:
    """Main trainer class for Qwen supervised fine-tuning"""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Loss tracking
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=f"qwen-sft-{config.model_name.split('/')[-1]}"
            )
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        num_gpus = torch.cuda.device_count()
        device_map = "balanced" if num_gpus > 1 else "auto"
        logger.info(f"Using {num_gpus} GPU(s) with device_map={device_map}")
        
        # Load tokenizer
        # Use LEFT padding for decoder-only models during training
        # This ensures proper alignment for causal language modeling
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="left"  # LEFT padding for causal LMs (decoder-only)
        )
        
        # Load model (no quantization to match qwen3_sft_yesno.py)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        
        # Enable gradient checkpointing to save memory during training
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled to save memory")
        
        # Apply LoRA if specified
        if self.config.use_lora:
            logger.info("Applying LoRA configuration...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                inference_mode=False,  # Ensure training mode
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            # Ensure model is in training mode
            self.model.train()
            
            # Check trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            logger.info(f"LoRA trainable parameters: {len(trainable_params)}")
            
            if len(trainable_params) == 0:
                logger.warning("No trainable parameters found! LoRA might not be properly configured.")
            
            # Log some trainable parameter names for debugging
            trainable_param_names = [name for name, param in self.model.named_parameters() if param.requires_grad]
            logger.info(f"Trainable parameter names: {trainable_param_names[:5]}...")  # Show first 5
            
            # Ensure all trainable parameters have gradients enabled
            for param in trainable_params:
                param.requires_grad = True
        
        logger.info("Model and tokenizer loaded successfully")
    
    def load_sft_dataset(self, dataset_file: str, split_ratio: float = 0.9, 
                        path_to_splits: str = None, split_type: str = "train",
                        original_dataset_file: Optional[str] = None,
                        balanced_train_dataset_file: Optional[str] = None,
                        balanced_val_dataset_file: Optional[str] = None) -> DatasetDict:
        """
        Load SFT dataset from file and filter by split.
        
        If balanced datasets are provided:
        - Concatenate original dataset with balanced train dataset for training
        - Use balanced val dataset for evaluation (no 0.9/0.1 split)
        
        Otherwise, use the original behavior with split_ratio.
        """
        # Check if using balanced datasets
        if balanced_train_dataset_file and balanced_val_dataset_file:
            logger.info("Using balanced dataset mode: concatenating original + balanced train for training, balanced val for eval")
            
            # Load original dataset (if provided)
            train_dataset = []
            if original_dataset_file and os.path.exists(original_dataset_file):
                logger.info(f"Loading original dataset from {original_dataset_file}")
                with open(original_dataset_file, 'r') as f:
                    original_data = json.load(f)
                train_dataset.extend(original_data)
                logger.info(f"Loaded {len(original_data)} examples from original dataset")
            
            # Load balanced train dataset
            logger.info(f"Loading balanced train dataset from {balanced_train_dataset_file}")
            with open(balanced_train_dataset_file, 'r') as f:
                balanced_train_data = json.load(f)
            train_dataset.extend(balanced_train_data)
            logger.info(f"Loaded {len(balanced_train_data)} examples from balanced train dataset")
            
            logger.info(f"Total train examples after concatenation: {len(train_dataset)}")
            
            # Load balanced val dataset for eval
            logger.info(f"Loading balanced val dataset from {balanced_val_dataset_file}")
            with open(balanced_val_dataset_file, 'r') as f:
                eval_dataset = json.load(f)
            logger.info(f"Loaded {len(eval_dataset)} examples from balanced val dataset")
            
        else:
            # Original behavior: load single dataset and split
            logger.info(f"Loading SFT dataset from {dataset_file}")
            
            with open(dataset_file, 'r') as f:
                dataset = json.load(f)
            
            logger.info(f"Loaded {len(dataset)} examples")
            
            # Filter by split if splits file is provided
            if path_to_splits and os.path.exists(path_to_splits):
                logger.info(f"Filtering dataset by {split_type} split from {path_to_splits}")
                import pandas as pd
                
                # Load splits
                splits_df = pd.read_csv(path_to_splits)
                split_patient_ids = set(splits_df[splits_df['split'] == split_type]['omop_person_id'].values)
                
                # Filter dataset to only include patients in the specified split
                filtered_dataset = []
                for example in dataset:
                    if 'patient_id' in example and example['patient_id'] in split_patient_ids:
                        filtered_dataset.append(example)
                
                logger.info(f"Filtered from {len(dataset)} to {len(filtered_dataset)} examples for {split_type} split")
                dataset = filtered_dataset
            
            # Split into train/eval
            split_idx = int(len(dataset) * split_ratio)
            train_dataset = dataset[:split_idx]
            eval_dataset = dataset[split_idx:]
            
            logger.info(f"Train examples: {len(train_dataset)}")
            logger.info(f"Eval examples: {len(eval_dataset)}")
        
        # Process datasets
        processor = SFTDatasetProcessor(self.tokenizer, self.config.max_length)
        
        train_hf_dataset = processor.process_dataset(train_dataset)
        eval_hf_dataset = processor.process_dataset(eval_dataset)
        
        return DatasetDict({
            "train": train_hf_dataset,
            "eval": eval_hf_dataset
        })
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Setup trainer"""
        logger.info("Setting up trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            eval_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to="wandb" if self.config.use_wandb else [],
            save_total_limit=3,
            bf16=True,  # Use bfloat16 instead of fp16 to match qwen3_sft_yesno.py
            dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        )
        
        # Use custom data collator with LEFT padding for decoder-only models
        # Since pad_sequence pads to the right, we use manual left padding
        class CustomDataCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            
            def __call__(self, features):
                batch = {}
                
                # Extract input_ids, labels, and attention_mask
                input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
                labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
                attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
                
                # Find max length
                max_len = max(len(ids) for ids in input_ids)
                
                # Left pad sequences: pad on the left by prepending pad tokens
                input_ids_padded = []
                labels_padded = []
                attention_mask_padded = []
                
                for ids, lbl, mask in zip(input_ids, labels, attention_mask):
                    pad_len = max_len - len(ids)
                    
                    # Prepend padding (left padding)
                    ids_padded = torch.cat([
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                        ids
                    ])
                    lbl_padded = torch.cat([
                        torch.full((pad_len,), -100, dtype=torch.long),
                        lbl
                    ])
                    mask_padded = torch.cat([
                        torch.zeros(pad_len, dtype=torch.long),
                        mask
                    ])
                    
                    input_ids_padded.append(ids_padded)
                    labels_padded.append(lbl_padded)
                    attention_mask_padded.append(mask_padded)
                
                # Stack into batch tensors
                batch["input_ids"] = torch.stack(input_ids_padded)
                batch["labels"] = torch.stack(labels_padded)
                batch["attention_mask"] = torch.stack(attention_mask_padded)
                
                return batch
        
        data_collator = CustomDataCollator(self.tokenizer)
        
        # Loss tracking callback
        class LossTrackingCallback(TrainerCallback):
            def __init__(self, trainer_instance):
                self.trainer_instance = trainer_instance
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                """Track losses during training and evaluation"""
                if logs is not None:
                    # Track training loss
                    if "loss" in logs:
                        self.trainer_instance.train_losses.append(logs["loss"])
                        self.trainer_instance.train_steps.append(state.global_step)
                        logger.info(f"Train loss at step {state.global_step}: {logs['loss']:.4f}")
                    
                    # Track evaluation loss
                    if "eval_loss" in logs:
                        self.trainer_instance.eval_losses.append(logs["eval_loss"])
                        self.trainer_instance.eval_steps.append(state.global_step)
                        logger.info(f"Eval loss at step {state.global_step}: {logs['eval_loss']:.4f}")
        
        # Custom trainer class to handle LoRA training
        class LoRATrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                # Hugging Face Trainer (>=0.45) may pass extra kwargs (e.g., num_items_in_batch)
                # that we don't need. Ignore them to stay compatible across versions.
                kwargs.pop("num_items_in_batch", None)
                
                """Custom loss computation for LoRA training"""
                labels = inputs.get("labels")
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Only check gradients during training, not evaluation
                if model.training and not loss.requires_grad:
                    logger.error("Loss does not require gradients during training!")
                    raise RuntimeError("Loss computation failed - no gradients during training")
                
                return (loss, outputs) if return_outputs else loss
        
        # Create loss tracking callback
        loss_callback = LossTrackingCallback(self)
        
        # Initialize trainer
        self.trainer = LoRATrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[loss_callback],
        )
        
        logger.info("Trainer setup complete")
    
    def train(self):
        """Start training"""
        logger.info("Starting training...")
        
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer first.")
        
        # Ensure model is in training mode
        self.model.train()
        
        # Check if any parameters require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found! Check LoRA configuration.")
        
        logger.info(f"Found {len(trainable_params)} trainable parameters")
        
        # Double-check that trainable parameters have gradients enabled
        for param in trainable_params:
            if not param.requires_grad:
                param.requires_grad = True
                logger.warning(f"Re-enabled gradients for parameter: {param.shape}")
        
        # Verify model is ready for training
        logger.info("Model is ready for training with LoRA adapters")
        
        # Train with explicit model preparation
        logger.info("Preparing model for training...")
        
        # Ensure model is in training mode
        self.model.train()
        
        # Explicitly enable gradients for all trainable parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.requires_grad = True
                logger.debug(f"Enabled gradients for {name}")
        
        # Train
        self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("Training completed!")
        
        # Plot losses
        logger.info("Plotting loss curves...")
        self.plot_losses()
    
    def generate_sample(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate a sample response"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer first.")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - max_new_tokens
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
    
    def plot_losses(self):
        """Plot training and evaluation losses"""
        if not self.train_losses and not self.eval_losses:
            logger.warning("No loss data to plot")
            return
        
        # Create output directory for plots
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot training loss
        if self.train_losses:
            ax.plot(self.train_steps, self.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
        
        # Plot evaluation loss
        if self.eval_losses:
            ax.plot(self.eval_steps, self.eval_losses, 'r-', label='Eval Loss', linewidth=2, alpha=0.8, marker='o', markersize=4)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Evaluation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "loss_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved loss plot to {plot_path}")
        
        plt.close()
        
        # Print summary statistics
        if self.train_losses:
            final_train_loss = self.train_losses[-1]
            initial_train_loss = self.train_losses[0] if len(self.train_losses) > 0 else None
            logger.info(f"\nTrain Loss Summary:")
            logger.info(f"  Initial: {initial_train_loss:.4f}")
            logger.info(f"  Final: {final_train_loss:.4f}")
            if initial_train_loss is not None:
                logger.info(f"  Reduction: {initial_train_loss - final_train_loss:.4f} ({(1 - final_train_loss/initial_train_loss)*100:.1f}%)")
        
        if self.eval_losses:
            final_eval_loss = self.eval_losses[-1]
            initial_eval_loss = self.eval_losses[0] if len(self.eval_losses) > 0 else None
            logger.info(f"\nEval Loss Summary:")
            logger.info(f"  Initial: {initial_eval_loss:.4f}")
            logger.info(f"  Final: {final_eval_loss:.4f}")
            if initial_eval_loss is not None:
                logger.info(f"  Reduction: {initial_eval_loss - final_eval_loss:.4f} ({(1 - final_eval_loss/initial_eval_loss)*100:.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen Supervised Fine-Tuning Pipeline")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./qwen_sft_output",
                       help="Output directory for trained model")
    parser.add_argument("--use_quantization", action="store_true", default=False,
                       help="Use quantization (disabled by default to match qwen3_sft_yesno.py)")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for efficient fine-tuning")
    
    # Training configuration
    parser.add_argument("--dataset_file", type=str, default=None,
                       help="Path to SFT dataset JSON file (required if not using balanced datasets)")
    parser.add_argument("--path_to_splits", type=str, default=None,
                       help="Path to splits CSV file")
    parser.add_argument("--split_type", type=str, default="train", choices=["train", "val", "test"],
                       help="Which split to use (train, val, test)")
    
    # Balanced dataset configuration
    parser.add_argument("--original_dataset_file", type=str, default=None,
                       help="Path to original SFT dataset JSON file (for balanced mode)")
    parser.add_argument("--balanced_train_dataset_file", type=str, default=None,
                       help="Path to balanced train SFT dataset JSON file (for balanced mode)")
    parser.add_argument("--balanced_val_dataset_file", type=str, default=None,
                       help="Path to balanced val SFT dataset JSON file (for balanced mode)")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                       help="Per device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Other
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--gpu_id", type=str, default=None,
                       help="GPU ID(s) to use (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate arguments
    if not args.balanced_train_dataset_file and not args.dataset_file:
        raise ValueError("Either --dataset_file or --balanced_train_dataset_file must be provided")
    
    if args.balanced_train_dataset_file and not args.balanced_val_dataset_file:
        raise ValueError("--balanced_val_dataset_file is required when using --balanced_train_dataset_file")
    
    # Set GPU
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    # Create configuration
    config = SFTConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_quantization=args.use_quantization,
        use_lora=args.use_lora,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_wandb=args.use_wandb,
        seed=args.seed
    )
    
    # Initialize trainer
    trainer = QwenSFTTrainer(config)
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Load dataset (use balanced mode if balanced datasets are provided)
    if args.balanced_train_dataset_file:
        logger.info("Using balanced dataset mode")
        datasets = trainer.load_sft_dataset(
            dataset_file=args.dataset_file or "",  # Not used in balanced mode, but required for signature
            original_dataset_file=args.original_dataset_file,
            balanced_train_dataset_file=args.balanced_train_dataset_file,
            balanced_val_dataset_file=args.balanced_val_dataset_file
        )
    else:
        logger.info("Using standard dataset mode")
        datasets = trainer.load_sft_dataset(
            dataset_file=args.dataset_file,
            path_to_splits=args.path_to_splits,
            split_type=args.split_type
        )
    
    # Setup trainer
    trainer.setup_trainer(datasets["train"], datasets["eval"])
    
    # Train
    trainer.train()
    
    # Generate a test sample
    logger.info("Generating test sample...")
    test_prompt = "Given a patient's electronic healthcare record (EHR) in Markdown format, predict whether the patient will develop acute MI.\n\nPatient Medical History:\n[Sample EHR data would go here]\n\nPlease provide your analysis and prediction."
    
    try:
        response = trainer.generate_sample(test_prompt)
        logger.info(f"Generated response: {response[:200]}...")
    except Exception as e:
        logger.warning(f"Could not generate test sample: {e}")
    
    logger.info("SFT pipeline completed!")


if __name__ == "__main__":
    main()
