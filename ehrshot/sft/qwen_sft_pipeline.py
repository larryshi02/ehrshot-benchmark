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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset, DatasetDict
from loguru import logger
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import wandb


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning"""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    trust_remote_code: bool = True
    use_quantization: bool = True
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
    eval_steps: int = 100
    save_steps: int = 500
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
    
    def format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for Qwen model"""
        formatted = ""
        
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            
            if role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        return formatted
    
    def tokenize_function(self, examples):
        """Tokenize examples for training"""
        # Format conversations
        formatted_texts = []
        for conversation in examples["conversations"]:
            formatted_text = self.format_conversation(conversation)
            formatted_texts.append(formatted_text)
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
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
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )
        
        # Load model
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        # Add quantization config if specified
        if self.config.use_quantization and self.config.quantization_config:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(**self.config.quantization_config)
        elif self.config.use_quantization:
            # Default quantization config
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
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
                        path_to_splits: str = None, split_type: str = "train") -> DatasetDict:
        """Load SFT dataset from file and filter by split"""
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
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to="wandb" if self.config.use_wandb else None,
            save_total_limit=3,
            fp16=True,
            gradient_checkpointing=False,
        )
        
        # Use standard data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Custom trainer class to handle LoRA training
        class LoRATrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                """Custom loss computation for LoRA training"""
                labels = inputs.get("labels")
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Only check gradients during training, not evaluation
                if model.training and not loss.requires_grad:
                    logger.error("Loss does not require gradients during training!")
                    raise RuntimeError("Loss computation failed - no gradients during training")
                
                return (loss, outputs) if return_outputs else loss
        
        # Initialize trainer
        self.trainer = LoRATrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
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
    
    def generate_sample(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate a sample response"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer first.")
        
        # Format prompt
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
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
        
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        
        return response


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen Supervised Fine-Tuning Pipeline")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./qwen_sft_output",
                       help="Output directory for trained model")
    parser.add_argument("--use_quantization", action="store_true", default=True,
                       help="Use quantization")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for efficient fine-tuning")
    
    # Training configuration
    parser.add_argument("--dataset_file", type=str, required=True,
                       help="Path to SFT dataset JSON file")
    parser.add_argument("--path_to_splits", type=str, default=None,
                       help="Path to splits CSV file")
    parser.add_argument("--split_type", type=str, default="train", choices=["train", "val", "test"],
                       help="Which split to use (train, val, test)")
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
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    # Load dataset
    datasets = trainer.load_sft_dataset(args.dataset_file, path_to_splits=args.path_to_splits, split_type=args.split_type)
    
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
