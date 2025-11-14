#!/usr/bin/env python3
"""
Clean Supervised Fine-Tuning pipeline for Qwen3-8B model on binary classification tasks.

Requirements:
1. Model outputs only "yes" or "no" - loss computed only on these tokens
2. Class-weighted loss for handling class imbalance
3. Evaluation on validation split (0.2 fraction) with F1-score, batch inference
4. Validation loss tracked and plotted
5. Hyperparameters tuned to limit evaluation overhead (<25% of runtime)
"""

import os
import json
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from datasets import Dataset
from loguru import logger
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import Counter


# Constants
IGNORE_INDEX = -100  # Mask tokens that don't contribute to loss

# Task queries
TASK_QUERIES = {
    'acute_mi': 'Will the patient develop an acute myocardial infarction in the next year?',
    'hyperlipidemia': 'Will the patient develop hyperlipidemia in the next year?',
    'hypertension': 'Will the patient develop hypertension in the next year?',
    'pancreatic_cancer': 'Will the patient develop pancreatic cancer in the next year?'
}


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    trust_remote_code: bool = True
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training
    output_dir: str = "./qwen3_sft_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # PRIMARY memory factor: controls peak GPU memory usage
    # NOTE: Only per_device_train_batch_size affects peak memory, NOT gradient_accumulation_steps!
    # Gradient accumulation does NOT reduce memory - it only affects when gradients are applied.
    # IMPORTANT: With max_length=20000, batch_size=1 is optimal to avoid memory bandwidth saturation
    # Larger batch sizes (2+) cause >2x slowdown due to memory bandwidth limits, making total time longer
    # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps = 1 * 8 = 8
    per_device_eval_batch_size: int = 1  # Set to 1 to avoid OOM during evaluation with long sequences (max_length=20000)
    gradient_accumulation_steps: int = 8  # Does NOT affect memory - only controls effective batch size
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_length: int = 20000
    
    # Evaluation
    eval_fraction: float = 0.2  # Use 20% of validation split
    eval_steps: int = 100  # Evaluate every N steps (tuned to limit overhead to <25% of runtime)
    eval_batch_size: int = 8  # Batch size for F1 evaluation (reduced to avoid OOM with long sequences)
    metric_for_best_model: str = "eval_f1"
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100  # Save checkpoints every N steps (must equal eval_steps for load_best_model_at_end)
    save_total_limit: int = 3  # Keep only the N best checkpoints
    
    # Other
    seed: int = 42
    dataloader_num_workers: int = 4
    task_name: str = "acute_mi"
    serialized_data_dir: str = "./serialized_multi_task_data"


class DatasetProcessor:
    """Processes datasets for binary classification (yes/no)"""
    
    def __init__(self, tokenizer, max_length: int, task_name: str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_name = task_name
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get task query
        self.query = TASK_QUERIES.get(
            task_name, 
            f"Will the patient develop {task_name.replace('_', ' ')} in the next year?"
        )
        
        # Tokenize "yes" and "no" for evaluation
        self.yes_tokens = self.tokenizer.encode("yes", add_special_tokens=False)
        self.no_tokens = self.tokenizer.encode("no", add_special_tokens=False)
        
        logger.info(f"Task: {task_name}")
        logger.info(f"Query: {self.query}")
        logger.info(f"'yes' token IDs: {self.yes_tokens}")
        logger.info(f"'no' token IDs: {self.no_tokens}")
    
    def create_prompt(self, context: str) -> str:
        """Create prompt from patient context"""
        prompt = (
            "User: You are a helpful medical assistant. "
            "Below is a patient's electronic healthcare record (EHR) in Markdown format. "
            f"Please answer the following query with only 'yes' or 'no'. "
            f"Query: {self.query}. "
            "Patient Medical History:\n\n"
            f"{context}\n\n"
            f"Answer the query with only 'yes' or 'no'. "
            f"Query: {self.query}. "
            "Answer: "
        )
        return prompt
    
    def create_answer(self, label_value: bool) -> str:
        """Create answer: 'yes' for True, 'no' for False"""
        return "yes" if label_value else "no"
    
    def tokenize_examples(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize examples for training.
        
        Loss is computed ONLY on the answer tokens ("yes" or "no").
        Patient context is conditioned on but masked in loss calculation.
        """
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        
        for i in range(len(examples["user_content"])):
            user_content = examples["user_content"][i]
            assistant_content = examples["assistant_content"][i]  # "yes" or "no"
            
            # Tokenize user content (patient context)
            user_tokens = self.tokenizer.encode(
                user_content,
                add_special_tokens=False,
                truncation=False
            )
            
            # Tokenize assistant content (answer: "yes" or "no")
            assistant_tokens = self.tokenizer.encode(
                assistant_content,
                add_special_tokens=False,
                truncation=False
            )
            
            # Truncate if needed (from the beginning of user content)
            total_length = len(user_tokens) + len(assistant_tokens)
            if total_length > self.max_length:
                max_user_tokens = self.max_length - len(assistant_tokens)
                if len(user_tokens) > max_user_tokens:
                    user_tokens = user_tokens[-max_user_tokens:]
            
            # Concatenate
            input_ids = user_tokens + assistant_tokens
            
            # Create labels:
            # - Mask all user tokens (IGNORE_INDEX) - patient context doesn't contribute to loss
            # - Include assistant tokens in loss - only "yes"/"no" tokens contribute
            labels = [IGNORE_INDEX] * len(user_tokens) + assistant_tokens
            
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
        processed_data = []
        for item in dataset:
            user_content = self.create_prompt(item["context"])
            assistant_content = self.create_answer(item["label_value"])
            processed_data.append({
                "user_content": user_content,
                "assistant_content": assistant_content,
                "label_value": item["label_value"]
            })
        
        hf_dataset = Dataset.from_list(processed_data)
        
        tokenized_dataset = hf_dataset.map(
            self.tokenize_examples,
            batched=True,
            remove_columns=["user_content", "assistant_content"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset


class FineTuningTrainer:
    """Main trainer for fine-tuning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Tracking for plots
        self.train_losses = []
        self.train_steps = []
        self.eval_f1_scores = []
        self.eval_steps = []  # Shared steps for F1 scores
        
        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    
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
        
        # Load model
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
        
        # Apply LoRA
        if self.config.use_lora:
            logger.info("Applying LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                inference_mode=False,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def load_datasets(self) -> Tuple[Dataset, Dataset, Dict[bool, float]]:
        """
        Load training and validation datasets.
        
        Returns:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (0.2 fraction of validation split)
            class_weights: Class weights for weighted loss
        """
        dataset_file = os.path.join(
            self.config.serialized_data_dir,
            f"{self.config.task_name}_all_splits.json"
        )
        
        logger.info(f"Loading dataset from {dataset_file}")
        
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        with open(dataset_file, 'r') as f:
            all_data = json.load(f)
        
        # Split data
        train_data = [item for item in all_data if item.get('split') == 'train']
        val_data = [item for item in all_data if item.get('split') == 'val']
        
        logger.info(f"Train examples: {len(train_data)}")
        logger.info(f"Validation examples: {len(val_data)}")
        
        if len(train_data) == 0 or len(val_data) == 0:
            raise ValueError("Missing train or validation data")
        
        # Use eval_fraction of validation data for evaluation
        eval_size = max(1, int(len(val_data) * self.config.eval_fraction))
        if eval_size < len(val_data):
            rng = np.random.RandomState(self.config.seed)
            indices = rng.choice(len(val_data), size=eval_size, replace=False)
            eval_data = [val_data[i] for i in sorted(indices)]
            logger.info(f"Using {len(eval_data)} validation samples ({self.config.eval_fraction*100:.1f}% of validation) for evaluation")
        else:
            eval_data = val_data
            logger.info(f"Using all {len(eval_data)} validation samples for evaluation")
        
        # Compute class weights (inverse frequency weighting)
        label_counts = Counter([item['label_value'] for item in train_data])
        total_samples = len(train_data)
        class_weights = {
            True: total_samples / (2 * label_counts.get(True, 1)),
            False: total_samples / (2 * label_counts.get(False, 1))
        }
        # Convert boolean counts to readable class names
        class_distribution = {"Yes": label_counts.get(True, 0), "No": label_counts.get(False, 0)}
        logger.info(f"Class distribution: {class_distribution}")
        logger.info(f"Class weights: Yes={class_weights[True]:.4f}, No={class_weights[False]:.4f}")
        
        # Process datasets
        processor = DatasetProcessor(self.tokenizer, self.config.max_length, self.config.task_name)
        train_dataset = processor.process_dataset(train_data)
        eval_dataset = processor.process_dataset(eval_data)
        
        # Store raw eval data for F1 computation
        self.eval_data_raw = eval_data
        self.processor = processor
        
        return train_dataset, eval_dataset, class_weights
    
    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        class_weights: Dict[bool, float]
    ):
        """Setup trainer with class-weighted loss and evaluation"""
        
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
            eval_strategy="steps",  # Changed from evaluation_strategy (deprecated)
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=True,
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            bf16=True,
            save_total_limit=self.config.save_total_limit,
            report_to=[],  # Disable wandb and other loggers (we use custom logging)
            prediction_loss_only=False,  # Compute predictions, not just loss (needed for F1)
            dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        )
        
        # Data collator with LEFT padding for decoder-only models
        # Since pad_sequence pads to the right, we use reverse-pad-reverse pattern for left padding
        class DataCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            
            def __call__(self, features):
                batch = {}
                input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
                labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
                attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
                label_values = [f.get("label_value", False) for f in features]
                
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
                        torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
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
                batch["label_values"] = label_values
                
                return batch
        
        data_collator = DataCollator(self.tokenizer)
        
        # Custom trainer with class-weighted loss
        class WeightedTrainer(Trainer):
            def __init__(self, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = {
                    True: torch.tensor(class_weights[True], dtype=torch.float32),
                    False: torch.tensor(class_weights[False], dtype=torch.float32)
                }
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """
                Compute class-weighted loss.
                
                Loss is computed ONLY on answer tokens ("yes"/"no").
                Patient context tokens are masked (IGNORE_INDEX).
                
                Args:
                    model: The model to compute loss for
                    inputs: Dictionary of inputs (input_ids, labels, attention_mask, label_values)
                    return_outputs: Whether to return model outputs
                    **kwargs: Additional arguments (e.g., num_items_in_batch) - ignored
                """
                labels = inputs.get("labels")
                label_values = inputs.get("label_values", None)
                
                # Remove label_values from model inputs
                model_inputs = {k: v for k, v in inputs.items() if k != "label_values"}
                outputs = model(**model_inputs)
                logits = outputs.logits
                
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Per-token loss
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Aggregate to per-sample loss
                batch_size = labels.size(0)
                per_token_loss = per_token_loss.view(batch_size, -1)
                valid_mask = (shift_labels != IGNORE_INDEX).view(batch_size, -1)
                per_sample_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
                
                # Apply class weights
                if label_values is not None and model.training:
                    device = per_sample_loss.device
                    sample_weights = torch.tensor([
                        self.class_weights[bool(lv)].item() for lv in label_values
                    ], device=device, dtype=per_sample_loss.dtype)
                    loss = (per_sample_loss * sample_weights).mean()
                else:
                    loss = per_sample_loss.mean()
                
                return (loss, outputs) if return_outputs else loss
            
            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
                """Override evaluate to compute F1 score only (no validation loss)"""
                # Clear CUDA cache before evaluation to free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Skip standard evaluation (which computes loss) to save memory
                # Only compute F1 score which is more memory-efficient
                eval_results = {}
                
                # Compute F1 score using batch inference
                if eval_dataset is None:
                    eval_dataset = self.eval_dataset
                
                f1_score_value = self._compute_f1_score()
                eval_results[f"{metric_key_prefix}_f1"] = f1_score_value
                
                # Clear cache after computation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                return eval_results
            
            def _compute_f1_score(self) -> float:
                """
                Compute F1 score using optimized batch inference.
                
                Uses direct logit comparison (much faster than text generation):
                - Single forward pass per batch
                - Compare probabilities of "yes" vs "no" tokens
                - No text generation or decoding needed
                """
                # Get references (set after trainer creation)
                eval_dataset_raw = getattr(self, 'eval_dataset_raw', None)
                processor = getattr(self, 'processor', None)
                config = getattr(self, 'config', None)
                tokenizer = getattr(self, 'tokenizer', None)
                
                if eval_dataset_raw is None or processor is None or config is None or tokenizer is None:
                    logger.error("Missing required attributes for F1 computation")
                    return 0.0
                
                self.model.eval()
                predictions = []
                ground_truths = []
                
                device = next(self.model.parameters()).device
                eval_batch_size = config.eval_batch_size
                
                logger.info(f"Computing F1 score on {len(eval_dataset_raw)} examples (batch size: {eval_batch_size})...")
                
                with torch.no_grad():
                    # Clear CUDA cache before evaluation to free up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    for batch_start in range(0, len(eval_dataset_raw), eval_batch_size):
                        batch_end = min(batch_start + eval_batch_size, len(eval_dataset_raw))
                        batch_examples = eval_dataset_raw[batch_start:batch_end]
                        
                        # Prepare batch
                        batch_prompts = []
                        batch_labels = []
                        for example in batch_examples:
                            prompt = processor.create_prompt(example["context"])
                            batch_prompts.append(prompt)
                            batch_labels.append(1 if example["label_value"] else 0)
                        
                        # Tokenize
                        batch_inputs = tokenizer(
                            batch_prompts,
                            return_tensors="pt",
                            truncation=True,
                            max_length=config.max_length,
                            padding=True
                        )
                        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                        
                        # Forward pass
                        with torch.cuda.amp.autocast():
                            outputs = self.model(**batch_inputs)
                            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                        
                        # Get logits at last token position (predicts next token: "yes" or "no")
                        batch_size = logits.size(0)
                        attention_mask = batch_inputs['attention_mask']
                        seq_lens = attention_mask.sum(dim=1)
                        last_token_indices = (seq_lens - 1).long().clamp(0, logits.size(1) - 1)
                        batch_indices = torch.arange(batch_size, device=device)
                        last_token_logits = logits[batch_indices, last_token_indices, :]
                        
                        # Get probabilities
                        probs = torch.softmax(last_token_logits, dim=-1)
                        
                        # Compare "yes" vs "no" probabilities
                        yes_token_id = processor.yes_tokens[0]
                        no_token_id = processor.no_tokens[0]
                        
                        yes_probs = probs[:, yes_token_id]
                        no_probs = probs[:, no_token_id]
                        
                        # Predict based on higher probability (move to CPU before numpy conversion)
                        batch_predictions = (yes_probs > no_probs).long().cpu().numpy()
                        
                        predictions.extend(batch_predictions.tolist())
                        ground_truths.extend(batch_labels)
                        
                        # Clear cache after each batch to prevent OOM
                        # Delete tensors that are no longer needed
                        del outputs, logits, last_token_logits, probs, yes_probs, no_probs, batch_inputs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Compute metrics
                try:
                    f1 = f1_score(ground_truths, predictions, zero_division=0)
                    precision = precision_score(ground_truths, predictions, zero_division=0)
                    recall = recall_score(ground_truths, predictions, zero_division=0)
                    accuracy = accuracy_score(ground_truths, predictions)
                    
                    logger.info(f"Eval metrics - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
                    return f1
                except Exception as e:
                    logger.error(f"Error computing F1 score: {e}")
                    return 0.0
        
        # Callback for tracking losses
        class LossTrackingCallback(TrainerCallback):
            def __init__(self, trainer_instance):
                self.trainer_instance = trainer_instance
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None:
                    # Track training loss
                    if "loss" in logs:
                        self.trainer_instance.train_losses.append(logs["loss"])
                        self.trainer_instance.train_steps.append(state.global_step)
                    
                    # Track F1 score
                    if "eval_f1" in logs:
                        self.trainer_instance.eval_f1_scores.append(logs["eval_f1"])
                        self.trainer_instance.eval_steps.append(state.global_step)
        
        # Create trainer
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[
                LossTrackingCallback(self),
                EarlyStoppingCallback(early_stopping_patience=3)
            ],
        )
        
        # Store references for F1 computation
        trainer.eval_dataset_raw = self.eval_data_raw
        trainer.processor = self.processor
        trainer.config = self.config
        trainer.tokenizer = self.tokenizer
        
        self.trainer = trainer
        
        logger.info("Trainer setup complete")
    
    def train(self):
        """Start training"""
        logger.info("Starting training...")
        
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer first.")
        
        self.model.train()
        self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("Training completed!")
        
        # Plot losses
        self.plot_losses()
    
    def plot_losses(self):
        """Plot training loss and validation F1 score"""
        if not self.train_losses and not self.eval_f1_scores:
            logger.warning("No data to plot")
            return
        
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots (2 plots: training loss and F1 score)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Training loss
        if self.train_losses:
            axes[0].plot(self.train_steps, self.train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
            axes[0].set_xlabel('Training Step', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].axis('off')
        
        # Plot 2: Validation F1 score
        if self.eval_f1_scores:
            axes[1].plot(self.eval_steps, self.eval_f1_scores, 'g-', label='Validation F1', linewidth=2, marker='^', markersize=3)
            axes[1].set_xlabel('Training Step', fontsize=12)
            axes[1].set_ylabel('F1 Score', fontsize=12)
            axes[1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([0, 1])
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {plot_path}")
        
        plt.close()
        
        # Print summary
        if self.train_losses:
            logger.info(f"Final train loss: {self.train_losses[-1]:.4f}")
        if self.eval_f1_scores:
            best_f1 = max(self.eval_f1_scores)
            best_step = self.eval_steps[self.eval_f1_scores.index(best_f1)]
            logger.info(f"Best validation F1: {best_f1:.4f} (at step {best_step})")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Qwen3-8B Fine-Tuning Pipeline")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Model name or path")
    parser.add_argument("--task_name", type=str, required=True,
                       choices=['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer'],
                       help="Task name")
    parser.add_argument("--serialized_data_dir", type=str, required=True,
                       help="Path to directory with serialized data files")
    parser.add_argument("--output_dir", type=str, default="./qwen3_sft_output",
                       help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="Evaluation steps (save_steps will be set to the same value)")
    parser.add_argument("--gpu_id", type=str, default=None,
                       help="GPU ID(s) to use (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set GPU
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    # Create config
    # Ensure save_steps equals eval_steps (required for load_best_model_at_end)
    # This ensures checkpoints are saved when evaluated, so the best model can be loaded
    config = TrainingConfig(
        model_name=args.model_name,
        task_name=args.task_name,
        serialized_data_dir=args.serialized_data_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,  # Set to same as eval_steps
        seed=args.seed
    )
    
    # Validate that save_steps equals eval_steps
    if config.save_steps != config.eval_steps:
        raise ValueError(
            f"save_steps ({config.save_steps}) must equal eval_steps ({config.eval_steps}) "
            "for load_best_model_at_end to work correctly. Checkpoints must be saved when evaluated."
        )
    
    logger.info(f"Evaluation every {config.eval_steps} steps, checkpoint saving every {config.save_steps} steps")
    
    # Initialize trainer
    trainer = FineTuningTrainer(config)
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Load datasets
    train_dataset, eval_dataset, class_weights = trainer.load_datasets()
    
    # Setup trainer
    trainer.setup_trainer(train_dataset, eval_dataset, class_weights)
    
    # Train
    trainer.train()
    
    logger.info(f"Fine-tuning completed for task: {args.task_name}!")


if __name__ == "__main__":
    main()

