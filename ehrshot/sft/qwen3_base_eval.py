#!/usr/bin/env python3
"""
Single-task evaluation script for base models.

This script evaluates ONE task per invocation. The shell script 
(run_base_model_multi_task_evaluation.sh) handles parallelization by launching
multiple instances of this script, one for each task on different GPUs.

Tasks that can be evaluated:
- Acute MI (new_acutemi)
- Hyperlipidemia (new_hyperlipidemia)
- Hypertension (new_hypertension)
- Pancreatic Cancer (new_pancan)

Uses VLLM for inference with multiple samples per patient and KV caching.
Each task runs on a separate GPU (one GPU per task for data parallel evaluation).
"""

import os
import fcntl
import sys
import json
import argparse
import re
# Import numpy first to avoid binary incompatibility issues
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Literal
from contextlib import contextmanager
from dataclasses import dataclass
from collections import Counter
import torch
import gc
# Import sklearn before VLLM to avoid numpy issues
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, precision_score, recall_score, f1_score
from transformers import AutoTokenizer
from loguru import logger


try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    print(f"ERROR: Failed to import VLLM: {e}")
    print("Please ensure VLLM is installed: pip install vllm")
    print("If you encounter numpy compatibility issues, try: pip install --upgrade numpy")
    raise


def extract_model_name_for_dir(base_model_name: str) -> str:
    """
    Extract a clean model name from the base model path for use in directory names.
    
    Examples:
        "Qwen/Qwen3-8B" -> "qwen3-8b"
        "Qwen/Qwen3-4B-Thinking" -> "qwen3-4b-thinking"
        "meta-llama/Llama-3-8B" -> "llama-3-8b"
    
    Args:
        base_model_name: Full model path/name (e.g., "Qwen/Qwen3-8B")
    
    Returns:
        Clean model name suitable for directory names (lowercase, normalized)
    """
    # Extract the last part of the path (model name without org/user)
    model_name = base_model_name.split("/")[-1]
    # Convert to lowercase and replace underscores/hyphens consistently
    model_name = model_name.lower()
    # Normalize separators (ensure consistent formatting)
    model_name = re.sub(r'[_\s]+', '-', model_name)
    return model_name

TASK_MAPPINGS = {
    'acute_mi': 'new_acutemi',
    'hyperlipidemia': 'new_hyperlipidemia',
    'hypertension': 'new_hypertension',
    'pancreatic_cancer': 'new_pancan'
}

TASK_QUERIES = {
    'acute_mi': 'Will the patient develop an acute myocardial infarction in the next year?',
    'hyperlipidemia': 'Will the patient develop hyperlipidemia in the next year?',
    'hypertension': 'Will the patient develop hypertension in the next year?',
    'pancreatic_cancer': 'Will the patient develop pancreatic cancer in the next year?'
}

# Import serialization modules will be done inside the methods that use them
# This avoids numpy compatibility issues with nptyping
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


@contextmanager
def _exclusive_file_lock(target_file: str):
    """Context manager providing an exclusive lock using a companion .lock file."""
    lock_file = f"{target_file}.lock"
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@dataclass
class MultiTaskEvaluationConfig:
    """
    Configuration for multi-task model evaluation.
    
    The output directory will automatically include the model name extracted from base_model_name.
    Quantization is not used - models are loaded in full precision (bfloat16).
    """
    
    # Model configuration
    base_model_name: str = "Qwen/Qwen3-8B"
    trust_remote_code: bool = True
    # Note: Quantization is not supported - models are loaded in bfloat16 precision
    
    # VLLM configuration
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism per task
    gpu_memory_utilization: float = 0.85  # GPU memory utilization (0-1)
    max_model_len: int = 20000  # Maximum sequence length supported by the model
    
    # Evaluation configuration
    max_new_tokens: int = 8500  # Maximum tokens to generate per sample
    temperature: float = 0.7  # Sampling temperature for diverse outputs
    top_p: float = 0.9  # Nucleus sampling parameter
    num_samples: int = 10  # Number of samples to generate per patient
    data_fraction: float = 1.0  # Fraction of test data to use (1.0 = all data, 0.05 = 5% for pilot runs)
    
    # Data configuration
    path_to_serialized_data: str = ""  # Path to directory with pre-serialized JSON files (required)
    
    # Output configuration
    output_dir: str = "./base_model_multi_task_results"  # Output directory for results


class MultiTaskBaseModelEvaluator:
    """
    Evaluator for base model on a single task using VLLM.
    
    Designed to be called once per task. The shell script handles parallelization
    by launching separate processes for each task on different GPUs.
    
    Models are loaded in bfloat16 precision (no quantization).
    Results are saved to a directory that includes the model name.
    """
    
    def __init__(self, config: MultiTaskEvaluationConfig):
        """Initialize the evaluator with configuration"""
        self.config = config
        self.tokenizer = None
        self.llm = None
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer using VLLM"""
        logger.info(f"Loading tokenizer: {self.config.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )
        
        # VLLM configuration - no quantization, using bfloat16 precision
        vllm_kwargs = {
            "model": self.config.base_model_name,
            "trust_remote_code": self.config.trust_remote_code,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "dtype": "bfloat16",  # Full precision bfloat16 (no quantization)
            "enable_prefix_caching": True,  # Enable KV cache reuse for efficiency
        }
        
        # Load base model with VLLM (no quantization)
        logger.info(f"Loading base model with VLLM from: {self.config.base_model_name}")
        logger.info(f"Model will be loaded in bfloat16 precision (no quantization)")
        self.llm = LLM(**vllm_kwargs)
        
        logger.info("VLLM model and tokenizer loaded successfully")
    
    def load_pre_serialized_data(self, task_name: str) -> List[Dict]:
        """Load pre-serialized EHR data from JSON file for a specific task"""
        # Map task name to JSON file
        json_file = os.path.join(self.config.path_to_serialized_data, f"{task_name}_all_splits.json")
        
        if not os.path.exists(json_file):
            logger.error(f"Serialized data file not found for task {task_name}: {json_file}")
            return []
        
        # Load JSON file
        logger.info(f"Loading pre-serialized data from: {json_file}")
        with open(json_file, 'r') as f:
            all_data = json.load(f)
        
        # Filter to test split only
        test_data = [item for item in all_data if item.get('split') == 'test']
        
        # Limit data to specified fraction (for pilot runs)
        if self.config.data_fraction < 1.0 and self.config.data_fraction > 0:
            original_count = len(test_data)
            # Use random sampling to get a representative subset
            # Fixed seed for reproducibility across runs
            np.random.seed(42)
            num_samples = max(1, int(len(test_data) * self.config.data_fraction))
            indices = np.random.choice(len(test_data), size=num_samples, replace=False)
            test_data = [test_data[i] for i in sorted(indices)]  # Sort for deterministic ordering
            logger.info(f"Limited test data from {original_count} to {len(test_data)} examples "
                       f"({self.config.data_fraction*100:.1f}% of test set) - PILOT RUN MODE")
        elif self.config.data_fraction <= 0:
            logger.error(f"Invalid data_fraction: {self.config.data_fraction}. Must be > 0")
            return []
        else:
            logger.info(f"Loaded {len(test_data)} test examples from {json_file} (full dataset)")
        
        return test_data
    
    def _create_prompt_from_context(self, task_name: str, context: str) -> str:
        """Create full prompt from raw EHR context by adding instructions and query"""
        # Get query for this task
        query = TASK_QUERIES.get(task_name, f"will the patient develop {task_name.replace('_', ' ')} in the next year")
        
        # Format prompt
        prompt = (
            f"You are a helpful medical assistant. "
            f"You will be given a patient's electronic healthcare record (EHR) in Markdown format and a query. "
            f"You will need to answer the query using clinical reasoning. "
            f"Query: {query} \n\n "
            f"Patient Medical History: \n\n {context} \n\n "
            f"Please answer the following query using clinical reasoning: {query}. "
            f"IMPORTANT: You must conclude your answer with your final prediction in the following format: "
            f"Final Answer: [Positive/Negative]<STOP> \n\n"
            f"Assistant: "
        )

        return prompt
    
    def _get_sampling_params(self, num_samples: int = 1) -> SamplingParams:
        """Get sampling parameters for VLLM generation"""
        return SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            repetition_penalty=1.1,
            n=num_samples,
            stop=["<STOP>"]
        )
    
    def generate_multiple_predictions(self, prompt: str, num_samples: int) -> List[str]:
        """Generate multiple predictions for a single prompt"""
        all_responses = []
        
        for sample_idx in range(num_samples):
            try:
                outputs = self.llm.generate([prompt], self._get_sampling_params())
                generated_text = outputs[0].outputs[0].text.strip()
                
                # Clean up any remaining artifacts
                if generated_text.startswith("<|im_end|>"):
                    generated_text = generated_text[len("<|im_end|>"):].strip()
                
                all_responses.append(generated_text)
            except Exception as e:
                logger.error(f"Error generating sample {sample_idx+1}: {e}")
                all_responses.append("Error in generation")
        
        return all_responses
    
    def generate_predictions_batch(self, examples: List[Dict], num_samples: int) -> List[Dict]:
        """Generate predictions for a batch of examples using VLLM native n sampling"""
        # Prepare prompts (one per patient)
        prompts = []
        valid_examples = []
        
        for example in examples:
            prompt = example['prompt']
            # Check prompt length
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            max_prompt_tokens = self.config.max_model_len - self.config.max_new_tokens
            
            if len(prompt_tokens) <= max_prompt_tokens:
                prompts.append(prompt)
                valid_examples.append(example)
            else:
                logger.warning(f"Skipping patient {example['patient_id']}: prompt too long ({len(prompt_tokens)} tokens)")
        
        if not prompts:
            return []
        
        # Generate using VLLM with native n-sampling for multi-output
        try:
            # Use n parameter to get multiple outputs from single prefill
            sampling_params = self._get_sampling_params(num_samples)
            
            outputs = self.llm.generate(prompts, sampling_params)
            results = []
            
            # Process outputs - each request now has n outputs
            for i, (output, example) in enumerate(zip(outputs, valid_examples)):
                # Extract all n samples from this output
                sample_responses = []
                for sample_idx in range(num_samples):
                    if sample_idx < len(output.outputs):
                        generated_text = output.outputs[sample_idx].text.strip()
                        
                        # Clean up any remaining artifacts (VLLM should already have stopped)
                        if generated_text.startswith("<|im_end|>"):
                            generated_text = generated_text[len("<|im_end|>"):].strip()
                        
                        sample_responses.append(generated_text)
                    else:
                        # Fallback if fewer outputs than requested
                        sample_responses.append("Error: missing output")
                
                results.append({
                    'patient_id': example['patient_id'],
                    'label_time': example.get('label_time'),  # Include label_time in results
                    'prompt': example.get('prompt'),  # Include original prompt in results
                    'responses': sample_responses,
                    'ground_truth': example['label_value']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch generation with n-sampling: {e}")
            # Fallback to individual generation
            results = []
            for example in examples:
                try:
                    responses = self.generate_multiple_predictions(example['prompt'], num_samples)
                    results.append({
                        'patient_id': example['patient_id'],
                        'label_time': example.get('label_time'),  # Include label_time in results
                        'prompt': example.get('prompt'),  # Include original prompt in results
                        'responses': responses,
                        'ground_truth': example['label_value']
                    })
                except Exception as e2:
                    logger.error(f"Error generating for patient {example['patient_id']}: {e2}")
                    results.append({
                        'patient_id': example['patient_id'],
                        'label_time': example.get('label_time'),  # Include label_time in results
                        'prompt': example.get('prompt'),  # Include original prompt in results
                        'responses': ["Error in generation"] * num_samples,
                        'ground_truth': example['label_value']
                    })
            return results
    
    def _extract_with_regex(self, response: str) -> Optional[int]:
        """Extract binary prediction using regex (fallback method)"""
        response_lower = response.lower()
        
        # Look for "Final Answer: [Positive/Negative]" pattern first
        if "final answer:" in response_lower:
            final_answer = response_lower.split("final answer:")[-1].strip()
            if "positive" in final_answer:
                return 1
            elif "negative" in final_answer:
                return 0
        
        return None
    
    def _aggregate_predictions(self, predictions: List[int]) -> float:
        """Aggregate multiple predictions into a single probability score"""
        positive_count = sum(predictions)
        total_count = len(predictions)
        return positive_count / total_count if total_count > 0 else 0.5
    
    def evaluate_task(self, task_name: str) -> Dict:
        """
        Evaluate base model on a single task.
        
        Args:
            task_name: Name of the task to evaluate (e.g., 'acute_mi')
        
        Returns:
            Dictionary containing evaluation results including metrics and predictions
        """
        logger.info(f"Evaluating task: {task_name}")
        
        # Load pre-serialized data
        logger.info(f"Using pre-serialized data from: {self.config.path_to_serialized_data}")
        data_examples = self.load_pre_serialized_data(task_name)
        if not data_examples:
            logger.warning(f"No pre-serialized data found for task {task_name}")
            return None
        
        # Convert pre-serialized data to format expected by evaluation loop
        # Each example contains: patient_id, label_time, label_value, and raw context
        serialized_examples = []
        for item in data_examples:
            # Create prompt from raw context (adds system instructions and task-specific query)
            prompt = self._create_prompt_from_context(task_name, item['context'])
            serialized_examples.append({
                'patient_id': item['patient_id'],
                'label_time': item['label_time'],
                'label_value': item['label_value'],
                'prompt': prompt
            })
        
        # Generate predictions in batches
        # Each batch processes multiple patients, with num_samples outputs per patient
        batch_size = 16  # Process 16 patients at a time with n-sampling
        all_aggregated_predictions = []
        all_predictions_list = []
        all_responses_list = []
        ground_truths = []
        patient_ids = []
        
        # Create task-specific output directory: output_dir/task_name/
        task_output_dir = os.path.join(self.config.output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        incremental_output_file = os.path.join(task_output_dir, f"{task_name}_detailed_results.json")
        is_first_write = True
        
        for i in range(0, len(serialized_examples), batch_size):
            batch = serialized_examples[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(serialized_examples) + batch_size - 1)//batch_size} for task {task_name} ({len(batch)} examples)")
            
            # Generate predictions for batch
            batch_results = self.generate_predictions_batch(batch, self.config.num_samples)
            
            # Process results and save incrementally
            for result in batch_results:
                responses = result['responses']
                predictions_with_none = [self._extract_with_regex(r) for r in responses]
                
                predictions = [p for p in predictions_with_none if p is not None]
                
                # Track invalid format (None) predictions
                num_invalid = sum(1 for p in predictions_with_none if p is None)
                num_total = len(predictions_with_none)
                
                aggregated_pred = self._aggregate_predictions(predictions)
                
                all_aggregated_predictions.append(aggregated_pred)
                all_predictions_list.append(predictions)
                all_responses_list.append(result['responses'])
                ground_truths.append(1 if result['ground_truth'] else 0)
                patient_ids.append(result['patient_id'])
                
                # Use label_time and prompt directly from result (already included in batch results)
                label_time = result.get('label_time')
                original_prompt = result.get('prompt')
                
                # Get a sample response for logging purposes
                sample_idx = np.random.randint(0, len(responses)) if responses else 0
                sample_response = responses[sample_idx] if responses else ""
                
                # Save this patient's results incrementally
                patient_output = {
                    'patient_id': result['patient_id'],
                    'label_time': label_time,
                    'ground_truth': result['ground_truth'],
                    'score': aggregated_pred,
                    'prompt': original_prompt,
                    'sample_response': sample_response,
                    'num_samples': len(responses),
                    'num_valid_predictions': len(predictions),
                    'num_invalid_predictions': num_invalid,
                    'invalid_format_percentage': float(num_invalid / num_total * 100) if num_total > 0 else 0.0,
                    'predictions': predictions
                }
                self._save_outputs_incremental([patient_output], incremental_output_file, is_first=is_first_write)
                is_first_write = False
        
        # Convert to numpy arrays
        aggregated_predictions = np.array(all_aggregated_predictions)
        ground_truths = np.array(ground_truths)
        
        # Calculate invalid format statistics
        total_responses = sum(len(responses) for responses in all_responses_list)
        total_valid_predictions = sum(len(preds) for preds in all_predictions_list)
        total_invalid_predictions = total_responses - total_valid_predictions
        invalid_format_percentage = (total_invalid_predictions / total_responses * 100) if total_responses > 0 else 0.0
        
        # Compute metrics
        try:
            auroc = roc_auc_score(ground_truths, aggregated_predictions)
        except ValueError:
            auroc = 0.5
        
        # Precision-recall curve
        precision_vals, recall_vals, thresholds = precision_recall_curve(ground_truths, aggregated_predictions)
        fpr, tpr, roc_thresholds = roc_curve(ground_truths, aggregated_predictions)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        # Binary predictions at optimal threshold
        binary_predictions = (aggregated_predictions >= optimal_threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(ground_truths, binary_predictions, zero_division=0)
        recall = recall_score(ground_truths, binary_predictions, zero_division=0)
        f1 = f1_score(ground_truths, binary_predictions, zero_division=0)
        
        results = {
            'task_name': task_name,
            'n_examples': len(serialized_examples),
            'n_samples_per_example': self.config.num_samples,
            'auroc': auroc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'optimal_threshold': optimal_threshold,
            'invalid_format_percentage': float(invalid_format_percentage),
            'total_responses': int(total_responses),
            'total_valid_predictions': int(total_valid_predictions),
            'total_invalid_predictions': int(total_invalid_predictions),
            'aggregated_predictions': aggregated_predictions.tolist(),
            'ground_truths': ground_truths.tolist(),
            'patient_ids': patient_ids,
            'all_predictions': all_predictions_list,
            'all_responses': all_responses_list
        }
        
        logger.info(f"Task {task_name} Results:")
        logger.info(f"  AUROC: {auroc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        logger.info(f"  Optimal Threshold: {optimal_threshold:.4f}")
        logger.info(f"  Invalid Format Percentage: {invalid_format_percentage:.2f}% ({total_invalid_predictions}/{total_responses})")
        
        # Save summary metrics to JSON file in task directory
        summary_metrics = {
            'task_name': task_name,
            'auroc': float(auroc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'optimal_threshold': float(optimal_threshold),
            'invalid_format_percentage': float(invalid_format_percentage),
            'total_responses': int(total_responses),
            'total_valid_predictions': int(total_valid_predictions),
            'total_invalid_predictions': int(total_invalid_predictions)
        }
        summary_file = os.path.join(task_output_dir, f"{task_name}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        logger.info(f"Summary metrics saved to {summary_file}")
        
        return results
    
    def _save_outputs_incremental(self, outputs: List[Dict], output_file: str, is_first: bool = False):
        """Save outputs incrementally to JSON file"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        if is_first:
            # Create new file
            with open(output_file, 'w') as f:
                json.dump(outputs, f, indent=2)
        else:
            # Read existing data, append new entries, and write back
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # If file doesn't exist or is corrupted, start fresh
                data = []
            
            data.extend(outputs)
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(outputs)} outputs to {output_file}")
    
    def save_results(self, task_results: Dict):
        """
        Save evaluation results for a single task.
        
        Saves:
        - JSON file with full results in task directory
        - CSV file with per-patient predictions in task directory
        - Updates the combined summary CSV in model directory (parent directory)
        
        Args:
            task_results: Dictionary containing evaluation results for one task
        """
        if not task_results:
            return
        
        task_name = task_results['task_name']
        # Task-specific directory: output_dir/task_name/
        task_output_dir = os.path.join(self.config.output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Save JSON file with full results in task directory
        task_json_file = os.path.join(task_output_dir, f"{task_name}_results.json")
        with open(task_json_file, 'w') as f:
            json.dump(task_results, f, indent=2)
        logger.info(f"Task {task_name} results saved to {task_json_file}")
        
        # Save CSV file with per-patient predictions in task directory
        task_csv_data = []
        for patient_id, ground_truth, aggregated_pred, predictions_list in zip(
            task_results['patient_ids'],
            task_results['ground_truths'],
            task_results['aggregated_predictions'],
            task_results['all_predictions']
        ):
            # Calculate binary prediction at optimal threshold
            binary_pred = 1 if aggregated_pred >= task_results['optimal_threshold'] else 0
            
            # Create row with per-patient information
            row = {
                'patient_id': patient_id,
                'ground_truth': ground_truth,
                'aggregated_prediction': aggregated_pred,
                'binary_prediction': binary_pred,
            }
            
            # Add individual sample predictions (binary predictions)
            for sample_idx, pred in enumerate(predictions_list):
                row[f'sample_{sample_idx+1}_prediction'] = pred
            
            task_csv_data.append(row)
        
        task_csv_df = pd.DataFrame(task_csv_data)
        task_csv_file = os.path.join(task_output_dir, f"{task_name}_results.csv")
        task_csv_df.to_csv(task_csv_file, index=False)
        logger.info(f"Task {task_name} CSV saved to {task_csv_file}")
        
        # Update combined summary CSV in model directory (parent of task directories)
        # This allows aggregating results from all tasks in one place
        summary_file = os.path.join(self.config.output_dir, "all_tasks_summary.csv")
        summary_row = {
            'task': task_name,
            'n_examples': task_results['n_examples'],
            'auroc': task_results['auroc'],
            'precision': task_results['precision'],
            'recall': task_results['recall'],
            'f1': task_results['f1'],
            'optimal_threshold': task_results['optimal_threshold']
        }
        
        # Use file lock to safely update the shared summary file
        with _exclusive_file_lock(summary_file):
            if os.path.exists(summary_file):
                try:
                    existing_df = pd.read_csv(summary_file)
                except (pd.errors.EmptyDataError, ValueError):
                    existing_df = pd.DataFrame(columns=list(summary_row.keys()))
            else:
                existing_df = pd.DataFrame(columns=list(summary_row.keys()))
            
            # Remove existing row for this task (if present) and add new one
            if 'task' in existing_df.columns:
                existing_df = existing_df[existing_df['task'] != task_name]
            
            # Add new row
            new_row_df = pd.DataFrame([summary_row])
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            updated_df.to_csv(summary_file, index=False)
        
        logger.info(f"Combined summary updated at {summary_file}")


def parse_args():
    """
    Parse command-line arguments for single-task evaluation.
    
    This script evaluates ONE task per invocation. The shell script handles parallelization
    by launching multiple instances of this script, one per task, on different GPUs.
    
    Note: Quantization is not supported - models are always loaded in bfloat16 precision.
    The output directory will automatically include the model name extracted from base_model_name.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate base model on a single task (one task per invocation). "
                    "The shell script handles parallelization across multiple tasks/GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single task on a specific GPU
  python %(prog)s --task_name acute_mi --gpu_id 0 --path_to_serialized_data ./data
  
  # The shell script (run_base_model_multi_task_evaluation.sh) handles parallel execution
  # by launching this script multiple times, once for each task on different GPUs.
        """
    )
    
    # Model configuration
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Base model name/path (e.g., 'Qwen/Qwen3-8B'). "
                            "Model name will be extracted and included in output directory.")
    
    # Data configuration
    parser.add_argument("--path_to_serialized_data", type=str, required=True,
                       help="Path to directory containing pre-serialized JSON files "
                            "(e.g., acute_mi_all_splits.json)")
    
    # VLLM configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of GPUs for tensor parallelism per task "
                            "(default: 1, for data parallel execution)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                       help="GPU memory utilization factor (0-1, default: 0.85)")
    parser.add_argument("--max_model_len", type=int, default=20000,
                       help="Maximum sequence length supported by the model (default: 20000)")
    
    # GPU configuration for data parallel execution
    parser.add_argument("--gpu_id", type=int, default=None,
                       help="GPU ID to use for this task (sets CUDA_VISIBLE_DEVICES). "
                            "Used by shell script for parallel task execution. "
                            "If None, uses all available GPUs.")
    
    # Task configuration
    parser.add_argument("--task_name", type=str, required=True,
                       choices=['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer'],
                       help="Task name to evaluate (required). "
                            "The shell script calls this script multiple times in parallel, "
                            "once for each task on a different GPU.")
    
    # Evaluation configuration
    parser.add_argument("--max_new_tokens", type=int, default=8500,
                       help="Maximum new tokens to generate per sample (default: 8500)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature for generation (default: 0.7)")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to generate per patient (default: 10)")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                       help="Fraction of test data to use (default: 1.0 = all data, "
                            "0.05 = 5%% for pilot runs). Uses random sampling with fixed seed for reproducibility.")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="",
                       help="Base output directory for results. "
                            "Model name will be automatically appended to create the final output directory. "
                            "If empty, defaults to './base_model_multi_task_results/{model_name}'")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set CUDA_VISIBLE_DEVICES for data parallel execution (one GPU per task)
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        logger.info(f"[GPU {args.gpu_id}] Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    # Compute output directory: base_dir/model_name
    model_name = extract_model_name_for_dir(args.base_model_name)
    if args.output_dir:
        # User provided output_dir - append model name if not already present
        base_output_dir = os.path.normpath(args.output_dir)
        if os.path.basename(base_output_dir) != model_name:
            output_dir = os.path.join(base_output_dir, model_name)
        else:
            output_dir = base_output_dir
    else:
        # Default: use ./base_model_multi_task_results/model_name
        output_dir = os.path.join("./base_model_multi_task_results", model_name)
    
    # Create configuration
    config = MultiTaskEvaluationConfig(
        base_model_name=args.base_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        data_fraction=args.data_fraction,
        path_to_serialized_data=args.path_to_serialized_data,
        output_dir=output_dir
    )
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.base_model_name} (directory name: {model_name})")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Task: {args.task_name}")
    logger.info(f"  Data fraction: {args.data_fraction} ({args.data_fraction*100:.1f}% of test set)")
    if args.gpu_id is not None:
        logger.info(f"  GPU: {args.gpu_id}")
    
    # Initialize evaluator
    evaluator = MultiTaskBaseModelEvaluator(config)
    
    # Load model
    evaluator.load_model_and_tokenizer()
    
    # Evaluate the specified task
    try:
        logger.info(f"Evaluating task: {args.task_name}")
        result = evaluator.evaluate_task(args.task_name)
        
        if result:
            # Save results for this task
            evaluator.save_results(result)
            logger.info(f"Task {args.task_name} evaluation completed successfully!")
        else:
            logger.error(f"Failed to evaluate task {args.task_name}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

