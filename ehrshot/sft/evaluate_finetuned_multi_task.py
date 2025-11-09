#!/usr/bin/env python3
"""
Multi-task evaluation pipeline for the finetuned Qwen model on 4 tasks:
- Acute MI (new_acutemi)
- Hyperlipidemia (new_hyperlipidemia)
- Hypertension (new_hypertension)
- Pancreatic Cancer (new_pancan)

Identical to the base-model evaluator except the default checkpoint points to
`qwen_sft_output`.
"""

import os
import fcntl
os.environ["VLLM_MP_START_METHOD"] = "spawn"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"   # older key
os.environ["VLLM_ENABLE_MP"] = "1"

# While debugging, avoid FlashAttention probing (removes that CUDA call path)
os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
import sys
import json
import argparse
import hashlib
import multiprocessing as mp
# Import numpy first to avoid binary incompatibility issues
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Literal
from contextlib import contextmanager
from dataclasses import dataclass
from collections import Counter
import torch
import torch.multiprocessing as torch_mp
import gc
# Import sklearn before VLLM to avoid numpy issues
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger

# Force Python and torch multiprocessing to use spawn start method before any CUDA init
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
try:
    torch_mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# Import VLLM last to minimize numpy compatibility issues
try:
    from vllm import LLM, SamplingParams
except ImportError as exc:
    print(f"ERROR: Failed to import VLLM: {exc}")
    print("Please ensure VLLM is installed: pip install vllm")
    print("If you encounter numpy compatibility issues, try: pip install --upgrade numpy")
    raise

# Task name mappings
TASK_MAPPINGS = {
    'acute_mi': 'new_acutemi',
    'hyperlipidemia': 'new_hyperlipidemia',
    'hypertension': 'new_hypertension',
    'pancreatic_cancer': 'new_pancan'
}

# Task-specific queries for prompt formatting
TASK_QUERIES = {
    'acute_mi': 'will the patient develop an acute myocardial infarction in the next year',
    'hyperlipidemia': 'will the patient develop hyperlipidemia in the next year',
    'hypertension': 'will the patient develop hypertension in the next year',
    'pancreatic_cancer': 'will the patient develop pancreatic cancer in the next year'
}

# Import serialization modules will be done inside the methods that use them
# This avoids numpy compatibility issues with nptyping
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

FINETUNED_MODEL_DEFAULT = os.path.join(os.path.dirname(__file__), "qwen_sft_output")
CACHE_MERGED_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ehrshot_merged_models")
os.makedirs(CACHE_MERGED_MODELS_DIR, exist_ok=True)


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


def _maybe_merge_peft_adapter(model_path: str) -> str:
    """
    If `model_path` points to a PEFT adapter directory, merge it with its base model
    (following the approach used in evaluate_sft_model.py) and return the merged model path.
    Otherwise, return the original path.
    """
    if not (model_path and os.path.isdir(model_path)):
        return model_path

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return model_path

    logger.info(f"Detected PEFT adapter at {model_path}. Merging with base model for VLLM.")

    with open(adapter_config_path, "r") as f:
        adapter_cfg = json.load(f)

    base_model_name = adapter_cfg.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            f"PEFT adapter at {model_path} missing 'base_model_name_or_path' in adapter_config.json"
        )

    abs_model_path = os.path.abspath(model_path)
    cache_key = hashlib.md5(abs_model_path.encode("utf-8")).hexdigest()[:8]
    base_safe = base_model_name.replace("/", "_").replace("-", "_")
    merged_dir = os.path.join(CACHE_MERGED_MODELS_DIR, f"{base_safe}_merged_{cache_key}")

    if os.path.exists(os.path.join(merged_dir, "config.json")):
        logger.info(f"Using cached merged model at {merged_dir}")
        return merged_dir

    logger.info(f"Merging adapter into base model '{base_model_name}' -> {merged_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(base_model, model_path)
    merged_model = peft_model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(merged_dir)

    del base_model, peft_model, merged_model
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Merged model saved to {merged_dir}")
    return merged_dir


@dataclass
class MultiTaskEvaluationConfig:
    """Configuration for multi-task model evaluation"""
    
    # Model configuration
    base_model_name: str = FINETUNED_MODEL_DEFAULT
    trust_remote_code: bool = True
    use_quantization: bool = False
    
    
    # VLLM configuration
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 20000
    
    # Evaluation configuration
    max_new_tokens: int = 8500
    temperature: float = 0.7  # Higher temperature for diverse samples
    top_p: float = 0.9
    num_samples: int = 10  # 10 samples per patient
    
    # Data configuration
    path_to_serialized_data: str = ""  # Path to directory with pre-serialized JSON files (required)
    
    # Output configuration
    output_dir: str = "./base_model_multi_task_results"


class MultiTaskBaseModelEvaluator:
    """Evaluator for base model on multiple tasks using VLLM"""
    
    def __init__(self, config: MultiTaskEvaluationConfig):
        self.config = config
        self.tokenizer = None
        self.llm = None
        self.tasks = ['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer']
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer using VLLM"""
        logger.info(f"Loading tokenizer: {self.config.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )
        
        # VLLM configuration
        vllm_kwargs = {
            "model": self.config.base_model_name,
            "trust_remote_code": self.config.trust_remote_code,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "dtype": "bfloat16",
            "enable_prefix_caching": True,  # Enable automatic prefix caching for KV reuse
        }
        
        # Add quantization if specified
        if self.config.use_quantization:
            vllm_kwargs["quantization"] = "bitsandbytes"
        
        # Load base model with VLLM
        logger.info(f"Loading base model with VLLM from: {self.config.base_model_name}")
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
        
        logger.info(f"Loaded {len(test_data)} test examples from {json_file}")
        return test_data
    
    def _create_prompt_from_context(self, task_name: str, context: str) -> str:
        """Create full prompt from raw EHR context by adding instructions and query"""
        # Get query for this task
        query = TASK_QUERIES.get(task_name, f"will the patient develop {task_name.replace('_', ' ')} in the next year")
        
        # Format prompt
        prompt = f'''System: You may reason step by step, but your final output MUST be in the following format: "Final Answer: [Positive/Negative]<STOP>".\n\n
User: You are a helpful medical assistant. Below is a patient's electronic healthcare record (EHR) in Markdown format. Please answer the following query using clinically grounded reasoning: {query}. Patient Medical History:\n\n\n\n{context}\n\nIMPORTANT: You must conclude with your final prediction in the format "Final Answer: [Positive/Negative]<STOP>".\n\n Assistant: '''

        return prompt
    
    def _get_sampling_params(self, num_samples: int = 1) -> SamplingParams:
        """Get sampling parameters for VLLM generation"""
        return SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            repetition_penalty=1.1,
            n=num_samples,
            stop=["<STOP>"],

            
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
        """Evaluate base model on a single task"""
        logger.info(f"Evaluating task: {task_name}")
        
        # Load pre-serialized data
        logger.info(f"Using pre-serialized data from: {self.config.path_to_serialized_data}")
        data_examples = self.load_pre_serialized_data(task_name)
        if not data_examples:
            logger.warning(f"No pre-serialized data found for task {task_name}")
            return None
        
        # Convert pre-serialized data to format expected by evaluation loop
        serialized_examples = []
        for item in data_examples:
            # Create prompt from raw context
            prompt = self._create_prompt_from_context(task_name, item['context'])
            serialized_examples.append({
                'patient_id': item['patient_id'],
                'label_time': item['label_time'],
                'label_value': item['label_value'],
                'prompt': prompt
            })
        
        # Generate predictions in batches
        batch_size = 16  # Process 16 patients at a time with n-sampling (10 outputs per prompt)
        all_aggregated_predictions = []
        all_predictions_list = []
        all_responses_list = []
        ground_truths = []
        patient_ids = []
        
        # Create output directory for incremental saving
        os.makedirs(self.config.output_dir, exist_ok=True)
        incremental_output_file = os.path.join(self.config.output_dir, f"{task_name}_detailed_results.json")
        is_first_write = True
        
        for i in range(0, len(serialized_examples), batch_size):
            batch = serialized_examples[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(serialized_examples) + batch_size - 1)//batch_size} for task {task_name} ({len(batch)} examples)")
            
            # Generate predictions for batch
            batch_results = self.generate_predictions_batch(batch, self.config.num_samples)
            
            # Process results and save incrementally
            # Collect all responses for batch processing
            all_batch_responses = []
            responses_per_result = []  # Track how many responses per result
            
            for result in batch_results:
                responses = result['responses']
                responses_per_result.append(len(responses))
                all_batch_responses.extend(responses)
            
            # Extract predictions directly via regex (no parser LLM)
            all_predictions_with_none = [self._extract_with_regex(r) for r in all_batch_responses]
            all_parser_outputs = [""] * len(all_batch_responses)
            
            # Group predictions back by patient
            response_idx = 0
            for i, result in enumerate(batch_results):
                num_responses = responses_per_result[i]
                predictions_with_none = all_predictions_with_none[response_idx:response_idx + num_responses]
                parser_outputs = all_parser_outputs[response_idx:response_idx + num_responses]
                response_idx += num_responses
                
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
                
                # Get sample response and corresponding parser output
                sample_idx = np.random.randint(0, len(responses)) if responses else 0
                sample_response = responses[sample_idx] if responses else ""
                sample_parser_output = parser_outputs[sample_idx] if parser_outputs else ""
                
                # Save this patient's results incrementally
                patient_output = {
                    'patient_id': result['patient_id'],
                    'label_time': label_time,
                    'ground_truth': result['ground_truth'],
                    'score': aggregated_pred,
                    'prompt': original_prompt,
                    'sample_response': sample_response,
                    'sample_parser_output': sample_parser_output,
                    'num_samples': len(responses),
                    'num_valid_predictions': len(predictions),
                    'num_invalid_predictions': num_invalid,
                    'invalid_format_percentage': float(num_invalid / num_total * 100) if num_total > 0 else 0.0,
                    'predictions': predictions,
                    'parser_outputs': parser_outputs  # Include all parser outputs for this patient
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
        
        # Save summary metrics to JSON file
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
        summary_file = os.path.join(self.config.output_dir, f"{task_name}_summary.json")
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
    
    def evaluate_all_tasks(self) -> Dict:
        """Evaluate base model on all tasks"""
        all_results = {}
        
        for task_name in self.tasks:
            try:
                result = self.evaluate_task(task_name)
                if result:
                    all_results[task_name] = result
            except Exception as e:
                logger.error(f"Error evaluating task {task_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return all_results
    
    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results - one JSON and CSV per task"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual JSON and CSV files for each task
        summary_data = []
        
        for task_name, task_results in results.items():
            # Save individual JSON file for this task
            task_json_file = os.path.join(output_dir, f"{task_name}_results.json")
            with open(task_json_file, 'w') as f:
                json.dump(task_results, f, indent=2)
            logger.info(f"Task {task_name} results saved to {task_json_file}")
            
            # Save individual CSV file for this task (detailed per-patient results)
            task_csv_data = []
            for patient_id, ground_truth, aggregated_pred, predictions_list, responses_list in zip(
                task_results['patient_ids'],
                task_results['ground_truths'],
                task_results['aggregated_predictions'],
                task_results['all_predictions'],
                task_results['all_responses']
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
                
                # Optionally add sample responses (truncated to 200 chars to keep CSV manageable)
                # Uncomment if you want full responses in CSV
                # for sample_idx, response in enumerate(responses_list):
                #     row[f'sample_{sample_idx+1}_response'] = response[:200]
                
                task_csv_data.append(row)
            
            task_csv_df = pd.DataFrame(task_csv_data)
            task_csv_file = os.path.join(output_dir, f"{task_name}_results.csv")
            task_csv_df.to_csv(task_csv_file, index=False)
            logger.info(f"Task {task_name} CSV saved to {task_csv_file}")
            
            # Add to summary for overall summary table
            summary_data.append({
                'task': task_name,
                'n_examples': task_results['n_examples'],
                'auroc': task_results['auroc'],
                'precision': task_results['precision'],
                'recall': task_results['recall'],
                'f1': task_results['f1'],
                'optimal_threshold': task_results['optimal_threshold']
            })
        
        # Save combined summary CSV (optional - for easy comparison)
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(output_dir, "all_tasks_summary.csv")
            with _exclusive_file_lock(summary_file):
                if os.path.exists(summary_file):
                    try:
                        existing_df = pd.read_csv(summary_file)
                    except (pd.errors.EmptyDataError, ValueError):
                        existing_df = pd.DataFrame(columns=summary_df.columns)
                else:
                    existing_df = pd.DataFrame(columns=summary_df.columns)

                if 'task' in existing_df.columns:
                    existing_df = existing_df.dropna(subset=['task'])
                else:
                    existing_df = pd.DataFrame(columns=summary_df.columns)

                # Ensure column order aligns with new summary
                for col in summary_df.columns:
                    if col not in existing_df.columns:
                        existing_df[col] = pd.NA
                existing_df = existing_df[summary_df.columns]

                updated_df = pd.concat(
                    [
                        existing_df[~existing_df['task'].isin(summary_df['task'])],
                        summary_df
                    ],
                    ignore_index=True
                )
                updated_df.to_csv(summary_file, index=False)

            logger.info(f"Combined summary updated at {summary_file}")
            
            # Print summary table
            print("\n" + "="*80)
            print("EVALUATION SUMMARY - BASE MODEL (Qwen3-8B)")
            print("="*80)
            print(summary_df.to_string(index=False))
            print("="*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base Qwen3-8B model on 4 tasks")
    
    # Model configuration
    parser.add_argument("--base_model_name", type=str, default=FINETUNED_MODEL_DEFAULT,
                       help="Finetuned model path or identifier (default: qwen_sft_output)")
    parser.add_argument("--use_quantization", action="store_true", default=False,
                       help="Use quantization")
    
    # Data configuration
    parser.add_argument("--path_to_serialized_data", type=str, required=True,
                       help="Path to directory with pre-serialized JSON files")
    
    # VLLM configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                       help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=20000,
                       help="Maximum model length")
    
    # GPU configuration for parallel execution
    parser.add_argument("--gpu_id", type=int, default=None,
                       help="GPU ID to use (sets CUDA_VISIBLE_DEVICES). If None, uses all available GPUs")
    
    # Task configuration
    parser.add_argument("--task_name", type=str, default=None,
                       choices=['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer', None],
                       help="Task name to evaluate. If None, evaluates all tasks")
    
    # Evaluation configuration
    parser.add_argument("--max_new_tokens", type=int, default=8500,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples per patient")
    
    # Parser LLM configuration (always enabled)
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./finetuned_multi_task_results",
                       help="Output directory for results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set CUDA_VISIBLE_DEVICES if gpu_id is specified
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    try:
        resolved_model_path = _maybe_merge_peft_adapter(args.base_model_name)
        if resolved_model_path != args.base_model_name:
            logger.info(f"Using merged finetuned model: {resolved_model_path}")
        else:
            logger.info(f"Using finetuned model path: {resolved_model_path}")

        # Create configuration
        config = MultiTaskEvaluationConfig(
            base_model_name=resolved_model_path,
            use_quantization=args.use_quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_samples=args.num_samples,
            path_to_serialized_data=args.path_to_serialized_data,
            output_dir=args.output_dir
        )
        
        # Initialize evaluator
        evaluator = MultiTaskBaseModelEvaluator(config)
        
        # Load model
        evaluator.load_model_and_tokenizer()
    except ValueError as e:
        if "numpy.dtype size changed" in str(e) or "binary incompatibility" in str(e).lower():
            print("\n" + "=" * 70)
            print("ERROR: Numpy compatibility issue detected!")
            print("=" * 70)
            print("\nThis is a common issue with VLLM and numpy version mismatches.")
            print("\nTo fix this, run one of the following:")
            print("\n  Option 1 (Quick fix):")
            print("    pip install --upgrade --force-reinstall numpy")
            print("    pip install --upgrade --force-reinstall vllm scikit-learn pandas")
            print("\n  Option 2 (Recommended - downgrade numpy):")
            print("    pip install 'numpy<2.0'")
            print("    pip install --upgrade --force-reinstall vllm scikit-learn pandas")
            print("\n  Option 3 (Fresh environment):")
            print("    python -m venv venv")
            print("    source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            print("    pip install numpy==1.24.3")
            print("    pip install -r requirements_vllm.txt")
            print("\n" + "=" * 70 + "\n")
            sys.exit(1)
        raise
    
    # Evaluate tasks
    try:
        if args.task_name is not None:
            # Evaluate single task
            logger.info(f"Evaluating single task: {args.task_name}")
            result = evaluator.evaluate_task(args.task_name)
            if result:
                # Save single task result
                results = {args.task_name: result}
                evaluator.save_results(results, args.output_dir)
                logger.info(f"Task {args.task_name} evaluation completed successfully!")
            else:
                logger.error(f"Failed to evaluate task {args.task_name}")
                sys.exit(1)
        else:
            # Evaluate all tasks
            results = evaluator.evaluate_all_tasks()
            
            # Save results
            evaluator.save_results(results, args.output_dir)
            
            logger.info("Multi-task evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

