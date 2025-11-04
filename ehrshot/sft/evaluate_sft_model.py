#!/usr/bin/env python3
"""
Evaluation pipeline for fine-tuned SFT model on test dataset.
Computes AUROC and other metrics for model performance evaluation.
"""

import os
import sys
import json
import argparse
# Import numpy first to avoid binary incompatibility issues
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import torch
import gc
# Import sklearn before VLLM to avoid numpy issues
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
# Import VLLM last to minimize numpy compatibility issues
try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    print(f"ERROR: Failed to import VLLM: {e}")
    print("Please ensure VLLM is installed: pip install vllm")
    print("If you encounter numpy compatibility issues, try: pip install --upgrade numpy")
    raise


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # Model configuration
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    peft_model_path: str = "./qwen_sft_output"
    trust_remote_code: bool = True
    use_quantization: bool = False  # Disable quantization to avoid GPU detection issues
    
    # VLLM configuration
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 32768
    
    # Evaluation configuration
    max_new_tokens: int = 16384
    temperature: float = 0.1  # Low temperature for more deterministic outputs
    top_p: float = 0.9
    
    # Output configuration
    output_dir: str = "./evaluation_results"
    save_predictions: bool = True
    save_plots: bool = True


class SFTModelEvaluator:
    """Evaluator for fine-tuned SFT model and baseline comparison using VLLM"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.tokenizer = None
        self.fine_tuned_llm = None
        self.baseline_llm = None
        self.vllm_kwargs: Optional[Dict[str, Any]] = None
        
    def _get_merged_model_path(self) -> str:
        """Get or create path for merged model (adapter merged into base model)"""
        # Create a cache directory for merged models
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ehrshot_merged_models")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a unique identifier from base model name and PEFT path
        import hashlib
        base_model_name_safe = self.config.base_model_name.replace("/", "_").replace("-", "_")
        peft_path_abs = os.path.abspath(self.config.peft_model_path)
        # Create a stable hash from the absolute path
        peft_path_hash = hashlib.md5(peft_path_abs.encode()).hexdigest()[:8]
        merged_model_name = f"{base_model_name_safe}_merged_{peft_path_hash}"
        merged_model_path = os.path.join(cache_dir, merged_model_name)
        
        return merged_model_path
    
    def _merge_peft_adapter(self) -> str:
        """Merge PEFT adapter with base model and return path to merged model"""
        merged_model_path = self._get_merged_model_path()
        
        # Check if merged model already exists
        if os.path.exists(merged_model_path) and os.path.exists(os.path.join(merged_model_path, "config.json")):
            logger.info(f"Using existing merged model at: {merged_model_path}")
            return merged_model_path
        
        # Check if PEFT model path exists
        if not os.path.exists(self.config.peft_model_path):
            logger.error(f"PEFT model path does not exist: {self.config.peft_model_path}")
            raise FileNotFoundError(f"PEFT model path does not exist: {self.config.peft_model_path}")
        
        if not os.path.exists(os.path.join(self.config.peft_model_path, "adapter_config.json")):
            logger.error(f"PEFT adapter config not found at: {self.config.peft_model_path}")
            raise FileNotFoundError(f"PEFT adapter config not found at: {self.config.peft_model_path}")
        
        logger.info(f"Merging PEFT adapter from {self.config.peft_model_path} with base model {self.config.base_model_name}")
        logger.info(f"This may take a few minutes...")
        
        # Load base model
        logger.info(f"Loading base model: {self.config.base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load PEFT adapter
        logger.info(f"Loading PEFT adapter from: {self.config.peft_model_path}")
        peft_model = PeftModel.from_pretrained(base_model, self.config.peft_model_path)
        
        # Merge adapter into base model
        logger.info("Merging adapter into base model...")
        merged_model = peft_model.merge_and_unload()
        
        # Save merged model
        logger.info(f"Saving merged model to: {merged_model_path}")
        os.makedirs(merged_model_path, exist_ok=True)
        merged_model.save_pretrained(merged_model_path)
        
        # Also save tokenizer for consistency
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        tokenizer.save_pretrained(merged_model_path)
        
        # Clean up memory
        del base_model, peft_model, merged_model
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Merged model saved successfully to: {merged_model_path}")
        return merged_model_path
        
    def load_model_and_tokenizer(self):
        """Load the fine-tuned model and tokenizer using VLLM"""
        logger.info(f"Loading tokenizer: {self.config.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )
        
        # Merge PEFT adapter with base model to create a merged model
        merged_model_path = None
        try:
            merged_model_path = self._merge_peft_adapter()
            logger.info(f"Using merged model for fine-tuned evaluation: {merged_model_path}")
        except Exception as e:
            logger.error(f"Failed to merge PEFT adapter: {e}")
            logger.warning("Falling back to base model (fine-tuned weights will NOT be used)")
            merged_model_path = None
        
        # VLLM configuration for fine-tuned model (with merged adapter if available)
        fine_tuned_model_path = merged_model_path if merged_model_path else self.config.base_model_name
        vllm_kwargs_fine_tuned = {
            "model": fine_tuned_model_path,
            "trust_remote_code": self.config.trust_remote_code,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "dtype": "float16",  # Explicitly set dtype to avoid GPU capability detection issues
        }
        
        # Add quantization if specified
        if self.config.use_quantization:
            vllm_kwargs_fine_tuned["quantization"] = "bitsandbytes"  # Use a supported quantization method
        
        # Load fine-tuned model (with merged adapter)
        logger.info(f"Loading fine-tuned model with VLLM from: {fine_tuned_model_path}")
        self.fine_tuned_llm = LLM(**vllm_kwargs_fine_tuned)
        
        # VLLM configuration for baseline model (always use base model)
        vllm_kwargs_baseline = {
            "model": self.config.base_model_name,
            "trust_remote_code": self.config.trust_remote_code,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "dtype": "float16",
        }
        
        if self.config.use_quantization:
            vllm_kwargs_baseline["quantization"] = "bitsandbytes"
        
        # Store baseline kwargs for later loading
        self.vllm_kwargs = vllm_kwargs_baseline
        
        # Defer baseline model loading to avoid double GPU allocation
        logger.info("Baseline VLLM model will be loaded after fine-tuned evaluation to conserve GPU memory")
        
        logger.info("VLLM models and tokenizer loaded successfully")

    def _truncate_after_final_answer(self, text: str, extra_tokens: int = 5) -> str:
        """Keep full prefix, then truncate AFTER first 'Final Answer' by extra_tokens.
        If trigger not found or tokenization fails, return original text."""
        if not isinstance(text, str):
            return text
        trigger = "Final Answer"
        idx = text.find(trigger)
        if idx < 0:
            return text
        prefix = text[:idx]  # everything before the trigger
        suffix = text[idx:]  # include the trigger itself
        try:
            token_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
            # Keep first token(s) including trigger plus extra_tokens
            keep = min(len(token_ids), 1 + extra_tokens)
            truncated = self.tokenizer.decode(token_ids[:keep])
            return (prefix + truncated).strip()
        except Exception:
            parts = suffix.split()
            keep = min(len(parts), 2 + extra_tokens)
            return (prefix + " " + " ".join(parts[:keep])).strip()
    
    def load_test_dataset(self, dataset_file: str, path_to_splits: str) -> List[Dict]:
        """Load test dataset from file and filter by test split"""
        logger.info(f"Loading test dataset from {dataset_file}")
        
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Filter by test split
        logger.info(f"Filtering dataset by test split from {path_to_splits}")
        splits_df = pd.read_csv(path_to_splits)
        test_patient_ids = set(splits_df[splits_df['split'] == 'test']['omop_person_id'].values)
        
        # Filter dataset to only include test patients
        test_dataset = []
        for example in dataset:
            if 'patient_id' in example and example['patient_id'] in test_patient_ids:
                test_dataset.append(example)
        
        logger.info(f"Filtered to {len(test_dataset)} test examples")
        return test_dataset
    
    def format_prompt(self, example: Dict) -> str:
        """Format example into prompt for model"""
        # Extract patient history from the conversation
        user_content = example['conversations'][0]['content']
        
        # Force the model to output "Final Answer: [Positive/Negative]" format
        # Remove any existing "Final Answer:" instruction and add our own
        # if "Final Answer:" in user_content:
        #     # Remove existing final answer instruction
        #     user_content = user_content.split("Final Answer:")[0].strip()
        
        # # Add our standardized instruction
        # user_content += "\n\nPlease provide your analysis, and conclude with your final prediction in the format Final Answer: [Positive/Negative]."
        
        # Return as single completion prompt (no chat format)
        return user_content
    
    def _get_sampling_params(self) -> SamplingParams:
        """Get sampling parameters for VLLM generation"""
        # Use VLLM's built-in stop parameters instead of custom stopping criteria
        # Stop generation when "Final Answer" is detected
        return SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            #stop=["Final Answer"],  # Stop when "Final Answer" is generated
            repetition_penalty=1.1,
        )
    
    def generate_prediction(self, example: Dict, use_baseline: bool = False) -> Dict:
        """Generate prediction for a single example using VLLM"""
        prompt = self.format_prompt(example)
        
        # Check prompt length
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        max_prompt_tokens = self.config.max_model_len - self.config.max_new_tokens
        
        if len(prompt_tokens) > max_prompt_tokens:
            logger.warning(f"Skipping patient {example['patient_id']}: prompt too long ({len(prompt_tokens)} tokens > {max_prompt_tokens} limit)")
            return {
                'patient_id': example['patient_id'],
                'ground_truth': example.get('label_value', None),
                'full_response': "SKIPPED: Prompt too long",
                'prediction': None,
                'prediction_text': "SKIPPED"
            }
        
        # Choose model
        llm = self.baseline_llm if use_baseline else self.fine_tuned_llm
        
        # Get sampling parameters
        sampling_params = self._get_sampling_params()
        
        # Generate using VLLM
        try:
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            # Post-process to keep 'Final Answer' + next 5 tokens if present
            generated_text = self._truncate_after_final_answer(generated_text, extra_tokens=5)
        except Exception as e:
            logger.error(f"Error generating for patient {example['patient_id']}: {e}")
            generated_text = "Error in generation"
        
        # Clean up any remaining artifacts
        if generated_text.startswith("<|im_end|>"):
            generated_text = generated_text[len("<|im_end|>"):].strip()
        
        # Fallback: If stopping criteria didn't work properly, complete the "Final Answer"
        if generated_text.strip().endswith("Final Answer"):
            # Try to extract prediction from context
            if "positive" in generated_text.lower() or "acute myocardial infarction" in generated_text.lower():
                generated_text += ": Positive"
            elif "negative" in generated_text.lower() or "no evidence" in generated_text.lower():
                generated_text += ": Negative"
            else:
                generated_text += ": [UNKNOWN]"
        
        return {
            'patient_id': example['patient_id'],
            'ground_truth': example.get('label_value', None),
            'full_response': generated_text,
            'prompt': prompt,
            'model_type': 'baseline' if use_baseline else 'fine_tuned'
        }
    
    def generate_predictions_batch(self, examples: List[Dict], use_baseline: bool = False) -> List[Dict]:
        """Generate predictions for a batch of examples using VLLM (faster)"""
        prompts = []
        valid_examples = []
        
        # Prepare prompts and filter valid examples
        for example in examples:
            prompt = self.format_prompt(example)
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            max_prompt_tokens = self.config.max_model_len - self.config.max_new_tokens
            
            if len(prompt_tokens) <= max_prompt_tokens:
                prompts.append(prompt)
                valid_examples.append(example)
            else:
                logger.warning(f"Skipping patient {example['patient_id']}: prompt too long")
        
        if not prompts:
            return []
        
        # Choose model
        llm = self.baseline_llm if use_baseline else self.fine_tuned_llm
        
        # Get sampling parameters
        sampling_params = self._get_sampling_params()
        
        # Generate using VLLM batch processing
        try:
            outputs = llm.generate(prompts, sampling_params)
            results = []
            
            for i, (output, example) in enumerate(zip(outputs, valid_examples)):
                generated_text = output.outputs[0].text.strip()
                # Post-process to keep 'Final Answer' + next 5 tokens if present
                generated_text = self._truncate_after_final_answer(generated_text, extra_tokens=5)
                
                # Clean up any remaining artifacts
                if generated_text.startswith("<|im_end|>"):
                    generated_text = generated_text[len("<|im_end|>"):].strip()
                
                # Fallback: If stopping criteria didn't work properly, complete the "Final Answer"
                if generated_text.strip().endswith("Final Answer"):
                    if "positive" in generated_text.lower() or "acute myocardial infarction" in generated_text.lower():
                        generated_text += ": Positive"
                    elif "negative" in generated_text.lower() or "no evidence" in generated_text.lower():
                        generated_text += ": Negative"
                    else:
                        generated_text += ": [UNKNOWN]"
                
                results.append({
                    'patient_id': example['patient_id'],
                    'ground_truth': example.get('label_value', None),
                    'full_response': generated_text,
                    'prompt': prompts[i],
                    'model_type': 'baseline' if use_baseline else 'fine_tuned'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            # Fallback to individual generation
            results = []
            for example in valid_examples:
                try:
                    result = self.generate_prediction(example, use_baseline)
                    results.append(result)
                except Exception as e2:
                    logger.error(f"Error generating for patient {example['patient_id']}: {e2}")
                    results.append({
                        'patient_id': example['patient_id'],
                        'ground_truth': example.get('label_value', None),
                        'full_response': "Error in generation",
                        'prompt': self.format_prompt(example),
                        'model_type': 'baseline' if use_baseline else 'fine_tuned'
                    })
            return results
    
    def extract_prediction_from_response(self, response: str) -> int:
        """Extract binary prediction from model response"""
        response_lower = response.lower()
        
        # Look for "Final Answer: [Positive/Negative]" pattern first
        if "final answer:" in response_lower:
            final_answer = response_lower.split("final answer:")[-1].strip()
            if "positive" in final_answer:
                return 1
            elif "negative" in final_answer:
                return 0
        
        # Look for positive/negative keywords
        if "positive" in response_lower:
            return 1
        elif "negative" in response_lower:
            return 0
        
        # Default to negative if unclear
        return 0
    
    def evaluate_dataset(self, test_dataset: List[Dict], output_dir: str = None) -> Dict:
        """Evaluate both fine-tuned and baseline models on test dataset"""
        logger.info(f"Evaluating models on {len(test_dataset)} test examples")
        
        # Set up output files for incremental saving
        fine_tuned_output_file = None
        baseline_output_file = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fine_tuned_output_file = os.path.join(output_dir, "fine_tuned_outputs.json")
            baseline_output_file = os.path.join(output_dir, "baseline_outputs.json")
        
        
        # Evaluate fine-tuned model
        logger.info("Evaluating fine-tuned model...")
        fine_tuned_results = self._evaluate_single_model(test_dataset, use_baseline=False, output_file=fine_tuned_output_file)
        
        
        # Free fine-tuned engine to make room for baseline
        try:
            del self.fine_tuned_llm
            self.fine_tuned_llm = None
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.warning(f"Could not fully free fine-tuned engine before loading baseline: {e}")
        
        # Load and evaluate baseline model
        logger.info("Loading baseline VLLM model for comparison...")
        if self.vllm_kwargs is None:
            raise RuntimeError("VLLM kwargs missing; load_model_and_tokenizer must be called first")
        self.baseline_llm = LLM(**self.vllm_kwargs)
        logger.info("Evaluating baseline model...")
        baseline_results = self._evaluate_single_model(test_dataset, use_baseline=True, output_file=baseline_output_file)
        
        # Optionally free baseline engine after evaluation
        try:
            del self.baseline_llm
            self.baseline_llm = None
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.warning(f"Could not fully free baseline engine after evaluation: {e}")
        

  
        # Combine results
        results = {
            'fine_tuned': fine_tuned_results,
            'baseline': baseline_results,
            'comparison': {
                'auroc_improvement': fine_tuned_results['auroc'] - baseline_results['auroc'],
                'accuracy_improvement': fine_tuned_results['accuracy'] - baseline_results['accuracy'],
                'precision_improvement': fine_tuned_results['precision'] - baseline_results['precision'],
                'recall_improvement': fine_tuned_results['recall'] - baseline_results['recall']
            },
            'n_examples': len(test_dataset)
        }
        
        # Log comparison results
        logger.info(f"Comparison Results:")
        logger.info(f"  AUROC - Fine-tuned: {fine_tuned_results['auroc']:.4f}, Baseline: {baseline_results['auroc']:.4f}, Improvement: {results['comparison']['auroc_improvement']:.4f}")
        logger.info(f"  Accuracy - Fine-tuned: {fine_tuned_results['accuracy']:.4f}, Baseline: {baseline_results['accuracy']:.4f}, Improvement: {results['comparison']['accuracy_improvement']:.4f}")
        logger.info(f"  Precision - Fine-tuned: {fine_tuned_results['precision']:.4f}, Baseline: {baseline_results['precision']:.4f}, Improvement: {results['comparison']['precision_improvement']:.4f}")
        logger.info(f"  Recall - Fine-tuned: {fine_tuned_results['recall']:.4f}, Baseline: {baseline_results['recall']:.4f}, Improvement: {results['comparison']['recall_improvement']:.4f}")
        
        return results
    
    def _evaluate_single_model(self, test_dataset: List[Dict], use_baseline: bool = False, 
                              output_file: str = None) -> Dict:
        """Evaluate a single model on test dataset using VLLM batch processing"""
        model_name = "baseline" if use_baseline else "fine_tuned"
        logger.info(f"Evaluating {model_name} model with VLLM batch processing...")
        
        # Process in batches for better performance
        batch_size = 32  # Adjust based on GPU memory
        all_results = []
        
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(test_dataset) + batch_size - 1)//batch_size} ({len(batch)} examples)")
            
            try:
                # Generate predictions for the batch
                batch_results = self.generate_predictions_batch(batch, use_baseline=use_baseline)
                all_results.extend(batch_results)
                
                # Process each result in the batch
                for j, (result, example) in enumerate(zip(batch_results, batch)):
                    # Extract prediction probability
                    pred_prob = self.extract_prediction_from_response(result['full_response'])
                    
                    # Create output entry with cleaned reasoning
                    output_entry = {
                        'patient_id': example['patient_id'],
                        'label_time': example.get('label_time', ''),
                        'ground_truth': example.get('label_value', 0),
                        'prediction': pred_prob,
                        'prediction_text': "Positive" if pred_prob >= 0.5 else "Negative",
                        'reasoning': result['full_response'],
                        'model_type': model_name,
                        'example_id': i + j + 1
                    }
                    
                    # Incrementally save outputs
                    if output_file:
                        self._save_outputs_incremental([output_entry], output_file, is_first=(i == 0 and j == 0))
                    
                    logger.info(f"Patient {example['patient_id']}: GT={example.get('label_value', 0)}, Pred={pred_prob:.3f} ({model_name})")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Fallback to individual processing for this batch
                for j, example in enumerate(batch):
                    try:
                        result = self.generate_prediction(example, use_baseline=use_baseline)
                        pred_prob = self.extract_prediction_from_response(result['full_response'])
                        
                        output_entry = {
                            'patient_id': example['patient_id'],
                            'label_time': example.get('label_time', ''),
                            'ground_truth': example.get('label_value', 0),
                            'prediction': pred_prob,
                            'prediction_text': "Positive" if pred_prob >= 0.5 else "Negative",
                            'reasoning': result['full_response'],
                            'model_type': model_name,
                            'example_id': i + j + 1
                        }
                        
                        if output_file:
                            self._save_outputs_incremental([output_entry], output_file, is_first=(i == 0 and j == 0))
                        
                        all_results.append(result)
                        
                    except Exception as e2:
                        logger.error(f"Error processing patient {example['patient_id']}: {e2}")
                        # Create error entry
                        output_entry = {
                            'patient_id': example['patient_id'],
                            'label_time': example.get('label_time', ''),
                            'ground_truth': example.get('label_value', 0),
                            'prediction': 0.5,
                            'prediction_text': "Error",
                            'reasoning': "Error in generation",
                            'model_type': model_name,
                            'example_id': i + j + 1,
                            'error': str(e2)
                        }
                        
                        if output_file:
                            self._save_outputs_incremental([output_entry], output_file, is_first=(i == 0 and j == 0))
        
        # Extract data for metrics calculation
        predictions = []
        ground_truths = []
        patient_ids = []
        full_responses = []
        
        for result in all_results:
            pred_prob = self.extract_prediction_from_response(result['full_response'])
            predictions.append(pred_prob)
            ground_truths.append(result['ground_truth'])
            patient_ids.append(result['patient_id'])
            full_responses.append(result['full_response'])
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        
        # Compute metrics
        try:
            auroc = roc_auc_score(ground_truths, predictions)
        except ValueError:
            auroc = 0.5  # Default if only one class
        
        # Additional metrics
        precision, recall, thresholds = precision_recall_curve(ground_truths, predictions)
        fpr, tpr, roc_thresholds = roc_curve(ground_truths, predictions)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        # Binary predictions at optimal threshold
        binary_predictions = (predictions >= optimal_threshold).astype(int)
        
        # Additional metrics
        accuracy = np.mean(binary_predictions == ground_truths)
        precision_at_optimal = precision[optimal_idx] if optimal_idx < len(precision) else 0.0
        recall_at_optimal = recall[optimal_idx] if optimal_idx < len(recall) else 0.0
        
        results = {
            'model_type': model_name,
            'auroc': auroc,
            'accuracy': accuracy,
            'precision': precision_at_optimal,
            'recall': recall_at_optimal,
            'optimal_threshold': optimal_threshold,
            'predictions': predictions.tolist(),
            'ground_truths': ground_truths.tolist(),
            'patient_ids': patient_ids,
            'full_responses': full_responses,
            'all_outputs': all_results
        }
        
        logger.info(f"{model_name.capitalize()} Results:")
        logger.info(f"  AUROC: {auroc:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision_at_optimal:.4f}")
        logger.info(f"  Recall: {recall_at_optimal:.4f}")
        logger.info(f"  Optimal Threshold: {optimal_threshold:.4f}")
        
        return results
    
    def _save_outputs_incremental(self, outputs: List[Dict], output_file: str, is_first: bool = False):
        """Save outputs incrementally to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
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
    
    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save predictions CSV for both models
        if self.config.save_predictions:
            # Fine-tuned model predictions
            fine_tuned_df = pd.DataFrame({
                'patient_id': results['fine_tuned']['patient_ids'],
                'ground_truth': results['fine_tuned']['ground_truths'],
                'prediction_prob': results['fine_tuned']['predictions'],
                'prediction_binary': (np.array(results['fine_tuned']['predictions']) >= results['fine_tuned']['optimal_threshold']).astype(int),
                'model_type': 'fine_tuned'
            })
            
            # Baseline model predictions
            baseline_df = pd.DataFrame({
                'patient_id': results['baseline']['patient_ids'],
                'ground_truth': results['baseline']['ground_truths'],
                'prediction_prob': results['baseline']['predictions'],
                'prediction_binary': (np.array(results['baseline']['predictions']) >= results['baseline']['optimal_threshold']).astype(int),
                'model_type': 'baseline'
            })
            
            # Combine and save
            combined_df = pd.concat([fine_tuned_df, baseline_df], ignore_index=True)
            predictions_file = os.path.join(output_dir, "predictions_comparison.csv")
            combined_df.to_csv(predictions_file, index=False)
            logger.info(f"Predictions saved to {predictions_file}")
        
        # Save plots
        if self.config.save_plots:
            self.plot_results(results, output_dir)
    
    def plot_results(self, results: Dict, output_dir: str):
        """Create evaluation plots comparing both models"""
        fine_tuned_preds = np.array(results['fine_tuned']['predictions'])
        baseline_preds = np.array(results['baseline']['predictions'])
        ground_truths = np.array(results['fine_tuned']['ground_truths'])
        
        # ROC Curves for both models
        fpr_ft, tpr_ft, _ = roc_curve(ground_truths, fine_tuned_preds)
        fpr_bl, tpr_bl, _ = roc_curve(ground_truths, baseline_preds)
        
        plt.figure(figsize=(15, 5))
        
        # ROC Curve Comparison
        plt.subplot(1, 3, 1)
        plt.plot(fpr_ft, tpr_ft, label=f'Fine-tuned (AUC = {results["fine_tuned"]["auroc"]:.3f})', color='blue')
        plt.plot(fpr_bl, tpr_bl, label=f'Baseline (AUC = {results["baseline"]["auroc"]:.3f})', color='red')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True)
        
        # Fine-tuned model prediction distribution
        plt.subplot(1, 3, 2)
        plt.hist(fine_tuned_preds[ground_truths == 0], alpha=0.7, label='Negative', bins=20, color='lightblue')
        plt.hist(fine_tuned_preds[ground_truths == 1], alpha=0.7, label='Positive', bins=20, color='darkblue')
        plt.axvline(results['fine_tuned']['optimal_threshold'], color='red', linestyle='--', label=f'Optimal Threshold')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Fine-tuned Model Distribution')
        plt.legend()
        plt.grid(True)
        
        # Baseline model prediction distribution
        plt.subplot(1, 3, 3)
        plt.hist(baseline_preds[ground_truths == 0], alpha=0.7, label='Negative', bins=20, color='lightcoral')
        plt.hist(baseline_preds[ground_truths == 1], alpha=0.7, label='Positive', bins=20, color='darkred')
        plt.axvline(results['baseline']['optimal_threshold'], color='red', linestyle='--', label=f'Optimal Threshold')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Baseline Model Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, "evaluation_comparison_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create metrics comparison plot
        self._plot_metrics_comparison(results, output_dir)
        
        logger.info(f"Plots saved to {plot_file}")
    
    def _plot_metrics_comparison(self, results: Dict, output_dir: str):
        """Create metrics comparison bar chart"""
        metrics = ['AUROC', 'Accuracy', 'Precision', 'Recall']
        fine_tuned_values = [
            results['fine_tuned']['auroc'],
            results['fine_tuned']['accuracy'],
            results['fine_tuned']['precision'],
            results['fine_tuned']['recall']
        ]
        baseline_values = [
            results['baseline']['auroc'],
            results['baseline']['accuracy'],
            results['baseline']['precision'],
            results['baseline']['recall']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, fine_tuned_values, width, label='Fine-tuned', color='blue', alpha=0.7)
        plt.bar(x + width/2, baseline_values, width, label='Baseline', color='red', alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (ft_val, bl_val) in enumerate(zip(fine_tuned_values, baseline_values)):
            plt.text(i - width/2, ft_val + 0.01, f'{ft_val:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, bl_val + 0.01, f'{bl_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        metrics_plot_file = os.path.join(output_dir, "metrics_comparison.png")
        plt.savefig(metrics_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics comparison plot saved to {metrics_plot_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SFT model on test dataset")
    
    # Model configuration
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model name")
    parser.add_argument("--peft_model_path", type=str, default="./qwen_sft_output",
                       help="Path to fine-tuned PEFT model")
    parser.add_argument("--use_quantization", action="store_true", default=False,
                       help="Use quantization")
    
    # Data configuration
    parser.add_argument("--dataset_file", type=str, required=True,
                       help="Path to SFT dataset JSON file")
    parser.add_argument("--path_to_splits", type=str, required=True,
                       help="Path to splits CSV file")
    
    # VLLM configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                       help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=16384,
                       help="Maximum model length")
    
    # Evaluation configuration
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--save_predictions", action="store_true", default=True,
                       help="Save detailed predictions")
    parser.add_argument("--save_plots", action="store_true", default=True,
                       help="Save evaluation plots")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        # Create configuration
        config = EvaluationConfig(
            base_model_name=args.base_model_name,
            peft_model_path=args.peft_model_path,
            use_quantization=args.use_quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_dir=args.output_dir,
            save_predictions=args.save_predictions,
            save_plots=args.save_plots
        )
        
        # Initialize evaluator
        evaluator = SFTModelEvaluator(config)
        
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
            print("    python fix_numpy_import.py")
            print("\n  Option 2 (Manual fix):")
            print("    pip install --upgrade --force-reinstall numpy")
            print("    pip install --upgrade --force-reinstall vllm scikit-learn pandas")
            print("\n  Option 3 (Fresh environment - recommended):")
            print("    python -m venv venv")
            print("    source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            print("    pip install numpy==1.24.3")
            print("    pip install -r requirements_vllm.txt")
            print("\n" + "=" * 70 + "\n")
            sys.exit(1)
        raise
    
    # Load test dataset
    test_dataset = evaluator.load_test_dataset(args.dataset_file, args.path_to_splits)
    
    if not test_dataset:
        logger.error("No test examples found. Exiting.")
        return
    
    # Evaluate model
    results = evaluator.evaluate_dataset(test_dataset, args.output_dir)
    
    # Save results
    evaluator.save_results(results, args.output_dir)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()

