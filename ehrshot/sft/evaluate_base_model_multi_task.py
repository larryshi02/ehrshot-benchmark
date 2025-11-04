#!/usr/bin/env python3
"""
Multi-task evaluation pipeline for base Qwen3-8B model on 4 tasks:
- Acute MI (new_acutemi)
- Hyperlipidemia (new_hyperlipidemia)
- Hypertension (new_hypertension)
- Pancreatic Cancer (new_pancan)

Uses VLLM for inference with 10 samples per patient and KV caching.
"""

import os
import sys
import json
import argparse
# Import numpy first to avoid binary incompatibility issues
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Literal
from dataclasses import dataclass
from collections import Counter
import torch
import gc
# Import sklearn before VLLM to avoid numpy issues
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, precision_score, recall_score, f1_score
from transformers import AutoTokenizer
from loguru import logger


PREDICTION_SCHEMA = {
  "type": "object",
  "properties": {
    "prediction": {
      "type": "string",
      "enum": ["positive", "negative"]
    }
  },
  "required": ["prediction"],
  "additionalProperties": False
}


# Import VLLM last to minimize numpy compatibility issues
try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError as e:
    print(f"ERROR: Failed to import VLLM: {e}")
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


@dataclass
class MultiTaskEvaluationConfig:
    """Configuration for multi-task model evaluation"""
    
    # Model configuration
    base_model_name: str = "Qwen/Qwen3-8B"  # Use Qwen3-8B as specified
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
    
    # Parser LLM configuration (always enabled)
    parser_model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    parser_gpu_memory_utilization: float = 0.1  # GPU memory utilization for parser LLM (low for small model)
    
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
        self.parser_llm = None
        self.parser_tokenizer = None
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
        
        # Load parser LLM (always enabled)
        self.load_parser_llm()
    
    def load_parser_llm(self):
        """Load the parser LLM (qwen2-1.5B-Instruct) with VLLM for structured output"""
        logger.info(f"Loading parser LLM: {self.config.parser_model_name}")
        
        # Load parser tokenizer
        self.parser_tokenizer = AutoTokenizer.from_pretrained(
            self.config.parser_model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )
        
        # VLLM configuration for parser LLM (smaller model, can use less memory)
        # For a 1.5B model, we need much less memory. Use a very low utilization
        # to fit alongside the base model. A 1.5B model in bfloat16 needs ~3-4 GiB.
        parser_vllm_kwargs = {
            "model": self.config.parser_model_name,
            "trust_remote_code": self.config.trust_remote_code,
            "tensor_parallel_size": 1,  # Smaller model, single GPU should be enough
            "gpu_memory_utilization": self.config.parser_gpu_memory_utilization,  # Configurable, default 0.1
            "max_model_len": 16384,  # Parser doesn't need long context
            "dtype": "bfloat16",
        }
        
        # Load parser LLM with VLLM
        logger.info(f"Loading parser LLM with VLLM from: {self.config.parser_model_name}")
        try:
            self.parser_llm = LLM(**parser_vllm_kwargs)
            logger.info("Parser LLM loaded successfully")
        except (ValueError, RuntimeError) as e:
            if "memory" in str(e).lower() or "free memory" in str(e).lower():
                logger.error(f"Failed to load parser LLM due to insufficient GPU memory: {e}")
                logger.error(f"Current parser GPU memory utilization: {self.config.parser_gpu_memory_utilization}")
                logger.error("Suggestions:")
                logger.error("  1. Reduce --parser_gpu_memory_utilization (e.g., --parser_gpu_memory_utilization 0.05)")
                logger.error("  2. Reduce base model's --gpu_memory_utilization to leave more room for parser")
                raise
            else:
                raise
    
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
        prompt = f'''System: You may reason step by step, but your final output MUST be in the following format: "Final Answer: [Positive/Negative]".\n\n
User: You are a helpful medical assistant. Below is a patient's electronic healthcare record (EHR) in Markdown format. Please answer the following query using clinically grounded reasoning: {query}. Patient Medical History:\n\n\n\n{context}\n\nIMPORTANT: You must conclude with your final prediction in the format "Final Answer: [Positive/Negative]".\n\n Assistant: '''

        return prompt
    
    def _get_sampling_params(self, num_samples: int = 1) -> SamplingParams:
        """Get sampling parameters for VLLM generation"""
        return SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            repetition_penalty=1.1,
            n=num_samples,
            stop=["Final Answer: Positive",
                  "Final Answer: Negative",
                  "**Final Answer**: Positive",
                  "**Final Answer**: Negative",
                  "Final Answer: **Positive**",
                  "Final Answer: **Negative**",
                  "**Final Answer**: **Positive**",
                  "**Final Answer**: **Negative**",
                  "** Final Answer **: Positive",
                  "** Final Answer **: Negative",
                  "** Final Answer **: **Positive**",
                  "** Final Answer **: **Negative**",
                  "** Final Answer**: Positive",
                  "** Final Answer**: Negative",
                  "** Final Answer**: **Positive**",
                  "** Final Answer**: **Negative**",

            ]
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
                        'responses': responses,
                        'ground_truth': example['label_value']
                    })
                except Exception as e2:
                    logger.error(f"Error generating for patient {example['patient_id']}: {e2}")
                    results.append({
                        'patient_id': example['patient_id'],
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
    
    def _create_parser_prompt(self, response: str, task_name: str = None) -> str:
        """Create parser prompt for a single response"""
        query = TASK_QUERIES.get(task_name, "unknown query") if task_name else "unknown query"
        return f"""You are a parser model. Below is an output from another LLM to reason about and predict the following: {query}. Output a single token that infers whether its final prediction is POSITIVE or NEGATIVE.

LLM Output:
{response}

Based on the above LLM output, determine if the final prediction is POSITIVE or NEGATIVE."""
    
    def _parse_parser_output(self, parser_output_text: str, original_response: str) -> Optional[int]:
        """Parse the structured output from parser LLM"""
        try:
            # Try to parse as JSON (vLLM should return JSON matching schema)
            parser_result = json.loads(parser_output_text)
            prediction_str = parser_result.get("prediction", "").upper()
            
            if prediction_str == "POSITIVE":
                return 1
            elif prediction_str == "NEGATIVE":
                return 0
            else:
                logger.warning(f"Parser LLM returned unexpected prediction: {prediction_str}")
                # Fallback to regex extraction
                return self._extract_with_regex(original_response)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse parser LLM output as JSON: {parser_output_text}, error: {e}")
            # Fallback: try to extract from text
            parser_output_lower = parser_output_text.lower()
            if "positive" in parser_output_lower:
                return 1
            elif "negative" in parser_output_lower:
                return 0
            else:
                # Final fallback to regex extraction
                return self._extract_with_regex(original_response)
    
    def _extract_with_parser_llm_batch(self, responses: List[str], task_name: str = None) -> Tuple[List[Optional[int]], List[str]]:
        """Extract binary predictions using parser LLM with structured output (batch processing)
        
        Returns:
            Tuple of (predictions, parser_outputs) where:
            - predictions: List of binary predictions (0/1) or None
            - parser_outputs: List of raw parser LLM output text
        """
        if not responses:
            return [], []
        
        try:
            # Get query for this task
            query = TASK_QUERIES.get(task_name, "unknown query") if task_name else "unknown query"
            
            # Create parser prompts for all responses
            parser_prompts = []
            for response in responses:
                parser_prompt = self._create_parser_prompt(response, task_name)
                parser_prompts.append(parser_prompt)
            

            parser_sampling_params = SamplingParams(
                temperature=0.1,  # Low temperature for deterministic parsing
                max_tokens=50,  # Increased to ensure we get output (structured output may need more tokens)
                guided_decoding=GuidedDecodingParams(json=PREDICTION_SCHEMA),  # Use structured output at inference level
            )
            
            # Generate with parser LLM in batch
            outputs = self.parser_llm.generate(parser_prompts, parser_sampling_params)
            
            # Parse all outputs and store raw parser outputs
            predictions = []
            parser_outputs = []
            for i, output in enumerate(outputs):
                if not output.outputs:
                    parser_output_text = ""
                else:
                    parser_output_text = output.outputs[0].text.strip()
                    
                    # Check if structured output is in a different field
                    if hasattr(output.outputs[0], 'structured_output'):
                        structured = output.outputs[0].structured_output
                        if structured:
                            parser_output_text = json.dumps(structured)
                    elif hasattr(output, 'structured_output'):
                        structured = output.structured_output
                        if structured:
                            parser_output_text = json.dumps(structured)
                    
                    # Check for other possible attributes
                    if not parser_output_text and hasattr(output.outputs[0], '__dict__'):
                        for attr_name in ['structured_output', 'json_output', 'output', 'content']:
                            if hasattr(output.outputs[0], attr_name):
                                attr_value = getattr(output.outputs[0], attr_name)
                                if attr_value:
                                    parser_output_text = str(attr_value) if not isinstance(attr_value, dict) else json.dumps(attr_value)
                                    break
                
                parser_outputs.append(parser_output_text)
                prediction = self._parse_parser_output(parser_output_text, responses[i])
                predictions.append(prediction)
            
            return predictions, parser_outputs
                    
        except Exception as e:
            logger.error(f"Error using parser LLM for batch extraction: {e}")
            # Fallback to regex extraction for all
            return [self._extract_with_regex(r) for r in responses], [""] * len(responses)
    
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
            # Collect all responses for batch processing with parser LLM
            all_batch_responses = []
            responses_per_result = []  # Track how many responses per result
            
            for result in batch_results:
                responses = result['responses']
                responses_per_result.append(len(responses))
                all_batch_responses.extend(responses)
            
            # Batch extract predictions using parser LLM
            if self.parser_llm is not None:
                # Use batch processing for parser LLM
                all_predictions_with_none, all_parser_outputs = self._extract_with_parser_llm_batch(all_batch_responses, task_name)
            else:
                # Fallback to regex if parser LLM failed to load
                logger.warning("Parser LLM not available, falling back to regex extraction")
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
                
                # Find the original prompt for this patient
                original_prompt = None
                label_time = None
                for ex in serialized_examples:
                    if ex['patient_id'] == result['patient_id']:
                        original_prompt = ex.get('prompt', None)
                        label_time = ex.get('label_time', None)
                        break
                
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
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Combined summary saved to {summary_file}")
            
            # Print summary table
            print("\n" + "="*80)
            print("EVALUATION SUMMARY - BASE MODEL (Qwen3-8B)")
            print("="*80)
            print(summary_df.to_string(index=False))
            print("="*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base Qwen3-8B model on 4 tasks")
    
    # Model configuration
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Base model name")
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
    
    # Evaluation configuration
    parser.add_argument("--max_new_tokens", type=int, default=8500,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples per patient")
    
    # Parser LLM configuration (always enabled)
    parser.add_argument("--parser_model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                       help="Parser LLM model name for structured output extraction")
    parser.add_argument("--parser_gpu_memory_utilization", type=float, default=0.1,
                       help="GPU memory utilization for parser LLM (default: 0.1, reduce if OOM)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./base_model_multi_task_results",
                       help="Output directory for results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        # Create configuration
        config = MultiTaskEvaluationConfig(
            base_model_name=args.base_model_name,
            use_quantization=args.use_quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_samples=args.num_samples,
            parser_model_name=args.parser_model_name,
            parser_gpu_memory_utilization=args.parser_gpu_memory_utilization,
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
    
    # Evaluate all tasks
    try:
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

