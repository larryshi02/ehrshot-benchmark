#!/usr/bin/env python3
"""
Single-task evaluation script for base Qwen3-8B model.

This script evaluates one task on one GPU. It is designed to be called by the 
parallel coordinator script (evaluate_base_model_parallel_tasks.py).

Execution flow:
1. Sets CUDA_VISIBLE_DEVICES to the specified GPU
2. Loads the model with tensor_parallel_size=1 on that GPU
3. Loads and evaluates the specified task
4. Saves results to output directory
5. Returns success/failure status

This script is meant to run in a separate process, allowing multiple tasks
to be evaluated in parallel across different GPUs.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import gc
import time
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
    raise

# Add script directory to path for imports (to import task_config)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import shared task configuration (worker script uses it, but coordinator validates it)
try:
    from task_config import TASK_QUERIES
except ImportError:
    # Fallback if task_config.py is not found
    # Note: Coordinator should validate tasks before launching workers
    TASK_QUERIES = {
        'acute_mi': 'will the patient develop an acute myocardial infarction in the next year',
        'hyperlipidemia': 'will the patient develop hyperlipidemia in the next year',
        'hypertension': 'will the patient develop hypertension in the next year',
        'pancreatic_cancer': 'will the patient develop pancreatic cancer in the next year'
    }

# Add parent directory to path for other imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


@dataclass
class SingleTaskEvaluationConfig:
    """Configuration for single-task model evaluation"""
    
    # Model configuration
    base_model_name: str = "Qwen/Qwen3-8B"
    trust_remote_code: bool = True
    use_quantization: bool = False
    
    # VLLM configuration (tensor_parallel_size=1 for single GPU)
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 20000
    
    # Evaluation configuration
    max_new_tokens: int = 8500
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples: int = 10
    
    # Parser LLM configuration
    parser_model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    parser_gpu_memory_utilization: float = 0.1
    
    # Data and output configuration
    path_to_serialized_data: str = ""
    output_dir: str = "./base_model_multi_task_results"
    task_name: str = ""
    gpu_id: int = 0


class SingleTaskEvaluator:
    """Evaluator for base model on a single task using VLLM on one GPU"""
    
    def __init__(self, config: SingleTaskEvaluationConfig):
        self.config = config
        self.tokenizer = None
        self.llm = None
        self.parser_llm = None
        self.parser_tokenizer = None
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer using VLLM on the specified GPU"""
        logger.info(f"[GPU {self.config.gpu_id}] Loading tokenizer: {self.config.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )
        
        # VLLM configuration with tensor_parallel_size=1
        vllm_kwargs = {
            "model": self.config.base_model_name,
            "trust_remote_code": self.config.trust_remote_code,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "dtype": "bfloat16",
            "enable_prefix_caching": True,
        }
        
        if self.config.use_quantization:
            vllm_kwargs["quantization"] = "bitsandbytes"
        
        # Load base model with VLLM
        logger.info(f"[GPU {self.config.gpu_id}] Loading base model with VLLM (tensor_parallel_size=1)")
        self.llm = LLM(**vllm_kwargs)
        logger.info(f"[GPU {self.config.gpu_id}] Base model loaded successfully")
        
        # Load parser LLM
        self.load_parser_llm()
    
    def load_parser_llm(self):
        """Load the parser LLM for structured output extraction"""
        logger.info(f"[GPU {self.config.gpu_id}] Loading parser LLM: {self.config.parser_model_name}")
        
        self.parser_tokenizer = AutoTokenizer.from_pretrained(
            self.config.parser_model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )
        
        parser_vllm_kwargs = {
            "model": self.config.parser_model_name,
            "trust_remote_code": self.config.trust_remote_code,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": self.config.parser_gpu_memory_utilization,
            "max_model_len": 16384,
            "dtype": "bfloat16",
        }
        
        try:
            self.parser_llm = LLM(**parser_vllm_kwargs)
            logger.info(f"[GPU {self.config.gpu_id}] Parser LLM loaded successfully")
        except (ValueError, RuntimeError) as e:
            if "memory" in str(e).lower():
                logger.error(f"[GPU {self.config.gpu_id}] Failed to load parser LLM due to memory: {e}")
                raise
            raise
    
    def load_pre_serialized_data(self, task_name: str) -> List[Dict]:
        """Load pre-serialized EHR data from JSON file for a specific task"""
        json_file = os.path.join(self.config.path_to_serialized_data, f"{task_name}_all_splits.json")
        
        if not os.path.exists(json_file):
            logger.error(f"[GPU {self.config.gpu_id}] Serialized data file not found: {json_file}")
            return []
        
        logger.info(f"[GPU {self.config.gpu_id}] Loading pre-serialized data from: {json_file}")
        with open(json_file, 'r') as f:
            all_data = json.load(f)
        
        # Filter to test split only
        test_data = [item for item in all_data if item.get('split') == 'test']
        logger.info(f"[GPU {self.config.gpu_id}] Loaded {len(test_data)} test examples")
        return test_data
    
    def _create_prompt_from_context(self, task_name: str, context: str) -> str:
        """Create full prompt from raw EHR context"""
        default_query = f"will the patient develop {task_name.replace('_', ' ')} in the next year"
        query = TASK_QUERIES.get(task_name, default_query)
        
        prompt = (
            f"You may reason step by step, "
            f"but your final output MUST be in the following format: \"Final Answer: [Positive/Negative]\".\n\n"
            f"User: You are a helpful medical assistant. "
            f"Below is a patient's electronic healthcare record (EHR) in Markdown format. "
            f"Please answer the following query using clinically grounded reasoning: {query}. "
            f"Patient Medical History:\n\n{context}\n\n"
            f"IMPORTANT: You must conclude with your final prediction in the format \"Final Answer: [Positive/Negative]\".\n\n"
            f"YOU HAVE TO FINISH YOUR ANSWER BY SAYING EITHER \"Final Answer: Positive\" OR \"Final Answer: Negative\"."
            f"AND DO NOT SAY ANYTHING ELSE AFTER THAT."
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
            stop=["Final Answer: Positive", "Final Answer: Negative",
                  "**Final Answer**: Positive", "**Final Answer**: Negative",
                  "Final Answer: **Positive**", "Final Answer: **Negative**",
                  "**Final Answer**: **Positive**", "**Final Answer**: **Negative**",
                  "** Final Answer **: Positive", "** Final Answer **: Negative",
                  "** Final Answer **: **Positive**", "** Final Answer **: **Negative**"]
        )
    
    def generate_predictions_batch(self, examples: List[Dict], num_samples: int) -> List[Dict]:
        """Generate predictions for a batch of examples using VLLM"""
        prompts = []
        valid_examples = []
        
        for example in examples:
            prompt = example['prompt']
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            max_prompt_tokens = self.config.max_model_len - self.config.max_new_tokens
            
            if len(prompt_tokens) <= max_prompt_tokens:
                prompts.append(prompt)
                valid_examples.append(example)
            else:
                logger.warning(f"[GPU {self.config.gpu_id}] Skipping patient {example['patient_id']}: prompt too long")
        
        if not prompts:
            return []
        
        try:
            sampling_params = self._get_sampling_params(num_samples)
            outputs = self.llm.generate(prompts, sampling_params)
            results = []
            
            for i, (output, example) in enumerate(zip(outputs, valid_examples)):
                sample_responses = []
                for sample_idx in range(num_samples):
                    if sample_idx < len(output.outputs):
                        generated_text = output.outputs[sample_idx].text.strip()
                        if generated_text.startswith("<|im_end|>"):
                            generated_text = generated_text[len("<|im_end|>"):].strip()
                        sample_responses.append(generated_text)
                    else:
                        sample_responses.append("Error: missing output")
                
                results.append({
                    'patient_id': example['patient_id'],
                    'label_time': example['label_time'],
                    'prompt': example['prompt'],
                    'responses': sample_responses,
                    'ground_truth': example['label_value']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"[GPU {self.config.gpu_id}] Error in batch generation: {e}")
            return []
    
    def _extract_with_regex(self, response: str) -> Optional[int]:
        """Extract binary prediction using regex (fallback method)"""
        response_lower = response.lower()
        
        if "final answer:" in response_lower:
            final_answer = response_lower.split("final answer:")[-1].strip()
            if "positive" in final_answer:
                return 1
            elif "negative" in final_answer:
                return 0
        
        return None
    
    def _create_parser_prompt(self, response: str, task_name: str = None) -> str:
        """Create parser prompt for a single response"""
        if task_name:
            default_query = f"will the patient develop {task_name.replace('_', ' ')} in the next year"
            query = TASK_QUERIES.get(task_name, default_query)
        else:
            query = "unknown query"
        
        return (
            f"You are a parser model. "
            f"Below is an output from another LLM to reason about and predict the following: {query}. "
            f"Output a single token that infers whether its final prediction is POSITIVE or NEGATIVE.\n\n"
            f"LLM Output:\n"
            f"{response}\n\n"
            f"Based on the above LLM output, determine if the final prediction is POSITIVE or NEGATIVE."
        )
    
    def _parse_parser_output(self, parser_output_text: str, original_response: str) -> Optional[int]:
        """Parse the structured output from parser LLM"""
        try:
            parser_result = json.loads(parser_output_text)
            prediction_str = parser_result.get("prediction", "").upper()
            
            if prediction_str == "POSITIVE":
                return 1
            elif prediction_str == "NEGATIVE":
                return 0
            else:
                return self._extract_with_regex(original_response)
        except (json.JSONDecodeError, KeyError):
            parser_output_lower = parser_output_text.lower()
            if "positive" in parser_output_lower:
                return 1
            elif "negative" in parser_output_lower:
                return 0
            else:
                return self._extract_with_regex(original_response)
    
    def _extract_with_parser_llm_batch(self, responses: List[str], task_name: str = None) -> Tuple[List[Optional[int]], List[str]]:
        """Extract binary predictions using parser LLM with structured output (batch processing)"""
        if not responses:
            return [], []
        
        try:
            parser_prompts = [self._create_parser_prompt(response, task_name) for response in responses]
            
            parser_sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=50,
                guided_decoding=GuidedDecodingParams(json=PREDICTION_SCHEMA),
            )
            
            outputs = self.parser_llm.generate(parser_prompts, parser_sampling_params)
            
            predictions = []
            parser_outputs = []
            for i, output in enumerate(outputs):
                if not output.outputs:
                    parser_output_text = ""
                else:
                    parser_output_text = output.outputs[0].text.strip()
                    
                    if hasattr(output.outputs[0], 'structured_output'):
                        structured = output.outputs[0].structured_output
                        if structured:
                            parser_output_text = json.dumps(structured)
                    elif hasattr(output, 'structured_output'):
                        structured = output.structured_output
                        if structured:
                            parser_output_text = json.dumps(structured)
                
                parser_outputs.append(parser_output_text)
                prediction = self._parse_parser_output(parser_output_text, responses[i])
                predictions.append(prediction)
            
            return predictions, parser_outputs
                    
        except Exception as e:
            logger.error(f"[GPU {self.config.gpu_id}] Error using parser LLM: {e}")
            return [self._extract_with_regex(r) for r in responses], [""] * len(responses)
    
    def _aggregate_predictions(self, predictions: List[int]) -> float:
        """Aggregate multiple predictions into a single probability score"""
        positive_count = sum(predictions)
        total_count = len(predictions)
        return positive_count / total_count if total_count > 0 else 0.5
    
    def _save_outputs_incremental(self, outputs: List[Dict], output_file: str, is_first: bool = False):
        """Save outputs incrementally to JSON file"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        if is_first:
            with open(output_file, 'w') as f:
                json.dump(outputs, f, indent=2)
        else:
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = []
            
            data.extend(outputs)
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"[GPU {self.config.gpu_id}] Saved {len(outputs)} outputs to {output_file}")
    
    def evaluate_task(self) -> Dict:
        """Evaluate base model on the configured task"""
        task_name = self.config.task_name
        logger.info(f"[GPU {self.config.gpu_id}] Starting evaluation for task: {task_name}")
        
        # Load pre-serialized data
        data_examples = self.load_pre_serialized_data(task_name)
        if not data_examples:
            logger.warning(f"[GPU {self.config.gpu_id}] No data found for task {task_name}")
            return None
        
        # Convert to format expected by evaluation loop
        serialized_examples = []
        for item in data_examples:
            prompt = self._create_prompt_from_context(task_name, item['context'])
            serialized_examples.append({
                'patient_id': item['patient_id'],
                'label_time': item['label_time'],
                'label_value': item['label_value'],
                'prompt': prompt
            })
        
        # Generate predictions in batches
        batch_size = 32
        all_aggregated_predictions = []
        all_predictions_list = []
        all_responses_list = []
        ground_truths = []
        patient_ids = []
        
        # Timing statistics for bottleneck analysis
        total_main_model_time = 0.0
        total_parser_llm_time = 0.0
        total_processing_time = 0.0
        num_batches_processed = 0
        
        # Create output directory for incremental saving
        os.makedirs(self.config.output_dir, exist_ok=True)
        incremental_output_file = os.path.join(self.config.output_dir, f"{task_name}_detailed_results.json")
        is_first_write = True
        
        for i in range(0, len(serialized_examples), batch_size):
            batch = serialized_examples[i:i + batch_size]
            batch_start_time = time.time()
            logger.info(f"[GPU {self.config.gpu_id}] Processing batch {i//batch_size + 1}/{(len(serialized_examples) + batch_size - 1)//batch_size} for task {task_name} ({len(batch)} examples)")
            
            # Time main model generation
            main_model_start = time.time()
            batch_results = self.generate_predictions_batch(batch, self.config.num_samples)
            main_model_time = time.time() - main_model_start
            total_main_model_time += main_model_time
            
            # Collect all responses for batch processing with parser LLM
            all_batch_responses = []
            responses_per_result = []
            
            for result in batch_results:
                responses = result['responses']
                responses_per_result.append(len(responses))
                all_batch_responses.extend(responses)
            
            # Time parser LLM processing
            parser_start = time.time()
            if self.parser_llm is not None:
                all_predictions_with_none, all_parser_outputs = self._extract_with_parser_llm_batch(all_batch_responses, task_name)
            else:
                logger.warning(f"[GPU {self.config.gpu_id}] Parser LLM not available, using regex")
                all_predictions_with_none = [self._extract_with_regex(r) for r in all_batch_responses]
                all_parser_outputs = [""] * len(all_batch_responses)
            parser_time = time.time() - parser_start
            total_parser_llm_time += parser_time
            
            batch_processing_time = time.time() - batch_start_time
            total_processing_time += batch_processing_time
            num_batches_processed += 1
            
            # Log timing for this batch
            num_responses = len(all_batch_responses)
            logger.info(
                f"[GPU {self.config.gpu_id}] Batch timing - "
                f"Main model: {main_model_time:.2f}s ({main_model_time/batch_processing_time*100:.1f}%), "
                f"Parser LLM: {parser_time:.2f}s ({parser_time/batch_processing_time*100:.1f}%), "
                f"Other: {batch_processing_time - main_model_time - parser_time:.2f}s, "
                f"Total: {batch_processing_time:.2f}s "
                f"({num_responses} responses, {num_responses/batch_processing_time:.1f} responses/s)"
            )
            
            # Group predictions back by patient
            response_idx = 0
            for j, result in enumerate(batch_results):
                num_responses = responses_per_result[j]
                predictions_with_none = all_predictions_with_none[response_idx:response_idx + num_responses]
                parser_outputs = all_parser_outputs[response_idx:response_idx + num_responses]
                response_idx += num_responses
                
                predictions = [p for p in predictions_with_none if p is not None]
                
                num_invalid = sum(1 for p in predictions_with_none if p is None)
                num_total = len(predictions_with_none)
                
                aggregated_pred = self._aggregate_predictions(predictions)
                
                all_aggregated_predictions.append(aggregated_pred)
                all_predictions_list.append(predictions)
                all_responses_list.append(result['responses'])
                ground_truths.append(1 if result['ground_truth'] else 0)
                patient_ids.append(result['patient_id'])
                
                # Find original prompt and label_time
                # original_prompt = None
                # label_time = None
                # for ex in serialized_examples:
                #     if ex['patient_id'] == result['patient_id']:
                #         original_prompt = ex.get('prompt', None)
                #         label_time = ex.get('label_time', None)
                #         break
                
                sample_idx = np.random.randint(0, len(result['responses'])) if result['responses'] else 0
                sample_response = result['responses'][sample_idx] if result['responses'] else ""
                sample_parser_output = parser_outputs[sample_idx] if parser_outputs else ""
                
                # Save incrementally
                patient_output = {
                    'patient_id': result['patient_id'],
                    'label_time': result['label_time'],
                    'ground_truth': result['ground_truth'],
                    'score': aggregated_pred,
                    'prompt': result['prompt'],
                    'sample_response': sample_response,
                    'sample_parser_output': sample_parser_output,
                    'num_samples': len(result['responses']),
                    'num_valid_predictions': len(predictions),
                    'num_invalid_predictions': num_invalid,
                    'invalid_format_percentage': float(num_invalid / num_total * 100) if num_total > 0 else 0.0,
                    'predictions': predictions,
                    'parser_outputs': parser_outputs
                }
                self._save_outputs_incremental([patient_output], incremental_output_file, is_first=is_first_write)
                is_first_write = False
        
        # Convert to numpy arrays
        aggregated_predictions = np.array(all_aggregated_predictions)
        ground_truths = np.array(ground_truths)
        
        # Calculate statistics
        total_responses = sum(len(responses) for responses in all_responses_list)
        total_valid_predictions = sum(len(preds) for preds in all_predictions_list)
        total_invalid_predictions = total_responses - total_valid_predictions
        invalid_format_percentage = (total_invalid_predictions / total_responses * 100) if total_responses > 0 else 0.0
        
        # Log timing summary for bottleneck analysis
        if num_batches_processed > 0:
            avg_main_model_time = total_main_model_time / num_batches_processed
            avg_parser_llm_time = total_parser_llm_time / num_batches_processed
            avg_total_time = total_processing_time / num_batches_processed
            main_model_percentage = (total_main_model_time / total_processing_time * 100) if total_processing_time > 0 else 0
            parser_llm_percentage = (total_parser_llm_time / total_processing_time * 100) if total_processing_time > 0 else 0
            
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[GPU {self.config.gpu_id}] PERFORMANCE SUMMARY - Task: {task_name}")
            logger.info("=" * 80)
            logger.info(f"Total batches processed: {num_batches_processed}")
            logger.info(f"Total processing time: {total_processing_time:.2f}s")
            logger.info(f"")
            logger.info(f"Main model generation:")
            logger.info(f"  Total time: {total_main_model_time:.2f}s ({main_model_percentage:.1f}%)")
            logger.info(f"  Average per batch: {avg_main_model_time:.2f}s")
            logger.info(f"  Throughput: {total_responses/total_main_model_time:.1f} responses/s" if total_main_model_time > 0 else "  Throughput: N/A")
            logger.info(f"")
            logger.info(f"Parser LLM processing:")
            logger.info(f"  Total time: {total_parser_llm_time:.2f}s ({parser_llm_percentage:.1f}%)")
            logger.info(f"  Average per batch: {avg_parser_llm_time:.2f}s")
            logger.info(f"  Throughput: {total_responses/total_parser_llm_time:.1f} responses/s" if total_parser_llm_time > 0 else "  Throughput: N/A")
            logger.info(f"")
            logger.info(f"Other processing (data handling, etc.):")
            other_time = total_processing_time - total_main_model_time - total_parser_llm_time
            other_percentage = (other_time / total_processing_time * 100) if total_processing_time > 0 else 0
            logger.info(f"  Total time: {other_time:.2f}s ({other_percentage:.1f}%)")
            logger.info(f"")
            if parser_llm_percentage > 20:
                logger.warning(f"⚠️  Parser LLM is using {parser_llm_percentage:.1f}% of total time - may be a bottleneck!")
            elif parser_llm_percentage > 10:
                logger.info(f"ℹ️  Parser LLM uses {parser_llm_percentage:.1f}% of total time")
            else:
                logger.info(f"✓ Parser LLM uses only {parser_llm_percentage:.1f}% of total time")
            logger.info("=" * 80)
            logger.info("")
        
        # Compute metrics
        try:
            auroc = roc_auc_score(ground_truths, aggregated_predictions)
        except ValueError:
            auroc = 0.5
        
        precision_vals, recall_vals, thresholds = precision_recall_curve(ground_truths, aggregated_predictions)
        fpr, tpr, roc_thresholds = roc_curve(ground_truths, aggregated_predictions)
        
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        binary_predictions = (aggregated_predictions >= optimal_threshold).astype(int)
        
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
            'all_responses': all_responses_list,
            # Performance metrics
            'performance': {
                'total_processing_time': float(total_processing_time),
                'total_main_model_time': float(total_main_model_time),
                'total_parser_llm_time': float(total_parser_llm_time),
                'main_model_percentage': float((total_main_model_time / total_processing_time * 100) if total_processing_time > 0 else 0.0),
                'parser_llm_percentage': float((total_parser_llm_time / total_processing_time * 100) if total_processing_time > 0 else 0.0),
                'num_batches_processed': int(num_batches_processed),
                'avg_batch_time': float(total_processing_time / num_batches_processed) if num_batches_processed > 0 else 0.0,
                'throughput_responses_per_sec': float(total_responses / total_processing_time) if total_processing_time > 0 else 0.0,
            }
        }
        
        logger.info(f"[GPU {self.config.gpu_id}] Task {task_name} Results:")
        logger.info(f"[GPU {self.config.gpu_id}]   AUROC: {auroc:.4f}")
        logger.info(f"[GPU {self.config.gpu_id}]   Precision: {precision:.4f}")
        logger.info(f"[GPU {self.config.gpu_id}]   Recall: {recall:.4f}")
        logger.info(f"[GPU {self.config.gpu_id}]   F1: {f1:.4f}")
        
        # Save summary metrics
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
        logger.info(f"[GPU {self.config.gpu_id}] Summary metrics saved to {summary_file}")
        
        return results
    
    def save_results(self, results: Dict):
        """Save evaluation results for the task"""
        if not results:
            return
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        task_name = results['task_name']
        
        # Save JSON file
        task_json_file = os.path.join(self.config.output_dir, f"{task_name}_results.json")
        with open(task_json_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"[GPU {self.config.gpu_id}] Task {task_name} results saved to {task_json_file}")
        
        # Save CSV file
        task_csv_data = []
        for patient_id, ground_truth, aggregated_pred, predictions_list in zip(
            results['patient_ids'],
            results['ground_truths'],
            results['aggregated_predictions'],
            results['all_predictions']
        ):
            binary_pred = 1 if aggregated_pred >= results['optimal_threshold'] else 0
            row = {
                'patient_id': patient_id,
                'ground_truth': ground_truth,
                'aggregated_prediction': aggregated_pred,
                'binary_prediction': binary_pred,
            }
            
            for sample_idx, pred in enumerate(predictions_list):
                row[f'sample_{sample_idx+1}_prediction'] = pred
            
            task_csv_data.append(row)
        
        task_csv_df = pd.DataFrame(task_csv_data)
        task_csv_file = os.path.join(self.config.output_dir, f"{task_name}_results.csv")
        task_csv_df.to_csv(task_csv_file, index=False)
        logger.info(f"[GPU {self.config.gpu_id}] Task {task_name} CSV saved to {task_csv_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base model on a single task on one GPU")
    
    # Required arguments
    parser.add_argument("--task_name", type=str, required=True,
                       help="Task name to evaluate")
    parser.add_argument("--gpu_id", type=int, required=True,
                       help="GPU ID to use (0-3)")
    
    # Model configuration
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Base model name")
    parser.add_argument("--use_quantization", action="store_true", default=False,
                       help="Use quantization")
    
    # Data configuration
    parser.add_argument("--path_to_serialized_data", type=str, required=True,
                       help="Path to directory with pre-serialized JSON files")
    
    # VLLM configuration
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
    
    # Parser LLM configuration
    parser.add_argument("--parser_model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                       help="Parser LLM model name")
    parser.add_argument("--parser_gpu_memory_utilization", type=float, default=0.1,
                       help="GPU memory utilization for parser LLM")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./base_model_multi_task_results",
                       help="Output directory for results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # CUDA_VISIBLE_DEVICES should be set by the coordinator script in the subprocess environment
    # If not set, set it to the specified GPU ID (for standalone execution)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={args.gpu_id} for task {args.task_name}")
    else:
        logger.info(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} for task {args.task_name} (GPU {args.gpu_id})")
    
    try:
        # Create configuration
        config = SingleTaskEvaluationConfig(
            base_model_name=args.base_model_name,
            use_quantization=args.use_quantization,
            tensor_parallel_size=1,  # Always 1 for single GPU
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_samples=args.num_samples,
            parser_model_name=args.parser_model_name,
            parser_gpu_memory_utilization=args.parser_gpu_memory_utilization,
            path_to_serialized_data=args.path_to_serialized_data,
            output_dir=args.output_dir,
            task_name=args.task_name,
            gpu_id=args.gpu_id
        )
        
        # Initialize evaluator
        evaluator = SingleTaskEvaluator(config)
        
        # Load model
        evaluator.load_model_and_tokenizer()
        
        # Evaluate task
        results = evaluator.evaluate_task()
        
        # Save results
        if results:
            evaluator.save_results(results)
            logger.info(f"[GPU {args.gpu_id}] Task {args.task_name} evaluation completed successfully!")
            return 0
        else:
            logger.error(f"[GPU {args.gpu_id}] Task {args.task_name} evaluation failed!")
            return 1
            
    except Exception as e:
        logger.error(f"[GPU {args.gpu_id}] Error evaluating task {args.task_name}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

