#!/usr/bin/env python3
"""
Multi-sample evaluation pipeline for fine-tuned SFT model on test dataset.
Computes scores over 10 samples for each data point with KV caching for faster inference.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel
from loguru import logger
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class FinalAnswerStoppingCriteria(StoppingCriteria):
    """Stop generation when 'Final Answer: [Positive/Negative]' is generated"""
    
    def __init__(self, tokenizer, stop_strings=["Final Answer: Positive", "Final Answer: Negative", "Final Answer:Positive", "Final Answer:Negative"]):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        # Tokenize stop strings for efficient comparison
        self.stop_token_ids = []
        for stop_string in stop_strings:
            tokens = tokenizer.encode(stop_string, add_special_tokens=False)
            self.stop_token_ids.append(tokens)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if any of the stop strings appear in the generated text
        for stop_tokens in self.stop_token_ids:
            if input_ids.shape[1] >= len(stop_tokens):
                # Check if the last tokens match the stop string
                last_tokens = input_ids[0, -len(stop_tokens):].tolist()
                if last_tokens == stop_tokens:
                    return True
        return False


@dataclass
class MultiSampleEvaluationConfig:
    """Configuration for multi-sample model evaluation"""
    
    # Model configuration
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    peft_model_path: str = "./qwen_sft_output"
    trust_remote_code: bool = True
    use_quantization: bool = True
    
    # Evaluation configuration
    max_length: int = 32768
    max_new_tokens: int = 16384
    temperature: float = 0.7  # Higher temperature for more diverse samples
    top_p: float = 0.9
    do_sample: bool = True
    num_samples: int = 10  # Number of samples per data point
    
    # KV Cache configuration
    use_kv_cache: bool = True
    cache_implementation: str = "torch"  # or "flash_attention_2"
    
    # Output configuration
    output_dir: str = "./multi_sample_evaluation_results"
    save_predictions: bool = True
    save_plots: bool = True


class MultiSampleSFTModelEvaluator:
    """Multi-sample evaluator for fine-tuned SFT model with KV caching"""
    
    def __init__(self, config: MultiSampleEvaluationConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.baseline_model = None
        self.generation_config = None
        
    def load_model_and_tokenizer(self):
        """Load the fine-tuned model and tokenizer with KV cache support"""
        logger.info(f"Loading base model: {self.config.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with KV cache support
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        # Add KV cache implementation if specified
        if self.config.use_kv_cache and self.config.cache_implementation:
            model_kwargs["attn_implementation"] = self.config.cache_implementation
        
        # Add quantization if specified
        if self.config.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            **model_kwargs
        )
        
        # Load PEFT model
        logger.info(f"Loading PEFT model from: {self.config.peft_model_path}")
        self.model = PeftModel.from_pretrained(base_model, self.config.peft_model_path)
        
        # Load baseline model for comparison
        logger.info("Loading baseline model for comparison...")
        self.baseline_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            **model_kwargs
        )
        
        # Set up generation config for multi-sample generation
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,  # Enable KV caching
        )
        
        logger.info("Models and tokenizer loaded successfully with KV cache support")
    
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
        if "Final Answer:" in user_content:
            user_content = user_content.split("Final Answer:")[0].strip()
        
        # Check if the content already ends with the instruction we want to add
        target_instruction = "Please provide your analysis, and conclude with your final prediction in the format Final Answer: [Positive/Negative]."
        if not user_content.strip().endswith(target_instruction.strip()):
            user_content += "\n\n" + target_instruction
        
        # Return as single completion prompt (no chat format)
        return user_content
    
    def _get_stopping_criteria(self):
        """Get stopping criteria for early stopping on 'Final Answer: [Positive/Negative]'"""
        stopping_criteria = FinalAnswerStoppingCriteria(self.tokenizer)
        return StoppingCriteriaList([stopping_criteria])
    
    def generate_multiple_predictions(self, example: Dict, use_baseline: bool = False) -> Dict:
        """Generate multiple predictions for a single example with KV caching"""
        prompt = self.format_prompt(example)
        
        # Check if prompt is too long for our token budget
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        max_prompt_tokens = self.config.max_length - self.config.max_new_tokens
        
        if len(prompt_tokens) > max_prompt_tokens:
            logger.warning(f"Skipping patient {example['patient_id']}: prompt too long ({len(prompt_tokens)} tokens > {max_prompt_tokens} limit)")
            return {
                'patient_id': example['patient_id'],
                'ground_truth': example.get('label_value', None),
                'predictions': [None] * self.config.num_samples,
                'prediction_texts': ["SKIPPED"] * self.config.num_samples,
                'full_responses': ["SKIPPED: Prompt too long"] * self.config.num_samples
            }
        
        # Choose model
        model = self.baseline_model if use_baseline else self.model
        
        # Tokenize once
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - self.config.max_new_tokens
        )
        
        # Ensure input_ids are integers and move to device
        inputs = {k: v.to(model.device) if k == 'input_ids' else v.to(model.device) for k, v in inputs.items()}
        if 'input_ids' in inputs:
            inputs['input_ids'] = inputs['input_ids'].long()
        
        # Generate multiple samples
        all_responses = []
        all_predictions = []
        
        # Use KV cache for faster generation
        with torch.no_grad():
                # First generation to warm up the cache
                outputs = model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    num_return_sequences=1,
                    stopping_criteria=self._get_stopping_criteria(),
                )
            
            # Generate remaining samples
            for sample_idx in range(self.config.num_samples):
                if sample_idx == 0:
                    # Use the first generation
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                        # Generate new sample
                        outputs = model.generate(
                            **inputs,
                            generation_config=self.generation_config,
                            num_return_sequences=1,
                            stopping_criteria=self._get_stopping_criteria(),
                        )
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # For single completion format, the response is the generated text after the prompt
                # Remove the original prompt from the response
                if prompt in response:
                    generated_text = response[len(prompt):].strip()
                else:
                    generated_text = response
                
                # Clean up any remaining artifacts
                if generated_text.startswith("<|im_end|>"):
                    generated_text = generated_text[len("<|im_end|>"):].strip()
                
                # Fallback: If stopping criteria didn't work, complete the "Final Answer"
                if generated_text.strip().endswith("Final Answer"):
                    # Try to extract prediction from context
                    if "positive" in generated_text.lower() or "acute myocardial infarction" in generated_text.lower():
                        generated_text += ": Positive"
                    elif "negative" in generated_text.lower() or "no evidence" in generated_text.lower():
                        generated_text += ": Negative"
                    else:
                        generated_text += ": [UNKNOWN]"
                
                # Clean up the response
                assistant_response = self._clean_response(generated_text)
                
                # Extract prediction
                prediction = self.extract_prediction_from_response(assistant_response)
                
                all_responses.append(assistant_response)
                all_predictions.append(prediction)
        
        # Compute aggregated prediction
        aggregated_prediction = self._aggregate_predictions(all_predictions)
        
        return {
            'patient_id': example['patient_id'],
            'ground_truth': example.get('label_value', None),
            'all_responses': all_responses,
            'all_predictions': all_predictions,
            'aggregated_prediction': aggregated_prediction,
            'prediction_confidence': self._compute_confidence(all_predictions),
            'prediction_consistency': self._compute_consistency(all_predictions),
            'model_type': 'baseline' if use_baseline else 'fine_tuned'
        }
    
    def _clean_response(self, response: str) -> str:
        """Clean up the model response"""
        # Remove any remaining prompt text that might be included
        if "To determine whether" in response:
            reasoning_start = response.find("To determine whether")
            response = response[reasoning_start:]
        elif "Based on" in response:
            reasoning_start = response.find("Based on")
            response = response[reasoning_start:]
        elif "I will analyze" in response:
            reasoning_start = response.find("I will analyze")
            response = response[reasoning_start:]
        
        return response
    
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
    
    def _aggregate_predictions(self, predictions: List[int]) -> float:
        """Aggregate multiple predictions into a single probability score"""
        # Simple majority voting with probability
        positive_count = sum(predictions)
        total_count = len(predictions)
        
        # Return the proportion of positive predictions
        return positive_count / total_count
    
    def _compute_confidence(self, predictions: List[int]) -> float:
        """Compute confidence based on prediction consistency"""
        # Higher confidence when predictions are more consistent
        positive_count = sum(predictions)
        total_count = len(predictions)
        
        # Confidence is the maximum of the two proportions
        positive_prop = positive_count / total_count
        negative_prop = (total_count - positive_count) / total_count
        
        return max(positive_prop, negative_prop)
    
    def _compute_consistency(self, predictions: List[int]) -> float:
        """Compute consistency score (1.0 = all same, 0.0 = split evenly)"""
        if not predictions:
            return 0.0
        
        # Count occurrences
        counts = Counter(predictions)
        max_count = max(counts.values())
        total_count = len(predictions)
        
        # Consistency is the proportion of the most common prediction
        return max_count / total_count
    
    def evaluate_dataset(self, test_dataset: List[Dict], output_dir: str = None) -> Dict:
        """Evaluate both fine-tuned and baseline models on test dataset with multi-sample scoring"""
        logger.info(f"Evaluating models on {len(test_dataset)} test examples with {self.config.num_samples} samples each")
        
        # Set up output files for incremental saving
        fine_tuned_output_file = None
        baseline_output_file = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fine_tuned_output_file = os.path.join(output_dir, "fine_tuned_multi_sample_outputs.json")
            baseline_output_file = os.path.join(output_dir, "baseline_multi_sample_outputs.json")
        
        # Evaluate fine-tuned model
        logger.info("Evaluating fine-tuned model with multi-sample scoring...")
        fine_tuned_results = self._evaluate_single_model_multi_sample(
            test_dataset, use_baseline=False, output_file=fine_tuned_output_file
        )
        
        # Evaluate baseline model
        logger.info("Evaluating baseline model with multi-sample scoring...")
        baseline_results = self._evaluate_single_model_multi_sample(
            test_dataset, use_baseline=True, output_file=baseline_output_file
        )
        
        # Combine results
        results = {
            'fine_tuned': fine_tuned_results,
            'baseline': baseline_results,
            'comparison': {
                'auroc_improvement': fine_tuned_results['auroc'] - baseline_results['auroc'],
                'accuracy_improvement': fine_tuned_results['accuracy'] - baseline_results['accuracy'],
                'precision_improvement': fine_tuned_results['precision'] - baseline_results['precision'],
                'recall_improvement': fine_tuned_results['recall'] - baseline_results['recall'],
                'confidence_improvement': fine_tuned_results['avg_confidence'] - baseline_results['avg_confidence'],
                'consistency_improvement': fine_tuned_results['avg_consistency'] - baseline_results['avg_consistency']
            },
            'n_examples': len(test_dataset),
            'n_samples_per_example': self.config.num_samples
        }
        
        # Log comparison results
        logger.info(f"Multi-Sample Comparison Results:")
        logger.info(f"  AUROC - Fine-tuned: {fine_tuned_results['auroc']:.4f}, Baseline: {baseline_results['auroc']:.4f}, Improvement: {results['comparison']['auroc_improvement']:.4f}")
        logger.info(f"  Accuracy - Fine-tuned: {fine_tuned_results['accuracy']:.4f}, Baseline: {baseline_results['accuracy']:.4f}, Improvement: {results['comparison']['accuracy_improvement']:.4f}")
        logger.info(f"  Precision - Fine-tuned: {fine_tuned_results['precision']:.4f}, Baseline: {baseline_results['precision']:.4f}, Improvement: {results['comparison']['precision_improvement']:.4f}")
        logger.info(f"  Recall - Fine-tuned: {fine_tuned_results['recall']:.4f}, Baseline: {baseline_results['recall']:.4f}, Improvement: {results['comparison']['recall_improvement']:.4f}")
        logger.info(f"  Avg Confidence - Fine-tuned: {fine_tuned_results['avg_confidence']:.4f}, Baseline: {baseline_results['avg_confidence']:.4f}, Improvement: {results['comparison']['confidence_improvement']:.4f}")
        logger.info(f"  Avg Consistency - Fine-tuned: {fine_tuned_results['avg_consistency']:.4f}, Baseline: {baseline_results['avg_consistency']:.4f}, Improvement: {results['comparison']['consistency_improvement']:.4f}")
        
        return results
    
    def _evaluate_single_model_multi_sample(self, test_dataset: List[Dict], use_baseline: bool = False, 
                                          output_file: str = None) -> Dict:
        """Evaluate a single model on test dataset with multi-sample scoring"""
        model_name = "baseline" if use_baseline else "fine_tuned"
        logger.info(f"Evaluating {model_name} model with {self.config.num_samples} samples per example...")
        
        aggregated_predictions = []
        ground_truths = []
        patient_ids = []
        all_sample_predictions = []
        all_confidences = []
        all_consistencies = []
        all_outputs = []
        
        for i, example in enumerate(test_dataset):
            logger.info(f"Processing example {i+1}/{len(test_dataset)} (Patient {example['patient_id']}) - {model_name} ({self.config.num_samples} samples)")
            
            try:
                # Generate multiple predictions
                result = self.generate_multiple_predictions(example, use_baseline=use_baseline)
                
                # Store results
                aggregated_predictions.append(result['aggregated_prediction'])
                ground_truths.append(example.get('label_value', 0))
                patient_ids.append(example['patient_id'])
                all_sample_predictions.append(result['all_predictions'])
                all_confidences.append(result['prediction_confidence'])
                all_consistencies.append(result['prediction_consistency'])
                
                # Create output entry
                output_entry = {
                    'patient_id': example['patient_id'],
                    'label_time': example.get('label_time', ''),
                    'ground_truth': example.get('label_value', 0),
                    'aggregated_prediction': result['aggregated_prediction'],
                    'prediction_confidence': result['prediction_confidence'],
                    'prediction_consistency': result['prediction_consistency'],
                    'all_predictions': result['all_predictions'],
                    'all_responses': result['all_responses'],
                    'model_type': model_name,
                    'example_id': i + 1,
                    'n_samples': self.config.num_samples
                }
                all_outputs.append(output_entry)
                
                # Incrementally save outputs
                if output_file:
                    self._save_outputs_incremental([output_entry], output_file, is_first=(i == 0))
                
                logger.info(f"Patient {example['patient_id']}: GT={example.get('label_value', 0)}, AggPred={result['aggregated_prediction']:.3f}, Conf={result['prediction_confidence']:.3f}, Cons={result['prediction_consistency']:.3f} ({model_name})")
                
            except Exception as e:
                logger.error(f"Error processing patient {example['patient_id']} with {model_name}: {e}")
                # Use default values for failed predictions
                aggregated_predictions.append(0.5)
                ground_truths.append(example.get('label_value', 0))
                patient_ids.append(example['patient_id'])
                all_sample_predictions.append([0] * self.config.num_samples)
                all_confidences.append(0.0)
                all_consistencies.append(0.0)
                
                # Create error output entry
                output_entry = {
                    'patient_id': example['patient_id'],
                    'label_time': example.get('label_time', ''),
                    'ground_truth': example.get('label_value', 0),
                    'aggregated_prediction': 0.5,
                    'prediction_confidence': 0.0,
                    'prediction_consistency': 0.0,
                    'all_predictions': [0] * self.config.num_samples,
                    'all_responses': ["Error in generation"] * self.config.num_samples,
                    'model_type': model_name,
                    'example_id': i + 1,
                    'n_samples': self.config.num_samples,
                    'error': str(e)
                }
                all_outputs.append(output_entry)
                
                # Incrementally save outputs
                if output_file:
                    self._save_outputs_incremental([output_entry], output_file, is_first=(i == 0))
        
        # Convert to numpy arrays
        aggregated_predictions = np.array(aggregated_predictions)
        ground_truths = np.array(ground_truths)
        
        # Compute metrics
        try:
            auroc = roc_auc_score(ground_truths, aggregated_predictions)
        except ValueError:
            auroc = 0.5  # Default if only one class
        
        # Additional metrics
        precision, recall, thresholds = precision_recall_curve(ground_truths, aggregated_predictions)
        fpr, tpr, roc_thresholds = roc_curve(ground_truths, aggregated_predictions)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        # Binary predictions at optimal threshold
        binary_predictions = (aggregated_predictions >= optimal_threshold).astype(int)
        
        # Additional metrics
        accuracy = np.mean(binary_predictions == ground_truths)
        precision_at_optimal = precision[optimal_idx] if optimal_idx < len(precision) else 0.0
        recall_at_optimal = recall[optimal_idx] if optimal_idx < len(recall) else 0.0
        
        # Multi-sample specific metrics
        avg_confidence = np.mean(all_confidences)
        avg_consistency = np.mean(all_consistencies)
        
        results = {
            'model_type': model_name,
            'auroc': auroc,
            'accuracy': accuracy,
            'precision': precision_at_optimal,
            'recall': recall_at_optimal,
            'optimal_threshold': optimal_threshold,
            'avg_confidence': avg_confidence,
            'avg_consistency': avg_consistency,
            'aggregated_predictions': aggregated_predictions.tolist(),
            'ground_truths': ground_truths.tolist(),
            'patient_ids': patient_ids,
            'all_sample_predictions': all_sample_predictions,
            'all_confidences': all_confidences,
            'all_consistencies': all_consistencies,
            'all_outputs': all_outputs,
            'n_samples_per_example': self.config.num_samples
        }
        
        logger.info(f"{model_name.capitalize()} Multi-Sample Results:")
        logger.info(f"  AUROC: {auroc:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision_at_optimal:.4f}")
        logger.info(f"  Recall: {recall_at_optimal:.4f}")
        logger.info(f"  Optimal Threshold: {optimal_threshold:.4f}")
        logger.info(f"  Avg Confidence: {avg_confidence:.4f}")
        logger.info(f"  Avg Consistency: {avg_consistency:.4f}")
        
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
        
        logger.info(f"Saved {len(outputs)} multi-sample outputs to {output_file}")
    
    def save_results(self, results: Dict, output_dir: str):
        """Save multi-sample evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, "multi_sample_evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Multi-sample results saved to {results_file}")
        
        # Save predictions CSV for both models
        if self.config.save_predictions:
            # Fine-tuned model predictions
            fine_tuned_df = pd.DataFrame({
                'patient_id': results['fine_tuned']['patient_ids'],
                'ground_truth': results['fine_tuned']['ground_truths'],
                'aggregated_prediction': results['fine_tuned']['aggregated_predictions'],
                'prediction_binary': (np.array(results['fine_tuned']['aggregated_predictions']) >= results['fine_tuned']['optimal_threshold']).astype(int),
                'confidence': results['fine_tuned']['all_confidences'],
                'consistency': results['fine_tuned']['all_consistencies'],
                'model_type': 'fine_tuned'
            })
            
            # Baseline model predictions
            baseline_df = pd.DataFrame({
                'patient_id': results['baseline']['patient_ids'],
                'ground_truth': results['baseline']['ground_truths'],
                'aggregated_prediction': results['baseline']['aggregated_predictions'],
                'prediction_binary': (np.array(results['baseline']['aggregated_predictions']) >= results['baseline']['optimal_threshold']).astype(int),
                'confidence': results['baseline']['all_confidences'],
                'consistency': results['baseline']['all_consistencies'],
                'model_type': 'baseline'
            })
            
            # Combine and save
            combined_df = pd.concat([fine_tuned_df, baseline_df], ignore_index=True)
            predictions_file = os.path.join(output_dir, "multi_sample_predictions_comparison.csv")
            combined_df.to_csv(predictions_file, index=False)
            logger.info(f"Multi-sample predictions saved to {predictions_file}")
        
        # Save plots
        if self.config.save_plots:
            self.plot_multi_sample_results(results, output_dir)
    
    def plot_multi_sample_results(self, results: Dict, output_dir: str):
        """Create evaluation plots for multi-sample results"""
        fine_tuned_preds = np.array(results['fine_tuned']['aggregated_predictions'])
        baseline_preds = np.array(results['baseline']['aggregated_predictions'])
        ground_truths = np.array(results['fine_tuned']['ground_truths'])
        
        # ROC Curves for both models
        fpr_ft, tpr_ft, _ = roc_curve(ground_truths, fine_tuned_preds)
        fpr_bl, tpr_bl, _ = roc_curve(ground_truths, baseline_preds)
        
        plt.figure(figsize=(20, 12))
        
        # ROC Curve Comparison
        plt.subplot(2, 4, 1)
        plt.plot(fpr_ft, tpr_ft, label=f'Fine-tuned (AUC = {results["fine_tuned"]["auroc"]:.3f})', color='blue')
        plt.plot(fpr_bl, tpr_bl, label=f'Baseline (AUC = {results["baseline"]["auroc"]:.3f})', color='red')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True)
        
        # Fine-tuned model prediction distribution
        plt.subplot(2, 4, 2)
        plt.hist(fine_tuned_preds[ground_truths == 0], alpha=0.7, label='Negative', bins=20, color='lightblue')
        plt.hist(fine_tuned_preds[ground_truths == 1], alpha=0.7, label='Positive', bins=20, color='darkblue')
        plt.axvline(results['fine_tuned']['optimal_threshold'], color='red', linestyle='--', label=f'Optimal Threshold')
        plt.xlabel('Aggregated Prediction Probability')
        plt.ylabel('Count')
        plt.title('Fine-tuned Model Distribution')
        plt.legend()
        plt.grid(True)
        
        # Baseline model prediction distribution
        plt.subplot(2, 4, 3)
        plt.hist(baseline_preds[ground_truths == 0], alpha=0.7, label='Negative', bins=20, color='lightcoral')
        plt.hist(baseline_preds[ground_truths == 1], alpha=0.7, label='Positive', bins=20, color='darkred')
        plt.axvline(results['baseline']['optimal_threshold'], color='red', linestyle='--', label=f'Optimal Threshold')
        plt.xlabel('Aggregated Prediction Probability')
        plt.ylabel('Count')
        plt.title('Baseline Model Distribution')
        plt.legend()
        plt.grid(True)
        
        # Confidence distribution
        plt.subplot(2, 4, 4)
        plt.hist(results['fine_tuned']['all_confidences'], alpha=0.7, label='Fine-tuned', bins=20, color='blue')
        plt.hist(results['baseline']['all_confidences'], alpha=0.7, label='Baseline', bins=20, color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True)
        
        # Consistency distribution
        plt.subplot(2, 4, 5)
        plt.hist(results['fine_tuned']['all_consistencies'], alpha=0.7, label='Fine-tuned', bins=20, color='blue')
        plt.hist(results['baseline']['all_consistencies'], alpha=0.7, label='Baseline', bins=20, color='red')
        plt.xlabel('Prediction Consistency')
        plt.ylabel('Count')
        plt.title('Consistency Distribution')
        plt.legend()
        plt.grid(True)
        
        # Confidence vs Performance
        plt.subplot(2, 4, 6)
        ft_correct = (np.array(results['fine_tuned']['aggregated_predictions']) >= results['fine_tuned']['optimal_threshold']).astype(int) == ground_truths
        bl_correct = (np.array(results['baseline']['aggregated_predictions']) >= results['baseline']['optimal_threshold']).astype(int) == ground_truths
        
        plt.scatter(results['fine_tuned']['all_confidences'], ft_correct.astype(int), alpha=0.6, label='Fine-tuned', color='blue')
        plt.scatter(results['baseline']['all_confidences'], bl_correct.astype(int), alpha=0.6, label='Baseline', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Correct Prediction (0/1)')
        plt.title('Confidence vs Performance')
        plt.legend()
        plt.grid(True)
        
        # Consistency vs Performance
        plt.subplot(2, 4, 7)
        plt.scatter(results['fine_tuned']['all_consistencies'], ft_correct.astype(int), alpha=0.6, label='Fine-tuned', color='blue')
        plt.scatter(results['baseline']['all_consistencies'], bl_correct.astype(int), alpha=0.6, label='Baseline', color='red')
        plt.xlabel('Consistency')
        plt.ylabel('Correct Prediction (0/1)')
        plt.title('Consistency vs Performance')
        plt.legend()
        plt.grid(True)
        
        # Metrics comparison
        plt.subplot(2, 4, 8)
        metrics = ['AUROC', 'Accuracy', 'Precision', 'Recall', 'Confidence', 'Consistency']
        fine_tuned_values = [
            results['fine_tuned']['auroc'],
            results['fine_tuned']['accuracy'],
            results['fine_tuned']['precision'],
            results['fine_tuned']['recall'],
            results['fine_tuned']['avg_confidence'],
            results['fine_tuned']['avg_consistency']
        ]
        baseline_values = [
            results['baseline']['auroc'],
            results['baseline']['accuracy'],
            results['baseline']['precision'],
            results['baseline']['recall'],
            results['baseline']['avg_confidence'],
            results['baseline']['avg_consistency']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, fine_tuned_values, width, label='Fine-tuned', color='blue', alpha=0.7)
        plt.bar(x + width/2, baseline_values, width, label='Baseline', color='red', alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Multi-Sample Performance Comparison')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, "multi_sample_evaluation_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Multi-sample evaluation plots saved to {plot_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SFT model with multi-sample scoring")
    
    # Model configuration
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model name")
    parser.add_argument("--peft_model_path", type=str, default="./qwen_sft_output",
                       help="Path to fine-tuned PEFT model")
    parser.add_argument("--use_quantization", action="store_true", default=True,
                       help="Use quantization")
    
    # Data configuration
    parser.add_argument("--dataset_file", type=str, required=True,
                       help="Path to SFT dataset JSON file")
    parser.add_argument("--path_to_splits", type=str, required=True,
                       help="Path to splits CSV file")
    
    # Evaluation configuration
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature for diversity")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples per data point")
    
    # KV Cache configuration
    parser.add_argument("--use_kv_cache", action="store_true", default=True,
                       help="Use KV caching for faster inference")
    parser.add_argument("--cache_implementation", type=str, default="torch",
                       choices=["torch", "flash_attention_2"],
                       help="KV cache implementation")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./multi_sample_evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--save_predictions", action="store_true", default=True,
                       help="Save detailed predictions")
    parser.add_argument("--save_plots", action="store_true", default=True,
                       help="Save evaluation plots")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create configuration
    config = MultiSampleEvaluationConfig(
        base_model_name=args.base_model_name,
        peft_model_path=args.peft_model_path,
        use_quantization=args.use_quantization,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        use_kv_cache=args.use_kv_cache,
        cache_implementation=args.cache_implementation,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        save_plots=args.save_plots
    )
    
    # Initialize evaluator
    evaluator = MultiSampleSFTModelEvaluator(config)
    
    # Load model
    evaluator.load_model_and_tokenizer()
    
    # Load test dataset
    test_dataset = evaluator.load_test_dataset(args.dataset_file, args.path_to_splits)
    
    if not test_dataset:
        logger.error("No test examples found. Exiting.")
        return
    
    # Evaluate model
    results = evaluator.evaluate_dataset(test_dataset, args.output_dir)
    
    # Save results
    evaluator.save_results(results, args.output_dir)
    
    logger.info("Multi-sample evaluation completed successfully!")


if __name__ == "__main__":
    main()
