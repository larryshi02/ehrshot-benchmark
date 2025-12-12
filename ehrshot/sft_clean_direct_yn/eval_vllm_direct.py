#!/usr/bin/env python3
"""
Direct Y/N Evaluation Script using vLLM

This script evaluates models trained for direct binary classification by computing
normalized probabilities over the vocabulary {"Positive", "Negative"}.

Key Features:
1. Computes P("Positive") / (P("Positive") + P("Negative")) using first-token logits
2. Single forward pass (no sampling) for deterministic evaluation
3. Supports both finetuned (LoRA) models and base models
4. Outputs predictions.csv compatible with existing metrics calculator

Usage:
    # Evaluate finetuned model
    python eval_vllm_direct.py \
        --model_base Qwen/Qwen3-8B \
        --lora_path /path/to/lora/adapter \
        --data_path /path/to/test_data.json \
        --task_name acute_mi \
        --output_dir /path/to/output

    # Evaluate base model (no LoRA)
    python eval_vllm_direct.py \
        --model_base Qwen/Qwen3-8B \
        --data_path /path/to/test_data.json \
        --task_name acute_mi \
        --output_dir /path/to/output
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Dict, Tuple, Optional


# Task-specific queries (same as original)
TASK_QUERIES = {
    'acute_mi': 'Will the patient develop an acute myocardial infarction in the next year?',
    'hyperlipidemia': 'Will the patient develop hyperlipidemia in the next year?',
    'hypertension': 'Will the patient develop hypertension in the next year?',
    'pancreatic_cancer': 'Will the patient develop pancreatic cancer in the next year?'
}


def construct_direct_yn_messages(context: str, task_name: str) -> List[Dict]:
    """
    Construct messages for direct Y/N prediction.
    Uses simplified prompt without reasoning instructions.
    """
    task_query = TASK_QUERIES[task_name]
    
    # Simplified system prompt (same as training data)
    system_content = (
        f"You are a medical expert. You will be provided with a patient's "
        f"Electronic Healthcare Record (EHR).\n"
        f"Your task is to predict: {task_query}\n\n"
        f"Respond with exactly one word: Positive or Negative.\n"
        f"- Positive: The patient WILL develop the condition.\n"
        f"- Negative: The patient will NOT develop the condition."
    )
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": context}
    ]


def get_token_ids(tokenizer, tokens: List[str]) -> Dict[str, int]:
    """
    Get token IDs for the classification tokens.
    
    Handles potential tokenization differences (e.g., with/without leading space).
    Returns a dict mapping token string to its most likely token ID.
    """
    token_ids = {}
    
    for token in tokens:
        # Try different variants: with leading space, without, capitalized, etc.
        variants = [token, f" {token}", token.lower(), f" {token.lower()}"]
        
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                # Found single-token encoding
                token_ids[token] = ids[0]
                break
        
        if token not in token_ids:
            # Use first token of multi-token encoding as fallback
            ids = tokenizer.encode(token, add_special_tokens=False)
            token_ids[token] = ids[0]
            print(f"Warning: '{token}' encodes to {len(ids)} tokens, using first token ID: {ids[0]}")
    
    return token_ids


def compute_normalized_probability(
    logits: torch.Tensor,
    positive_id: int,
    negative_id: int
) -> float:
    """
    Compute normalized probability of "Positive" over {"Positive", "Negative"}.
    
    P(Positive) = exp(logit_positive) / (exp(logit_positive) + exp(logit_negative))
               = softmax([logit_positive, logit_negative])[0]
    
    Args:
        logits: Logits tensor for the next token prediction
        positive_id: Token ID for "Positive"
        negative_id: Token ID for "Negative"
    
    Returns:
        Normalized probability of "Positive" (float between 0 and 1)
    """
    # Extract logits for our two classes
    binary_logits = torch.tensor([logits[positive_id], logits[negative_id]])
    
    # Apply softmax to get normalized probabilities
    probs = torch.softmax(binary_logits, dim=0)
    
    # Return probability of "Positive" (first element)
    return probs[0].item()


def main():
    parser = argparse.ArgumentParser(description="Direct Y/N Evaluation with vLLM")
    parser.add_argument("--model_base", type=str, required=True,
                        help="Base model name/path (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter. Omit or pass 'None' for base model evaluation.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test data JSON file")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=list(TASK_QUERIES.keys()),
                        help="Task name for prompt construction")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine if we're using LoRA
    use_lora = args.lora_path is not None and args.lora_path.lower() != "none"
    
    print("=" * 60)
    print("Direct Y/N Evaluation")
    print("=" * 60)
    print(f"Base Model: {args.model_base}")
    print(f"LoRA Path: {args.lora_path if use_lora else 'None (Base Model)'}")
    print(f"Task: {args.task_name}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load Test Data
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading test data...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for test split if available, otherwise use all
    if any(d.get("split") == "test" for d in data):
        test_data = [d for d in data if d.get("split") == "test"]
        print(f"  Found {len(test_data)} test examples (filtered by split='test')")
    else:
        test_data = data
        print(f"  Using all {len(test_data)} examples (no split field found)")

    # -------------------------------------------------------------------------
    # 2. Initialize vLLM with prompt_logprobs enabled
    # -------------------------------------------------------------------------
    print("\n[2/4] Initializing vLLM...")
    
    llm = LLM(
        model=args.model_base,
        enable_lora=use_lora,
        max_model_len=12288,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
    )
    
    tokenizer = llm.get_tokenizer()
    
    # Get token IDs for our classification tokens
    token_ids = get_token_ids(tokenizer, ["Positive", "Negative"])
    positive_id = token_ids["Positive"]
    negative_id = token_ids["Negative"]
    print(f"  Token IDs: Positive={positive_id}, Negative={negative_id}")

    # -------------------------------------------------------------------------
    # 3. Prepare Prompts and Run Inference
    # -------------------------------------------------------------------------
    print("\n[3/4] Preparing prompts and running inference...")
    
    prompts = []
    metadata = []
    
    for item in test_data:
        # Get context from 'context' field or from conversations
        if 'context' in item:
            context = item['context']
        elif 'conversations' in item:
            # Extract user content from conversations
            for conv in item['conversations']:
                if conv['role'] == 'user':
                    context = conv['content']
                    break
        else:
            raise ValueError(f"Cannot find context in item: {item.keys()}")
        
        # Construct messages for direct Y/N
        messages = construct_direct_yn_messages(context, args.task_name)
        
        # Apply chat template with thinking disabled
        # Note: enable_thinking=False tells Qwen3 to not use thinking mode
        try:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        prompts.append(prompt)
        
        # Get ground truth - handle both naming conventions
        ground_truth = item.get('label_value', item.get('ground_truth'))
        
        metadata.append({
            "patient_id": item.get('patient_id'),
            "label_time": item.get('label_time'),
            "ground_truth": ground_truth,
        })

    # Sampling params for getting logprobs of next token
    # We request logprobs but only generate 1 token (to get the logits)
    sampling_params = SamplingParams(
        n=1,
        temperature=0,  # Greedy for deterministic behavior
        max_tokens=1,   # Only need first token
        logprobs=50,    # Get top 50 logprobs (should include Positive/Negative)
    )

    # LoRA request if applicable
    lora_request = None
    if use_lora:
        lora_request = LoRARequest("adapter", 1, args.lora_path)

    # Run inference
    print(f"  Running inference on {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # -------------------------------------------------------------------------
    # 4. Process Outputs and Compute Probabilities
    # -------------------------------------------------------------------------
    print("\n[4/4] Computing normalized probabilities...")
    
    detailed_logs = []
    csv_rows = []
    
    for i, output in enumerate(outputs):
        meta = metadata[i]
        
        # Get logprobs from the first (and only) output
        output_logprobs = output.outputs[0].logprobs
        
        if output_logprobs and len(output_logprobs) > 0:
            # Get the logprobs dict for the first generated token position
            first_token_logprobs = output_logprobs[0]
            
            # Extract log probabilities for Positive and Negative
            # logprobs is a dict mapping token_id -> Logprob object
            logprob_positive = None
            logprob_negative = None
            
            # Check if our tokens are in the top-k logprobs
            for token_id, logprob_obj in first_token_logprobs.items():
                if token_id == positive_id:
                    logprob_positive = logprob_obj.logprob
                elif token_id == negative_id:
                    logprob_negative = logprob_obj.logprob
            
            # If tokens not in top-k, use very low probability
            if logprob_positive is None:
                logprob_positive = -100.0  # Effectively 0 probability
                print(f"  Warning: 'Positive' not in top logprobs for example {i}")
            if logprob_negative is None:
                logprob_negative = -100.0
                print(f"  Warning: 'Negative' not in top logprobs for example {i}")
            
            # Compute normalized probability
            # P(Positive) = exp(lp_pos) / (exp(lp_pos) + exp(lp_neg))
            #             = 1 / (1 + exp(lp_neg - lp_pos))  [numerically stable]
            log_diff = logprob_negative - logprob_positive
            prob_positive = 1.0 / (1.0 + np.exp(log_diff))
            
        else:
            # Fallback if no logprobs available
            print(f"  Warning: No logprobs available for example {i}, using 0.5")
            prob_positive = 0.5
            logprob_positive = 0.0
            logprob_negative = 0.0
        
        # Get generated text for logging
        generated_text = output.outputs[0].text.strip()
        
        # Store detailed log
        detailed_logs.append({
            **meta,
            "probability_positive": prob_positive,
            "logprob_positive": logprob_positive,
            "logprob_negative": logprob_negative,
            "generated_token": generated_text,
            "prompt_length": len(prompts[i]),
        })
        
        # Store CSV row (compatible with existing metrics calculator)
        csv_rows.append({
            "patient_id": meta['patient_id'],
            "label_time": meta['label_time'],
            "target_task": args.task_name,
            "ground_truth": meta['ground_truth'],
            "probability_score": prob_positive,  # This is P(Positive)
            "has_valid_samples": True  # Always valid for direct probability computation
        })

    # -------------------------------------------------------------------------
    # 5. Save Results
    # -------------------------------------------------------------------------
    print("\nSaving results...")
    
    # Save detailed logs (for debugging/analysis)
    detailed_path = os.path.join(args.output_dir, "detailed_logs.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, indent=2)
    print(f"  Saved detailed logs to: {detailed_path}")
    
    # Save predictions CSV (compatible with metrics calculator)
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  Saved predictions to: {csv_path}")
    
    # Save run statistics
    run_stats = {
        "model_base": args.model_base,
        "lora_path": args.lora_path if use_lora else None,
        "task_name": args.task_name,
        "num_examples": len(test_data),
        "positive_token_id": positive_id,
        "negative_token_id": negative_id,
        "method": "normalized_probability",
        "total_responses": len(test_data),
        "total_valid_predictions": len(test_data),  # All are valid with this method
        "total_invalid_predictions": 0,
        "invalid_format_percentage": 0.0
    }
    stats_path = os.path.join(args.output_dir, "run_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(run_stats, f, indent=2)
    print(f"  Saved run stats to: {stats_path}")
    
    # Print summary statistics
    probs = [r['probability_score'] for r in csv_rows]
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"  Mean P(Positive): {np.mean(probs):.4f}")
    print(f"  Std P(Positive): {np.std(probs):.4f}")
    print(f"  Min P(Positive): {np.min(probs):.4f}")
    print(f"  Max P(Positive): {np.max(probs):.4f}")
    
    # Calculate accuracy based on threshold 0.5
    y_true = [1 if r['ground_truth'] else 0 for r in csv_rows]
    y_pred = [1 if p > 0.5 else 0 for p in probs]
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
    print(f"  Accuracy (threshold=0.5): {accuracy:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
