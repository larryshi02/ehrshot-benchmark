#!/usr/bin/env python3
"""
Evaluate Clinical Outcome Prediction via Normalized Token Probabilities

Computes P(Positive) for binary classification by extracting log-probabilities
of "Positive" and "Negative" tokens from the model's first output token.

Method:
    P(Positive) = exp(logprob_pos) / (exp(logprob_pos) + exp(logprob_neg))
               = sigmoid(logprob_pos - logprob_neg)

Supports:
    - Base model evaluation (Qwen3-8B)
    - Finetuned model evaluation (with LoRA adapter)

Usage:
    # Finetuned model
    python eval_vllm_direct.py \\
        --model_base Qwen/Qwen3-8B \\
        --lora_path finetuned_models/originalEHR_train_all/acute_mi \\
        --data_path /dev/shm/ehrshot-data/serialized_multi_task_data/acute_mi_all_splits.json \\
        --task_name acute_mi \\
        --output_dir eval_results/originalEHR_train_all_finetuned/acute_mi

    # Base model (no LoRA)
    python eval_vllm_direct.py \\
        --model_base Qwen/Qwen3-8B \\
        --data_path /dev/shm/ehrshot-data/serialized_multi_task_data/acute_mi_all_splits.json \\
        --task_name acute_mi \\
        --output_dir eval_results/originalEHR_base/acute_mi

Output Files (saved to output_dir):
    predictions.csv     - Columns: patient_id, label_time, target_task, ground_truth, 
                                   probability_score, has_valid_samples
    detailed_logs.json  - Per-example logprobs and metadata
    run_stats.json      - Model config and summary statistics
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Dict, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

TASK_QUERIES = {
    'acute_mi': 'Will the patient develop an acute myocardial infarction in the next year?',
    'hyperlipidemia': 'Will the patient develop hyperlipidemia in the next year?',
    'hypertension': 'Will the patient develop hypertension in the next year?',
    'pancreatic_cancer': 'Will the patient develop pancreatic cancer in the next year?'
}

# Default logprob fallback when token not in top-k
DEFAULT_MISSING_LOGPROB = -100.0


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def construct_direct_yn_messages(context: str, task_name: str) -> List[Dict]:
    """
    Construct chat messages for direct Y/N prediction.
    
    Format matches training data: minimal system prompt + sandwich pattern.
    """
    task_query = TASK_QUERIES[task_name]
    
    system_content = "You are a medical expert specializing in clinical risk prediction."
    
    user_content = (
        f"Based on the patient's Electronic Healthcare Record below, predict: {task_query}\n\n"
        f"--- Patient EHR ---\n\n"
        f"{context}\n\n"
        f"--- End of EHR ---\n\n"
        f"Respond with exactly one word: Positive or Negative.\n"
        f"- Positive: The patient WILL develop the condition.\n"
        f"- Negative: The patient will NOT develop the condition."
    )
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


# =============================================================================
# TOKENIZATION HELPERS
# =============================================================================

def get_token_id(tokenizer, token: str) -> Tuple[int, str]:
    """
    Get token ID for a classification token.
    
    Tries variants (with/without leading space) to find single-token encoding.
    
    Returns:
        Tuple of (token_id, variant_used)
    """
    variants = [token, f" {token}", token.lower(), f" {token.lower()}"]
    
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], variant
    
    # Fallback: use first token of multi-token encoding
    ids = tokenizer.encode(token, add_special_tokens=False)
    return ids[0], f"{token} (multi-token, using first)"


def get_classification_token_ids(tokenizer) -> Tuple[int, int]:
    """
    Get token IDs for Positive and Negative classification tokens.
    
    Returns:
        Tuple of (positive_id, negative_id)
    """
    positive_id, pos_variant = get_token_id(tokenizer, "Positive")
    negative_id, neg_variant = get_token_id(tokenizer, "Negative")
    
    print(f"  Token IDs: Positive={positive_id} ({pos_variant}), Negative={negative_id} ({neg_variant})")
    
    return positive_id, negative_id


# =============================================================================
# PROBABILITY COMPUTATION
# =============================================================================

def compute_prob_positive(
    logprob_positive: float, 
    logprob_negative: float
) -> float:
    """
    Compute normalized probability of Positive over {Positive, Negative}.
    
    P(Positive) = exp(lp_pos) / (exp(lp_pos) + exp(lp_neg))
                = 1 / (1 + exp(lp_neg - lp_pos))  [numerically stable sigmoid]
    """
    log_diff = logprob_negative - logprob_positive
    return 1.0 / (1.0 + np.exp(log_diff))


def extract_logprobs_from_output(
    output_logprobs: List[Dict],
    positive_id: int,
    negative_id: int
) -> Tuple[float, float, int, int]:
    """
    Extract logprobs for Positive and Negative from vLLM output.
    
    Returns:
        Tuple of (logprob_positive, logprob_negative, pos_found, neg_found)
        where pos_found/neg_found are 1 if found in top-k, 0 otherwise
    """
    if not output_logprobs or len(output_logprobs) == 0:
        return DEFAULT_MISSING_LOGPROB, DEFAULT_MISSING_LOGPROB, 0, 0
    
    first_token_logprobs = output_logprobs[0]
    
    logprob_positive = DEFAULT_MISSING_LOGPROB
    logprob_negative = DEFAULT_MISSING_LOGPROB
    pos_found = 0
    neg_found = 0
    
    for token_id, logprob_obj in first_token_logprobs.items():
        if token_id == positive_id:
            logprob_positive = logprob_obj.logprob
            pos_found = 1
        elif token_id == negative_id:
            logprob_negative = logprob_obj.logprob
            neg_found = 1
    
    return logprob_positive, logprob_negative, pos_found, neg_found


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data(data_path: str) -> List[Dict]:
    """Load and filter test data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for test split if available
    if any(d.get("split") == "test" for d in data):
        test_data = [d for d in data if d.get("split") == "test"]
        print(f"  Found {len(test_data)} test examples (filtered by split='test')")
    else:
        test_data = data
        print(f"  Using all {len(test_data)} examples (no split field found)")
    
    return test_data


def extract_context(item: Dict) -> str:
    """Extract EHR context from a data item."""
    if 'context' in item:
        return item['context'].strip()
    elif 'conversations' in item:
        for conv in item['conversations']:
            if conv['role'] == 'user':
                return conv['content'].strip()
    raise ValueError(f"Cannot find context in item: {item.keys()}")


def parse_ground_truth(value) -> int:
    """Convert ground truth to int (0 or 1), handling various formats."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        return 1 if value.lower() in ('true', '1', 'yes', 'positive') else 0
    return 0


# =============================================================================
# MAIN FUNCTION
# =============================================================================

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
    test_data = load_test_data(args.data_path)

    # -------------------------------------------------------------------------
    # 2. Initialize vLLM
    # -------------------------------------------------------------------------
    print("\n[2/4] Initializing vLLM...")
    
    llm = LLM(
        model=args.model_base,
        enable_lora=use_lora,
        max_model_len=12288,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
        max_logprobs=50,
    )
    
    tokenizer = llm.get_tokenizer()
    positive_id, negative_id = get_classification_token_ids(tokenizer)

    # -------------------------------------------------------------------------
    # 3. Prepare Prompts and Run Inference
    # -------------------------------------------------------------------------
    print("\n[3/4] Preparing prompts and running inference...")
    
    prompts = []
    metadata = []
    
    for item in test_data:
        context = extract_context(item)
        messages = construct_direct_yn_messages(context, args.task_name)
        
        # Apply chat template with thinking disabled (Qwen3 specific)
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
        ground_truth_raw = item.get('label_value', item.get('ground_truth'))
        
        metadata.append({
            "patient_id": item.get('patient_id'),
            "label_time": item.get('label_time'),
            "ground_truth_raw": ground_truth_raw,
            "ground_truth": parse_ground_truth(ground_truth_raw),
        })

    # Sampling params: temperature=1 for unscaled probs, logprobs=100 for top-k
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        max_tokens=1,
        logprobs=50,
    )

    # LoRA request if applicable
    lora_request = LoRARequest("adapter", 1, args.lora_path) if use_lora else None

    # Run inference
    print(f"  Running inference on {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # -------------------------------------------------------------------------
    # 4. Process Outputs and Compute Probabilities
    # -------------------------------------------------------------------------
    print("\n[4/4] Computing normalized probabilities...")
    
    detailed_logs = []
    csv_rows = []
    missing_pos_count = 0
    missing_neg_count = 0
    
    for i, output in enumerate(outputs):
        meta = metadata[i]
        output_logprobs = output.outputs[0].logprobs
        
        # Extract logprobs
        logprob_pos, logprob_neg, pos_found, neg_found = extract_logprobs_from_output(
            output_logprobs, positive_id, negative_id
        )
        
        missing_pos_count += (1 - pos_found)
        missing_neg_count += (1 - neg_found)
        
        # Compute normalized probability
        prob_positive = compute_prob_positive(logprob_pos, logprob_neg)
        
        # Get generated text for logging
        generated_text = output.outputs[0].text.strip()
        
        # Store detailed log
        detailed_logs.append({
            "patient_id": meta['patient_id'],
            "label_time": meta['label_time'],
            "ground_truth": meta['ground_truth'],
            "probability_positive": prob_positive,
            "logprob_positive": logprob_pos,
            "logprob_negative": logprob_neg,
            "pos_in_topk": pos_found,
            "neg_in_topk": neg_found,
            "generated_token": generated_text,
            "prompt_length": len(prompts[i]),
        })
        
        # Store CSV row (compatible with eval_vllm_compute_metrics.py)
        csv_rows.append({
            "patient_id": meta['patient_id'],
            "label_time": meta['label_time'],
            "target_task": args.task_name,
            "ground_truth": meta['ground_truth'],
            "probability_score": prob_positive,
            "has_valid_samples": True
        })

    # Print warnings summary (not per-example)
    if missing_pos_count > 0:
        print(f"  Warning: 'Positive' not in top-100 logprobs for {missing_pos_count}/{len(outputs)} examples")
    if missing_neg_count > 0:
        print(f"  Warning: 'Negative' not in top-100 logprobs for {missing_neg_count}/{len(outputs)} examples")

    # -------------------------------------------------------------------------
    # 5. Save Results
    # -------------------------------------------------------------------------
    print("\nSaving results...")
    
    # Save detailed logs
    detailed_path = os.path.join(args.output_dir, "detailed_logs.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, indent=2)
    print(f"  Saved detailed logs to: {detailed_path}")
    
    # Save predictions CSV
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
        "missing_positive_count": missing_pos_count,
        "missing_negative_count": missing_neg_count,
    }
    stats_path = os.path.join(args.output_dir, "run_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(run_stats, f, indent=2)
    print(f"  Saved run stats to: {stats_path}")
    
    # Print summary statistics
    probs = [r['probability_score'] for r in csv_rows]
    y_true = [r['ground_truth'] for r in csv_rows]
    y_pred = [1 if p > 0.5 else 0 for p in probs]
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"  Mean P(Positive): {np.mean(probs):.4f}")
    print(f"  Std P(Positive): {np.std(probs):.4f}")
    print(f"  Min P(Positive): {np.min(probs):.4f}")
    print(f"  Max P(Positive): {np.max(probs):.4f}")
    print(f"  Accuracy (threshold=0.5): {accuracy:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
