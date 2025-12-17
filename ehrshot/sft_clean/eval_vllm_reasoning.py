#!/usr/bin/env python3
"""
Evaluate Clinical Outcome Prediction via Sampling-Based Reasoning

Generates multiple reasoning responses per test example and computes probability
scores based on the frequency of Positive/Negative final answers.

Method:
    - Sample N=10 responses per test example
    - Parse "Final Answer: Positive/Negative" from each response
    - P(Positive) = count(Positive) / count(valid_responses)

Supports:
    - Base model evaluation (Qwen3-8B)
    - Finetuned model evaluation (with LoRA adapter)

Usage:
    # Finetuned model
    python eval_vllm_reasoning.py \\
        --model_base Qwen/Qwen3-8B \\
        --lora_path finetuned_models/originalEHR_train_orig/acute_mi \\
        --data_path /dev/shm/ehrshot-data/serialized_multi_task_data/acute_mi_all_splits.json \\
        --task_name acute_mi \\
        --output_dir eval_results/originalEHR_train_orig_finetuned/acute_mi

    # Base model (no LoRA)
    python eval_vllm_reasoning.py \\
        --model_base Qwen/Qwen3-8B \\
        --data_path /dev/shm/ehrshot-data/serialized_multi_task_data/acute_mi_all_splits.json \\
        --task_name acute_mi \\
        --output_dir eval_results/originalEHR_base/acute_mi

Output Files (saved to output_dir):
    predictions.csv     - Columns: patient_id, label_time, target_task, ground_truth, 
                                   probability_score, has_valid_samples
    detailed_logs.json  - Per-example responses, parsed answers, and metadata
    run_stats.json      - Model config and summary statistics
    summary.json        - Computed metrics (AUROC, AUPRC) after metrics computation
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Dict, Tuple, Optional
import re


# =============================================================================
# CONFIGURATION
# =============================================================================

TASK_QUERIES = {
    'acute_mi': 'Will the patient develop an acute myocardial infarction in the next year?',
    'hyperlipidemia': 'Will the patient develop hyperlipidemia in the next year?',
    'hypertension': 'Will the patient develop hypertension in the next year?',
    'pancreatic_cancer': 'Will the patient develop pancreatic cancer in the next year?'
}

# Sampling configuration
NUM_SAMPLES = 10
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 4096


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def construct_reasoning_messages(context: str, task_name: str) -> List[Dict]:
    """
    Construct chat messages for reasoning-based prediction.
    
    Format matches training data: system prompt + sandwich pattern with reasoning request.
    """
    task_query = TASK_QUERIES[task_name]
    
    system_content = "You are a medical expert specializing in clinical risk prediction."
    
    user_content = (
        f"Based on the patient's Electronic Healthcare Record below, predict: {task_query}\n\n"
        f"Analyze the patient's condition and risk factors step by step.\n\n"
        f"--- Patient EHR ---\n\n"
        f"{context}\n\n"
        f"--- End of EHR ---\n\n"
        f"Provide your response in the following format:\n"
        f"<think>\n"
        f"[Your clinical reasoning]\n"
        f"</think>\n\n"
        f"Final Answer: [Positive/Negative]\n\n"
        f"IMPORTANT: YOU HAVE TO FINISH YOUR ANSWER BY SAYING EITHER \"Final Answer: Positive\" OR \"Final Answer: Negative\". "
        f"AND DO NOT SAY ANYTHING ELSE AFTER THAT."
    )
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_final_answer(response_text: str) -> int:
    """
    Parse the final answer from a reasoning response.
    
    Looks for "Final Answer: Positive" or "Final Answer: Negative" patterns.
    
    Returns:
        1 if Positive, 0 if Negative, -1 if parsing failed
    """
    text = response_text.strip().lower()
    
    # Try multiple patterns
    patterns = [
        r"final answer:\s*\[?positive\]?",
        r"final answer:\s*\[?negative\]?",
        r"final answer:\s*positive",
        r"final answer:\s*negative",
    ]
    
    # Check for Positive
    if re.search(r"final answer:\s*\[?positive\]?", text):
        return 1
    
    # Check for Negative
    if re.search(r"final answer:\s*\[?negative\]?", text):
        return 0
    
    # Fallback: check last line
    lines = text.strip().split('\n')
    last_line = lines[-1].strip().lower() if lines else ""
    
    if "positive" in last_line and "negative" not in last_line:
        return 1
    if "negative" in last_line and "positive" not in last_line:
        return 0
    
    return -1


def compute_probability_from_samples(parsed_answers: List[int]) -> Tuple[float, int, int]:
    """
    Compute probability of Positive from parsed sample answers.
    
    Args:
        parsed_answers: List of parsed answers (1=Positive, 0=Negative, -1=Invalid)
    
    Returns:
        Tuple of (probability, num_valid, num_invalid)
    """
    valid_answers = [a for a in parsed_answers if a != -1]
    num_valid = len(valid_answers)
    num_invalid = len(parsed_answers) - num_valid
    
    if num_valid == 0:
        # No valid answers - return 0.5 as neutral probability
        return 0.5, num_valid, num_invalid
    
    probability = sum(valid_answers) / num_valid
    return probability, num_valid, num_invalid


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
    parser = argparse.ArgumentParser(description="Reasoning-Based Evaluation with vLLM")
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
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES,
                        help=f"Number of samples per example (default: {NUM_SAMPLES})")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine if we're using LoRA
    use_lora = args.lora_path is not None and args.lora_path.lower() != "none"
    
    print("=" * 60)
    print("Reasoning-Based Evaluation")
    print("=" * 60)
    print(f"Base Model: {args.model_base}")
    print(f"LoRA Path: {args.lora_path if use_lora else 'None (Base Model)'}")
    print(f"Task: {args.task_name}")
    print(f"Num Samples: {args.num_samples}")
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
    )
    
    tokenizer = llm.get_tokenizer()

    # -------------------------------------------------------------------------
    # 3. Prepare Prompts and Run Inference
    # -------------------------------------------------------------------------
    print("\n[3/4] Preparing prompts and running inference...")
    
    prompts = []
    metadata = []
    
    for item in test_data:
        context = extract_context(item)
        messages = construct_reasoning_messages(context, args.task_name)
        
        # Apply chat template
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

    # Sampling parameters for reasoning
    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    # LoRA request if applicable
    lora_request = LoRARequest("adapter", 1, args.lora_path) if use_lora else None

    # Run inference
    print(f"  Running inference on {len(prompts)} examples ({args.num_samples} samples each)...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # -------------------------------------------------------------------------
    # 4. Process Outputs and Compute Probabilities
    # -------------------------------------------------------------------------
    print("\n[4/4] Parsing responses and computing probabilities...")
    
    detailed_logs = []
    csv_rows = []
    
    global_total_samples = 0
    global_valid_samples = 0
    global_invalid_samples = 0
    
    for i, output in enumerate(outputs):
        meta = metadata[i]
        
        # Get all sampled responses
        responses = [o.text for o in output.outputs]
        parsed_answers = [parse_final_answer(r) for r in responses]
        
        # Compute probability
        prob, num_valid, num_invalid = compute_probability_from_samples(parsed_answers)
        
        # Update global stats
        global_total_samples += len(responses)
        global_valid_samples += num_valid
        global_invalid_samples += num_invalid
        
        # Store detailed log
        detailed_logs.append({
            "patient_id": meta['patient_id'],
            "label_time": meta['label_time'],
            "ground_truth": meta['ground_truth'],
            "probability_positive": prob,
            "num_samples": len(responses),
            "num_valid": num_valid,
            "num_invalid": num_invalid,
            "parsed_answers": parsed_answers,
            "model_responses": responses,
        })
        
        # Store CSV row (compatible with eval_compute_metrics.py)
        csv_rows.append({
            "patient_id": meta['patient_id'],
            "label_time": meta['label_time'],
            "target_task": args.task_name,
            "ground_truth": meta['ground_truth'],
            "probability_score": prob,
            "has_valid_samples": num_valid > 0
        })

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
    invalid_pct = global_invalid_samples / global_total_samples if global_total_samples > 0 else 0
    run_stats = {
        "model_base": args.model_base,
        "lora_path": args.lora_path if use_lora else None,
        "task_name": args.task_name,
        "num_examples": len(test_data),
        "num_samples_per_example": args.num_samples,
        "method": "sampling_based",
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "total_samples": global_total_samples,
        "valid_samples": global_valid_samples,
        "invalid_samples": global_invalid_samples,
        "invalid_percentage": invalid_pct,
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
    print(f"  Total Examples: {len(test_data)}")
    print(f"  Total Samples: {global_total_samples}")
    print(f"  Valid Samples: {global_valid_samples} ({100*(1-invalid_pct):.1f}%)")
    print(f"  Invalid Samples: {global_invalid_samples} ({100*invalid_pct:.1f}%)")
    print(f"  Mean P(Positive): {np.mean(probs):.4f}")
    print(f"  Std P(Positive): {np.std(probs):.4f}")
    print(f"  Accuracy (threshold=0.5): {accuracy:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
