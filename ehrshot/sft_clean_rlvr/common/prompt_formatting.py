"""
Prompt Formatting Utilities for RLVR.

This module handles:
1. Loading SFT data and extracting prompts (dropping assistant responses)
2. Creating balanced datasets for RLVR
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .constants import TASKS, DATA_PATHS


def extract_prompt_from_conversation(conversation: List[Dict]) -> List[Dict]:
    """
    Extract system and user messages from a conversation, dropping assistant response.
    
    The SFT data already has properly formatted prompts - we just need to
    remove the assistant turn (which contains GPT-5's reasoning trace).
    
    Args:
        conversation: List of {"role": str, "content": str} dicts
        
    Returns:
        List of message dicts (system + user only) for GRPO prompt
    """
    prompt_messages = []
    
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        
        if role in ["system", "user"]:
            prompt_messages.append({"role": role, "content": content})
        # Skip assistant - we want the model to generate its own response
    
    return prompt_messages


def load_sft_data_as_grpo_format(
    task_name: str,
    split: str = "train_orig",
    balance_classes: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Load SFT training data and convert to GRPO format.
    
    GRPO format:
    {
        "prompt_id": str,          # Unique identifier
        "prompt": List[Dict],      # Chat messages (system + user only)
        "ground_truth": bool,      # Label for reward computation
        "task": str,               # Task name
    }
    
    The prompts in SFT data are already properly formatted - we just
    extract system + user messages and drop the assistant response.
    
    Args:
        task_name: One of TASKS
        split: "train_orig" or "train_all"
        balance_classes: If True, sample negatives to match positives
        seed: Random seed for balanced sampling
        
    Returns:
        List of GRPO-formatted examples
    """
    sft_path = DATA_PATHS.get_sft_train_path(task_name, split)
    
    with open(sft_path, 'r') as f:
        sft_data = json.load(f)
    
    # Separate positives and negatives
    positives = []
    negatives = []
    
    for i, example in enumerate(sft_data):
        conv = example["conversations"]
        
        # Extract prompt (system + user only, drop assistant)
        prompt_messages = extract_prompt_from_conversation(conv)
        
        grpo_example = {
            "prompt_id": f"{task_name}_{example.get('patient_id', i)}_{example.get('label_time', i)}",
            "prompt": prompt_messages,
            "ground_truth": example["label_value"],
            "task": task_name,
            "patient_id": example.get("patient_id"),
            "label_time": example.get("label_time"),
        }
        
        if example["label_value"]:
            positives.append(grpo_example)
        else:
            negatives.append(grpo_example)
    
    # Balance classes if requested
    if balance_classes:
        random.seed(seed)
        n_positives = len(positives)
        if len(negatives) > n_positives:
            negatives = random.sample(negatives, n_positives)
        elif len(positives) > len(negatives):
            positives = random.sample(positives, len(negatives))
    
    # Combine and shuffle
    all_examples = positives + negatives
    random.seed(seed)
    random.shuffle(all_examples)
    
    return all_examples


def load_aggregated_grpo_data(
    tasks: List[str] = None,
    split: str = "train_orig",
    balance_classes: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Load and combine GRPO data from multiple tasks.
    
    Args:
        tasks: List of task names (default: all tasks)
        split: "train_orig" or "train_all"
        balance_classes: If True, balance within each task
        seed: Random seed
        
    Returns:
        Combined list of GRPO-formatted examples from all tasks
    """
    if tasks is None:
        tasks = TASKS
    
    all_examples = []
    
    for task in tasks:
        task_examples = load_sft_data_as_grpo_format(
            task_name=task,
            split=split,
            balance_classes=balance_classes,
            seed=seed
        )
        all_examples.extend(task_examples)
        print(f"  Loaded {len(task_examples)} examples from {task}")
    
    # Shuffle combined data
    random.seed(seed)
    random.shuffle(all_examples)
    
    return all_examples


def create_ground_truth_map(examples: List[Dict]) -> Dict[str, bool]:
    """
    Create a mapping from prompt_id to ground_truth for reward computation.
    
    Args:
        examples: List of GRPO-formatted examples
        
    Returns:
        Dict mapping prompt_id -> ground_truth
    """
    return {ex["prompt_id"]: ex["ground_truth"] for ex in examples}


# =============================================================================
# Testing / Validation
# =============================================================================

if __name__ == "__main__":
    print("Testing prompt formatting utilities...")
    print("-" * 60)
    
    # Test loading single task
    print("\n1. Loading single task (acute_mi, balanced):")
    examples = load_sft_data_as_grpo_format("acute_mi", balance_classes=True)
    print(f"   Total examples: {len(examples)}")
    positives = sum(1 for ex in examples if ex["ground_truth"])
    negatives = len(examples) - positives
    print(f"   Positives: {positives}, Negatives: {negatives}")
    
    # Show sample
    print("\n   Sample example:")
    sample = examples[0]
    print(f"   - prompt_id: {sample['prompt_id']}")
    print(f"   - ground_truth: {sample['ground_truth']}")
    print(f"   - task: {sample['task']}")
    print(f"   - num messages in prompt: {len(sample['prompt'])}")
    for msg in sample['prompt']:
        print(f"     - {msg['role']}: {msg['content'][:80]}...")
    
    # Test aggregated loading
    print("\n2. Loading aggregated data (all tasks, balanced):")
    all_examples = load_aggregated_grpo_data(balance_classes=True)
    print(f"   Total examples: {len(all_examples)}")
    
    # Count per task
    task_counts = {}
    for ex in all_examples:
        task = ex["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
    print("   Per-task counts:")
    for task, count in sorted(task_counts.items()):
        print(f"     {task}: {count}")
    
    print("\n" + "-" * 60)
    print("Done!")
