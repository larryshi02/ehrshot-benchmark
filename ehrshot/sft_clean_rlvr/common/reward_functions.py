"""
Verifiable Reward Functions for Clinical Outcome Prediction.

This module provides reward computation for GRPO training. The rewards are
"verifiable" because they can be automatically computed by comparing the
model's final answer against the ground truth label.

Reward scheme:
    +1.0 : Correct prediction (Final Answer matches ground truth)
    -1.0 : Incorrect prediction (Final Answer does not match ground truth)
     0.0 : Malformed output (cannot parse Final Answer)
"""

import re
from typing import List, Union


def parse_final_answer(response_text: str) -> int:
    """
    Parse the final answer from a model response.
    
    Looks for "Final Answer: Positive" or "Final Answer: Negative" patterns.
    This is the same parsing logic used in eval_vllm_reasoning.py for consistency.
    
    Args:
        response_text: The model's generated response
        
    Returns:
        1 if Positive, 0 if Negative, -1 if parsing failed
    """
    if not response_text:
        return -1
        
    text = response_text.strip().lower()
    
    # Primary patterns - look for explicit "Final Answer:" format
    if re.search(r"final answer:\s*\[?positive\]?", text):
        return 1
    if re.search(r"final answer:\s*\[?negative\]?", text):
        return 0
    
    # Fallback: check last few lines for clear positive/negative
    lines = text.strip().split('\n')
    for line in reversed(lines[-5:]):  # Check last 5 lines
        line = line.strip().lower()
        if "final answer" in line:
            if "positive" in line and "negative" not in line:
                return 1
            if "negative" in line and "positive" not in line:
                return 0
    
    # Could not parse
    return -1


def compute_reward(response: str, ground_truth: bool) -> float:
    """
    Compute reward for a single response.
    
    Args:
        response: Model's generated response text
        ground_truth: True if positive case, False if negative case
        
    Returns:
        +1.0 for correct, -1.0 for incorrect, 0.0 for unparseable
    """
    parsed = parse_final_answer(response)
    
    if parsed == -1:
        # Malformed output - neutral reward (don't reward or penalize)
        return 0.0
    
    # Convert ground truth to int for comparison
    expected = 1 if ground_truth else 0
    correct = (parsed == expected)
    
    return 1.0 if correct else -1.0


def reward_function_batch(
    completions: List[str],
    ground_truths: List[bool],
    **kwargs
) -> List[float]:
    """
    Batch reward computation for GRPO trainer.
    
    This function is called by TRL's GRPOTrainer during training.
    It receives a batch of completions and their corresponding ground truths,
    and returns rewards for each completion.
    
    Args:
        completions: List of model-generated responses
        ground_truths: List of ground truth labels (True/False)
        **kwargs: Additional arguments (ignored, for compatibility)
        
    Returns:
        List of rewards, one per completion
    """
    assert len(completions) == len(ground_truths), \
        f"Mismatch: {len(completions)} completions vs {len(ground_truths)} ground_truths"
    
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        reward = compute_reward(completion, gt)
        rewards.append(reward)
    
    return rewards


# =============================================================================
# TRL-Compatible Reward Function Factory
# =============================================================================

def create_reward_function(ground_truth_map: dict):
    """
    Create a reward function that can be used with TRL's GRPOTrainer.
    
    TRL expects a reward function with signature:
        reward_fn(completions: List[str], prompts: List[str], ...) -> List[float]
    
    We need to map prompts back to their ground truth labels.
    
    Args:
        ground_truth_map: Dict mapping prompt_id -> ground_truth (bool)
        
    Returns:
        Callable reward function compatible with TRL
    """
    def reward_fn(completions: List[str], prompts: List[str] = None, 
                  prompt_ids: List[str] = None, **kwargs) -> List[float]:
        """
        Reward function for TRL GRPOTrainer.
        
        Uses prompt_ids to look up ground truth labels.
        """
        if prompt_ids is None:
            raise ValueError("prompt_ids required for reward computation")
        
        rewards = []
        for completion, pid in zip(completions, prompt_ids):
            gt = ground_truth_map.get(pid)
            if gt is None:
                raise ValueError(f"Unknown prompt_id: {pid}")
            reward = compute_reward(completion, gt)
            rewards.append(reward)
        
        return rewards
    
    return reward_fn


# =============================================================================
# Testing / Validation
# =============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        # (response, ground_truth, expected_reward)
        ("<think>Patient has risk factors...</think>\n\nFinal Answer: Positive", True, 1.0),
        ("<think>Patient has risk factors...</think>\n\nFinal Answer: Positive", False, -1.0),
        ("<think>Low risk...</think>\n\nFinal Answer: Negative", False, 1.0),
        ("<think>Low risk...</think>\n\nFinal Answer: Negative", True, -1.0),
        ("Some rambling without final answer...", True, 0.0),
        ("Final Answer: [Positive]", True, 1.0),  # Bracket format
        ("FINAL ANSWER: NEGATIVE", False, 1.0),  # Case insensitive
        ("", True, 0.0),  # Empty response
    ]
    
    print("Testing reward functions:")
    print("-" * 60)
    
    all_passed = True
    for response, gt, expected in test_cases:
        reward = compute_reward(response, gt)
        passed = abs(reward - expected) < 0.01
        status = "✓" if passed else "✗"
        print(f"{status} Response: '{response[:50]}...' | GT: {gt} | Reward: {reward} (expected {expected})")
        if not passed:
            all_passed = False
    
    print("-" * 60)
    print(f"All tests passed: {all_passed}")
