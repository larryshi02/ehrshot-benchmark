"""
Common utilities for RLVR (Reinforcement Learning with Verifiable Rewards).

This module provides shared functionality for both Phase 1 (single GPU validation)
and Phase 2 (multi-GPU aggregated training).
"""

from .constants import TASKS, TASK_QUERIES, DATA_PATHS, MODEL_CONFIG, GRPO_CONFIG
from .reward_functions import parse_final_answer, compute_reward, reward_function_batch
from .prompt_formatting import (
    extract_prompt_from_conversation,
    load_sft_data_as_grpo_format,
    load_aggregated_grpo_data,
    create_ground_truth_map,
)

__all__ = [
    # Constants
    "TASKS",
    "TASK_QUERIES", 
    "DATA_PATHS",
    "MODEL_CONFIG",
    "GRPO_CONFIG",
    # Reward functions
    "parse_final_answer",
    "compute_reward",
    "reward_function_batch",
    # Data loading
    "extract_prompt_from_conversation",
    "load_sft_data_as_grpo_format",
    "load_aggregated_grpo_data",
    "create_ground_truth_map",
]

