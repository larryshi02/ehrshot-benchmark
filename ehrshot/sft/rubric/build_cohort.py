#!/usr/bin/env python3
"""
Build diverse cohorts for each EHRSHOT task using k-means + medoids selection.

For each task (acute MI, hyperlipidemia, hypertension, pancreatic cancer),
selects 20 positive and 20 negative samples using k-means clustering on
patient embeddings, then selecting medoids (points closest to cluster centroids).
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from loguru import logger

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import text encoder classes
from serialization.text_encoder import (
    GTEQwen2_7B_InstructEncoder, 
    GTEQwen2_1_5B_InstructEncoder,
    TextEncoder
)

# Task names and mappings
TASKS = ['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer']

# Map task names to instruction keys in task_to_instructions.json
TASK_TO_INSTRUCTION_KEY = {
    'acute_mi': 'new_acutemi',
    'hyperlipidemia': 'new_hyperlipidemia',
    'hypertension': 'new_hypertension',
    'pancreatic_cancer': 'new_pancan'
}


def load_task_data(task_name: str, data_dir: str, split: str = 'train') -> List[Dict]:
    """Load serialized data for a task, filtered by split."""
    json_file = os.path.join(data_dir, f"{task_name}_all_splits.json")
    
    if not os.path.exists(json_file):
        logger.error(f"Data file not found for task {task_name}: {json_file}")
        return []
    
    logger.info(f"Loading data from {json_file} (split: {split})")
    with open(json_file, 'r') as f:
        all_data = json.load(f)
    
    # Filter to only the specified split
    data = [ex for ex in all_data if ex.get('split') == split]
    
    logger.info(f"Loaded {len(data)} {split} examples for task {task_name} (out of {len(all_data)} total)")
    return data


def load_task_instructions(task_to_instructions_file: str) -> Dict[str, str]:
    """Load task instructions from JSON file."""
    if not os.path.exists(task_to_instructions_file):
        logger.warning(f"Task instructions file not found: {task_to_instructions_file}")
        return {}
    
    with open(task_to_instructions_file, 'r') as f:
        task_to_instructions = json.load(f)
    
    logger.info(f"Loaded task instructions from {task_to_instructions_file}")
    return task_to_instructions


def compute_embeddings(
    texts: List[str], 
    text_encoder: TextEncoder,
    instructions: Optional[List[str]] = None
) -> np.ndarray:
    """
    Compute embeddings for a list of texts using Qwen2LLMEncoder.
    
    Args:
        texts: List of text strings to embed
        text_encoder: TextEncoder instance with Qwen2LLMEncoder
        instructions: Optional list of instructions (one per text, or None for all)
    
    Returns:
        numpy array of embeddings with shape (len(texts), embedding_dim)
    """
    if instructions is None:
        # Use empty instructions if not provided
        instructions = [""] * len(texts)
    
    logger.info(f"Computing embeddings for {len(texts)} texts using Qwen2LLMEncoder")
    
    # Use TextEncoder.encode_texts which handles instruction formatting
    embeddings = text_encoder.encode_texts(instructions, texts, cache_dir=None)
    
    logger.info(f"Computed embeddings with shape {embeddings.shape}")
    return embeddings


def select_medoids(X: np.ndarray, kmeans: KMeans, example_indices: np.ndarray) -> List[int]:
    """
    Select medoids (points closest to cluster centroids) from k-means clusters.
    
    Args:
        X: Embedding matrix of shape (n_samples, n_features)
        kmeans: Fitted KMeans model
        example_indices: Array of example indices in the original data list
    
    Returns:
        List of selected example indices (medoids)
    """
    selected_indices = []
    
    for cluster_id in range(kmeans.n_clusters):
        # Get indices of points in this cluster
        cluster_mask = kmeans.labels_ == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Get cluster centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        
        # Get embeddings of points in this cluster
        cluster_embeddings = X[cluster_indices]
        
        # Compute distances from centroid to all points in cluster
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Select point closest to centroid (medoid)
        medoid_idx_in_cluster = np.argmin(distances)
        medoid_idx = cluster_indices[medoid_idx_in_cluster]
        
        selected_indices.append(example_indices[medoid_idx])
    
    return selected_indices


def select_diverse_samples(
    data: List[Dict],
    text_encoder: TextEncoder,
    task_name: str,
    task_to_instructions: Dict[str, str],
    n_pos: int = 20,
    n_neg: int = 20
) -> Tuple[List[Dict], List[Dict]]:
    """
    Select diverse samples using k-means + medoids.
    
    Args:
        data: List of data examples with 'label_value' and 'context' fields
        text_encoder: TextEncoder instance with Qwen2LLMEncoder
        task_name: Name of the task (for instruction lookup)
        task_to_instructions: Dictionary mapping task instruction keys to instruction strings
        n_pos: Number of positive samples to select
        n_neg: Number of negative samples to select
    
    Returns:
        Tuple of (selected_positive_samples, selected_negative_samples)
    """
    # Get instruction for this task
    instruction_key = TASK_TO_INSTRUCTION_KEY.get(task_name, '')
    instruction = task_to_instructions.get(instruction_key, "")
    
    # Build full instruction with prefix if available
    instruction_prefix = task_to_instructions.get("instruction_prefix", "")
    if instruction_prefix and instruction:
        full_instruction = f"{instruction_prefix} {instruction}"
    elif instruction:
        full_instruction = instruction
    else:
        full_instruction = ""
    
    logger.info(f"Using instruction for task {task_name}: {full_instruction[:100]}..." if full_instruction else "No instruction found")
    
    # Separate positive and negative examples
    positives = [ex for ex in data if ex.get('label_value', False) is True]
    negatives = [ex for ex in data if ex.get('label_value', False) is False]
    
    logger.info(f"Found {len(positives)} positive and {len(negatives)} negative examples")
    
    # Check if we have enough samples
    if len(positives) < n_pos:
        logger.warning(f"Only {len(positives)} positive examples available, requested {n_pos}")
        n_pos = len(positives)
    
    if len(negatives) < n_neg:
        logger.warning(f"Only {len(negatives)} negative examples available, requested {n_neg}")
        n_neg = len(negatives)
    
    selected_positives = []
    selected_negatives = []
    
    # Process positives
    if n_pos > 0 and len(positives) > 0:
        logger.info(f"Selecting {n_pos} diverse positive samples...")
        pos_texts = [ex.get('context', '') for ex in positives]
        pos_instructions = [full_instruction] * len(pos_texts)
        pos_embeddings = compute_embeddings(pos_texts, text_encoder, pos_instructions)
        pos_example_indices = np.array(range(len(positives)))
        
        # Run k-means on positives
        k_pos = min(n_pos, len(positives))
        kmeans_pos = KMeans(n_clusters=k_pos, random_state=42, n_init=10)
        kmeans_pos.fit(pos_embeddings)
        
        # Select medoids
        selected_pos_indices = select_medoids(pos_embeddings, kmeans_pos, pos_example_indices)
        
        # Get selected examples by index
        selected_positives = [positives[i] for i in selected_pos_indices]
        
        logger.info(f"Selected {len(selected_positives)} positive samples")
    
    # Process negatives
    if n_neg > 0 and len(negatives) > 0:
        logger.info(f"Selecting {n_neg} diverse negative samples...")
        neg_texts = [ex.get('context', '') for ex in negatives]
        neg_instructions = [full_instruction] * len(neg_texts)
        neg_embeddings = compute_embeddings(neg_texts, text_encoder, neg_instructions)
        neg_example_indices = np.array(range(len(negatives)))
        
        # Run k-means on negatives
        k_neg = min(n_neg, len(negatives))
        kmeans_neg = KMeans(n_clusters=k_neg, random_state=42, n_init=10)
        kmeans_neg.fit(neg_embeddings)
        
        # Select medoids
        selected_neg_indices = select_medoids(neg_embeddings, kmeans_neg, neg_example_indices)
        
        # Get selected examples by index
        selected_negatives = [negatives[i] for i in selected_neg_indices]
        
        logger.info(f"Selected {len(selected_negatives)} negative samples")
    
    return selected_positives, selected_negatives


def process_task(
    task_name: str,
    data_dir: str,
    output_dir: str,
    text_encoder: TextEncoder,
    task_to_instructions: Dict[str, str],
    n_pos: int = 20,
    n_neg: int = 20
) -> None:
    """Process a single task to select diverse samples."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing task: {task_name}")
    logger.info(f"{'='*60}")
    
    # Load data
    data = load_task_data(task_name, data_dir)
    
    if not data:
        logger.warning(f"No data found for task {task_name}, skipping")
        return
    
    # Select diverse samples
    selected_pos, selected_neg = select_diverse_samples(
        data, text_encoder, task_name, task_to_instructions, n_pos=n_pos, n_neg=n_neg
    )
    
    # Combine selected samples
    selected_samples = selected_pos + selected_neg
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{task_name}_cohort.json")
    
    with open(output_file, 'w') as f:
        json.dump(selected_samples, f, indent=2)
    
    logger.info(f"\nTask {task_name} Summary:")
    logger.info(f"  Selected {len(selected_pos)} positive samples")
    logger.info(f"  Selected {len(selected_neg)} negative samples")
    logger.info(f"  Total: {len(selected_samples)} samples")
    logger.info(f"  Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Build diverse cohorts for EHRSHOT tasks using k-means + medoids"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="ehrshot/sft/serialized_multi_task_data",
        help="Directory containing serialized task data JSON files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ehrshot/sft/rubric/cohorts",
        help="Output directory for selected cohorts"
    )
    
    parser.add_argument(
        "--n_pos",
        type=int,
        default=20,
        help="Number of positive samples to select per task"
    )
    
    parser.add_argument(
        "--n_neg",
        type=int,
        default=20,
        help="Number of negative samples to select per task"
    )
    
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=8192,
        help="Maximum input length for embeddings (default: 8192)"
    )
    
    parser.add_argument(
        "--task_to_instructions",
        type=str,
        default="ehrshot/serialization/task_to_instructions.json",
        help="Path to task instructions JSON file"
    )
    
    parser.add_argument(
        "--use_fallback_encoder",
        action="store_true",
        help="Use GTEQwen2-1.5B instead of 7B (smaller model, may avoid flash_attn issues)"
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=TASKS,
        help="List of tasks to process (default: all tasks)"
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(project_root, args.data_dir)
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    if not os.path.isabs(args.task_to_instructions):
        args.task_to_instructions = os.path.join(project_root, args.task_to_instructions)
    
    # Load task instructions
    task_to_instructions = load_task_instructions(args.task_to_instructions)
    
    # Initialize text encoder with Qwen2LLMEncoder
    encoder_class = GTEQwen2_1_5B_InstructEncoder if args.use_fallback_encoder else GTEQwen2_7B_InstructEncoder
    encoder_name = "GTEQwen2-1.5B-Instruct" if args.use_fallback_encoder else "GTEQwen2-7B-Instruct"
    
    logger.info(f"Initializing {encoder_name} with max_input_length={args.max_input_length}")
    
    try:
        qwen_encoder = encoder_class(max_input_length=args.max_input_length)
        text_encoder = TextEncoder(qwen_encoder)
        logger.info("Text encoder initialized successfully")
    except (ImportError, OSError) as e:
        error_str = str(e).lower()
        if "flash_attn" in error_str or "cuda_home" in error_str or "nvcc" in error_str:
            logger.error("="*60)
            logger.error("FLASH_ATTN / CUDA ERROR DETECTED")
            logger.error("="*60)
            logger.error(f"The {encoder_name} model requires flash_attn, but there's a compatibility issue.")
            logger.error("")
            logger.error("OPTION 1: Fix flash_attn installation (recommended if you have CUDA toolkit):")
            logger.error("")
            logger.error("  a) Find your CUDA installation:")
            logger.error("     find /usr/local -name 'nvcc' 2>/dev/null")
            logger.error("     # or check: ls /usr/local/cuda*/bin/nvcc")
            logger.error("")
            logger.error("  b) Set CUDA_HOME and install:")
            logger.error("     export CUDA_HOME=/usr/local/cuda-12.1  # Adjust path as needed")
            logger.error("     export PATH=$CUDA_HOME/bin:$PATH")
            logger.error("     pip uninstall flash-attn")
            logger.error("     MAX_JOBS=4 pip install flash-attn --no-build-isolation")
            logger.error("")
            logger.error("OPTION 2: Use fallback encoder (smaller model, may work without flash_attn):")
            logger.error("")
            logger.error("     ./run_build_cohort.sh --use_fallback_encoder")
            logger.error("     # or")
            logger.error("     python build_cohort.py --use_fallback_encoder ...")
            logger.error("")
            logger.error("OPTION 3: Use pre-built wheel (if available for your system):")
            logger.error("")
            logger.error("     pip install flash-attn --index-url https://download.pytorch.org/whl/cu121")
            logger.error("")
            logger.error("Original error:")
            logger.error(str(e))
            logger.error("="*60)
            sys.exit(1)
        else:
            # Re-raise if it's a different ImportError/OSError
            raise
    except Exception as e:
        logger.error(f"Failed to initialize text encoder: {e}")
        logger.error("This might be due to:")
        logger.error("  - Missing dependencies (flash_attn, transformers, torch)")
        logger.error("  - CUDA/GPU compatibility issues")
        logger.error("  - Model download issues")
        logger.error("")
        logger.error("Try using --use_fallback_encoder as an alternative")
        raise
    
    # Process each task
    for task_name in args.tasks:
        try:
            process_task(
                task_name=task_name,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                text_encoder=text_encoder,
                task_to_instructions=task_to_instructions,
                n_pos=args.n_pos,
                n_neg=args.n_neg
            )
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*60)
    logger.info("All tasks processed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

