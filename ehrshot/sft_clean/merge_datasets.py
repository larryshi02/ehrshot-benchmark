#!/usr/bin/env python3
"""
Merge and Shuffle Task Datasets for Aggregated Multi-Task Training

Combines individual task SFT datasets into a single shuffled dataset for
training a unified model across all clinical prediction tasks.

Usage:
    python merge_datasets.py \\
        --data_dir /dev/shm/ehrshot-data/data_gpt-5-mini_sft_reasoning_originalEHR \\
        --split train_orig \\
        --output_file /dev/shm/ehrshot-data/data_gpt-5-mini_sft_reasoning_originalEHR/train_orig/merged_sft_dataset.json

Features:
    - Concatenates all 4 task datasets
    - Randomly shuffles the merged dataset
    - Adds task_name field to each example for tracking
    - Reports statistics on dataset composition
"""

import os
import json
import random
import argparse
from typing import List, Dict


TASKS = ["acute_mi", "hyperlipidemia", "hypertension", "pancreatic_cancer"]


def load_task_dataset(data_dir: str, split: str, task: str) -> List[Dict]:
    """Load a single task's SFT dataset."""
    filepath = os.path.join(data_dir, split, f"{task}_sft_dataset.json")
    
    if not os.path.exists(filepath):
        print(f"  ⚠️  Warning: {filepath} not found, skipping...")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add task_name to each example for tracking
    for item in data:
        item['task_name'] = task
    
    return data


def merge_and_shuffle(datasets: Dict[str, List[Dict]], seed: int = 42) -> List[Dict]:
    """Merge all task datasets and shuffle."""
    merged = []
    
    for task, data in datasets.items():
        merged.extend(data)
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(merged)
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge task datasets for multi-task training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Base directory containing task datasets (e.g., data_gpt-5-mini_sft_reasoning_originalEHR)")
    parser.add_argument("--split", type=str, required=True, choices=["train_orig", "train_all", "val_small"],
                        help="Data split to merge")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path (default: {data_dir}/{split}/merged_sft_dataset.json)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    args = parser.parse_args()
    
    # Default output path
    if args.output_file is None:
        args.output_file = os.path.join(args.data_dir, args.split, "merged_sft_dataset.json")
    
    print("=" * 60)
    print("Merge Task Datasets")
    print("=" * 60)
    print(f"Data Directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Output File: {args.output_file}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Load all task datasets
    print("\nLoading task datasets...")
    datasets = {}
    total_examples = 0
    
    for task in TASKS:
        data = load_task_dataset(args.data_dir, args.split, task)
        datasets[task] = data
        print(f"  {task}: {len(data)} examples")
        total_examples += len(data)
    
    print(f"\nTotal examples before merge: {total_examples}")
    
    # Merge and shuffle
    print("\nMerging and shuffling...")
    merged = merge_and_shuffle(datasets, seed=args.seed)
    
    print(f"Total examples after merge: {len(merged)}")
    
    # Verify shuffle quality by checking task distribution in first 100 examples
    first_100_tasks = [ex['task_name'] for ex in merged[:100]]
    task_counts = {task: first_100_tasks.count(task) for task in TASKS}
    print(f"\nTask distribution in first 100 examples: {task_counts}")
    
    # Save merged dataset
    print(f"\nSaving to: {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Merge Complete!")
    print("=" * 60)
    print(f"Output file: {args.output_file}")
    print(f"Total examples: {len(merged)}")
    print("\nPer-task breakdown:")
    for task in TASKS:
        count = sum(1 for ex in merged if ex.get('task_name') == task)
        pct = 100 * count / len(merged)
        print(f"  {task}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
