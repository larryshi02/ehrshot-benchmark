#!/usr/bin/env python3
"""
Serialize patient data for 4 tasks into JSON files with splits.
Creates one JSON file per task with all splits (train, val, test).
Each example includes a 'split' field and raw EHR context (no instructions or queries).
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Optional
from datetime import datetime
# Import numpy first to avoid binary incompatibility issues
import numpy as np
import pandas as pd
from loguru import logger

# Import serialization modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llm_featurizer import load_labeled_patients_with_tasks, LLMFeaturizer, preprocess_llm_featurizer
from serialization.ehr_serializer import UniqueThenListVisitsWOAllCondsWithValuesStrategy
from femr.extension.datasets import PatientDatabase
from transformers import AutoTokenizer

# Task name mappings
TASK_MAPPINGS = {
    'acute_mi': 'new_acutemi',
    'hyperlipidemia': 'new_hyperlipidemia',
    'hypertension': 'new_hypertension',
    'pancreatic_cancer': 'new_pancan'
}


def load_splits(path_to_splits: str) -> Dict[str, set]:
    """Load train/val/test splits from CSV file"""
    splits_df = pd.read_csv(path_to_splits)
    
    splits = {
        'train': set(splits_df[splits_df['split'] == 'train']['omop_person_id'].values),
        'val': set(splits_df[splits_df['split'] == 'val']['omop_person_id'].values),
        'test': set(splits_df[splits_df['split'] == 'test']['omop_person_id'].values)
    }
    
    logger.info(f"Loaded splits - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    return splits


def load_task_labels(task_name: str, path_to_labels_dir: str, splits: Dict[str, set]) -> Dict[str, List[Dict]]:
    """Load labels for a specific task and separate by split"""
    # Map task name to directory name
    task_dir = TASK_MAPPINGS.get(task_name, task_name)
    label_file = os.path.join(path_to_labels_dir, task_dir, 'labeled_patients.csv')
    
    if not os.path.exists(label_file):
        logger.error(f"Label file not found for task {task_name}: {label_file}")
        return {}
    
    # Load labels
    task_patients_to_labels = load_labeled_patients_with_tasks(label_file)
    
    # Separate examples by split
    examples_by_split = {'train': [], 'val': [], 'test': []}
    
    for patient_id, labels in task_patients_to_labels.items():
        # Determine split
        split = None
        if patient_id in splits['train']:
            split = 'train'
        elif patient_id in splits['val']:
            split = 'val'
        elif patient_id in splits['test']:
            split = 'test'
        else:
            logger.warning(f"Patient {patient_id} not in any split, skipping")
            continue
        
        for label in labels:
            examples_by_split[split].append({
                'patient_id': patient_id,
                'label_time': label.time.isoformat(),
                'label_value': label.value,
                'label': label,
                'split': split
            })
    
    # Log counts
    for split_name in ['train', 'val', 'test']:
        logger.info(f"Loaded {len(examples_by_split[split_name])} {split_name} examples for task {task_name}")
    
    return examples_by_split


def truncate_context_to_fit(
    context: str, 
    tokenizer, 
    max_context_tokens: int = 3700
) -> str:
    """
    Intelligently truncate EHR context to fit within token limit.
    Preserves most recent information (most recent visits first) and truncates at token boundaries.
    Similar to truncation strategy used in the pipeline's text encoder.
    
    Args:
        context: Raw EHR serialization text
        tokenizer: Tokenizer to use for tokenization
        max_context_tokens: Maximum tokens for context (default 3700, leaving ~400 for prompt)
    
    Returns:
        Truncated context string
    """
    try:
        # Tokenize the context to get token offsets
        encoding = tokenizer(
            context,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            padding=False
        )
        
        token_ids = encoding['input_ids']
        offset_mapping = encoding['offset_mapping']
        
        # If context fits, return as-is
        if len(token_ids) <= max_context_tokens:
            return context
        
        # Truncate to max_context_tokens, preserving the start (most recent information)
        # Since serialization is already sorted most recent first, we keep the beginning
        num_offsets = len(offset_mapping)
        end_idx = min(max_context_tokens, num_offsets)
        
        # Get the character range for the last token we want to keep
        if end_idx > 0:
            last_offset = offset_mapping[end_idx - 1]
            # Truncate at the end of the last token
            truncated_context = context[:last_offset[1]]
        else:
            truncated_context = ""
        
        logger.debug(f"Truncated context from {len(token_ids)} tokens to {end_idx} tokens")
        return truncated_context
        
    except Exception as e:
        logger.warning(f"Error truncating context: {e}, returning original context")
        return context


def serialize_patient_history(
    task_name: str, 
    examples_by_split: Dict[str, List[Dict]], 
    path_to_database: str,
    task_to_instructions: Dict[str, str],
    excluded_ontologies: List[str],
    num_aggregated_events: int,
    tokenizer: Optional[AutoTokenizer] = None,
    max_context_tokens: int = 3700
) -> Dict[str, List[Dict]]:
    """
    Serialize patient history for all splits - returns raw EHR without instructions.
    
    Args:
        task_name: Name of the task
        examples_by_split: Examples grouped by split (train/val/test)
        path_to_database: Path to patient database
        task_to_instructions: Task instructions dictionary
        excluded_ontologies: List of ontologies to exclude
        num_aggregated_events: Number of aggregated events to include
        tokenizer: Optional tokenizer for intelligent truncation (if None, no truncation)
        max_context_tokens: Maximum tokens for context (default 3700, leaving ~400 for prompt)
    
    Returns:
        Dictionary of serialized examples by split
    """
    # Load database
    logger.info(f"Loading database from: {path_to_database}")
    database = PatientDatabase(path_to_database)
    
    # Combine all examples
    all_examples = []
    for split_name in ['train', 'val', 'test']:
        all_examples.extend(examples_by_split[split_name])
    
    if not all_examples:
        logger.warning(f"No examples to serialize for task {task_name}")
        return {}
    
    # Prepare labels for LLMFeaturizer
    all_patients_to_labels = {}
    for example in all_examples:
        pid = example['patient_id']
        if pid not in all_patients_to_labels:
            all_patients_to_labels[pid] = []
        all_patients_to_labels[pid].append(example['label'])
    
    # Create serialization strategy
    serialization_strategy = UniqueThenListVisitsWOAllCondsWithValuesStrategy(
        num_aggregated_events=num_aggregated_events
    )
    
    # Initialize LLMFeaturizer
    llm_featurizer = LLMFeaturizer(
        embedding_size=1,
        serialization_strategy=serialization_strategy,
        task_to_instructions=task_to_instructions,
        excluded_ontologies=excluded_ontologies,
        filter_aggregated_events=True,
        time_window=None
    )
    
    # Preprocess featurizer to generate serializations
    logger.info(f"Serializing patient histories for task {task_name}...")
    llm_featurizer = preprocess_llm_featurizer(
        path_to_database,
        llm_featurizer,
        all_patients_to_labels,
        1  # num_threads
    )
    
    # Create mapping from (pid, label_time) to serialization
    pid_time_to_serialization = {}
    for (pid_key, label_idx), (inst, serialization_text) in llm_featurizer.pid_label_idx_serializations.items():
        if pid_key in all_patients_to_labels:
            patient_labels = all_patients_to_labels[pid_key]
            if label_idx < len(patient_labels):
                label = patient_labels[label_idx]
                pid_time_to_serialization[(pid_key, label.time)] = serialization_text
    
    # Create serialized examples with splits
    serialized_examples_by_split = {'train': [], 'val': [], 'test': []}
    
    for example in all_examples:
        pid = example['patient_id']
        label = example['label']
        label_time = label.time
        split = example['split']
        
        # Find serialization for this patient-label pair
        key = (pid, label_time)
        if key in pid_time_to_serialization:
            serialization = pid_time_to_serialization[key]
            
            # Apply intelligent truncation if tokenizer is provided (for 4k strategy)
            if tokenizer is not None:
                serialization = truncate_context_to_fit(serialization, tokenizer, max_context_tokens)
            
            # Store only the raw EHR context (no instructions, no query)
            serialized_example = {
                'patient_id': pid,
                'label_time': example['label_time'],
                'label_value': example['label_value'],
                'context': serialization,  # Raw EHR serialization (possibly truncated)
                'split': split
            }
            
            serialized_examples_by_split[split].append(serialized_example)
        else:
            logger.warning(f"No serialization found for patient {pid} at time {label_time}")
    
    # Log counts
    for split_name in ['train', 'val', 'test']:
        logger.info(f"Serialized {len(serialized_examples_by_split[split_name])} {split_name} examples for task {task_name}")
    
    return serialized_examples_by_split


def serialize_task(
    task_name: str,
    path_to_database: str,
    path_to_labels_dir: str,
    path_to_splits: str,
    output_dir: str,
    task_to_instructions: Dict[str, str],
    excluded_ontologies: List[str],
    num_aggregated_events: int,
    tokenizer: Optional[AutoTokenizer] = None,
    max_context_tokens: int = 8192
) -> None:
    """Serialize all splits for a single task"""
    logger.info(f"Processing task: {task_name}")
    
    # Load splits
    splits = load_splits(path_to_splits)
    
    # Load task labels by split
    examples_by_split = load_task_labels(task_name, path_to_labels_dir, splits)
    
    if not any(examples_by_split.values()):
        logger.warning(f"No examples found for task {task_name}, skipping")
        return
    
    # Serialize patient histories
    serialized_examples_by_split = serialize_patient_history(
        task_name,
        examples_by_split,
        path_to_database,
        task_to_instructions,
        excluded_ontologies,
        num_aggregated_events,
        tokenizer=tokenizer,
        max_context_tokens=max_context_tokens
    )
    
    if not any(serialized_examples_by_split.values()):
        logger.warning(f"No serialized examples for task {task_name}, skipping")
        return
    
    # Combine all splits into single list
    all_serialized_examples = []
    for split_name in ['train', 'val', 'test']:
        all_serialized_examples.extend(serialized_examples_by_split[split_name])
    
    # Save to JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{task_name}_all_splits.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_serialized_examples, f, indent=2)
    
    logger.info(f"Saved {len(all_serialized_examples)} examples to {output_file}")
    
    # Print summary
    print(f"\nTask {task_name} Summary:")
    print(f"  Train: {len(serialized_examples_by_split['train'])}")
    print(f"  Val: {len(serialized_examples_by_split['val'])}")
    print(f"  Test: {len(serialized_examples_by_split['test'])}")
    print(f"  Total: {len(all_serialized_examples)}")


def main():
    parser = argparse.ArgumentParser(description="Serialize patient data for 4 tasks with splits")
    
    # Data configuration
    parser.add_argument("--path_to_database", type=str, required=True,
                       help="Path to patient database")
    parser.add_argument("--path_to_labels_dir", type=str, required=True,
                       help="Path to labels directory")
    parser.add_argument("--path_to_splits", type=str, required=True,
                       help="Path to splits CSV file")
    parser.add_argument("--task_to_instructions", type=str,
                       default="ehrshot/serialization/task_to_instructions.json",
                       help="Path to task instructions JSON file")
    
    # Serialization configuration
    parser.add_argument("--excluded_ontologies", type=str,
                       default="LOINC,Domain,CARE_SITE,ICDO3,Medicare Specialty,CMS Place of Service,OMOP Extension,Condition Type",
                       help="Comma-separated list of ontologies to exclude")
    parser.add_argument("--num_aggregated_events", type=int, default=3,
                       help="Number of aggregated events to include")
    
    # Truncation configuration (for 4k strategy)
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2-7B-Instruct",
                       help="Tokenizer to use for intelligent truncation (default: Qwen/Qwen2-7B-Instruct). Set to empty string to disable truncation.")
    parser.add_argument("--max_context_tokens", type=int, default=8192,
                       help="Maximum tokens for context (default: 8192)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for JSON files")
    
    args = parser.parse_args()
    
    # Load task instructions
    task_to_instructions = {}
    if args.task_to_instructions and os.path.exists(args.task_to_instructions):
        with open(args.task_to_instructions, 'r') as f:
            task_to_instructions = json.load(f)
        logger.info(f"Loaded task instructions from {args.task_to_instructions}")
    else:
        logger.warning(f"Task instructions file not found: {args.task_to_instructions}")
    
    # Parse excluded ontologies
    excluded_ontologies = [o.strip() for o in args.excluded_ontologies.split(',')]
    
    # Load tokenizer for intelligent truncation (if specified)
    tokenizer = None
    if args.tokenizer_name:
        logger.info(f"Loading tokenizer for truncation: {args.tokenizer_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name,
                trust_remote_code=True
            )
            logger.info(f"Tokenizer loaded successfully. Max context tokens: {args.max_context_tokens}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {args.tokenizer_name}: {e}. Truncation will be disabled.")
            tokenizer = None
    else:
        logger.info("No tokenizer specified. Truncation disabled.")
    
    # Process each task
    tasks = ['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer']
    
    for task_name in tasks:
        try:
            serialize_task(
                task_name=task_name,
                path_to_database=args.path_to_database,
                path_to_labels_dir=args.path_to_labels_dir,
                path_to_splits=args.path_to_splits,
                output_dir=args.output_dir,
                task_to_instructions=task_to_instructions,
                excluded_ontologies=excluded_ontologies,
                num_aggregated_events=args.num_aggregated_events,
                tokenizer=tokenizer,
                max_context_tokens=args.max_context_tokens
            )
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("Serialization completed!")


if __name__ == "__main__":
    main()

