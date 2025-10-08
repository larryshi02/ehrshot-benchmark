#!/usr/bin/env python3
"""
Complete Azure Reasoning Pipeline

Combines data preparation from FEMR database and reasoning trace generation
into a single script for streamlined workflow.
"""

import os
import sys
import json
import argparse
import collections
from datetime import datetime
from typing import Dict, List, Tuple, Any
from loguru import logger

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from llm_featurizer import load_labeled_patients_with_tasks, LLMFeaturizer, preprocess_llm_featurizer
    from serialization.ehr_serializer import UniqueThenListVisitsWOAllCondsWithValuesStrategy
    from femr.extension import datasets as extension_datasets
    from azure_reasoning_pipeline import AzureReasoningGenerator, ReasoningExample, AzureReasoningConfig
    import openai
    FEMR_AVAILABLE = True
except ImportError as e:
    logger.error(f"Required modules not available: {e}")
    logger.error("Please ensure you're in the EHRSHOT_ENV conda environment")
    FEMR_AVAILABLE = False

PatientDatabase = extension_datasets.PatientDatabase if FEMR_AVAILABLE else None


def load_instructions(task_to_instructions_file: str) -> Dict[str, str]:
    """Load task instructions from JSON file"""
    with open(task_to_instructions_file, 'r') as f:
        return json.load(f)


def create_training_examples(serialized_texts: List[str], 
                           patient_ids: List[int], 
                           label_times: List[datetime], 
                           label_values: List[bool],
                           task_instructions: List[str],
                           max_sequence_length: int) -> List[Dict[str, Any]]:
    """Create training examples from serialized data"""
    examples = []
    
    for i, (text, pid, label_time, label_value, instruction) in enumerate(
        zip(serialized_texts, patient_ids, label_times, label_values, task_instructions)
    ):
        # Truncate text if too long
        if len(text) > max_sequence_length:
            text = text[:max_sequence_length]
        
        example = {
            "patient_id": pid,
            "label_time": label_time.isoformat(),
            "label_value": label_value,
            "input_text": text,
            "target_text": "Positive" if label_value else "Negative",
            "full_sequence": f"{text}\n\nPrediction: {'Positive' if label_value else 'Negative'}",
            "task_instruction": instruction
        }
        examples.append(example)
    
    return examples


def split_data(examples: List[Dict[str, Any]], 
               train_split: float = 0.8, 
               val_split: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test sets"""
    import random
    random.shuffle(examples)
    
    n = len(examples)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    
    train_data = examples[:train_end]
    val_data = examples[train_end:val_end]
    test_data = examples[val_end:]
    
    return train_data, val_data, test_data


def prepare_data(args) -> List[Dict[str, Any]]:
    """Prepare data from FEMR database"""
    if not FEMR_AVAILABLE:
        raise RuntimeError("FEMR environment not available. Please activate EHRSHOT_ENV conda environment.")
    
    logger.info("Starting data preparation from FEMR database...")
    
    # Load task instructions
    task_to_instructions = load_instructions(args.task_to_instructions)
    logger.info(f"Loaded instructions for {len(task_to_instructions)} tasks")
    
    # Find all task directories
    task_dirs = [d for d in os.listdir(args.path_to_labels_dir) 
                 if os.path.isdir(os.path.join(args.path_to_labels_dir, d))]
    
    # Filter for specific task if specified
    if args.specific_task:
        if args.specific_task in task_dirs:
            task_dirs = [args.specific_task]
            logger.info(f"Filtering for specific task: {args.specific_task}")
        else:
            logger.error(f"Task '{args.specific_task}' not found. Available tasks: {task_dirs}")
            raise ValueError(f"Task '{args.specific_task}' not found in {args.path_to_labels_dir}")
    
    logger.info(f"Found {len(task_dirs)} task directories: {task_dirs}")
    
    # Load and combine labels from all tasks
    all_patients_to_labels: Dict[int, List[Tuple[datetime, str]]] = collections.defaultdict(list)
    for task_dir in task_dirs:
        label_file = os.path.join(args.path_to_labels_dir, task_dir, 'labeled_patients.csv')
        if not os.path.exists(label_file):
            logger.warning(f"Label file not found for task {task_dir}: {label_file}")
            continue
            
        task_patients_to_labels = load_labeled_patients_with_tasks(label_file)
        for patient_id, labels in task_patients_to_labels.items():
            all_patients_to_labels[patient_id].extend(labels)
    
    logger.info(f"Loaded labels from {len(task_dirs)} tasks")
    logger.info(f"Total patients: {len(all_patients_to_labels)}")
    logger.info(f"Total labels: {sum(len(labels) for labels in all_patients_to_labels.values())}")
    
    # Limit the number of patients if specified
    if args.num_samples:
        # Take a random sample of patients
        import random
        patient_ids_list = list(all_patients_to_labels.keys())
        random.shuffle(patient_ids_list)
        selected_patient_ids = set(patient_ids_list[:args.num_samples])
        all_patients_to_labels = {pid: labels for pid, labels in all_patients_to_labels.items() 
                                 if pid in selected_patient_ids}
        logger.info(f"Limited to {len(all_patients_to_labels)} patients for processing")
    
    # Load database
    logger.info(f"Loading database from {args.path_to_database}")
    database = PatientDatabase(args.path_to_database)
    
    # Create serialization strategy
    serialization_strategy = UniqueThenListVisitsWOAllCondsWithValuesStrategy(
        num_aggregated_events=args.num_aggregated
    )
    
    # Process patients in batches to avoid loading all at once
    batch_size = 50  # Process 50 patients at a time
    patient_ids_list = list(all_patients_to_labels.keys())
    total_batches = (len(patient_ids_list) + batch_size - 1) // batch_size
    
    serialized_texts = []
    patient_ids = []
    label_times = []
    label_values = []
    task_instructions = []
    
    logger.info(f"Processing {len(patient_ids_list)} patients in {total_batches} batches of {batch_size}")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(patient_ids_list))
        batch_patient_ids = patient_ids_list[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_patient_ids)} patients)")
        
        # Create a batch-specific patients_to_labels dict
        batch_patients_to_labels = {pid: all_patients_to_labels[pid] for pid in batch_patient_ids}
        
        # Initialize LLMFeaturizer for this batch
        llm_featurizer = LLMFeaturizer(
            embedding_size=1,  # Not used for data preparation
            serialization_strategy=serialization_strategy,
            task_to_instructions=task_to_instructions,
            excluded_ontologies=args.excluded_ontologies.split(',') if args.excluded_ontologies else [],
            filter_aggregated_events=True,
            time_window=None
        )
        
        # Preprocess featurizer for this batch only
        logger.info(f"Preprocessing batch {batch_idx + 1} featurizers...")
        llm_featurizer = preprocess_llm_featurizer(
            args.path_to_database, 
            llm_featurizer, 
            batch_patients_to_labels, 
            args.num_threads
        )
        
        # Process this batch
        for (pid, label_idx), (instruction, text) in sorted(llm_featurizer.pid_label_idx_serializations.items()):
            serialized_texts.append(text)
            label_time, label_value = batch_patients_to_labels[pid][label_idx]
            patient_ids.append(pid)
            label_times.append(label_time)
            label_values.append(label_value)
            task_instructions.append(instruction)
        
        logger.info(f"Batch {batch_idx + 1} complete. Total examples so far: {len(serialized_texts)}")
    
    logger.info(f"Extracted {len(serialized_texts)} serialized patient histories total")
    
    # Limit total examples if specified (in addition to patient limit)
    if args.num_samples and len(serialized_texts) > args.num_samples:
        logger.info(f"Limiting total examples from {len(serialized_texts)} to {args.num_samples}")
        serialized_texts = serialized_texts[:args.num_samples]
        patient_ids = patient_ids[:args.num_samples]
        label_times = label_times[:args.num_samples]
        label_values = label_values[:args.num_samples]
        task_instructions = task_instructions[:args.num_samples]
    
    # Create training examples
    logger.info("Creating training examples...")
    examples = create_training_examples(
        serialized_texts, 
        patient_ids, 
        label_times, 
        label_values,
        task_instructions,
        args.max_sequence_length
    )
    
    logger.info(f"Created {len(examples)} training examples")
    return examples


def generate_reasoning_traces(examples: List[Dict[str, Any]], args) -> List[ReasoningExample]:
    """Generate reasoning traces using Azure OpenAI"""
    logger.info("Starting reasoning trace generation...")
    
    # Initialize Azure reasoning generator
    config = AzureReasoningConfig(
        endpoint=args.azure_endpoint,
        deployment=args.azure_deployment,
        api_key=args.azure_api_key,
        api_version=args.azure_api_version,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    generator = AzureReasoningGenerator(config)
    
    # Convert examples to ReasoningExample objects
    reasoning_examples = []
    for example in examples[:args.max_examples]:
        reasoning_example = ReasoningExample(
            patient_id=example["patient_id"],
            label_time=example["label_time"],
            label_value=example["label_value"],
            input_text=example["input_text"],
            target_text=example["target_text"],
            full_sequence=example["full_sequence"],
            task_instruction=example["task_instruction"]
        )
        reasoning_examples.append(reasoning_example)
    
    logger.info(f"Generating reasoning traces for {len(reasoning_examples)} examples...")
    
    # Generate reasoning traces
    for i, example in enumerate(reasoning_examples):
        logger.info(f"Processing example {i+1}/{len(reasoning_examples)}")
        generator.generate_reasoning(example)
    
    logger.info("Reasoning trace generation complete!")
    return reasoning_examples


def save_reasoning_traces(examples: List[ReasoningExample], output_file: str):
    """Save reasoning traces to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert all examples to a list of dictionaries
    data = [example.to_dict() for example in examples]
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(examples)} reasoning traces to {output_file}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Complete Azure Reasoning Pipeline")
    
    # Data preparation arguments
    parser.add_argument("--path_to_database", required=True,
                       help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True,
                       help="Path to directory containing saved labels")
    parser.add_argument("--task_to_instructions", required=True,
                       help="Path to task to instructions file")
    parser.add_argument("--specific_task", type=str, default=None,
                       help="Optional: Filter to generate patients and predict only for this specific task")
    parser.add_argument("--num_samples", type=int,
                       help="Limit number of samples (not patients) to generate")
    parser.add_argument("--num_threads", type=int, default=1,
                       help="Number of threads to use")
    parser.add_argument("--serialization_strategy", default="unique_then_list_visits_wo_allconds_w_values_4k",
                       help="Serialization strategy to use")
    parser.add_argument("--excluded_ontologies", default="no_labs_single",
                       help="Ontologies to exclude")
    parser.add_argument("--num_aggregated", type=int, default=3,
                       help="Number of aggregated values to use")
    parser.add_argument("--time_window_days", type=int, default=0,
                       help="Number of days before label time to consider")
    parser.add_argument("--max_sequence_length", type=int, default=4096,
                       help="Maximum sequence length for training")
    
    # Reasoning generation arguments
    parser.add_argument("--max_examples", type=int, default=10,
                       help="Maximum number of example patients to extract from FEMR database")
    parser.add_argument("--output_file", required=True,
                       help="Path to output JSON file for reasoning traces")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature for Azure OpenAI")
    parser.add_argument("--max_tokens", type=int, default=4096,
                       help="Maximum tokens for Azure OpenAI responses")
    
    # Azure OpenAI arguments (secure defaults)
    parser.add_argument("--azure_endpoint", 
                       default=None,
                       help="Azure OpenAI endpoint URL (defaults to environment variable or config file)")
    parser.add_argument("--azure_deployment", 
                       default=None,
                       help="Azure OpenAI deployment name (defaults to environment variable or config file)")
    parser.add_argument("--azure_api_key", 
                       default=None,
                       help="Azure OpenAI API key (defaults to environment variable or config file)")
    parser.add_argument("--azure_api_version", 
                       default=None,
                       help="Azure OpenAI API version (defaults to environment variable or config file)")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    try:
        # Step 1: Prepare data from FEMR database
        logger.info("=== STEP 1: DATA PREPARATION ===")
        examples = prepare_data(args)
        
        # Step 2: Generate reasoning traces
        logger.info("=== STEP 2: REASONING GENERATION ===")
        reasoning_examples = generate_reasoning_traces(examples, args)
        
        # Step 3: Save results
        logger.info("=== STEP 3: SAVING RESULTS ===")
        save_reasoning_traces(reasoning_examples, args.output_file)
        
        logger.success(f"Complete pipeline finished! Results saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
