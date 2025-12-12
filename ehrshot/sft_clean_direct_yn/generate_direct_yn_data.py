#!/usr/bin/env python3
"""
Generate Direct Y/N Training Data

This script transforms the existing SFT datasets (with reasoning) into 
direct classification datasets where the assistant response is simply 
"Positive" or "Negative" without any reasoning.

Key differences from reasoning-based SFT:
1. System prompt is simplified - no instructions for <think> tags
2. Assistant response is just the label token
3. Thinking mode is explicitly disabled via system prompt suffix

Usage:
    python generate_direct_yn_data.py --input_dir <path> --output_dir <path>
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any


# Task-specific queries (same as original)
TASK_QUERIES = {
    'acute_mi': 'Will the patient develop an acute myocardial infarction in the next year?',
    'hyperlipidemia': 'Will the patient develop hyperlipidemia in the next year?',
    'hypertension': 'Will the patient develop hypertension in the next year?',
    'pancreatic_cancer': 'Will the patient develop pancreatic cancer in the next year?'
}


def get_task_name_from_filename(filename: str) -> str:
    """Extract task name from filename like 'acute_mi_sft_dataset.json'."""
    for task in TASK_QUERIES.keys():
        if filename.startswith(task):
            return task
    return None


def create_direct_yn_system_prompt(task_instruction: str) -> str:
    """
    Create a simplified system prompt for direct Y/N classification.
    
    This prompt:
    1. Does NOT ask for reasoning or <think> tags
    2. Includes /no_think suffix to disable Qwen3's thinking mode
    3. Asks for a direct single-word response
    """
    system_content = (
        f"You are a medical expert. You will be provided with a patient's "
        f"Electronic Healthcare Record (EHR).\n"
        f"Your task is to predict: {task_instruction}\n\n"
        f"Respond with exactly one word: Positive or Negative.\n"
        f"- Positive: The patient WILL develop the condition.\n"
        f"- Negative: The patient will NOT develop the condition."
    )
    return system_content


def transform_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single entry from reasoning-based format to direct Y/N format.
    
    Input format (from existing sft_dataset.json):
    {
        "conversations": [
            {"role": "system", "content": "...reasoning instructions..."},
            {"role": "user", "content": "...EHR context..."},
            {"role": "assistant", "content": "<think>...</think>\n\nFinal Answer: Positive/Negative"}
        ],
        "patient_id": ...,
        "label_time": ...,
        "label_value": true/false,
        "task_instruction": "..."
    }
    
    Output format:
    {
        "conversations": [
            {"role": "system", "content": "...simple direct prompt..."},
            {"role": "user", "content": "...EHR context..."},
            {"role": "assistant", "content": "Positive" or "Negative"}
        ],
        "patient_id": ...,
        "label_time": ...,
        "label_value": true/false,
        "task_instruction": "..."
    }
    """
    # Extract original data
    original_conversations = entry["conversations"]
    patient_id = entry.get("patient_id")
    label_time = entry.get("label_time")
    label_value = entry.get("label_value")
    task_instruction = entry.get("task_instruction", "")
    
    # Get the user content (EHR context) from original conversations
    user_content = ""
    for conv in original_conversations:
        if conv["role"] == "user":
            user_content = conv["content"]
            break
    
    # Determine the direct response based on label_value
    # label_value = True  -> Patient WILL develop condition -> "Positive"
    # label_value = False -> Patient will NOT develop condition -> "Negative"
    direct_response = "Positive" if label_value else "Negative"
    
    # Create new simplified system prompt
    system_content = create_direct_yn_system_prompt(task_instruction)
    
    # Build new conversations
    new_conversations = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": direct_response}
    ]
    
    # Return transformed entry
    return {
        "conversations": new_conversations,
        "patient_id": patient_id,
        "label_time": label_time,
        "label_value": label_value,
        "task_instruction": task_instruction
    }


def process_file(input_path: str, output_path: str) -> int:
    """
    Process a single JSON file and write transformed data.
    
    Returns the number of entries processed.
    """
    print(f"  Processing: {os.path.basename(input_path)}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    transformed_data = []
    for entry in data:
        transformed_entry = transform_entry(entry)
        transformed_data.append(transformed_entry)
    
    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=4)
    
    print(f"    -> Wrote {len(transformed_data)} entries to {os.path.basename(output_path)}")
    return len(transformed_data)


def main():
    parser = argparse.ArgumentParser(
        description="Generate direct Y/N training data from reasoning-based SFT datasets"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/demirel/clinical-reasoning/ehrshot-benchmark/ehrshot/sft_clean/data_gpt-5-mini/sft_rt_yn",
        help="Input directory containing train_all/, val_small/ with *_sft_dataset.json files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/demirel/clinical-reasoning/ehrshot-benchmark/ehrshot/sft_clean_direct_yn/data",
        help="Output directory for transformed data"
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process each subfolder (train_all, val_small)
    subfolders = ["train_all", "val_small"]
    total_entries = 0
    
    for subfolder in subfolders:
        input_subfolder = input_dir / subfolder
        output_subfolder = output_dir / subfolder
        
        if not input_subfolder.exists():
            print(f"Skipping missing subfolder: {input_subfolder}")
            continue
        
        print(f"Processing subfolder: {subfolder}")
        
        # Find all *_sft_dataset.json files
        for json_file in sorted(input_subfolder.glob("*_sft_dataset.json")):
            # Skip balanced files (used for validation, different naming)
            if "balanced" in json_file.name:
                # For val_small, the balanced files are the main ones
                output_path = output_subfolder / json_file.name
            else:
                output_path = output_subfolder / json_file.name
            
            count = process_file(str(json_file), str(output_path))
            total_entries += count
        
        print()
    
    print(f"Done! Total entries processed: {total_entries}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
