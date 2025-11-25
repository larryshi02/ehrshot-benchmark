#!/usr/bin/env python3
"""
Create rubric instructions for each EHRSHOT task using GPT-5.

Reads cohort files (from build_cohort.py) and feeds them into GPT-5 to generate
step-by-step instructions for creating a fixed rubric format for each task.
The goal is to transform input sequences into a structured rubric format.
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Any, Optional
from pathlib import Path

from loguru import logger
from openai import AzureOpenAI

# Import Azure config from the backwards reasoning pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from azure_backwards_reasoning_multi_task_pipeline import AzureBackwardsReasoningConfig

# Task names
TASKS = ['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer']



TASK_PREDICTION_QUERIES: Dict[str, str] = {
    "acute_mi": "Will the patient develop an acute myocardial infarction in the next year?",
    "hypertension": "Will the patient develop hypertension in the next year?",
    "hyperlipidemia": "Will the patient develop hyperlipidemia in the next year?",
    "pancreatic_cancer": "Will the patient develop pancreatic cancer in the next year?",
}

def load_cohort_file(cohort_file: str) -> List[Dict]:
    """Load cohort examples from a JSON file."""
    if not os.path.exists(cohort_file):
        logger.error(f"Cohort file not found: {cohort_file}")
        return []
    
    with open(cohort_file, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from {cohort_file}")
    return data


def format_example_for_prompt(example: Dict, task_name: str) -> str:
    """Format a single example for inclusion in the prompt."""
    label = "Positive" if example.get('label_value', False) else "Negative"
    context = example.get('context', '')
    
    # Truncate context if too long (keep first 2000 chars to fit in prompt)
    if len(context) > 2000:
        context = context[:2000] + "... [truncated]"
    
    return f"""Example {example.get('patient_id', 'unknown')} ({label}):
{context}
---"""


def create_rubric_prompt(
    task_name: str,
    examples: List[Dict],
    task_instruction: str
) -> str:
    """Create a prompt asking GPT-5 to generate rubric instructions."""
    

    
    # Format examples
    formatted_examples = []
    for ex in examples:
        formatted_examples.append(format_example_for_prompt(ex, task_name))
        
    random.shuffle(formatted_examples)
    
    examples_text = "NEW EXAMPLE\n\n".join(formatted_examples)
    examples_text = f"NEW EXAMPLE\n\n{examples_text}"
    
    prompt = f"""You are a medical expert tasked with creating a structured rubric for evaluating clinical predictions.

Task: {task_name.replace('_', ' ').title()}
Task Query: {task_instruction}

I have provided you with {len(examples)} diverse patient examples from the training data. Each example contains:
- Patient demographics
- Recent body metrics and vital signs
- Recent lab results
- Past medical visits with detailed information
- General medical events

Your task is to create a detailed **step-by-step instruction/rubric** that can be used to systematically evaluate any patient's EHR and transform it into a structured rubric format. This rubric should:

1. Identify key clinical indicators and risk factors relevant to the prediction task
2. Structure the evaluation in a clear, reproducible format
3. Guide systematic analysis of patient data

The rubric should be detailed enough that it can be applied consistently to new patient cases, and should capture the clinical reasoning patterns evident in the provided examples. However, it should not reference any specific examples or data points.

Please provide:
1. A step-by-step instruction/rubric template
2. Key categories or dimensions to evaluate (e.g., risk factors, clinical indicators, lab values, medical history patterns)
3. How to structure the analysis

Within the rubric, encourage the user to fill out the fields, but not jump to any conclusions about the patient's risk for the task.

Here are the {len(examples)} diverse examples:

{examples_text}

Now, create a comprehensive, step-by-step rubric instruction that can transform any patient's EHR input into this structured evaluation format:"""
    
    return prompt

def generate_rubric_instructions(
    task_name: str,
    examples: List[Dict],
    task_instruction: str,
    config: AzureBackwardsReasoningConfig
) -> Dict[str, Any]:
    """Generate rubric instructions using Azure GPT-5."""
    
    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
    )
    
    prompt = create_rubric_prompt(task_name, examples, task_instruction)
    
    logger.info(f"Generating rubric instructions for task: {task_name}")
    logger.info(f"Prompt length: {len(prompt)} characters")
    
    try:
        completion_params = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical expert AI assistant specializing in creating structured "
                        "clinical evaluation rubrics. You excel at analyzing patient data patterns "
                        "and creating systematic evaluation frameworks."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": config.max_completion_tokens,
            "temperature": config.temperature,
            "model": config.deployment,
        }
        
        # Only include top_p if the model supports it
        if "gpt-5" not in config.deployment.lower():
            completion_params["top_p"] = config.top_p
        
        response = client.chat.completions.create(**completion_params)
        
        rubric_instructions = response.choices[0].message.content.strip()
        
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "total_tokens": response.usage.total_tokens if response.usage else None,
        }
        
        logger.info(f"Generated rubric instructions ({usage.get('total_tokens', 0)} tokens)")
        
        return {
            "task_name": task_name,
            "task_instruction": task_instruction,
            "rubric_instructions": rubric_instructions,
            "num_examples_used": len(examples),
            "usage": usage,
            "prompt": prompt,
        }
        
    except Exception as exc:
        logger.error(f"Error generating rubric instructions: {exc}")
        raise




def process_task(
    task_name: str,
    cohort_dir: str,
    task_instructions: Dict[str, str],
    config: AzureBackwardsReasoningConfig,
    output_file: str
) -> None:
    """Process a single task to generate rubric instructions."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing task: {task_name}")
    logger.info(f"{'='*60}")
    
    # Load cohort file
    cohort_file = os.path.join(cohort_dir, f"{task_name}_cohort.json")
    examples = load_cohort_file(cohort_file)
    
    if not examples:
        logger.warning(f"No examples found for task {task_name}, skipping")
        return
    
    logger.info(f"Using {len(examples)} examples from cohort")
    
    # Get task instruction
    task_instruction = task_instructions.get(task_name, "")
    if not task_instruction:
        logger.warning(f"No instruction found for task {task_name}")
        task_instruction = f"Will the patient develop {task_name.replace('_', ' ')} in the next year?"
    
    # Generate rubric instructions
    result = generate_rubric_instructions(
        task_name, examples, task_instruction, config
    )
    
    # Save result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved rubric instructions to: {output_file}")
    logger.info(f"  Examples used: {result['num_examples_used']}")
    logger.info(f"  Tokens used: {result['usage'].get('total_tokens', 0)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate rubric instructions for EHRSHOT tasks using GPT-5"
    )
    
    parser.add_argument(
        "--cohort_dir",
        type=str,
        default="ehrshot/sft/rubric/cohorts",
        help="Directory containing cohort JSON files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ehrshot/sft/rubric/rubrics",
        help="Output directory for rubric instruction files"
    )
    
    parser.add_argument(
        "--task_to_instructions",
        type=str,
        default="ehrshot/serialization/task_to_instructions.json",
        help="Path to task instructions JSON file"
    )
    
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens for GPT-5 (default: 4096)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for GPT-5 generation (default: 0.3)"
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
    
    if not os.path.isabs(args.cohort_dir):
        args.cohort_dir = os.path.join(project_root, args.cohort_dir)
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    if not os.path.isabs(args.task_to_instructions):
        args.task_to_instructions = os.path.join(project_root, args.task_to_instructions)
    
    # Load task instructions
    task_instructions = TASK_PREDICTION_QUERIES
    
    # Initialize Azure config
    config = AzureBackwardsReasoningConfig(
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )
    
    # Process each task
    for task_name in args.tasks:
        try:
            output_file = os.path.join(args.output_dir, f"{task_name}_rubric.json")
            process_task(
                task_name=task_name,
                cohort_dir=args.cohort_dir,
                task_instructions=task_instructions,
                config=config,
                output_file=output_file
            )
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*60)
    logger.info("Rubric generation completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

