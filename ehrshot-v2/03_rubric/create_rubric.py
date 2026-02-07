#!/usr/bin/env python3
"""
Generate task-specific rubric instructions using GPT-5-mini.

For each task, this script:
  1. Loads the 40-patient cohort (from build_cohort.py).
  2. Formats examples with their ground-truth labels.
  3. Prompts GPT-5-mini to produce a step-by-step rubric template.
  4. Saves the rubric JSON.

Inputs:
  --cohort_dir  : Directory with {task}/cohort.json files.
  --output_dir  : Where to write rubric JSONs.
  --tasks       : Space-separated list (default: all 15).

Outputs:
  {output_dir}/{task}/rubric.json

GPT-5-mini parameters: max_completion_tokens=16384, temperature=1.

Connects to:
  - Upstream  : build_cohort.py
  - Downstream: apply_rubric.py
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from openai import AzureOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, SEED
from config.azure import AzureConfig


def _format_example(example: dict, task_query: str) -> str:
    label = "Yes" if example.get("label", False) else "No"
    ctx = example.get("serialization", "")
    pid = example.get("patient_id", "unknown")
    return f"Example {pid} (Ground Truth: {label}):\n{ctx}\n---"


def _build_prompt(task: str, examples: List[dict], task_query: str) -> str:
    random.seed(SEED)
    formatted = [_format_example(e, task_query) for e in examples]
    random.shuffle(formatted)
    examples_text = "\nNEW EXAMPLE\n\n".join(formatted)

    return f"""You are a medical expert tasked with creating a structured rubric for evaluating clinical predictions.

Task: {task.replace('_', ' ').title()}
Task Query: {task_query}

I have provided you with {len(examples)} diverse patient examples from the training data.
Each example is labeled as Yes or No based on the actual outcome.

Your task is to create a detailed step-by-step instruction/rubric that can be used to
systematically evaluate any patient's EHR and transform it into a structured rubric format.

The rubric should:
1. Identify key clinical indicators and risk factors relevant to the prediction task.
2. Structure the evaluation in a clear, reproducible format.
3. Guide systematic analysis of patient data.

Within the rubric, encourage the evaluator to fill out the fields but NOT jump to any
conclusions about the patient's risk. The rubric will be used by other models to transform
patient EHRs for the given task.

Please provide:
1. A step-by-step instruction/rubric template.
2. Key categories or dimensions to evaluate.
3. How to structure the analysis.

Here are the {len(examples)} diverse examples:

NEW EXAMPLE

{examples_text}

Now, create a comprehensive step-by-step rubric instruction:"""


def generate_rubric(task: str, examples: List[dict],
                    task_query: str, config: AzureConfig) -> Dict[str, Any]:
    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
    )
    prompt = _build_prompt(task, examples, task_query)
    logger.info(f"  prompt length: {len(prompt)} chars")

    resp = client.chat.completions.create(
        model=config.deployment,
        messages=[
            {"role": "system",
             "content": ("You are a medical expert AI assistant specializing in "
                         "creating structured clinical evaluation rubrics.")},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=config.max_completion_tokens,
        temperature=config.temperature,
    )
    text = resp.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
    }
    logger.info(f"  rubric generated ({usage.get('completion_tokens', '?')} completion tokens)")
    return {
        "task": task,
        "task_query": task_query,
        "rubric_instructions": text,
        "num_examples": len(examples),
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--azure_config", default=None)
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    return p.parse_args()


def main():
    args = parse_args()
    config = AzureConfig.from_json(args.azure_config)
    # Enforce plan parameters
    config.max_completion_tokens = 16384
    config.temperature = 1.0

    for task in args.tasks:
        logger.info(f"\n{'='*60}\nCreating rubric for: {task}\n{'='*60}")
        cohort_path = Path(args.cohort_dir) / task / "cohort.json"
        if not cohort_path.exists():
            logger.warning(f"  cohort not found at {cohort_path}, skipping")
            continue
        with open(cohort_path) as f:
            examples = json.load(f)

        result = generate_rubric(task, examples, TASKS[task], config)

        out_dir = Path(args.output_dir) / task
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "rubric.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  saved -> {out_dir / 'rubric.json'}")

    logger.success("All rubrics created.")


if __name__ == "__main__":
    main()
