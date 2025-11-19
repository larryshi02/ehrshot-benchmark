#!/usr/bin/env python3
"""
Azure-based forward reasoning pipeline for unsupervised fine-tuning across multiple tasks.

Key features:
  - Generates unsupervised reasoning traces (no label exposure) for train and val sets
  - Uses the same prompt as evaluation (forward reasoning, no ground truth)
  - Creates balanced train set: 1 negative per (patient_id, label_time) and matching positives
  - Creates val set same as backwards reasoning: 100 pos, 100 neg with 3 traces each
    (or 50 pos, 50 neg with 6 traces for pancreatic_cancer)
  - Uses prompt caching (cached inputs) to save costs
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any

from loguru import logger
from openai import AzureOpenAI


class AzureForwardReasoningConfig:
    """Configuration for Azure-based forward reasoning trace generation."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        deployment: str = "gpt-5-mini",
        model: str = "gpt-5-mini",
        max_completion_tokens: int = 4096,
        temperature: float = 1,
        top_p: float = 1.0,
    ):
        config_file = os.path.join(os.path.dirname(__file__), "azure_config.json")

        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
            self.endpoint = endpoint or config.get(
                "endpoint", "https://ehrshot.openai.azure.com/"
            )
            self.api_key = api_key or config.get("api_key", os.getenv("AZURE_OPENAI_API_KEY"))
            self.api_version = api_version or config.get("api_version", "2024-12-01-preview")
            self.deployment = deployment or config.get("deployment", "gpt-5-mini")
            self.model = model or config.get("model", "gpt-5-mini")
        else:
            self.endpoint = endpoint or os.getenv(
                "AZURE_OPENAI_ENDPOINT", "https://ehrshot.openai.azure.com/"
            )
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.api_version = api_version or os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            )
            self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
            self.model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-5-mini")

        if not self.api_key or self.api_key == "your-api-key-here":
            raise ValueError(
                "Azure OpenAI API key not found! Please set AZURE_OPENAI_API_KEY or create azure_config.json"
            )

        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p


class SFTExample:
    """Data structure for supervised fine-tuning examples."""

    def __init__(
        self,
        patient_id: int,
        label_time: str,
        label_value: bool,
        input_text: str,
        target_text: str,
        task_instruction: str,
        forward_reasoning: Optional[str] = None,
        final_prediction: Optional[str] = None,
        forward_prompt: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ):
        self.patient_id = patient_id
        self.label_time = label_time
        self.label_value = label_value
        self.input_text = input_text
        self.target_text = target_text
        self.task_instruction = task_instruction
        self.forward_reasoning = forward_reasoning
        self.final_prediction = final_prediction
        self.forward_prompt = forward_prompt
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "label_time": self.label_time,
            "label_value": self.label_value,
            "input_text": self.input_text,
            "target_text": self.target_text,
            "task_instruction": self.task_instruction,
            "forward_reasoning": self.forward_reasoning,
            "final_prediction": self.final_prediction,
            "forward_prompt": self.forward_prompt,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class AzureForwardReasoningGenerator:
    """Generates forward reasoning traces using Azure OpenAI (no label exposure)."""

    def __init__(self, config: AzureForwardReasoningConfig):
        self.config = config
        self.client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
        )

    def _create_forward_prompt(
        self,
        input_text: str,
        task_instruction: str,
    ) -> str:
        """
        Create forward reasoning prompt (no label exposure) - same as evaluation script.
        
        Note: While we could optimize for prompt caching by putting static content first,
        we match the evaluation script exactly for consistency.
        """
        return (
            f"Below is a patient's electronic healthcare record (EHR) in Markdown format. "
            f"Please answer the following query using clinical reasoning: {task_instruction}. "
            f"Patient Medical History:\n\n{input_text}\n\n"
            f"Demonstrate your reasoning step by step, and then provide your final answer in the format \"Final Answer: [Positive/Negative]<STOP>\".\n\n"
        )

    def _generate_response(
        self, prompt: str
    ) -> Tuple[str, Dict[str, Optional[int]]]:
        """
        Generate response using prompt caching (cached inputs).
        
        Prompt caching is automatic when prompts share a common prefix (>=1024 tokens).
        By structuring prompts with static content first and dynamic content last,
        OpenAI can cache the static portion, reducing costs.
        """
        try:
            # Build base parameters
            completion_params = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical reasoning assistant that delivers accurate, "
                            "well-structured medical assessments."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": self.config.max_completion_tokens,
                "temperature": self.config.temperature,
                "model": self.config.deployment,
            }
            
            # Prompt caching (cached inputs) is automatic - no parameter needed
            # OpenAI automatically caches prompts >=1024 tokens with common prefixes
            
            # Only include top_p if the model supports it (some models like gpt-5-mini don't)
            if "gpt-5" not in self.config.deployment.lower():
                completion_params["top_p"] = self.config.top_p
            
            response = self.client.chat.completions.create(**completion_params)

            content = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            }

            return content, usage
        except Exception as exc:
            logger.error(f"Error generating response: {exc}")
            return f"Error: {exc}", {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }

    def generate_forward_reasoning(
        self, example: SFTExample
    ) -> SFTExample:
        """
        Generate forward reasoning trace with prompt caching optimization.
        
        The prompt is structured with static content first (which can be cached)
        and dynamic patient-specific content last. This enables OpenAI's automatic
        prompt caching to reduce costs on the cached portion.
        """
        prompt = self._create_forward_prompt(
            example.input_text,
            example.task_instruction,
        )

        reasoning_response, reasoning_usage = self._generate_response(prompt)

        example.forward_reasoning = reasoning_response
        # Extract final prediction from response
        example.final_prediction = self._extract_prediction(reasoning_response)
        example.forward_prompt = prompt
        example.prompt_tokens = reasoning_usage["prompt_tokens"] or 0
        example.completion_tokens = reasoning_usage["completion_tokens"] or 0
        example.total_tokens = reasoning_usage["total_tokens"] or 0

        return example

    @staticmethod
    def _extract_prediction(response_text: str) -> Optional[str]:
        """Extract final prediction from response text."""
        response_lower = response_text.lower()
        
        # Look for "Final Answer: [Positive/Negative]" pattern
        if "final answer" in response_lower:
            # Split on "final answer" and take everything after it
            final_answer = response_lower.split("final answer")[-1].strip()
            # Remove any leading colons, dashes, brackets, or whitespace
            final_answer = final_answer.lstrip(":[]- ").strip()
            
            # Check if "positive" or "negative" appears anywhere in the substring
            if "positive" in final_answer:
                return "Positive"
            elif "negative" in final_answer:
                return "Negative"
        
        return None


TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "val"

DEFAULT_TASKS: Tuple[str, ...] = (
    "acute_mi",
    "hypertension",
    "hyperlipidemia",
    "pancreatic_cancer",
)

TASK_PREDICTION_QUERIES: Dict[str, str] = {
    "acute_mi": "Will the patient develop an acute myocardial infarction in the next year?",
    "hypertension": "Will the patient develop hypertension in the next year?",
    "hyperlipidemia": "Will the patient develop hyperlipidemia in the next year?",
    "pancreatic_cancer": "Will the patient develop pancreatic cancer in the next year?",
}


@dataclass
class TaskProcessingStats:
    """Simple container for reporting per-task statistics."""

    task_name: str
    num_train_examples: int
    num_val_examples: int
    total_tokens: int


def load_serialized_examples(
    serialized_data_path: str,
    *,
    split: str = TRAIN_SPLIT_NAME,
) -> List[dict]:
    """
    Load serialized examples for a specific split from the given JSON file.

    Args:
        serialized_data_path: Absolute path to the serialized JSON file.
        split: Which split to keep (default: train).

    Returns:
        List of serialized example dicts filtered by the requested split.
    """
    with open(serialized_data_path, "r") as f:
        all_examples = json.load(f)

    filtered_examples = [
        example for example in all_examples if example.get("split") == split
    ]

    logger.info(
        f"Loaded {len(filtered_examples)} {split} examples from {serialized_data_path}"
    )
    return filtered_examples


def build_sft_example(
    serialized_example: dict,
    task_instruction: str,
) -> SFTExample:
    """
    Convert a serialized example into the SFTExample structure used by the pipeline.
    """
    label_value = serialized_example.get("label_value", False)
    target_text = "Positive" if label_value else "Negative"

    return SFTExample(
        patient_id=serialized_example["patient_id"],
        label_time=serialized_example["label_time"],
        label_value=label_value,
        input_text=serialized_example["context"],
        target_text=target_text,
        task_instruction=task_instruction,
    )


def create_balanced_train_set(
    train_examples: List[SFTExample],
) -> Tuple[List[SFTExample], List[SFTExample], int]:
    """
    Create a balanced train set: 1 negative per (patient_id, label_time) and multiple traces per positive.
    
    Strategy:
    1. Group examples by (patient_id, label_time)
    2. For each group, take 1 negative (if available)
    3. Count total negatives across all groups
    4. Use all positive examples, calculate how many traces per positive to match total negatives
    
    Returns:
        Tuple of (negatives list, positives list, traces_per_positive)
    """
    # Group by (patient_id, label_time)
    grouped: Dict[Tuple[int, str], Dict[str, List[SFTExample]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )
    
    for example in train_examples:
        key = (example.patient_id, example.label_time)
        if example.label_value:
            grouped[key]["positive"].append(example)
        else:
            grouped[key]["negative"].append(example)
    
    # Collect 1 negative per (patient_id, label_time)
    negatives = []
    all_positives = []
    
    for (patient_id, label_time), groups in grouped.items():
        if groups["negative"]:
            # Take 1 negative per group
            negatives.append(groups["negative"][0])
        # Collect all positives
        all_positives.extend(groups["positive"])
    
    total_negatives = len(negatives)
    
    # Calculate how many traces per positive to generate
    if len(all_positives) == 0:
        traces_per_positive = 0
    else:
        # Round to nearest integer
        traces_per_positive = round(total_negatives / len(all_positives))
    
    logger.info(
        f"Created balanced train set: {total_negatives} negatives, {len(all_positives)} positives "
        f"(will generate {traces_per_positive} traces per positive = {len(all_positives) * traces_per_positive} total positive traces)"
    )
    
    return negatives, all_positives, traces_per_positive


class BalancedMultiTaskForwardReasoningPipeline:
    """Generates balanced forward reasoning traces for training and validation sets."""

    def __init__(
        self,
        config: AzureForwardReasoningConfig,
        serialized_data_dir: str,
        task_instructions: Dict[str, str],
        max_workers: int = 1,
        random_seed: int = 42,
    ):
        self.config = config
        self.serialized_data_dir = serialized_data_dir
        self.task_instructions = task_instructions
        self.max_workers = max_workers
        self.random_seed = random_seed
        random.seed(random_seed)
        # Thread locks for safe file writing
        self._file_locks: Dict[str, threading.Lock] = {}

    def _resolve_task_instruction(self, task_name: str) -> str:
        if task_name in TASK_PREDICTION_QUERIES:
            return TASK_PREDICTION_QUERIES[task_name]

        if task_name in self.task_instructions:
            return self.task_instructions[task_name]

        raise KeyError(
            f"No instruction mapping found for task '{task_name}'. "
            f"Known tasks: {sorted(TASK_PREDICTION_QUERIES.keys())}"
        )

    def _load_examples_for_task(self, task_name: str, split: str) -> List[SFTExample]:
        serialized_filename = f"{task_name}_all_splits.json"
        serialized_path = os.path.join(self.serialized_data_dir, serialized_filename)

        if not os.path.exists(serialized_path):
            raise FileNotFoundError(
                f"Serialized data file not found for task '{task_name}': {serialized_path}"
            )

        serialized_examples = load_serialized_examples(serialized_path, split=split)

        instruction = self._resolve_task_instruction(task_name)
        return [build_sft_example(example, instruction) for example in serialized_examples]

    def process_task(
        self,
        task_name: str,
        *,
        train_output_file: Optional[str] = None,
        train_sft_dataset_file: Optional[str] = None,
        val_output_file: Optional[str] = None,
        val_sft_dataset_file: Optional[str] = None,
    ) -> TaskProcessingStats:
        """
        Generate forward reasoning traces for training and validation examples.
        """
        logger.info(f"=== Processing task: {task_name} ===")

        # Load train examples and create balanced set
        train_examples = self._load_examples_for_task(task_name, TRAIN_SPLIT_NAME)
        train_negatives, train_positives, traces_per_positive = create_balanced_train_set(train_examples)
        
        logger.info(
            f"[{task_name}] Balanced train set: {len(train_negatives)} negatives, "
            f"{len(train_positives)} positives (generating {traces_per_positive} traces per positive)"
        )

        # Load validation examples
        val_examples = self._load_examples_for_task(task_name, VAL_SPLIT_NAME)
        val_positive = [e for e in val_examples if e.label_value]
        val_negative = [e for e in val_examples if not e.label_value]
        
        logger.info(
            f"[{task_name}] Validation set: {len(val_positive)} positive, {len(val_negative)} negative"
        )

        # Sample validation examples (same as backwards reasoning)
        if task_name == "pancreatic_cancer":
            val_sample_pos = min(50, len(val_positive))
            val_sample_neg = min(50, len(val_negative))
            traces_per_val_example = 6
        else:
            val_sample_pos = min(100, len(val_positive))
            val_sample_neg = min(100, len(val_negative))
            traces_per_val_example = 3

        if len(val_positive) < val_sample_pos:
            logger.warning(
                f"[{task_name}] Only {len(val_positive)} positive validation examples available, "
                f"using all of them instead of {val_sample_pos}"
            )
            val_sample_pos = len(val_positive)
        
        if len(val_negative) < val_sample_neg:
            logger.warning(
                f"[{task_name}] Only {len(val_negative)} negative validation examples available, "
                f"using all of them instead of {val_sample_neg}"
            )
            val_sample_neg = len(val_negative)

        sampled_val_positive = random.sample(val_positive, val_sample_pos) if val_positive else []
        sampled_val_negative = random.sample(val_negative, val_sample_neg) if val_negative else []
        
        logger.info(
            f"[{task_name}] Sampled {len(sampled_val_positive)} positive and "
            f"{len(sampled_val_negative)} negative validation examples, "
            f"generating {traces_per_val_example} traces per example"
        )

        # Initialize file locks
        for output_file in [train_output_file, train_sft_dataset_file, val_output_file, val_sft_dataset_file]:
            if output_file and output_file not in self._file_locks:
                self._file_locks[output_file] = threading.Lock()

        # Process training examples
        train_stats = self._process_training_samples(
            task_name,
            train_negatives,
            train_positives,
            traces_per_positive,
            train_output_file,
            train_sft_dataset_file,
        )

        # Process validation examples (both positive and negative, multiple traces each)
        val_stats = self._process_validation_samples(
            task_name,
            sampled_val_positive + sampled_val_negative,
            traces_per_val_example,
            val_output_file,
            val_sft_dataset_file,
        )

        total_train_examples = len(train_negatives) + (len(train_positives) * traces_per_positive)
        return TaskProcessingStats(
            task_name=task_name,
            num_train_examples=total_train_examples,
            num_val_examples=(len(sampled_val_positive) + len(sampled_val_negative)) * traces_per_val_example,
            total_tokens=train_stats + val_stats,
        )

    def _process_training_samples(
        self,
        task_name: str,
        negatives: List[SFTExample],
        positives: List[SFTExample],
        traces_per_positive: int,
        output_file: Optional[str],
        sft_dataset_file: Optional[str],
    ) -> int:
        """Process training examples: 1 trace per negative, multiple traces per positive."""
        if not negatives and not positives:
            logger.info(f"[{task_name}] No training examples to process")
            return 0

        if self.max_workers > 1:
            return self._process_training_parallel(
                task_name, negatives, positives, traces_per_positive, output_file, sft_dataset_file
            )
        else:
            return self._process_training_sequential(
                task_name, negatives, positives, traces_per_positive, output_file, sft_dataset_file
            )

    def _process_training_sequential(
        self,
        task_name: str,
        negatives: List[SFTExample],
        positives: List[SFTExample],
        traces_per_positive: int,
        output_file: Optional[str],
        sft_dataset_file: Optional[str],
    ) -> int:
        """Process training examples sequentially: 1 trace per negative, multiple traces per positive."""
        total_tokens = 0
        generator = AzureForwardReasoningGenerator(self.config)
        total_traces = len(negatives) + (len(positives) * traces_per_positive)
        trace_count = 0

        # Process negatives (1 trace each)
        for idx, example in enumerate(negatives):
            trace_count += 1
            logger.info(
                f"[{task_name}] Generating training trace {trace_count}/{total_traces} "
                f"(Patient {example.patient_id}, negative)"
            )

            trace_example = generator.generate_forward_reasoning(example)

            if trace_example.total_tokens:
                total_tokens += trace_example.total_tokens

            if output_file:
                is_first = trace_count == 1
                self.save_sft_traces_incremental(trace_example, output_file, is_first=is_first)

            if sft_dataset_file:
                is_first = trace_count == 1
                self.save_sft_dataset_incremental(
                    trace_example, sft_dataset_file, is_first=is_first
                )

        # Process positives (multiple traces each)
        for pos_idx, example in enumerate(positives):
            for trace_idx in range(traces_per_positive):
                trace_count += 1
                logger.info(
                    f"[{task_name}] Generating training trace {trace_count}/{total_traces} "
                    f"(Patient {example.patient_id}, positive, trace {trace_idx + 1}/{traces_per_positive})"
                )

                trace_example = generator.generate_forward_reasoning(example)

                if trace_example.total_tokens:
                    total_tokens += trace_example.total_tokens

                if output_file:
                    is_first = trace_count == 1
                    self.save_sft_traces_incremental(trace_example, output_file, is_first=is_first)

                if sft_dataset_file:
                    is_first = trace_count == 1
                    self.save_sft_dataset_incremental(
                        trace_example, sft_dataset_file, is_first=is_first
                    )

        logger.info(
            f"[{task_name}] Generated {total_traces} training traces ({len(negatives)} negatives, "
            f"{len(positives) * traces_per_positive} positives). Total tokens: {total_tokens}"
        )
        return total_tokens

    def _process_training_parallel(
        self,
        task_name: str,
        negatives: List[SFTExample],
        positives: List[SFTExample],
        traces_per_positive: int,
        output_file: Optional[str],
        sft_dataset_file: Optional[str],
    ) -> int:
        """Process training examples in parallel: 1 trace per negative, multiple traces per positive."""
        total_tokens = 0
        completed_count = 0
        tokens_lock = threading.Lock()

        # Create all tasks: 1 per negative, traces_per_positive per positive
        tasks = []
        for example in negatives:
            tasks.append((example, 0, "negative"))  # (example, trace_idx, label_type)
        for example in positives:
            for trace_idx in range(traces_per_positive):
                tasks.append((example, trace_idx, "positive"))

        total_tasks = len(tasks)

        def process_single_trace(task_tuple: Tuple[SFTExample, int, str]) -> Tuple[SFTExample, int, str]:
            example, trace_idx, label_type = task_tuple
            generator = AzureForwardReasoningGenerator(self.config)
            return generator.generate_forward_reasoning(example), trace_idx, label_type

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(process_single_trace, task): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                try:
                    trace_example, trace_idx, label_type = future.result()
                    completed_count += 1

                    if trace_example.total_tokens:
                        with tokens_lock:
                            total_tokens += trace_example.total_tokens

                    logger.info(
                        f"[{task_name}] Training progress: {completed_count}/{total_tasks} "
                        f"(Total tokens: {total_tokens})"
                    )

                    is_first = completed_count == 1
                    if output_file:
                        with self._file_locks[output_file]:
                            self.save_sft_traces_incremental(
                                trace_example, output_file, is_first=is_first
                            )

                    if sft_dataset_file:
                        with self._file_locks[sft_dataset_file]:
                            self.save_sft_dataset_incremental(
                                trace_example, sft_dataset_file, is_first=is_first
                            )

                except Exception as exc:
                    task = future_to_task[future]
                    logger.error(
                        f"[{task_name}] Training task failed (Patient {task[0].patient_id}, "
                        f"{task[2]}, trace {task[1]}): {exc}"
                    )

        logger.info(
            f"[{task_name}] Generated {completed_count}/{total_tasks} training traces "
            f"({len(negatives)} negatives, {len(positives) * traces_per_positive} positives). "
            f"Total tokens: {total_tokens}"
        )
        return total_tokens

    def _process_validation_samples(
        self,
        task_name: str,
        val_examples: List[SFTExample],
        traces_per_example: int,
        output_file: Optional[str],
        sft_dataset_file: Optional[str],
    ) -> int:
        """Process validation examples, generating multiple traces per example."""
        if not val_examples:
            logger.info(f"[{task_name}] No validation examples to process")
            return 0

        total_tokens = 0
        total_generated = 0

        # Create all tasks: for each validation example, generate traces_per_example traces
        tasks = []
        for example in val_examples:
            for trace_idx in range(traces_per_example):
                tasks.append((example, trace_idx))

        if self.max_workers > 1:
            return self._process_validation_parallel(
                task_name, tasks, output_file, sft_dataset_file
            )
        else:
            return self._process_validation_sequential(
                task_name, tasks, output_file, sft_dataset_file
            )

    def _process_validation_sequential(
        self,
        task_name: str,
        tasks: List[Tuple[SFTExample, int]],
        output_file: Optional[str],
        sft_dataset_file: Optional[str],
    ) -> int:
        """Process validation examples sequentially."""
        total_tokens = 0
        generator = AzureForwardReasoningGenerator(self.config)

        for idx, (example, trace_idx) in enumerate(tasks):
            logger.info(
                f"[{task_name}] Generating validation trace {idx + 1}/{len(tasks)} "
                f"(Patient {example.patient_id}, trace {trace_idx + 1})"
            )

            trace_example = SFTExample(
                patient_id=example.patient_id,
                label_time=example.label_time,
                label_value=example.label_value,
                input_text=example.input_text,
                target_text=example.target_text,
                task_instruction=example.task_instruction,
            )

            trace_example = generator.generate_forward_reasoning(trace_example)

            if trace_example.total_tokens:
                total_tokens += trace_example.total_tokens

            if output_file:
                is_first = idx == 0
                self.save_sft_traces_incremental(trace_example, output_file, is_first=is_first)

            if sft_dataset_file:
                is_first = idx == 0
                self.save_sft_dataset_incremental(
                    trace_example, sft_dataset_file, is_first=is_first
                )

        logger.info(
            f"[{task_name}] Generated {len(tasks)} validation traces. Total tokens: {total_tokens}"
        )
        return total_tokens

    def _process_validation_parallel(
        self,
        task_name: str,
        tasks: List[Tuple[SFTExample, int]],
        output_file: Optional[str],
        sft_dataset_file: Optional[str],
    ) -> int:
        """Process validation examples in parallel."""
        total_tokens = 0
        completed_count = 0
        total_tasks = len(tasks)
        tokens_lock = threading.Lock()

        def process_single_trace(task_tuple: Tuple[SFTExample, int]) -> Tuple[SFTExample, int]:
            example, trace_idx = task_tuple
            generator = AzureForwardReasoningGenerator(self.config)

            trace_example = SFTExample(
                patient_id=example.patient_id,
                label_time=example.label_time,
                label_value=example.label_value,
                input_text=example.input_text,
                target_text=example.target_text,
                task_instruction=example.task_instruction,
            )

            trace_example = generator.generate_forward_reasoning(trace_example)
            return trace_example, trace_idx

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(process_single_trace, task): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                try:
                    trace_example, trace_idx = future.result()
                    completed_count += 1

                    if trace_example.total_tokens:
                        with tokens_lock:
                            total_tokens += trace_example.total_tokens

                    logger.info(
                        f"[{task_name}] Validation progress: {completed_count}/{total_tasks} "
                        f"(Total tokens: {total_tokens})"
                    )

                    is_first = completed_count == 1
                    if output_file:
                        with self._file_locks[output_file]:
                            self.save_sft_traces_incremental(
                                trace_example, output_file, is_first=is_first
                            )

                    if sft_dataset_file:
                        with self._file_locks[sft_dataset_file]:
                            self.save_sft_dataset_incremental(
                                trace_example, sft_dataset_file, is_first=is_first
                            )

                except Exception as exc:
                    task = future_to_task[future]
                    logger.error(
                        f"[{task_name}] Validation task failed (Patient {task[0].patient_id}, "
                        f"trace {task[1]}): {exc}"
                    )

        logger.info(
            f"[{task_name}] Generated {completed_count}/{total_tasks} validation traces. "
            f"Total tokens: {total_tokens}"
        )
        return total_tokens

    @staticmethod
    def save_sft_traces_incremental(
        example: SFTExample, output_file: str, *, is_first: bool = False
    ) -> None:
        """Save SFT trace incrementally. Thread-safe when called with a lock."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if is_first:
            with open(output_file, "w") as f:
                json.dump([example.to_dict()], f, indent=2)
            return

        try:
            with open(output_file, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(example.to_dict())

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def save_sft_dataset_incremental(
        example: SFTExample, output_file: str, *, is_first: bool = False
    ) -> None:
        """Save SFT dataset entry incrementally. Thread-safe when called with a lock."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        conversation = [
            {
                "role": "user",
                "content": (
                    f'''System: You may reason step by step, but your final output MUST be in the following format: "Final Answer: [Positive/Negative]<STOP>".\n\n
User: You are a helpful medical assistant. Below is a patient's electronic healthcare record (EHR) in Markdown format. Please answer the following query using clinically grounded reasoning: {example.task_instruction}. Patient Medical History:\n\n\n\n{example.input_text}\n\nIMPORTANT: You must conclude with your final prediction in the format "Final Answer: [Positive/Negative]<STOP>".\n\n Assistant: '''
                ),
            },
            {
                "role": "assistant",
                "content": example.forward_reasoning or "",
            },
        ]

        sft_entry = {
            "conversations": conversation,
            "patient_id": example.patient_id,
            "label_time": example.label_time,
            "label_value": example.label_value,
            "task_instruction": example.task_instruction,
        }

        if is_first:
            with open(output_file, "w") as f:
                json.dump([sft_entry], f, indent=2)
            return

        try:
            with open(output_file, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(sft_entry)

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Azure-based balanced forward reasoning pipeline for multiple tasks."
    )

    parser.add_argument(
        "--serialized_data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "serialized_multi_task_data"),
        help="Directory containing serialized *_all_splits.json files.",
    )
    parser.add_argument(
        "--task_to_instructions",
        type=str,
        default=os.path.join("ehrshot", "serialization", "task_to_instructions.json"),
        help="Path to task instruction mappings.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="List of task names to process sequentially.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data_gpt-5-mini_forward"),
        help="Base directory to save all output files. Creates train/ and val/ subdirectories.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for Azure OpenAI generation.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens to generate per response.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel workers for concurrent API calls (default: 1 = sequential). "
        "Increase for faster processing, but be mindful of rate limits.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for train/val set sampling.",
    )

    return parser.parse_args()


def load_task_instructions(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Task instruction JSON file not found: {path}")

    with open(path, "r") as f:
        instructions = json.load(f)

    logger.info(f"Loaded task instructions from {path}")
    return instructions


def build_output_path(
    base_dir: Optional[str], split: str, task_name: str, suffix: str
) -> Optional[str]:
    if not base_dir:
        return None
    split_dir = os.path.join(base_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    return os.path.join(split_dir, f"{task_name}_{suffix}.json")


def main() -> None:
    args = parse_args()

    config = AzureForwardReasoningConfig(
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    instructions = load_task_instructions(args.task_to_instructions)
    pipeline = BalancedMultiTaskForwardReasoningPipeline(
        config=config,
        serialized_data_dir=args.serialized_data_dir,
        task_instructions=instructions,
        max_workers=args.max_workers,
        random_seed=args.random_seed,
    )
    
    if args.max_workers > 1:
        logger.info(f"Using parallel processing with {args.max_workers} workers")
    else:
        logger.info("Using sequential processing")

    total_tokens_across_tasks = 0
    processed_stats: List[TaskProcessingStats] = []

    for task_name in args.tasks:
        train_output_file = build_output_path(
            args.output_dir, "train_all", task_name, "forward_reasoning"
        )
        train_sft_file = build_output_path(
            args.output_dir, "train_all", task_name, "forward_sft_dataset"
        )
        val_output_file = build_output_path(
            args.output_dir, "val", task_name, "forward_reasoning"
        )
        val_sft_file = build_output_path(
            args.output_dir, "val", task_name, "forward_sft_dataset"
        )

        stats = pipeline.process_task(
            task_name,
            train_output_file=train_output_file,
            train_sft_dataset_file=train_sft_file,
            val_output_file=val_output_file,
            val_sft_dataset_file=val_sft_file,
        )

        total_tokens_across_tasks += stats.total_tokens
        processed_stats.append(stats)

    logger.info("=== Forward reasoning summary ===")
    for stats in processed_stats:
        logger.info(
            f"Task {stats.task_name}: "
            f"{stats.num_train_examples} training examples, "
            f"{stats.num_val_examples} validation examples, "
            f"{stats.total_tokens} tokens."
        )

    logger.info(f"Total tokens across all tasks: {total_tokens_across_tasks}")
    logger.info("Forward reasoning pipeline completed.")


if __name__ == "__main__":
    main()

