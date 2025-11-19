#!/usr/bin/env python3
"""
Azure-based backwards reasoning pipeline for supervised fine-tuning across multiple tasks.

Key differences from `azure_backwards_reasoning_pipeline.py`:
  - Processes all four benchmark tasks sequentially (acute mi, hypertension, hyperlipidemia, pancreatic cancer)
  - Loads pre-serialized EHR contexts from `serialized_multi_task_data` instead of querying FEMR directly
  - Generates backwards reasoning traces only for the train split of each task
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any

from loguru import logger
from openai import AzureOpenAI
class AzureBackwardsReasoningConfig:
    """Configuration for Azure-based backwards reasoning trace generation."""

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
        backwards_reasoning: Optional[List[str]] = None,
        final_prediction: Optional[str] = None,
        backwards_prompt: Optional[str] = None,
        prompt_tokens: Optional[List[int]] = None,
        completion_tokens: Optional[List[int]] = None,
        total_tokens: Optional[List[int]] = None,
    ):
        self.patient_id = patient_id
        self.label_time = label_time
        self.label_value = label_value
        self.input_text = input_text
        self.target_text = target_text
        self.task_instruction = task_instruction
        self.backwards_reasoning = backwards_reasoning or []
        self.final_prediction = final_prediction
        self.backwards_prompt = backwards_prompt
        self.prompt_tokens = prompt_tokens or []
        self.completion_tokens = completion_tokens or []
        self.total_tokens = total_tokens or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "label_time": self.label_time,
            "label_value": self.label_value,
            "input_text": self.input_text,
            "target_text": self.target_text,
            "task_instruction": self.task_instruction,
            "backwards_reasoning": self.backwards_reasoning,
            "final_prediction": self.final_prediction,
            "backwards_prompt": self.backwards_prompt,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class AzureBackwardsReasoningGenerator:
    """Generates backwards reasoning traces using Azure OpenAI."""

    def __init__(self, config: AzureBackwardsReasoningConfig):
        self.config = config
        self.client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
        )

    def _create_backwards_prompt(
        self,
        input_text: str,
        ground_truth: str,
        task_instruction: str,
    ) -> str:
        correct_answer = "Positive" if ground_truth == "Positive" else "Negative"

        return (
            "You are a medical expert tasked with explaining clinical reasoning. You have been"
            " given a patient's electronic healthcare record (EHR) in Markdown format and the"
            " correct prediction outcome. Your task is to provide a detailed, step-by-step"
            " reasoning process that explains why this outcome is correct, WITHOUT explicitly"
            " stating that you know the answer in advance.\n\n"
            "Focus on identifying the key clinical indicators, risk factors, and evidence from"
            " the patient's medical history that support the correct prediction. Structure your"
            " reasoning logically and provide specific examples from the patient data.\n\n"
            "Only after providing your reasoning, conclude with your final answer (Positive or"
            " Negative) in the following format:\n"
            "Final Answer: [Positive/Negative]<STOP>\n\n"
            f"Patient's Electronic Healthcare Record:\n{input_text}\n\n"
            f"Task: {task_instruction}\n\n"
            f"Correct Prediction: {correct_answer}\n\n"
            "Now, provide a detailed reasoning process that explains why this prediction is"
            " correct based on the clinical evidence:\n\n"
            "Reasoning:"
        )

    def _generate_response(self, prompt: str) -> Tuple[str, Dict[str, Optional[int]]]:
        try:
            # Build base parameters
            completion_params = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a medical expert AI assistant that provides detailed clinical"
                            " reasoning for medical predictions. You excel at analyzing patient data"
                            " and explaining clinical decision-making processes."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": self.config.max_completion_tokens,
                "temperature": self.config.temperature,
                "model": self.config.deployment,
            }
            
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

    def generate_backwards_reasoning(self, example: SFTExample, num_traces: int = 10) -> SFTExample:
        prompt = self._create_backwards_prompt(
            example.input_text,
            example.target_text,
            example.task_instruction,
        )

        # Generate multiple reasoning traces
        reasoning_responses = []
        prompt_tokens_list = []
        completion_tokens_list = []
        total_tokens_list = []

        for i in range(num_traces):
            reasoning_response, reasoning_usage = self._generate_response(prompt)
            reasoning_responses.append(reasoning_response)
            prompt_tokens_list.append(reasoning_usage["prompt_tokens"] or 0)
            completion_tokens_list.append(reasoning_usage["completion_tokens"] or 0)
            total_tokens_list.append(reasoning_usage["total_tokens"] or 0)

        example.backwards_reasoning = reasoning_responses
        example.final_prediction = example.target_text
        example.backwards_prompt = prompt
        example.prompt_tokens = prompt_tokens_list
        example.completion_tokens = completion_tokens_list
        example.total_tokens = total_tokens_list

        return example


TRAIN_SPLIT_NAME = "train"

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
    num_examples: int
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


class MultiTaskBackwardsReasoningPipeline:
    """Generates backwards reasoning traces sequentially across multiple clinical tasks."""

    def __init__(
        self,
        config: AzureBackwardsReasoningConfig,
        serialized_data_dir: str,
        task_instructions: Dict[str, str],
        max_workers: int = 1,
    ):
        self.config = config
        self.serialized_data_dir = serialized_data_dir
        self.task_instructions = task_instructions
        self.generator = AzureBackwardsReasoningGenerator(config)
        self.max_workers = max_workers
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

    def _load_train_examples_for_task(self, task_name: str) -> List[SFTExample]:
        serialized_filename = f"{task_name}_all_splits.json"
        serialized_path = os.path.join(self.serialized_data_dir, serialized_filename)

        if not os.path.exists(serialized_path):
            raise FileNotFoundError(
                f"Serialized data file not found for task '{task_name}': {serialized_path}"
            )

        serialized_examples = load_serialized_examples(
            serialized_path, split=TRAIN_SPLIT_NAME
        )

        instruction = self._resolve_task_instruction(task_name)
        return [build_sft_example(example, instruction) for example in serialized_examples]

    def process_task(
        self,
        task_name: str,
        *,
        max_examples: Optional[int] = None,
        patient_ids: Optional[Iterable[int]] = None,
        output_file: Optional[str] = None,
        sft_dataset_file: Optional[str] = None,
    ) -> TaskProcessingStats:
        """
        Generate backwards reasoning traces for a single task.
        """
        logger.info(f"=== Processing task: {task_name} ===")

        examples = self._load_train_examples_for_task(task_name)

        if patient_ids:
            patient_ids_set = {int(pid) for pid in patient_ids}
            before_filter = len(examples)
            examples = [
                example for example in examples if example.patient_id in patient_ids_set
            ]
            logger.info(
                f"Filtered examples by patient IDs: {before_filter} -> {len(examples)}"
            )

        if not examples:
            logger.warning(f"No train examples available for task '{task_name}'. Skipping.")
            return TaskProcessingStats(task_name, 0, 0)

        if max_examples is not None and max_examples < len(examples):
            logger.info(f"Limiting to first {max_examples} examples for task '{task_name}'.")
            examples = examples[:max_examples]

        # Initialize file locks for thread-safe writing
        if output_file:
            if output_file not in self._file_locks:
                self._file_locks[output_file] = threading.Lock()
        if sft_dataset_file:
            if sft_dataset_file not in self._file_locks:
                self._file_locks[sft_dataset_file] = threading.Lock()

        # Process examples in parallel if max_workers > 1
        if self.max_workers > 1:
            return self._process_task_parallel(
                task_name, examples, output_file, sft_dataset_file
            )
        else:
            return self._process_task_sequential(
                task_name, examples, output_file, sft_dataset_file
            )

    def _process_task_sequential(
        self,
        task_name: str,
        examples: List[SFTExample],
        output_file: Optional[str],
        sft_dataset_file: Optional[str],
    ) -> TaskProcessingStats:
        """Process examples sequentially (original implementation)."""
        total_tokens = 0
        for index, example in enumerate(examples):
            logger.info(
                f"[{task_name}] Generating reasoning for example {index + 1}/{len(examples)} "
                f"(Patient {example.patient_id})"
            )
            example = self.generator.generate_backwards_reasoning(example, num_traces=10)

            if example.total_tokens:
                example_total = sum(example.total_tokens)
                total_tokens += example_total
                logger.info(
                    f"[{task_name}] Example {index + 1} used {example_total} tokens "
                    f"(Task total: {total_tokens})"
                )

            if output_file:
                is_first = index == 0
                self.save_sft_traces_incremental(example, output_file, is_first=is_first)

            if sft_dataset_file:
                is_first = index == 0
                self.save_sft_dataset_incremental(
                    example, sft_dataset_file, is_first=is_first
                )

        logger.info(
            f"[{task_name}] Completed {len(examples)} examples. Total tokens used: {total_tokens}"
        )

        return TaskProcessingStats(task_name, len(examples), total_tokens)

    def _process_task_parallel(
        self,
        task_name: str,
        examples: List[SFTExample],
        output_file: Optional[str],
        sft_dataset_file: Optional[str],
    ) -> TaskProcessingStats:
        """Process examples in parallel using ThreadPoolExecutor."""
        total_tokens = 0
        completed_count = 0
        total_examples = len(examples)
        tokens_lock = threading.Lock()
        
        def process_single_example(example: SFTExample, index: int) -> Tuple[SFTExample, int]:
            """Process a single example and return it with its index."""
            logger.info(
                f"[{task_name}] Generating reasoning for example {index + 1}/{total_examples} "
                f"(Patient {example.patient_id})"
            )
            
            # Create a new generator instance for thread safety
            generator = AzureBackwardsReasoningGenerator(self.config)
            processed_example = generator.generate_backwards_reasoning(example, num_traces=10)
            
            if processed_example.total_tokens:
                example_total = sum(processed_example.total_tokens)
                logger.info(
                    f"[{task_name}] Example {index + 1} (Patient {example.patient_id}) "
                    f"used {example_total} tokens"
                )
            
            return processed_example, index

        # Process examples in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_example, example, idx): idx
                for idx, example in enumerate(examples)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    processed_example, original_index = future.result()
                    completed_count += 1
                    
                    # Update token count (thread-safe)
                    if processed_example.total_tokens:
                        example_total = sum(processed_example.total_tokens)
                        with tokens_lock:
                            total_tokens += example_total
                    
                    logger.info(
                        f"[{task_name}] Progress: {completed_count}/{total_examples} completed "
                        f"(Total tokens: {total_tokens})"
                    )
                    
                    # Save results (thread-safe)
                    is_first = completed_count == 1
                    if output_file:
                        with self._file_locks[output_file]:
                            self.save_sft_traces_incremental(
                                processed_example, output_file, is_first=is_first
                            )
                    
                    if sft_dataset_file:
                        with self._file_locks[sft_dataset_file]:
                            self.save_sft_dataset_incremental(
                                processed_example, sft_dataset_file, is_first=is_first
                            )
                            
                except Exception as exc:
                    original_index = future_to_index[future]
                    logger.error(
                        f"[{task_name}] Example {original_index + 1} failed: {exc}"
                    )

        logger.info(
            f"[{task_name}] Completed {completed_count}/{total_examples} examples. "
            f"Total tokens used: {total_tokens}"
        )

        return TaskProcessingStats(task_name, completed_count, total_tokens)

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

        # Create one SFT entry for each reasoning trace
        sft_entries = []
        for reasoning in example.backwards_reasoning:
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
                    "content": reasoning or "",
                },
            ]

            sft_entry = {
                "conversations": conversation,
                "patient_id": example.patient_id,
                "label_time": example.label_time,
                "label_value": example.label_value,
                "task_instruction": example.task_instruction,
            }
            sft_entries.append(sft_entry)

        if is_first:
            with open(output_file, "w") as f:
                json.dump(sft_entries, f, indent=2)
            return

        try:
            with open(output_file, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.extend(sft_entries)

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Azure-based backwards reasoning pipeline for multiple tasks (train split only)."
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
        "--patient_ids",
        nargs="+",
        type=int,
        help="Optional list of patient IDs to filter (applied per task).",
    )
    parser.add_argument(
        "--max_examples_per_task",
        type=int,
        help="Limit the number of examples processed per task.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data_gpt-5-mini"),
        help="Directory to save backwards reasoning traces (per-task JSON files).",
    )
    parser.add_argument(
        "--sft_dataset_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data_gpt-5-mini"),
        help="Directory to save SFT dataset entries (per-task JSON files).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
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

    return parser.parse_args()


def load_task_instructions(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Task instruction JSON file not found: {path}")

    with open(path, "r") as f:
        instructions = json.load(f)

    logger.info(f"Loaded task instructions from {path}")
    return instructions


def build_output_path(
    base_dir: Optional[str], task_name: str, suffix: str
) -> Optional[str]:
    if not base_dir:
        return None
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{task_name}_{suffix}.json")


def main() -> None:
    args = parse_args()

    config = AzureBackwardsReasoningConfig(
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    instructions = load_task_instructions(args.task_to_instructions)
    pipeline = MultiTaskBackwardsReasoningPipeline(
        config=config,
        serialized_data_dir=args.serialized_data_dir,
        task_instructions=instructions,
        max_workers=args.max_workers,
    )
    
    if args.max_workers > 1:
        logger.info(f"Using parallel processing with {args.max_workers} workers")
    else:
        logger.info("Using sequential processing")

    total_tokens_across_tasks = 0
    processed_stats: List[TaskProcessingStats] = []

    for task_name in args.tasks:
        task_output_file = build_output_path(args.output_dir, task_name, "backwards_reasoning_large")
        task_sft_file = build_output_path(args.sft_dataset_dir, task_name, "train_sft_large")

        stats = pipeline.process_task(
            task_name,
            max_examples=args.max_examples_per_task,
            patient_ids=args.patient_ids,
            output_file=task_output_file,
            sft_dataset_file=task_sft_file,
        )

        total_tokens_across_tasks += stats.total_tokens
        processed_stats.append(stats)

    logger.info("=== Multi-task backwards reasoning summary ===")
    for stats in processed_stats:
        logger.info(
            f"Task {stats.task_name}: {stats.num_examples} examples, {stats.total_tokens} tokens."
        )

    logger.info(f"Total tokens across all tasks: {total_tokens_across_tasks}")
    logger.info("Multi-task backwards reasoning pipeline completed.")


if __name__ == "__main__":
    main()

