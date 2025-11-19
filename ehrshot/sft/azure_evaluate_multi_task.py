#!/usr/bin/env python3
"""
Multi-task evaluation pipeline that mirrors `evaluate_base_model_multi_task.py`
but uses Azure OpenAI (GPT-5 deployment) instead of a locally hosted Qwen model.

Key differences:
  - Loads pre-serialized EHR contexts from `serialized_multi_task_data`.
  - Evaluates only the test split for each task.
  - Uses Azure Chat Completions (sequential calls; no batching).
  - Supports multiple samples per patient via repeated calls (optional).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from openai import AzureOpenAI
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from azure_backwards_reasoning_multi_task_pipeline import (
    AzureBackwardsReasoningConfig,
)


DEFAULT_TASKS: Tuple[str, ...] = (
    "acute_mi",
    "hyperlipidemia",
    "hypertension",
    "pancreatic_cancer",
)

TASK_PREDICTION_QUERIES: Dict[str, str] = {
    "acute_mi": "Will the patient develop an acute myocardial infarction in the next year?",
    "hyperlipidemia": "Will the patient develop hyperlipidemia in the next year?",
    "hypertension": "Will the patient develop hypertension in the next year?",
    "pancreatic_cancer": "Will the patient develop pancreatic cancer in the next year?",
}


@dataclass
class EvaluationSample:
    patient_id: int
    label_time: str
    ground_truth: bool
    prompt: str


@dataclass
class PredictionRecord:
    patient_id: int
    label_time: str
    ground_truth: int
    sample_predictions: List[int]
    sample_responses: List[str]
    sample_token_usage: List[Dict[str, int]]
    score: float


class AzureGPT5Evaluator:
    """Evaluates Azure GPT-5 on the benchmark tasks using serialized test data."""

    def __init__(
        self,
        config: AzureBackwardsReasoningConfig,
        *,
        temperature: float,
        max_completion_tokens: int,
        num_samples: int,
        max_workers: int = 1,
    ):
        self.config = config
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.num_samples = num_samples
        self.max_workers = max_workers

        self.client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
        )
        
        # Thread-safe file locks for incremental writing
        self._file_locks: Dict[str, threading.Lock] = {}

    @staticmethod
    def load_test_examples(serialized_dir: str, task_name: str) -> List[dict]:
        path = os.path.join(serialized_dir, f"{task_name}_all_splits.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Serialized data file not found for task '{task_name}': {path}"
            )

        with open(path, "r") as f:
            data = json.load(f)

        test_examples = [row for row in data if row.get("split") == "test"]
        logger.info(f"[{task_name}] Loaded {len(test_examples)} test examples from {path}")
        return test_examples

    @staticmethod
    def filter_by_patients(
        examples: List[dict],
        patient_ids: Optional[Iterable[int]],
    ) -> List[dict]:
        if not patient_ids:
            return examples
        patient_set = {int(pid) for pid in patient_ids}
        filtered = [example for example in examples if example["patient_id"] in patient_set]
        logger.info(
            f"Filtered to {len(filtered)}/{len(examples)} examples after applying patient ID filter."
        )
        return filtered

    @staticmethod
    def build_prompt(task_name: str, context: str) -> str:
        query = TASK_PREDICTION_QUERIES.get(
            task_name,
            f"Will the patient develop {task_name.replace('_', ' ')} in the next year?",
        )
        
        
        return (
            f"Below is a patient's electronic healthcare record (EHR) in Markdown format. "
            f"Please answer the following query using clinical reasoning: {query}. "
            f"Patient Medical History:\n\n{context}\n\n"
            f"Demonstrate your reasoning step by step, and then provide your final answer in the format \"Final Answer: [Positive/Negative]\".\n\n"
            
        )

    def _make_chat_completion(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        # Build base parameters
        completion_params = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are a clinical reasoning assistant that delivers accurate, "
                        f"well-structured medical assessments."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "model": self.config.deployment,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }
        
        response = self.client.chat.completions.create(**completion_params)

        message = response.choices[0].message.content.strip()
        usage = response.usage or None
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        return message, usage_dict

    @staticmethod
    def parse_prediction(response_text: str) -> Optional[int]:
        """Extract binary prediction by looking for 'final answer' and checking for positive/negative."""
        response_lower = response_text.lower()
        
        # Look for "Final Answer: [Positive/Negative]" pattern
        if "final answer" in response_lower:
            # Split on "final answer" and take everything after it
            final_answer = response_lower.split("final answer")[-1].strip()
            # Remove any leading colons, dashes, brackets, or whitespace
            final_answer = final_answer.lstrip(":[]- ").strip()
            
            # Check if "positive" or "negative" appears anywhere in the substring
            if "positive" in final_answer:
                return 1
            elif "negative" in final_answer:
                return 0
        
        return None

    def evaluate_example(self, sample: EvaluationSample) -> PredictionRecord:
        responses: List[str] = []
        predictions: List[int] = []
        usages: List[Dict[str, int]] = []

        for attempt in range(self.num_samples):
            try:
                response, usage = self._make_chat_completion(sample.prompt)
            except Exception as exc:
                logger.error(
                    f"Error during Azure completion (patient {sample.patient_id}, attempt {attempt + 1}): {exc}"
                )
                responses.append(f"Error: {exc}")
                usages.append({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                predictions.append(None)  # type: ignore[arg-type]
                continue

            responses.append(response)
            usages.append(usage)
            prediction = self.parse_prediction(response)
            if prediction is None:
                logger.warning(
                    f"Could not parse final answer for patient {sample.patient_id}. "
                    "Defaulting to None for this sample."
                )
            predictions.append(prediction)  # type: ignore[arg-type]

        valid_predictions = [p for p in predictions if p is not None]
        score = (
            float(np.mean(valid_predictions)) if valid_predictions else 0.5
        )

        return PredictionRecord(
            patient_id=sample.patient_id,
            label_time=sample.label_time,
            ground_truth=1 if sample.ground_truth else 0,
            sample_predictions=[p if p is not None else -1 for p in predictions],
            sample_responses=responses,
            sample_token_usage=usages,
            score=score,
        )

    def _load_existing_records(self, output_dir: str, task_name: str) -> List[PredictionRecord]:
        """Load existing records from JSON file if it exists (for resuming)."""
        detailed_path = os.path.join(output_dir, f"{task_name}_detailed_results.json")
        if not os.path.exists(detailed_path):
            return []
        
        try:
            with open(detailed_path, "r") as f:
                existing_data = json.load(f)
            
            records = []
            for item in existing_data:
                record = PredictionRecord(
                    patient_id=item["patient_id"],
                    label_time=item["label_time"],
                    ground_truth=item["ground_truth"],
                    sample_predictions=item["sample_predictions"],
                    sample_responses=item["sample_responses"],
                    sample_token_usage=item["sample_token_usage"],
                    score=item["score"],
                )
                records.append(record)
            
            logger.info(f"[{task_name}] Loaded {len(records)} existing records from {detailed_path}")
            return records
        except Exception as exc:
            logger.warning(f"[{task_name}] Failed to load existing records: {exc}. Starting fresh.")
            return []

    def _reset_output_file(self, output_dir: str, task_name: str) -> None:
        """Reset/clear the output JSON file before starting evaluation."""
        os.makedirs(output_dir, exist_ok=True)
        detailed_path = os.path.join(output_dir, f"{task_name}_detailed_results.json")
        with open(detailed_path, "w") as f:
            json.dump([], f, indent=2)
        logger.info(f"[{task_name}] Reset output file: {detailed_path}")

    def _append_record_incrementally(
        self, record: PredictionRecord, output_dir: str, task_name: str
    ) -> None:
        """Append a single record incrementally to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        detailed_path = os.path.join(output_dir, f"{task_name}_detailed_results.json")
        
        # Read existing records from file
        existing_payload = []
        if os.path.exists(detailed_path):
            try:
                with open(detailed_path, "r") as f:
                    existing_payload = json.load(f)
                if not isinstance(existing_payload, list):
                    existing_payload = []
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning(f"[{task_name}] Failed to read existing file: {exc}. Starting fresh.")
                existing_payload = []
        
        # Convert new record to JSON-serializable format
        new_record = {
            "patient_id": record.patient_id,
            "label_time": record.label_time,
            "ground_truth": record.ground_truth,
            "score": record.score,
            "sample_predictions": record.sample_predictions,
            "sample_responses": record.sample_responses,
            "sample_token_usage": record.sample_token_usage,
        }
        
        # Append new record (no duplicate checking)
        existing_payload.append(new_record)
        
        # Write back to file (atomic write using temp file)
        temp_path = detailed_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(existing_payload, f, indent=2)
        os.replace(temp_path, detailed_path)  # Atomic rename

    def evaluate_task(
        self,
        task_name: str,
        *,
        serialized_dir: str,
        patient_ids: Optional[Iterable[int]],
        output_dir: str,
    ) -> Dict[str, float]:
        logger.info(f"=== Evaluating task: {task_name} ===")
        raw_examples = self.load_test_examples(serialized_dir, task_name)
        raw_examples = self.filter_by_patients(raw_examples, patient_ids)

        if not raw_examples:
            logger.warning(f"No examples available for task '{task_name}'. Skipping.")
            return {}

        evaluation_samples = [
            EvaluationSample(
                patient_id=example["patient_id"],
                label_time=example["label_time"],
                ground_truth=example["label_value"],
                prompt=self.build_prompt(task_name, example["context"]),
            )
            for example in raw_examples
        ]

        # Reset the output file before starting
        self._reset_output_file(output_dir, task_name)

        # Initialize file lock for thread-safe writing
        detailed_path = os.path.join(output_dir, f"{task_name}_detailed_results.json")
        if detailed_path not in self._file_locks:
            self._file_locks[detailed_path] = threading.Lock()

        # Start with empty list
        detailed_records: List[PredictionRecord] = []
        
        logger.info(
            f"[{task_name}] Starting evaluation. Will process {len(evaluation_samples)} examples."
        )
        
        if self.max_workers > 1:
            return self._evaluate_task_parallel(
                task_name, evaluation_samples, output_dir, detailed_records
            )
        else:
            return self._evaluate_task_sequential(
                task_name, evaluation_samples, output_dir, detailed_records
            )
    
    def _evaluate_task_sequential(
        self,
        task_name: str,
        evaluation_samples: List[EvaluationSample],
        output_dir: str,
        detailed_records: List[PredictionRecord],
    ) -> Dict[str, float]:
        """Process examples sequentially (original implementation)."""
        for idx, sample in enumerate(evaluation_samples, start=1):
            logger.info(
                f"[{task_name}] Processing example {idx}/{len(evaluation_samples)} "
                f"(Patient {sample.patient_id})"
            )
            record = self.evaluate_example(sample)
            detailed_records.append(record)
            
            # Append record incrementally to file
            self._append_record_incrementally(record, output_dir, task_name)
            logger.debug(
                f"[{task_name}] Appended record for patient {sample.patient_id} incrementally"
            )

        metrics = self._compute_and_save_metrics(task_name, detailed_records, output_dir)
        return metrics
    
    def _evaluate_task_parallel(
        self,
        task_name: str,
        evaluation_samples: List[EvaluationSample],
        output_dir: str,
        detailed_records: List[PredictionRecord],
    ) -> Dict[str, float]:
        """Process examples in parallel using ThreadPoolExecutor."""
        total_examples = len(evaluation_samples)
        completed_count = 0
        records_lock = threading.Lock()
        detailed_path = os.path.join(output_dir, f"{task_name}_detailed_results.json")
        
        def process_single_example(sample: EvaluationSample, index: int) -> Tuple[PredictionRecord, int]:
            """Process a single example and return it with its index."""
            logger.info(
                f"[{task_name}] Processing example {index + 1}/{total_examples} "
                f"(Patient {sample.patient_id})"
            )
            
            # Create a new evaluator instance for thread safety
            evaluator = AzureGPT5Evaluator(
                self.config,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                num_samples=self.num_samples,
                max_workers=1,  # Don't nest parallelization
            )
            
            record = evaluator.evaluate_example(sample)
            logger.debug(
                f"[{task_name}] Completed example {index + 1} (Patient {sample.patient_id})"
            )
            
            return record, index

        # Process examples in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_example, sample, idx): idx
                for idx, sample in enumerate(evaluation_samples)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    record, original_index = future.result()
                    completed_count += 1
                    
                    # Add to records list (thread-safe)
                    with records_lock:
                        detailed_records.append(record)
                    
                    logger.info(
                        f"[{task_name}] Progress: {completed_count}/{total_examples} completed"
                    )
                    
                    # Append record incrementally to file (thread-safe)
                    with self._file_locks[detailed_path]:
                        self._append_record_incrementally(record, output_dir, task_name)
                    
                    logger.debug(
                        f"[{task_name}] Appended record for patient {record.patient_id} incrementally"
                    )
                            
                except Exception as exc:
                    original_index = future_to_index[future]
                    logger.error(
                        f"[{task_name}] Example {original_index + 1} failed: {exc}"
                    )

        logger.info(
            f"[{task_name}] Completed {completed_count}/{total_examples} examples."
        )

        metrics = self._compute_and_save_metrics(task_name, detailed_records, output_dir)
        return metrics

    def _compute_and_save_metrics(
        self,
        task_name: str,
        records: List[PredictionRecord],
        output_dir: str,
    ) -> Dict[str, float]:
        os.makedirs(output_dir, exist_ok=True)

        scores = np.array([record.score for record in records])
        ground_truths = np.array([record.ground_truth for record in records])

        try:
            auroc = roc_auc_score(ground_truths, scores)
        except ValueError:
            auroc = 0.5

        precision_vals, recall_vals, thresholds = precision_recall_curve(
            ground_truths, scores
        )
        fpr, tpr, roc_thresholds = roc_curve(ground_truths, scores)

        j_scores = tpr - fpr
        optimal_idx = int(np.argmax(j_scores))
        optimal_threshold = roc_thresholds[optimal_idx] if roc_thresholds.size else 0.5

        binary_predictions = (scores >= optimal_threshold).astype(int)

        precision = precision_score(ground_truths, binary_predictions, zero_division=0)
        recall = recall_score(ground_truths, binary_predictions, zero_division=0)
        f1 = f1_score(ground_truths, binary_predictions, zero_division=0)

        total_prompt_tokens = sum(
            usage["prompt_tokens"] for record in records for usage in record.sample_token_usage
        )
        total_completion_tokens = sum(
            usage["completion_tokens"] for record in records for usage in record.sample_token_usage
        )
        total_tokens = sum(
            usage["total_tokens"] for record in records for usage in record.sample_token_usage
        )

        detailed_path = os.path.join(output_dir, f"{task_name}_detailed_results.json")
        detailed_payload = [
            {
                "patient_id": record.patient_id,
                "label_time": record.label_time,
                "ground_truth": record.ground_truth,
                "score": record.score,
                "sample_predictions": record.sample_predictions,
                "sample_responses": record.sample_responses,
                "sample_token_usage": record.sample_token_usage,
            }
            for record in records
        ]
        with open(detailed_path, "w") as f:
            json.dump(detailed_payload, f, indent=2)
        logger.info(f"[{task_name}] Detailed outputs saved to {detailed_path}")

        summary_path = os.path.join(output_dir, f"{task_name}_summary.json")
        summary_payload = {
            "task_name": task_name,
            "n_examples": len(records),
            "num_samples_per_example": self.num_samples,
            "auroc": float(auroc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "optimal_threshold": float(optimal_threshold),
            "total_prompt_tokens": int(total_prompt_tokens),
            "total_completion_tokens": int(total_completion_tokens),
            "total_tokens": int(total_tokens),
            "precision_curve": {
                "precision": precision_vals.tolist(),
                "recall": recall_vals.tolist(),
                "thresholds": thresholds.tolist(),
            },
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
            },
        }
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        logger.info(f"[{task_name}] Summary metrics saved to {summary_path}")

        csv_rows = []
        for record, binary_pred in zip(records, binary_predictions):
            row = {
                "patient_id": record.patient_id,
                "label_time": record.label_time,
                "ground_truth": record.ground_truth,
                "score": record.score,
                "binary_prediction": int(binary_pred),
            }
            for idx, prediction in enumerate(record.sample_predictions, start=1):
                row[f"sample_{idx}_prediction"] = prediction
            csv_rows.append(row)

        csv_df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(output_dir, f"{task_name}_results.csv")
        csv_df.to_csv(csv_path, index=False)
        logger.info(f"[{task_name}] CSV results saved to {csv_path}")

        return {
            "task_name": task_name,
            "n_examples": len(records),
            "auroc": float(auroc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "optimal_threshold": float(optimal_threshold),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Azure GPT-5 on the EHRShot multi-task benchmark (test split)."
    )

    parser.add_argument(
        "--serialized_data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "serialized_multi_task_data"),
        help="Directory containing serialized *_all_splits.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "azure_gpt5_multi_task_results"),
        help="Directory to store evaluation outputs.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Tasks to evaluate sequentially.",
    )
    parser.add_argument(
        "--patient_ids",
        nargs="+",
        type=int,
        help="Optional list of patient IDs to restrict evaluation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per patient (sequential calls).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Sampling temperature for Azure GPT-5.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate for each completion.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel workers for concurrent API calls (default: 1 = sequential). "
        "Increase for faster processing, but be mindful of rate limits.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = AzureBackwardsReasoningConfig(
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    evaluator = AzureGPT5Evaluator(
        config,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
    )
    
    if args.max_workers > 1:
        logger.info(f"Using parallel processing with {args.max_workers} workers")
    else:
        logger.info("Using sequential processing")

    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []

    for task_name in args.tasks:
        try:
            metrics = evaluator.evaluate_task(
                task_name=task_name,
                serialized_dir=args.serialized_data_dir,
                patient_ids=args.patient_ids,
                output_dir=args.output_dir,
            )
            if metrics:
                summary_rows.append(metrics)
        except Exception as exc:
            logger.error(f"Task {task_name} evaluation failed: {exc}")
            raise

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(args.output_dir, "all_tasks_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Combined summary saved to {summary_csv}")

        print("\n" + "=" * 80)
        print("AZURE GPT-5 MULTI-TASK EVALUATION SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80 + "\n")
    else:
        logger.warning("No tasks were successfully evaluated. Summary not generated.")


if __name__ == "__main__":
    main()

