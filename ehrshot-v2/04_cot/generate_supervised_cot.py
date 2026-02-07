#!/usr/bin/env python3
"""
Generate supervised Chain-of-Thought reasoning traces using GPT-5-mini.

"Supervised" means the ground-truth label IS provided in the generation prompt,
so the model creates a reasoning trace that works backward from the answer.

For each patient, GPT-5-mini produces:
  <think>
  1. ### PATIENT SNAPSHOT
  2. ### MAIN RISK FACTORS
  3. ### PROTECTIVE FACTORS
  4. ### WHAT'S UNKNOWN / COULD SWING THE RISK
  5. ### WEIGHING AND AGGREGATING THE EVIDENCE
  6. ### CONCLUSION AND AN OVERALL RISK IMPRESSION

  Final Answer: Yes/No
  </think>

Inputs:
  --sft_dir   : Plaintext SFT directory (data/sft/plaintext).
  --output_dir: Where to write CoT SFT datasets.
  --tasks     : Space-separated list (default: all 15).
  --splits    : Which splits to process (default: train val).

The train+val traces are needed so that finetune_reasoning.py can monitor
eval loss on the val set during training.

Outputs:
  {output_dir}/{split}/{task}.json  (SFT conversation format with reasoning)

GPT-5-mini parameters: max_completion_tokens=16384, temperature=1.
Incremental saving + resumability via (patient_id, label_time) dedup.

Connects to:
  - Upstream  : 02_create_sft (plaintext)
  - Downstream: 05_train/finetune_reasoning.py
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

from loguru import logger
from openai import AzureOpenAI, RateLimitError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, SYSTEM_MESSAGE
from config.azure import AzureConfig


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _generation_prompt(ehr_text: str, task_query: str, label: bool) -> str:
    answer = "Yes" if label else "No"
    return (
        "You are provided with a patient's Electronic Healthcare Record (EHR) "
        f"and the known ground-truth outcome for the following prediction task: {task_query}\n\n"
        "### YOUR GOAL\n"
        "Create a high-fidelity clinical reasoning trace to bridge the EHR data "
        "to the final outcome through step-by-step clinical reasoning.\n"
        "IMPORTANT: DO NOT explicitly state that you were given the ground truth, "
        "nor use it as a shortcut while interpreting evidence.\n"
        "The reasoning should feel like a blind prediction process.\n"
        "All relevant information, including those that contradict the ground truth, "
        "should be included in the reasoning trace.\n"
        "Only use the ground truth to help weigh the evidence toward the end.\n\n"
        f"### INPUT DATA\n"
        f"TASK: {task_query}\n"
        f"GROUND TRUTH OUTCOME: {answer}\n"
        f"--- START OF EHR DATA ---\n{ehr_text}\n--- END OF EHR DATA ---\n\n"
        "### OUTPUT FORMAT\n"
        "Your output MUST follow this exact structure:\n"
        "<think>\n"
        "1. ### PATIENT SNAPSHOT\n"
        "2. ### MAIN RISK FACTORS\n"
        "3. ### PROTECTIVE FACTORS\n"
        "4. ### WHAT'S UNKNOWN / COULD SWING THE RISK\n"
        "5. ### WEIGHING AND AGGREGATING THE EVIDENCE\n"
        "6. ### CONCLUSION AND AN OVERALL RISK IMPRESSION\n\n"
        f"Final Answer: {answer}\n</think>"
    )


def _sft_user_prompt(task_query: str, ehr_text: str) -> str:
    """The user prompt that will be used during fine-tuning (not generation)."""
    return (
        f"Based on the patient's EHR below, predict: {task_query}\n\n"
        "Analyze the patient's condition and risk factors step by step.\n\n"
        f"--- Patient EHR ---\n\n{ehr_text}\n\n--- End of EHR ---\n\n"
        f"Based on the above EHR, predict: {task_query}\n"
        "Provide your response in the following format:\n"
        "<think>\n"
        "1. ### PATIENT SNAPSHOT\n"
        "2. ### MAIN RISK FACTORS\n"
        "3. ### PROTECTIVE FACTORS\n"
        "4. ### WHAT'S UNKNOWN / COULD SWING THE RISK\n"
        "5. ### WEIGHING AND AGGREGATING THE EVIDENCE\n"
        "6. ### CONCLUSION AND AN OVERALL RISK IMPRESSION\n\n"
        "Final Answer: Yes/No\n</think>"
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _generate_trace(record: dict, task_query: str,
                    config: AzureConfig, max_retries: int = 5) -> Dict[str, Any]:
    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
    )
    # Extract the original EHR text from the SFT user prompt
    ehr_text = record["conversations"][1]["content"]
    # Get between the EHR markers
    start = ehr_text.find("--- Patient EHR ---\n") + len("--- Patient EHR ---\n")
    end = ehr_text.find("\n--- End of EHR ---")
    ehr_text = ehr_text[start:end]

    label = record["label_value"]
    prompt = _generation_prompt(ehr_text, task_query, label)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=config.deployment,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=config.max_completion_tokens,
                temperature=config.temperature,
            )
            trace = resp.choices[0].message.content.strip()
            if trace:
                answer = "Yes" if label else "No"
                return {
                    "conversations": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user",
                         "content": _sft_user_prompt(task_query, ehr_text)},
                        {"role": "assistant", "content": trace},
                    ],
                    "patient_id": record["patient_id"],
                    "label_time": record["label_time"],
                    "label_value": label,
                    "task": record["task"],
                    "task_query": task_query,
                }
        except RateLimitError:
            delay = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Rate limit (patient {record['patient_id']}), "
                           f"retry in {delay:.1f}s")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Error for patient {record['patient_id']}: {e}")
            break

    return None  # skip on failure


# ---------------------------------------------------------------------------
# Incremental save / resume
# ---------------------------------------------------------------------------

def _load_done(path: Path) -> Set:
    if not path.exists():
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
        return {(e["patient_id"], e["label_time"]) for e in data}
    except Exception:
        return set()


def _save_inc(entry: dict, path: Path, lock: threading.Lock):
    with lock:
        data = []
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:
                pass
        data.append(entry)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sft_dir", required=True,
                   help="Plaintext SFT dir (data/sft/plaintext)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write CoT SFT datasets")
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--azure_config", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = AzureConfig.from_json(args.azure_config)
    config.max_completion_tokens = 16384
    config.temperature = 1.0
    lock = threading.Lock()

    for task in args.tasks:
        for split in args.splits:
            src = Path(args.sft_dir) / split / f"{task}.json"
            if not src.exists():
                logger.info(f"  {task}/{split}: no data, skipping")
                continue
            with open(src) as f:
                records = json.load(f)

            out_dir = Path(args.output_dir) / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{task}.json"
            done = _load_done(out_path)
            todo = [r for r in records
                    if (r["patient_id"], r["label_time"]) not in done]
            logger.info(f"{task}/{split}: {len(todo)} to generate "
                        f"({len(done)} already done)")
            if not todo:
                continue

            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futs = {
                    pool.submit(_generate_trace, r, TASKS[task], config): r
                    for r in todo
                }
                for fut in as_completed(futs):
                    result = fut.result()
                    if result is not None:
                        _save_inc(result, out_path, lock)

            logger.info(f"  {task}/{split} done -> {out_path}")

    logger.success("Supervised CoT generation complete.")


if __name__ == "__main__":
    main()
