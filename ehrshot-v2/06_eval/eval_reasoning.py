#!/usr/bin/env python3
"""
Evaluate a reasoning-fine-tuned model via sampling.

For each test example the script:
  1. Samples N=10 responses (temperature=0.7, top_p=0.9, max_tokens=4096).
  2. Parses "Final Answer: Yes/No" from each response.
  3. Computes P(Yes) = count(Yes) / count(valid_responses).

Inputs:
  --test_file  : Test SFT JSON (data/sft/cot_supervised/test/{task}.json
                 or data/sft/plaintext/test/{task}.json).
  --lora_path  : Path to LoRA adapter (or "base").
  --output_dir : Where to write predictions.csv.

Outputs:
  {output_dir}/predictions.csv  (same schema as eval_direct.py)

Connects to:
  - Upstream  : 05_train/finetune_reasoning.py
  - Downstream: compute_metrics.py
"""

import argparse
import csv
import json
import os
import re
import sys

from loguru import logger
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import FINETUNE_MODEL, SEED

N_SAMPLES = 10


def _parse_answer(text: str) -> str | None:
    """Extract 'Yes' or 'No' from 'Final Answer: ...'."""
    m = re.search(r"Final\s+Answer\s*:\s*(Yes|No)", text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return None


def run_eval(test_data, llm, lora_request, sampling_params):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL, trust_remote_code=True)

    formatted_prompts = []
    for entry in test_data:
        convos = entry["conversations"][:2]
        text = tokenizer.apply_chat_template(
            convos, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        formatted_prompts.append(text)

    # Repeat each prompt N_SAMPLES times
    expanded = []
    for p in formatted_prompts:
        expanded.extend([p] * N_SAMPLES)

    outputs = llm.generate(expanded, sampling_params, lora_request=lora_request)

    results = []
    for i, entry in enumerate(test_data):
        batch = outputs[i * N_SAMPLES : (i + 1) * N_SAMPLES]
        yes_count = 0
        valid_count = 0
        for out in batch:
            text = out.outputs[0].text
            answer = _parse_answer(text)
            if answer is not None:
                valid_count += 1
                if answer == "Yes":
                    yes_count += 1

        prob = yes_count / valid_count if valid_count > 0 else 0.5
        results.append({
            "patient_id": entry["patient_id"],
            "label_time": entry["label_time"],
            "ground_truth": 1 if entry["label_value"] else 0,
            "probability_score": prob,
            "target_task": entry["task"],
        })
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_file", required=True)
    p.add_argument("--lora_path", default="base")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    return p.parse_args()


def main():
    args = parse_args()
    test_data = json.load(open(args.test_file))
    logger.info(f"Loaded {len(test_data)} test examples")

    enable_lora = args.lora_path != "base"
    llm = LLM(
        model=FINETUNE_MODEL,
        enable_lora=enable_lora,
        max_lora_rank=16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=12288,
        trust_remote_code=True,
    )
    lora_request = None
    if enable_lora:
        lora_request = LoRARequest("adapter", 1, args.lora_path)

    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        n=1,  # we expand prompts ourselves
        seed=SEED,  # reproducible sampling
    )

    results = run_eval(test_data, llm, lora_request, sampling_params)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "predictions.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id", "label_time", "ground_truth",
            "probability_score", "target_task"])
        w.writeheader()
        w.writerows(results)

    logger.success(f"Saved {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
