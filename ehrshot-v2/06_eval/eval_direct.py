#!/usr/bin/env python3
"""
Evaluate a direct-prediction (Yes/No) model using vLLM logprobs.

For each test example the script:
  1. Constructs the same prompt used during training (sandwich style).
  2. Runs inference with max_tokens=1 and logprobs enabled.
  3. Extracts P(Yes) = sigmoid(logprob_yes - logprob_no).

Inputs:
  --test_file   : Test SFT JSON (data/sft/{repr}/test/{task}.json).
  --lora_path   : Path to LoRA adapter (or "base" for base model).
  --output_dir  : Where to write predictions.csv.

Outputs:
  {output_dir}/predictions.csv
    Columns: patient_id, label_time, ground_truth, probability_score, target_task

Connects to:
  - Upstream  : 05_train/finetune_direct.py
  - Downstream: compute_metrics.py
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

from loguru import logger
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import FINETUNE_MODEL


def load_test_data(path: str):
    with open(path) as f:
        return json.load(f)


def run_eval(test_data, llm, lora_request, sampling_params):
    """Run inference and extract P(Yes) from logprobs."""
    # Build prompts (just the user turn -- the model generates the assistant turn)
    prompts = []
    for entry in test_data:
        convos = entry["conversations"][:2]  # system + user only
        # We need to apply chat template for vLLM
        prompts.append(convos)

    # For vLLM with chat format, we use the generate interface
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL, trust_remote_code=True)

    formatted_prompts = []
    for convos in prompts:
        text = tokenizer.apply_chat_template(
            convos, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        formatted_prompts.append(text)

    outputs = llm.generate(
        formatted_prompts,
        sampling_params,
        lora_request=lora_request,
    )

    results = []
    for entry, output in zip(test_data, outputs):
        logprobs_list = output.outputs[0].logprobs
        prob_yes = 0.5  # default

        if logprobs_list and len(logprobs_list) > 0:
            first_token_logprobs = logprobs_list[0]
            # Search through top logprobs for Yes/No tokens
            logprob_yes = None
            logprob_no = None
            for token_id, logprob_obj in first_token_logprobs.items():
                decoded = logprob_obj.decoded_token.strip().lower()
                if decoded == "yes":
                    logprob_yes = logprob_obj.logprob
                elif decoded == "no":
                    logprob_no = logprob_obj.logprob

            if logprob_yes is not None and logprob_no is not None:
                prob_yes = 1.0 / (1.0 + math.exp(-(logprob_yes - logprob_no)))
            elif logprob_yes is not None:
                prob_yes = math.exp(logprob_yes)
            elif logprob_no is not None:
                prob_yes = 1.0 - math.exp(logprob_no)

        results.append({
            "patient_id": entry["patient_id"],
            "label_time": entry["label_time"],
            "ground_truth": 1 if entry["label_value"] else 0,
            "probability_score": prob_yes,
            "target_task": entry["task"],
        })
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_file", required=True)
    p.add_argument("--lora_path", default="base",
                   help="Path to LoRA adapter dir, or 'base' for base model")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    return p.parse_args()


def main():
    args = parse_args()

    test_data = load_test_data(args.test_file)
    logger.info(f"Loaded {len(test_data)} test examples from {args.test_file}")

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
        max_tokens=1,
        logprobs=50,
        temperature=0.0,
    )

    results = run_eval(test_data, llm, lora_request, sampling_params)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "predictions.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patient_id", "label_time", "ground_truth",
            "probability_score", "target_task"])
        writer.writeheader()
        writer.writerows(results)

    logger.success(f"Saved {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
