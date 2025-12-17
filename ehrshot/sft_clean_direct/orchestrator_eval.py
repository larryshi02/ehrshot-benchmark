#!/usr/bin/env python3
"""
Orchestrator for Parallel Model Evaluation

Runs evaluation across multiple GPUs in parallel for clinical outcome prediction tasks.
Supports two modes:
  - base: Evaluate base Qwen3-8B model on all 4 tasks
  - finetuned: Evaluate 4 task-specific finetuned models on their respective tasks

Usage:
    # Base model evaluation (original EHR data)
    python orchestrator_eval.py --mode base --gpus 0,1,2,3 \\
        --output_dir eval_results/originalEHR_base

    # Base model evaluation (rubricified EHR data)
    python orchestrator_eval.py --mode base --gpus 0,1,2,3 \\
        --data_type rubricified --output_dir eval_results/rubricifiedEHR_base

    # Finetuned model evaluation
    python orchestrator_eval.py --mode finetuned --gpus 0,1,2,3 \\
        --lora_base_dir finetuned_models/originalEHR_train_all \\
        --output_dir eval_results/originalEHR_train_all_finetuned

Output Structure:
    output_dir/
    ├── acute_mi/predictions.csv
    ├── hyperlipidemia/predictions.csv
    ├── hypertension/predictions.csv
    ├── pancreatic_cancer/predictions.csv
    └── metrics_summary.csv
"""

import os
import sys
import time
import argparse
import subprocess
import queue
import threading
from typing import List, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL = "Qwen/Qwen3-8B"
TASKS = ["acute_mi", "pancreatic_cancer", "hypertension", "hyperlipidemia"]

# Test data paths
ORIGINAL_TEST_DATA = "/dev/shm/ehrshot-data/serialized_multi_task_data"
RUBRICIFIED_TEST_DATA = "/dev/shm/ehrshot-data/serialized_multi_task_data_rubricified"

# Worker scripts
EVAL_WORKER_SCRIPT = "eval_vllm_direct.py"
METRICS_SCRIPT = "eval_vllm_compute_metrics.py"


# =============================================================================
# HELPERS
# =============================================================================

def find_lora_checkpoint(task_dir: str) -> Optional[str]:
    """
    Find the LoRA adapter path for a finetuned task.
    
    The best model is saved directly to task_dir when training completes
    (via load_best_model_at_end=True in TrainingArguments).
    
    Returns the path to the LoRA adapter (task_dir itself), or None if not found.
    """
    if not os.path.isdir(task_dir):
        return None
    
    # Check if adapter files exist directly in task_dir (best model saved here)
    if os.path.exists(os.path.join(task_dir, "adapter_config.json")):
        return task_dir
    
    return None


# =============================================================================
# GPU WORKER
# =============================================================================

def evaluation_worker(
    gpu_id: int, 
    job_queue: queue.Queue, 
    model_name: str,
    test_data_dir: str,
    output_dir: str
):
    """
    Worker thread that runs evaluation jobs on a specific GPU.
    
    Pulls jobs from the queue until empty. Each job is (task_name, lora_path).
    If lora_path is None, evaluates the base model; otherwise evaluates finetuned.
    """
    while True:
        try:
            job = job_queue.get(timeout=1)
        except queue.Empty:
            break

        task_name, lora_path = job
        
        # Build paths
        data_path = os.path.join(test_data_dir, f"{task_name}_all_splits.json")
        task_output_dir = os.path.join(output_dir, task_name)
        log_file = os.path.join(task_output_dir, "inference.log")
        os.makedirs(task_output_dir, exist_ok=True)

        # Job description for logging
        job_desc = f"Finetuned({task_name})" if lora_path else f"Base -> {task_name}"

        # Check data exists
        if not os.path.exists(data_path):
            print(f"⚠️  [GPU {gpu_id}] Skipping {job_desc}: Data not found at {data_path}")
            job_queue.task_done()
            continue

        print(f"🚀 [GPU {gpu_id}] STARTING: {job_desc}")

        # Build command
        cmd = [
            "python", os.path.join(SCRIPT_DIR, EVAL_WORKER_SCRIPT),
            "--model_base", model_name,
            "--data_path", data_path,
            "--task_name", task_name,
            "--output_dir", task_output_dir
        ]
        
        if lora_path:
            cmd.extend(["--lora_path", lora_path])

        # Run with specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with open(log_file, "w") as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            print(f"✅ [GPU {gpu_id}] FINISHED: {job_desc}")
        else:
            print(f"❌ [GPU {gpu_id}] FAILED: {job_desc} (see {log_file})")

        job_queue.task_done()


# =============================================================================
# EVALUATION MODES
# =============================================================================

def evaluate_base_model(
    model_name: str,
    test_data_dir: str,
    output_dir: str,
    gpu_ids: List[int]
) -> str:
    """
    Evaluate the base model on all 4 tasks in parallel.
    
    Each GPU processes one task. The same base model is used for all tasks.
    """
    print("=" * 60)
    print("EVALUATION MODE: Base Model")
    print("=" * 60)
    print(f"  Model:     {model_name}")
    print(f"  Tasks:     {TASKS}")
    print(f"  GPUs:      {gpu_ids}")
    print(f"  Test Data: {test_data_dir}")
    print(f"  Output:    {output_dir}/")
    print("=" * 60)
    
    # Queue jobs (one per task, no LoRA)
    job_queue = queue.Queue()
    for task in TASKS:
        job_queue.put((task, None))  # None = no LoRA = base model
    
    print(f"\nQueued {job_queue.qsize()} jobs. Starting workers...")
    
    # Start worker threads
    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(
            target=evaluation_worker,
            args=(gpu_id, job_queue, model_name, test_data_dir, output_dir)
        )
        t.start()
        threads.append(t)
        time.sleep(2)  # Stagger starts
    
    for t in threads:
        t.join()
    
    return output_dir


def evaluate_finetuned_models(
    model_name: str,
    lora_base_dir: str,
    test_data_dir: str,
    output_dir: str,
    gpu_ids: List[int]
) -> Optional[str]:
    """
    Evaluate task-specific finetuned models on their respective tasks.
    
    Each task has its own LoRA adapter in lora_base_dir/{task}/. 
    Each model is evaluated only on the task it was trained for.
    """
    print("=" * 60)
    print("EVALUATION MODE: Finetuned Models")
    print("=" * 60)
    print(f"  Base Model: {model_name}")
    print(f"  LoRA Dir:   {lora_base_dir}")
    print(f"  Tasks:      {TASKS}")
    print(f"  GPUs:       {gpu_ids}")
    print(f"  Test Data:  {test_data_dir}")
    print(f"  Output:     {output_dir}/")
    print("=" * 60)
    
    # Queue jobs (one per task with its LoRA)
    job_queue = queue.Queue()
    print("\nLocating LoRA adapters...")
    
    for task in TASKS:
        task_lora_dir = os.path.join(lora_base_dir, task)
        lora_path = find_lora_checkpoint(task_lora_dir)
        
        if lora_path:
            job_queue.put((task, lora_path))
            print(f"  ✓ {task}: {lora_path}")
        else:
            print(f"  ✗ {task}: LoRA not found at {task_lora_dir}")
    
    if job_queue.qsize() == 0:
        print("\n❌ No LoRA adapters found. Exiting.")
        return None
    
    print(f"\nQueued {job_queue.qsize()} jobs. Starting workers...")
    
    # Start worker threads
    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(
            target=evaluation_worker,
            args=(gpu_id, job_queue, model_name, test_data_dir, output_dir)
        )
        t.start()
        threads.append(t)
        time.sleep(2)
    
    for t in threads:
        t.join()
    
    return output_dir


def compute_metrics(eval_dir: str):
    """Run metrics computation on evaluation results."""
    print("\n" + "=" * 60)
    print("Computing Metrics...")
    print("=" * 60)
    
    cmd = ["python", os.path.join(SCRIPT_DIR, METRICS_SCRIPT), "--eval_root", eval_dir]
    subprocess.run(cmd)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate parallel model evaluation on multiple GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True,
        choices=["base", "finetuned"],
        help="'base' = evaluate base model on all tasks, 'finetuned' = evaluate task-specific models"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU IDs (default: 0,1,2,3)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--lora_base_dir",
        type=str,
        default=None,
        help="Directory with task-specific LoRA adapters (required for --mode finetuned)"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="original",
        choices=["original", "rubricified"],
        help="Test data type (default: original)"
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help="Custom test data directory (overrides --data_type)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./eval_results)"
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip metrics computation after evaluation"
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    
    # Determine test data directory
    if args.test_data_dir:
        test_data_dir = args.test_data_dir
    elif args.data_type == "rubricified":
        test_data_dir = RUBRICIFIED_TEST_DATA
    else:
        test_data_dir = ORIGINAL_TEST_DATA
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    if args.mode == "base":
        eval_dir = evaluate_base_model(
            model_name=args.model,
            test_data_dir=test_data_dir,
            output_dir=output_dir,
            gpu_ids=gpu_ids
        )
    else:
        if not args.lora_base_dir:
            print("ERROR: --lora_base_dir is required for --mode finetuned")
            print("Example: --lora_base_dir finetuned_models/originalEHR_train_all")
            sys.exit(1)
        
        eval_dir = evaluate_finetuned_models(
            model_name=args.model,
            lora_base_dir=args.lora_base_dir,
            test_data_dir=test_data_dir,
            output_dir=output_dir,
            gpu_ids=gpu_ids
        )
    
    # Compute metrics
    if eval_dir and not args.skip_metrics:
        compute_metrics(eval_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results: {eval_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
