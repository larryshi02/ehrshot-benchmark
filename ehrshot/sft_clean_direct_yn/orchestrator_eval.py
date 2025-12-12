#!/usr/bin/env python3
"""
Orchestrator for Direct Y/N Evaluation

This script manages parallel evaluation on multiple GPUs for:
1. Base model evaluation on all 4 tasks (--mode base)
2. Finetuned model evaluation on their respective tasks (--mode finetuned)

Usage:
    # Evaluate base Qwen3-8B on all 4 tasks using GPUs 0,1,2,3
    python orchestrator_eval.py --mode base --gpus 0,1,2,3

    # Evaluate 4 finetuned models on their own tasks using GPUs 0,1,2,3
    python orchestrator_eval.py --mode finetuned --gpus 0,1,2,3
    
    # Custom LoRA directory
    python orchestrator_eval.py --mode finetuned --lora_base_dir /path/to/lora/models
"""

import os
import sys
import time
import argparse
import subprocess
import queue
import threading
from pathlib import Path
from typing import List, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default model
DEFAULT_MODEL_BASE = "Qwen/Qwen3-8B"

# Tasks
TASKS = ["acute_mi", "pancreatic_cancer", "hypertension", "hyperlipidemia"]

# Test data location (same as sft_clean/orchestrator.py)
DEFAULT_TEST_DATA_BASE = "/dev/shm/ehrshot-data/serialized_multi_task_data"

# Worker script
WORKER_SCRIPT = "eval_vllm_direct.py"
METRICS_SCRIPT = "eval_vllm_compute_metrics.py"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_latest_checkpoint(base_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a training output directory.
    Returns the checkpoint path, or the base_dir if no checkpoints found.
    """
    if not os.path.isdir(base_dir):
        return None
    
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    checkpoints = [d for d in subdirs if d.startswith("checkpoint-")]
    
    if not checkpoints:
        # No checkpoint subdirs - check if adapter files exist in base_dir
        if os.path.exists(os.path.join(base_dir, "adapter_config.json")):
            return base_dir
        return None
    
    # Sort by checkpoint number
    try:
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        return os.path.join(base_dir, checkpoints[-1])
    except ValueError:
        return base_dir


def get_default_lora_base_dir(model_name: str, lora_rank: int = 16) -> str:
    """Generate default LoRA base directory path."""
    model_short = model_name.split('/')[-1].lower()
    return os.path.join(SCRIPT_DIR, f"output_direct_yn_{model_short}_r{lora_rank}")


# =============================================================================
# GPU WORKER
# =============================================================================

def gpu_worker(
    gpu_id: int, 
    task_queue: queue.Queue, 
    model_base: str,
    test_data_base: str,
    output_root: str
):
    """
    Worker thread that processes jobs from the queue on a specific GPU.
    
    Each job is a tuple: (task_name, lora_path_or_none)
    - If lora_path is None, evaluates base model
    - If lora_path is provided, evaluates finetuned model
    """
    while True:
        try:
            job = task_queue.get(timeout=1)
        except queue.Empty:
            break

        task_name, lora_path = job
        
        # Setup paths (format: {task}_all_splits.json)
        data_path = os.path.join(test_data_base, f"{task_name}_all_splits.json")
        
        if lora_path:
            # Finetuned model
            output_dir = os.path.join(output_root, "finetuned", task_name)
            job_desc = f"Finetuned({task_name})"
        else:
            # Base model
            output_dir = os.path.join(output_root, "base_model", task_name)
            job_desc = f"Base -> {task_name}"
        
        log_file = os.path.join(output_dir, "inference.log")
        os.makedirs(output_dir, exist_ok=True)

        # Check data exists
        if not os.path.exists(data_path):
            print(f"⚠️  [GPU {gpu_id}] Skipping {job_desc}: Data not found at {data_path}")
            task_queue.task_done()
            continue

        print(f"🚀 [GPU {gpu_id}] STARTING: {job_desc}")

        # Build command
        cmd = [
            "python", os.path.join(SCRIPT_DIR, WORKER_SCRIPT),
            "--model_base", model_base,
            "--data_path", data_path,
            "--task_name", task_name,
            "--output_dir", output_dir
        ]
        
        # Add LoRA path if finetuned
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
            print(f"❌ [GPU {gpu_id}] FAILED: {job_desc} (Check {log_file})")

        task_queue.task_done()


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run_base_model_evaluation(
    model_base: str,
    test_data_base: str,
    output_root: str,
    gpu_ids: List[int]
):
    """
    Evaluate base model on all 4 tasks in parallel.
    """
    print("=" * 60)
    print("MODE: Base Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_base}")
    print(f"Tasks: {TASKS}")
    print(f"GPUs: {gpu_ids}")
    print(f"Output: {output_root}/base_model/")
    print("=" * 60)
    
    # Create job queue - one job per task, all with lora_path=None
    job_queue = queue.Queue()
    
    print("\n--- Queueing Jobs ---")
    for task in TASKS:
        job_queue.put((task, None))  # None = no LoRA = base model
        print(f"  Queued: Base model -> {task}")
    
    print(f"\nTotal Jobs: {job_queue.qsize()}")
    print("Starting Workers...")
    
    # Start worker threads
    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, job_queue, model_base, test_data_base, output_root)
        )
        t.start()
        threads.append(t)
        time.sleep(2)  # Stagger starts to reduce CPU spike
    
    # Wait for completion
    for t in threads:
        t.join()
    
    return os.path.join(output_root, "base_model")


def run_finetuned_evaluation(
    model_base: str,
    lora_base_dir: str,
    test_data_base: str,
    output_root: str,
    gpu_ids: List[int]
):
    """
    Evaluate 4 finetuned models on their respective tasks in parallel.
    Each model is only evaluated on the task it was finetuned on.
    """
    print("=" * 60)
    print("MODE: Finetuned Model Evaluation (Same-Task Only)")
    print("=" * 60)
    print(f"Base Model: {model_base}")
    print(f"LoRA Base Dir: {lora_base_dir}")
    print(f"Tasks: {TASKS}")
    print(f"GPUs: {gpu_ids}")
    print(f"Output: {output_root}/finetuned/")
    print("=" * 60)
    
    # Create job queue - one job per task with corresponding LoRA
    job_queue = queue.Queue()
    
    print("\n--- Queueing Jobs ---")
    for task in TASKS:
        # Find LoRA adapter for this task
        task_lora_dir = os.path.join(lora_base_dir, task)
        lora_path = get_latest_checkpoint(task_lora_dir)
        
        if lora_path:
            job_queue.put((task, lora_path))
            print(f"  Queued: {task} (LoRA: {lora_path})")
        else:
            print(f"  ⚠️  Skipped: {task} (LoRA not found at {task_lora_dir})")
    
    if job_queue.qsize() == 0:
        print("\n❌ No valid LoRA adapters found. Exiting.")
        return None
    
    print(f"\nTotal Jobs: {job_queue.qsize()}")
    print("Starting Workers...")
    
    # Start worker threads
    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, job_queue, model_base, test_data_base, output_root)
        )
        t.start()
        threads.append(t)
        time.sleep(2)
    
    # Wait for completion
    for t in threads:
        t.join()
    
    return os.path.join(output_root, "finetuned")


def run_metrics_computation(eval_dir: str):
    """Run the metrics computation script on evaluation results."""
    print("\n" + "=" * 60)
    print("Computing Metrics...")
    print("=" * 60)
    
    cmd = [
        "python", os.path.join(SCRIPT_DIR, METRICS_SCRIPT),
        "--eval_root", eval_dir
    ]
    
    subprocess.run(cmd)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrator for Direct Y/N Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate base model on all tasks
  python orchestrator_eval.py --mode base --gpus 0,1,2,3

  # Evaluate finetuned models on their own tasks
  python orchestrator_eval.py --mode finetuned --gpus 0,1,2,3

  # Custom settings
  python orchestrator_eval.py --mode finetuned --gpus 0,1 --model Qwen/Qwen3-8B
        """
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True,
        choices=["base", "finetuned"],
        help="Evaluation mode: 'base' for base model, 'finetuned' for task-specific models"
    )
    
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU IDs to use (default: 0,1,2,3)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_BASE,
        help=f"Base model name (default: {DEFAULT_MODEL_BASE})"
    )
    
    parser.add_argument(
        "--lora_base_dir",
        type=str,
        default=None,
        help="Directory containing task-specific LoRA adapters (default: auto-detect)"
    )
    
    parser.add_argument(
        "--test_data_base",
        type=str,
        default=DEFAULT_TEST_DATA_BASE,
        help="Directory containing test data JSON files"
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
    
    # Set output directory
    output_root = args.output_dir or os.path.join(SCRIPT_DIR, "eval_results")
    os.makedirs(output_root, exist_ok=True)
    
    # Set LoRA base directory
    lora_base_dir = args.lora_base_dir or get_default_lora_base_dir(args.model)
    
    # Run evaluation based on mode
    if args.mode == "base":
        eval_dir = run_base_model_evaluation(
            model_base=args.model,
            test_data_base=args.test_data_base,
            output_root=output_root,
            gpu_ids=gpu_ids
        )
    else:  # finetuned
        eval_dir = run_finetuned_evaluation(
            model_base=args.model,
            lora_base_dir=lora_base_dir,
            test_data_base=args.test_data_base,
            output_root=output_root,
            gpu_ids=gpu_ids
        )
    
    # Compute metrics
    if eval_dir and not args.skip_metrics:
        run_metrics_computation(eval_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved to: {eval_dir}")


if __name__ == "__main__":
    main()
