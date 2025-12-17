#!/usr/bin/env python3
"""
Orchestrator for Parallel Model Evaluation (Reasoning Experiments)

Runs evaluation across multiple GPUs in parallel for clinical outcome prediction tasks.

Modes:
  - base: Evaluate base Qwen3-8B model on all 4 tasks
  - aggregated: Evaluate aggregated model on all 4 tasks  
  - per_task: Evaluate 4 task-specific models on ALL 4 tasks (16 jobs, diagonal first)

Usage:
    # Base model
    python orchestrator_eval.py --mode base --output_dir eval_results/base

    # Aggregated model
    python orchestrator_eval.py --mode aggregated \\
        --lora_path finetuned_models/aggregated \\
        --output_dir eval_results/aggregated

    # Per-task models (full 4x4 matrix)
    python orchestrator_eval.py --mode per_task \\
        --lora_base_dir finetuned_models/per_task \\
        --output_dir eval_results/per_task_matrix
"""

import os
import sys
import time
import argparse
import subprocess
import queue
import threading
import json
from typing import List, Optional, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL = "Qwen/Qwen3-8B"
TASKS = ["acute_mi", "pancreatic_cancer", "hypertension", "hyperlipidemia"]

# Test data paths
TEST_DATA_PATHS = {
    "originalEHR": "/dev/shm/ehrshot-data/serialized_multi_task_data",
    "rubricified": "/dev/shm/ehrshot-data/serialized_multi_task_data_rubricified",
}

# Worker scripts
EVAL_WORKER_SCRIPT = "eval_vllm_reasoning.py"
METRICS_SCRIPT = "eval_compute_metrics.py"


# =============================================================================
# HELPERS
# =============================================================================

def find_lora_checkpoint(task_dir: str) -> Optional[str]:
    """
    Find the LoRA adapter path for a finetuned task.
    
    The best model is saved directly to task_dir when training completes
    (via load_best_model_at_end=True in TrainingArguments).
    
    Returns the path to the LoRA adapter, or None if not found.
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

def compute_single_task_metrics(task_dir: str, task_name: str) -> Optional[dict]:
    """
    Compute metrics for a single task's predictions.csv file.
    
    Returns summary dict with AUROC/AUPRC, or None if predictions not found.
    """
    csv_path = os.path.join(task_dir, "predictions.csv")
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        
        # Filter to valid samples
        if 'has_valid_samples' in df.columns:
            valid_df = df[df['has_valid_samples'] == True]
            if valid_df.empty:
                valid_df = df
        else:
            valid_df = df
        
        y_true = valid_df['ground_truth'].values
        y_scores = valid_df['probability_score'].values
        
        # Quick metrics (no bootstrap for speed)
        auroc = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auprc = auc(recall, precision)
        
        summary = {
            "task_name": task_name,
            "num_examples": len(valid_df),
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
        }
        
        # Save quick summary
        with open(os.path.join(task_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
        
    except Exception as e:
        print(f"    ⚠️ Metrics error for {task_name}: {e}")
        return None


def evaluation_worker(
    gpu_id: int, 
    job_queue: queue.Queue, 
    model_name: str,
    test_data_dir: str,
    compute_metrics_on_complete: bool = False
):
    """
    Worker thread that runs evaluation jobs on a specific GPU.
    
    Each job is a tuple: (task_to_eval, lora_path, output_dir)
    If lora_path is None, evaluates the base model.
    
    If compute_metrics_on_complete=True, computes and prints metrics immediately
    after each job completes.
    """
    while True:
        try:
            job = job_queue.get(timeout=1)
        except queue.Empty:
            break

        task_name, lora_path, output_dir = job
        
        # Build paths
        data_path = os.path.join(test_data_dir, f"{task_name}_all_splits.json")
        log_file = os.path.join(output_dir, "inference.log")
        os.makedirs(output_dir, exist_ok=True)

        # Job description for logging
        if lora_path:
            # Extract source model name from lora_path
            source_model = os.path.basename(os.path.dirname(lora_path)) if lora_path else "base"
            if source_model == os.path.basename(lora_path):
                source_model = os.path.basename(lora_path)
            job_desc = f"{source_model} -> {task_name}"
        else:
            job_desc = f"Base -> {task_name}"

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
            "--output_dir", output_dir
        ]
        
        if lora_path:
            cmd.extend(["--lora_path", lora_path])

        # Run with specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with open(log_file, "w") as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            # Compute metrics immediately if requested
            if compute_metrics_on_complete:
                metrics = compute_single_task_metrics(output_dir, task_name)
                if metrics:
                    print(f"✅ [GPU {gpu_id}] FINISHED: {job_desc} | AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f}")
                else:
                    print(f"✅ [GPU {gpu_id}] FINISHED: {job_desc} (metrics pending)")
            else:
                print(f"✅ [GPU {gpu_id}] FINISHED: {job_desc}")
        else:
            print(f"❌ [GPU {gpu_id}] FAILED: {job_desc} (see {log_file})")

        job_queue.task_done()


def run_parallel_jobs(
    jobs: List[Tuple], 
    model_name: str, 
    test_data_dir: str, 
    gpu_ids: List[int],
    compute_metrics_on_complete: bool = False
):
    """Run jobs in parallel across available GPUs."""
    job_queue = queue.Queue()
    
    for job in jobs:
        job_queue.put(job)
    
    print(f"\nQueued {job_queue.qsize()} jobs. Starting workers on GPUs: {gpu_ids}...")
    if compute_metrics_on_complete:
        print("📊 Metrics will be computed and displayed as each job completes.\n")
    
    # Start worker threads
    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(
            target=evaluation_worker,
            args=(gpu_id, job_queue, model_name, test_data_dir, compute_metrics_on_complete)
        )
        t.start()
        threads.append(t)
        time.sleep(2)  # Stagger starts
    
    for t in threads:
        t.join()


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
    Mode: base
    Evaluate the base model on all 4 tasks in parallel.
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
    
    # Create jobs: (task, lora_path=None, output_dir)
    jobs = [(task, None, os.path.join(output_dir, task)) for task in TASKS]
    
    run_parallel_jobs(jobs, model_name, test_data_dir, gpu_ids, compute_metrics_on_complete=True)
    
    return output_dir


def evaluate_aggregated_model(
    model_name: str,
    lora_path: str,
    test_data_dir: str,
    output_dir: str,
    gpu_ids: List[int]
) -> str:
    """
    Mode: aggregated
    Evaluate a single aggregated model on all 4 tasks in parallel.
    """
    print("=" * 60)
    print("EVALUATION MODE: Aggregated Model")
    print("=" * 60)
    print(f"  Model:     {model_name}")
    print(f"  LoRA:      {lora_path}")
    print(f"  Tasks:     {TASKS}")
    print(f"  GPUs:      {gpu_ids}")
    print(f"  Test Data: {test_data_dir}")
    print(f"  Output:    {output_dir}/")
    print("=" * 60)
    
    # Verify LoRA exists
    if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        print(f"❌ ERROR: LoRA adapter not found at {lora_path}")
        return None
    
    # Create jobs: same LoRA for all tasks
    jobs = [(task, lora_path, os.path.join(output_dir, task)) for task in TASKS]
    
    run_parallel_jobs(jobs, model_name, test_data_dir, gpu_ids, compute_metrics_on_complete=True)
    
    return output_dir


def evaluate_per_task(
    model_name: str,
    lora_base_dir: str,
    test_data_dir: str,
    output_dir: str,
    gpu_ids: List[int]
) -> str:
    """
    Mode: per_task
    Evaluate 4 task-specific models on ALL 4 tasks (full 4x4 matrix = 16 jobs).
    Diagonal jobs run first, metrics displayed as each job completes.
    """
    print("=" * 60)
    print("EVALUATION MODE: Per-Task (4x4 Matrix)")
    print("=" * 60)
    print(f"  Model:     {model_name}")
    print(f"  LoRA Dir:  {lora_base_dir}")
    print(f"  Tasks:     {TASKS}")
    print(f"  GPUs:      {gpu_ids}")
    print(f"  Test Data: {test_data_dir}")
    print(f"  Output:    {output_dir}/")
    print("=" * 60)
    
    # Find LoRA adapters
    lora_paths = {}
    print("\nLocating LoRA adapters...")
    
    for task in TASKS:
        task_lora_dir = os.path.join(lora_base_dir, task)
        lora_path = find_lora_checkpoint(task_lora_dir)
        
        if lora_path:
            lora_paths[task] = lora_path
            print(f"  ✓ {task}: {lora_path}")
        else:
            print(f"  ✗ {task}: LoRA not found at {task_lora_dir}")
    
    if not lora_paths:
        print("\n❌ No LoRA adapters found. Exiting.")
        return None
    
    # Build all 16 jobs with smart ordering:
    # 1. Diagonal jobs first (highest priority - model on its own task)
    # 2. Off-diagonal jobs interleaved by runtime
    
    diagonal_jobs = []
    off_diagonal_jobs = []
    
    for source_task in lora_paths:
        for target_task in TASKS:
            target_output = os.path.join(output_dir, source_task, target_task)
            job = (target_task, lora_paths[source_task], target_output)
            
            if source_task == target_task:
                diagonal_jobs.append(job)
            else:
                off_diagonal_jobs.append(job)
    
    # Interleave off-diagonal by runtime (slow/fast tasks)
    slow_tasks = ["acute_mi", "pancreatic_cancer"]
    slow_jobs = [j for j in off_diagonal_jobs if j[0] in slow_tasks]
    fast_jobs = [j for j in off_diagonal_jobs if j[0] not in slow_tasks]
    
    interleaved_off_diag = []
    while slow_jobs or fast_jobs:
        if slow_jobs:
            interleaved_off_diag.append(slow_jobs.pop(0))
            interleaved_off_diag.append(slow_jobs.pop(0))
        if fast_jobs:
            interleaved_off_diag.append(fast_jobs.pop(0))
            interleaved_off_diag.append(fast_jobs.pop(0))
    
    # Final job order: diagonal first, then interleaved off-diagonal
    all_jobs = diagonal_jobs + interleaved_off_diag
    
    print(f"\n📋 Total jobs: {len(all_jobs)} ({len(diagonal_jobs)} diagonal + {len(interleaved_off_diag)} off-diagonal)")
    print("   Diagonal jobs will complete first, showing results immediately.\n")
    
    # Run all jobs with incremental metrics
    run_parallel_jobs(all_jobs, model_name, test_data_dir, gpu_ids, compute_metrics_on_complete=True)
    
    return output_dir


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(eval_dir: str, is_matrix: bool = False):
    """Run metrics computation on evaluation results."""
    print("\n" + "=" * 60)
    print("Computing Metrics...")
    print("=" * 60)
    
    if is_matrix:
        # For matrix mode, compute metrics for each source model separately
        for source_task in TASKS:
            source_dir = os.path.join(eval_dir, source_task)
            if os.path.exists(source_dir):
                print(f"\nComputing metrics for model: {source_task}")
                cmd = ["python", os.path.join(SCRIPT_DIR, METRICS_SCRIPT), "--eval_root", source_dir]
                subprocess.run(cmd)
        
        # Also create summary matrix
        create_matrix_summary(eval_dir)
    else:
        cmd = ["python", os.path.join(SCRIPT_DIR, METRICS_SCRIPT), "--eval_root", eval_dir]
        subprocess.run(cmd)


def create_matrix_summary(eval_dir: str):
    """Create a summary matrix CSV for cross-task evaluation."""
    import pandas as pd
    
    auroc_matrix = pd.DataFrame(index=TASKS, columns=TASKS)
    auprc_matrix = pd.DataFrame(index=TASKS, columns=TASKS)
    
    for source_task in TASKS:
        for target_task in TASKS:
            summary_path = os.path.join(eval_dir, source_task, target_task, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                auroc = summary.get('auroc', [0, 0, 0])
                auprc = summary.get('auprc', [0, 0, 0])
                
                auroc_matrix.loc[source_task, target_task] = f"{auroc[0]:.3f} ({auroc[1]:.3f}-{auroc[2]:.3f})"
                auprc_matrix.loc[source_task, target_task] = f"{auprc[0]:.3f} ({auprc[1]:.3f}-{auprc[2]:.3f})"
            else:
                auroc_matrix.loc[source_task, target_task] = "N/A"
                auprc_matrix.loc[source_task, target_task] = "N/A"
    
    auroc_matrix.to_csv(os.path.join(eval_dir, "final_auroc_matrix.csv"))
    auprc_matrix.to_csv(os.path.join(eval_dir, "final_auprc_matrix.csv"))
    
    print(f"\nSaved: {os.path.join(eval_dir, 'final_auroc_matrix.csv')}")
    print(f"Saved: {os.path.join(eval_dir, 'final_auprc_matrix.csv')}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate parallel model evaluation for reasoning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True,
        choices=["base", "aggregated", "per_task"],
        help="Evaluation mode: base, aggregated, or per_task (4x4 matrix)"
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
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (for aggregated mode)"
    )
    parser.add_argument(
        "--lora_base_dir",
        type=str,
        default=None,
        help="Directory with task-specific LoRA adapters (for per_task modes)"
    )
    parser.add_argument(
        "--data_repr",
        type=str,
        default="originalEHR",
        choices=["originalEHR", "rubricified"],
        help="Data representation type (default: originalEHR)"
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help="Custom test data directory (overrides --data_repr)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
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
    else:
        test_data_dir = TEST_DATA_PATHS.get(args.data_repr)
        if not test_data_dir:
            print(f"ERROR: Unknown data representation: {args.data_repr}")
            sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation based on mode
    eval_dir = None
    is_matrix = False
    
    if args.mode == "base":
        eval_dir = evaluate_base_model(
            model_name=args.model,
            test_data_dir=test_data_dir,
            output_dir=args.output_dir,
            gpu_ids=gpu_ids
        )
    
    elif args.mode == "aggregated":
        if not args.lora_path:
            print("ERROR: --lora_path is required for aggregated mode")
            sys.exit(1)
        
        eval_dir = evaluate_aggregated_model(
            model_name=args.model,
            lora_path=args.lora_path,
            test_data_dir=test_data_dir,
            output_dir=args.output_dir,
            gpu_ids=gpu_ids
        )
    
    elif args.mode == "per_task":
        if not args.lora_base_dir:
            print("ERROR: --lora_base_dir is required for per_task mode")
            sys.exit(1)
        
        eval_dir = evaluate_per_task(
            model_name=args.model,
            lora_base_dir=args.lora_base_dir,
            test_data_dir=test_data_dir,
            output_dir=args.output_dir,
            gpu_ids=gpu_ids
        )
        is_matrix = True
    
    # Compute metrics
    if eval_dir and not args.skip_metrics:
        compute_metrics(eval_dir, is_matrix=is_matrix)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results: {eval_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
