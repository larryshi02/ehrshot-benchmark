#!/usr/bin/env python3
"""
Parallel multi-task evaluation coordinator for base Qwen3-8B model.

This script coordinates parallel evaluation across multiple GPUs, where each GPU
loads the model independently and evaluates one task at a time.

Execution flow when main() is called:
1. parse_args() - Parses command-line arguments and configuration
2. main() - Entry point that:
   a. Validates inputs and sets up output directories
   b. Determines available GPUs (default: 4 GPUs 0-3)
   c. Partitions tasks into batches (e.g., 4 tasks per batch for 4 GPUs)
   d. For each batch:
      - Launches parallel processes using subprocess, one per GPU
      - Each process runs evaluate_base_model_single_task.py with:
        * Specific GPU ID (CUDA_VISIBLE_DEVICES set in the subprocess)
        * Specific task name
        * tensor_parallel_size=1 (single GPU per task)
      - Waits for all processes in the batch to complete
      - Collects results and handles any failures
   e. Aggregates all results and generates summary
3. Each subprocess (evaluate_base_model_single_task.py) independently:
   - Sets CUDA_VISIBLE_DEVICES to its assigned GPU
   - Loads the model with tensor_parallel_size=1
   - Evaluates its assigned task
   - Saves results to the shared output directory

This architecture allows:
- Multiple tasks to be evaluated in parallel (one per GPU)
- Support for more tasks than GPUs (batches are processed sequentially)
- Each GPU loads its own model instance (tensor_parallel_size=1)
- Independent failure handling per task/GPU
"""

import os
import sys
import json
import argparse
import subprocess
import time
from typing import List, Dict, Optional, Tuple
import pandas as pd
from loguru import logger

# Add script directory to path for imports (to import task_config)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import shared task configuration (coordinator owns and validates the config)
try:
    from task_config import DEFAULT_TASKS, TASK_QUERIES
except ImportError:
    # Fallback if task_config.py is not found
    DEFAULT_TASKS = ['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer']
    TASK_QUERIES = {
        'acute_mi': 'will the patient develop an acute myocardial infarction in the next year',
        'hyperlipidemia': 'will the patient develop hyperlipidemia in the next year',
        'hypertension': 'will the patient develop hypertension in the next year',
        'pancreatic_cancer': 'will the patient develop pancreatic cancer in the next year'
    }


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Parallel multi-task evaluation coordinator - distributes tasks across GPUs"
    )
    
    # Task configuration
    parser.add_argument("--tasks", type=str, nargs='+', default=DEFAULT_TASKS,
                       help=f"List of tasks to evaluate (default: {DEFAULT_TASKS})")
    
    # GPU configuration
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="Number of GPUs to use (default: 4)")
    parser.add_argument("--gpu_start_id", type=int, default=0,
                       help="Starting GPU ID (default: 0, so GPUs 0-3 will be used)")
    
    # Model configuration
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Base model name")
    parser.add_argument("--use_quantization", action="store_true", default=False,
                       help="Use quantization")
    
    # Data configuration
    parser.add_argument("--path_to_serialized_data", type=str, required=True,
                       help="Path to directory with pre-serialized JSON files")
    
    # VLLM configuration
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                       help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=20000,
                       help="Maximum model length")
    
    # Evaluation configuration
    parser.add_argument("--max_new_tokens", type=int, default=8500,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples per patient")
    
    # Parser LLM configuration
    parser.add_argument("--parser_model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                       help="Parser LLM model name")
    parser.add_argument("--parser_gpu_memory_utilization", type=float, default=0.1,
                       help="GPU memory utilization for parser LLM")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./base_model_multi_task_results",
                       help="Output directory for results")
    
    # Script path (for subprocess calls)
    parser.add_argument("--single_task_script", type=str,
                       default=None,
                       help="Path to evaluate_base_model_single_task.py (auto-detected if not provided)")
    
    return parser.parse_args()


def get_script_dir() -> str:
    """Get the directory containing this script"""
    return os.path.dirname(os.path.abspath(__file__))


def partition_tasks_into_batches(tasks: List[str], num_gpus: int) -> List[List[str]]:
    """
    Partition tasks into batches of size num_gpus.
    
    Each batch can be processed in parallel across num_gpus GPUs.
    If there are more tasks than GPUs, multiple batches are created.
    
    Args:
        tasks: List of task names to evaluate
        num_gpus: Number of GPUs available (batch size)
    
    Returns:
        List of batches, where each batch is a list of task names
    """
    batches = []
    for i in range(0, len(tasks), num_gpus):
        batch = tasks[i:i + num_gpus]
        batches.append(batch)
    return batches


def run_single_task_evaluation(
    task_name: str,
    gpu_id: int,
    args: argparse.Namespace,
    single_task_script: str
) -> Tuple[bool, str]:
    """
    Run evaluation for a single task on a specific GPU using subprocess.
    
    Args:
        task_name: Name of the task to evaluate
        gpu_id: GPU ID to use (0-indexed, will be used as CUDA_VISIBLE_DEVICES)
        args: Parsed command-line arguments
        single_task_script: Path to evaluate_base_model_single_task.py
    
    Returns:
        Tuple of (success: bool, output_message: str)
    """
    logger.info(f"Starting task '{task_name}' on GPU {gpu_id}")
    
    # Build command for subprocess
    cmd = [
        sys.executable,
        single_task_script,
        "--task_name", task_name,
        "--gpu_id", str(gpu_id),
        "--base_model_name", args.base_model_name,
        "--path_to_serialized_data", args.path_to_serialized_data,
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
        "--max_model_len", str(args.max_model_len),
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--num_samples", str(args.num_samples),
        "--parser_model_name", args.parser_model_name,
        "--parser_gpu_memory_utilization", str(args.parser_gpu_memory_utilization),
        "--output_dir", args.output_dir,
    ]
    
    if args.use_quantization:
        cmd.append("--use_quantization")
    
    # Set CUDA_VISIBLE_DEVICES in the subprocess environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        # Run subprocess and capture output
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=None  # No timeout (tasks may take a long time)
        )
        
        if result.returncode == 0:
            logger.info(f"Task '{task_name}' on GPU {gpu_id} completed successfully")
            return True, result.stdout
        else:
            logger.error(f"Task '{task_name}' on GPU {gpu_id} failed with return code {result.returncode}")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
            return False, result.stderr
        
    except Exception as e:
        logger.error(f"Error running task '{task_name}' on GPU {gpu_id}: {e}")
        return False, str(e)


def run_parallel_batch(
    batch: List[str],
    gpu_start_id: int,
    args: argparse.Namespace,
    single_task_script: str
) -> Dict[str, Tuple[bool, str]]:
    """
    Run a batch of tasks in parallel across multiple GPUs.
    
    Each task in the batch is assigned to a different GPU and runs in parallel.
    
    Args:
        batch: List of task names to evaluate in parallel
        gpu_start_id: Starting GPU ID (tasks will use GPUs starting from this ID)
        args: Parsed command-line arguments
        single_task_script: Path to evaluate_base_model_single_task.py
    
    Returns:
        Dictionary mapping task_name -> (success: bool, output_message: str)
    """
    import concurrent.futures
    
    results = {}
    
    # Use ThreadPoolExecutor to run tasks in parallel
    # Note: Each task runs in a subprocess with its own GPU, so threading is fine here
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
        # Submit all tasks to the executor
        future_to_task = {}
        for i, task_name in enumerate(batch):
            gpu_id = gpu_start_id + i
            future = executor.submit(
                run_single_task_evaluation,
                task_name,
                gpu_id,
                args,
                single_task_script
            )
            future_to_task[future] = task_name
        
        # Wait for all tasks to complete and collect results
        for future in concurrent.futures.as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                success, message = future.result()
                results[task_name] = (success, message)
            except Exception as e:
                logger.error(f"Task '{task_name}' raised an exception: {e}")
                results[task_name] = (False, str(e))
    
    return results


def aggregate_results(output_dir: str, tasks: List[str]) -> pd.DataFrame:
    """
    Aggregate results from all tasks into a summary DataFrame.
    
    Reads the summary JSON files generated by each task evaluation.
    
    Args:
        output_dir: Directory containing the results
        tasks: List of task names that were evaluated
    
    Returns:
        DataFrame with summary metrics for all tasks
    """
    summary_data = []
    
    for task_name in tasks:
        summary_file = os.path.join(output_dir, f"{task_name}_summary.json")
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    task_metrics = json.load(f)
                    summary_data.append(task_metrics)
            except Exception as e:
                logger.warning(f"Failed to read summary for task {task_name}: {e}")
                summary_data.append({
                    'task_name': task_name,
                    'auroc': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'error': str(e)
                })
        else:
            logger.warning(f"Summary file not found for task {task_name}: {summary_file}")
            summary_data.append({
                'task_name': task_name,
                'auroc': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'error': 'Summary file not found'
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        return df
    else:
        return pd.DataFrame()


def validate_tasks(tasks: List[str]) -> bool:
    """
    Validate that all tasks have corresponding queries in TASK_QUERIES.
    
    Args:
        tasks: List of task names to validate
    
    Returns:
        True if all tasks are valid, False otherwise
    """
    invalid_tasks = [task for task in tasks if task not in TASK_QUERIES]
    if invalid_tasks:
        logger.error(f"Invalid tasks (not found in TASK_QUERIES): {invalid_tasks}")
        logger.info(f"Available tasks: {list(TASK_QUERIES.keys())}")
        return False
    return True


def main():
    """Main entry point for parallel multi-task evaluation"""
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.path_to_serialized_data):
        logger.error(f"Serialized data directory not found: {args.path_to_serialized_data}")
        sys.exit(1)
    
    # Validate that all tasks have valid queries (coordinator owns the configuration)
    if not validate_tasks(args.tasks):
        logger.error("Task validation failed. Please check task names.")
        sys.exit(1)
    
    # Determine path to single-task script
    if args.single_task_script:
        single_task_script = args.single_task_script
    else:
        script_dir = get_script_dir()
        single_task_script = os.path.join(script_dir, "evaluate_base_model_single_task.py")
    
    if not os.path.exists(single_task_script):
        logger.error(f"Single-task evaluation script not found: {single_task_script}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("Parallel Multi-Task Evaluation Coordinator")
    logger.info("=" * 80)
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Number of GPUs: {args.num_gpus}")
    logger.info(f"GPU IDs: {list(range(args.gpu_start_id, args.gpu_start_id + args.num_gpus))}")
    logger.info(f"Base Model: {args.base_model_name}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Partition tasks into batches
    batches = partition_tasks_into_batches(args.tasks, args.num_gpus)
    logger.info(f"Partitioned {len(args.tasks)} tasks into {len(batches)} batch(es)")
    for i, batch in enumerate(batches):
        logger.info(f"  Batch {i+1}: {batch}")
    
    # Track results for all tasks
    all_results = {}
    all_task_results = {}
    
    # Process each batch sequentially
    for batch_idx, batch in enumerate(batches):
        logger.info("")
        logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} task(s)")
        logger.info("-" * 80)
        
        # Run batch in parallel
        batch_start_time = time.time()
        batch_results = run_parallel_batch(
            batch,
            args.gpu_start_id,
            args,
            single_task_script
        )
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        
        # Store results
        all_results.update(batch_results)
        
        # Log batch completion
        logger.info(f"Batch {batch_idx + 1} completed in {batch_duration:.2f} seconds")
        for task_name, (success, message) in batch_results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  {task_name}: {status}")
            if not success:
                logger.warning(f"    Error: {message[:200]}")  # Truncate long messages
        
        # Collect task results from output files
        for task_name in batch:
            results_file = os.path.join(args.output_dir, f"{task_name}_results.json")
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        all_task_results[task_name] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load results for {task_name}: {e}")
    
    # Aggregate and save summary
    logger.info("")
    logger.info("Aggregating results...")
    summary_df = aggregate_results(args.output_dir, args.tasks)
    
    if not summary_df.empty:
        summary_file = os.path.join(args.output_dir, "all_tasks_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary saved to {summary_file}")
        
        # Print summary table
        logger.info("")
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY - BASE MODEL (Qwen3-8B)")
        logger.info("=" * 80)
        print(summary_df.to_string(index=False))
        logger.info("=" * 80)
    
    # Final status report
    logger.info("")
    logger.info("=" * 80)
    logger.info("Parallel evaluation completed!")
    logger.info("=" * 80)
    successful_tasks = [task for task, (success, _) in all_results.items() if success]
    failed_tasks = [task for task, (success, _) in all_results.items() if not success]
    
    logger.info(f"Successful tasks ({len(successful_tasks)}/{len(args.tasks)}): {successful_tasks}")
    if failed_tasks:
        logger.warning(f"Failed tasks ({len(failed_tasks)}/{len(args.tasks)}): {failed_tasks}")
    
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)
    
    # Exit with error code if any tasks failed
    if failed_tasks:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

