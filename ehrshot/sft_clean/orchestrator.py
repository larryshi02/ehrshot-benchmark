import os
import sys
import time
import subprocess
import queue
import threading
from pathlib import Path

# --- CONFIGURATION ---
EVAL_NAME = "eval_experiment_hfpeft_yn"
MODEL_BASE = "Qwen/Qwen3-8B"
DATA_ROOT = "/dev/shm/ehrshot-data/serialized_multi_task_data"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, EVAL_NAME)
TRAINING_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_qwen3-8b_r16_data_gpt-5-mini_balanced_new_sft_supervised_reasoning_traces")

WORKER_SCRIPT = "eval_vllm_worker.py"
AGGREGATOR_SCRIPT = "eval_vllm_metric_calculator.py" # Check spelling!

# SOURCE_TASKS = ["acute_mi", "pancreatic_cancer", "hypertension", "hyperlipidemia"]

SOURCE_TASKS = ["hypertension", "pancreatic_cancer", "hyperlipidemia", "acute_mi"]
TARGET_TASKS = ["acute_mi", "pancreatic_cancer", "hypertension", "hyperlipidemia"]

TASK_GRID = [
    # ["hypertension", "hypertension"],
    # ["hyperlipidemia", "hyperlipidemia"],
    ["hypertension", "pancreatic_cancer"],
    ["hypertension", "acute_mi"],
    ["hyperlipidemia", "acute_mi"],
    ["hyperlipidemia", "pancreatic_cancer"],
    ["hyperlipidemia", "hypertension"],
    ["hypertension", "hyperlipidemia"],
    # ["acute_mi", "acute_mi"],
    # ["pancreatic_cancer", "pancreatic_cancer"],
    ["acute_mi", "pancreatic_cancer"],
    ["pancreatic_cancer", "acute_mi"],
    ["acute_mi", "hypertension"],
    ["acute_mi", "hyperlipidemia"],
    ["pancreatic_cancer", "hypertension"],
    ["pancreatic_cancer", "hyperlipidemia"]
]

GPU_IDS = [2, 3] # The GPUs you want to use

# --- HELPER: Find Latest Checkpoint ---
def get_latest_checkpoint(base_dir):
    if not os.path.isdir(base_dir):
        return None
    
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    checkpoints = [d for d in subdirs if d.startswith("checkpoint-")]
    
    if not checkpoints:
        return base_dir # Fallback to root if no checkpoints
    
    # Sort by number: checkpoint-100, checkpoint-500
    try:
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        return os.path.join(base_dir, checkpoints[-1])
    except:
        return base_dir

# --- WORKER THREAD ---
def gpu_worker(gpu_id, task_queue):
    """
    Keeps pulling jobs from the queue until empty.
    Runs them on the specific GPU_ID.
    """
    while True:
        try:
            # Try to get a job. If empty for 1 sec, stop.
            job = task_queue.get(timeout=1) 
        except queue.Empty:
            break

        source_model, target_task = job
        
        # 1. Setup Paths
        model_dir = os.path.join(TRAINING_OUTPUT_DIR, source_model)
        lora_path = get_latest_checkpoint(model_dir)
        
        if not lora_path:
            print(f"⚠️  [GPU {gpu_id}] Skipping {source_model}: Dir not found.")
            task_queue.task_done()
            continue

        data_path = os.path.join(DATA_ROOT, f"{target_task}_all_splits.json")
        target_out_dir = os.path.join(OUTPUT_ROOT, source_model, target_task)
        log_file = os.path.join(target_out_dir, "inference.log")

        os.makedirs(target_out_dir, exist_ok=True)

        print(f"🚀 [GPU {gpu_id}] STARTING: Model={source_model} -> Task={target_task}")

        # 2. Construct Command
        cmd = [
            "python", os.path.join(SCRIPT_DIR, WORKER_SCRIPT),
            "--model_base", MODEL_BASE,
            "--lora_path", lora_path,
            "--data_path", data_path,
            "--task_name", target_task,
            "--output_dir", target_out_dir
        ]

        # 3. Run Subprocess with specific GPU visibility
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with open(log_file, "w") as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            print(f"✅ [GPU {gpu_id}] FINISHED: {source_model} -> {target_task}")
        else:
            print(f"❌ [GPU {gpu_id}] FAILED: {source_model} -> {target_task} (Check {log_file})")

        task_queue.task_done()

# --- MAIN ---
def main():
    # 1. Create Output Root
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 2. Build Job Queue (Cross Product of Models x Tasks)
    # We create 16 total jobs
    job_queue = queue.Queue()
    
    # print("--- Queueing Jobs ---")
    # for source_model in SOURCE_TASKS:
    #     for target_task in TARGET_TASKS:
    #         job_queue.put((source_model, target_task))
    #         print(f"  Queued: {source_model} evaluated on {target_task}")

    print("--- Queueing Jobs ---")
    for source_model, target_task in TASK_GRID:
        job_queue.put((source_model, target_task))
        print(f"  Queued: {source_model} evaluated on {target_task}")
    
    print(f"\nTotal Jobs: {job_queue.qsize()}")
    print("Starting Workers...")

    # 3. Start Workers
    threads = []
    for gpu_id in GPU_IDS:
        t = threading.Thread(target=gpu_worker, args=(gpu_id, job_queue))
        t.start()
        threads.append(t)
        time.sleep(2) # Stagger starts slightly to reduce tokenization CPU spike

    # 4. Wait for all threads
    for t in threads:
        t.join()

    print("\n---------------------------------------------------")
    print("All Inference Jobs Complete. Running Aggregator...")
    
    # 5. Run Aggregator
    subprocess.run([
        "python", os.path.join(SCRIPT_DIR, AGGREGATOR_SCRIPT),
        "--eval_root", OUTPUT_ROOT
    ])

if __name__ == "__main__":
    main()