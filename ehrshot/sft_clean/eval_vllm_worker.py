import os
import json
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

TASK_QUERIES = {
    'acute_mi': 'Will the patient develop an acute myocardial infarction in the next year?',
    'hyperlipidemia': 'Will the patient develop hyperlipidemia in the next year?',
    'hypertension': 'Will the patient develop hypertension in the next year?',
    'pancreatic_cancer': 'Will the patient develop pancreatic cancer in the next year?'
}

def construct_messages(context: str, task_name: str) -> list:
    task_query = TASK_QUERIES[task_name]
    system_content = (
        f"You are a medical expert. You will be provided with a patient's Electronic Healthcare Record (EHR).\n"
        f"Your task is to predict: {task_query}\n"
        f"Think step-by-step to analyze the patient's condition and risk factors.\n"
        f"Format your response as follows:\n"
        f"<think>\n"
        f"[Detailed clinical reasoning]\n"
        f"</think>\n\n"
        f"Final Answer: [Positive/Negative]"
    )
    user_content = context
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

def parse_response(text: str) -> int:
    text = text.strip().lower()
    if "final answer: [positive]" in text or "final answer: positive" in text:
        return 1
    if "final answer: [negative]" in text or "final answer: negative" in text:
        return 0
    return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, required=True)
    # LoRA is now optional
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter. Pass 'None' or omit for Base Model.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine if we are using LoRA
    use_lora = args.lora_path is not None and args.lora_path.lower() != "none"

    # 1. Load Test Data
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    test_data = [d for d in data if d.get("split") == "test"]

    # 2. vLLM Inference
    print(f"Initializing vLLM (LoRA Enabled: {use_lora})...")
    llm = LLM(
        model=args.model_base,
        enable_lora=use_lora, # Only enable if we have a path
        max_model_len=12288,  # Updated to 4096 * 3
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        disable_log_stats=True
    )
    
    # Get tokenizer to apply chat template
    tokenizer = llm.get_tokenizer()

    # 3. Prepare Prompts
    prompts = []
    metadata = []
    for item in test_data:
        messages = construct_messages(item['context'], args.task_name)
        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        metadata.append({
            "patient_id": item['patient_id'],
            "label_time": item['label_time'],
            "ground_truth": item['label_value'],
            "prompt": prompt
        })

    sampling_params = SamplingParams(
        n=10, 
        temperature=0.7, 
        top_p=0.9,
        max_tokens=4096, # Updated to 4096
    )

    # Handle Request Object
    lora_request = None
    if use_lora:
        lora_request = LoRARequest("adapter", 1, args.lora_path)
    
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # 4. Process Responses
    detailed_logs = []
    csv_rows = []
    
    global_total = 0
    global_valid = 0
    global_invalid = 0

    for i, output in enumerate(outputs):
        meta = metadata[i]
        responses = [o.text for o in output.outputs]
        parsed = [parse_response(r) for r in responses]
        
        valid_preds = [p for p in parsed if p != -1]
        num_valid = len(valid_preds)
        num_invalid = 10 - num_valid
        
        global_total += 10
        global_valid += num_valid
        global_invalid += num_invalid

        score = sum(valid_preds) / num_valid if num_valid > 0 else 0.0
        
        detailed_logs.append({
            **meta,
            "score": score,
            "model_responses": responses,
            "parser_outputs": parsed,
            "num_samples": 10,
            "num_valid_predictions": num_valid,
            "num_invalid_predictions": num_invalid,
            "invalid_format_percentage": num_invalid / 10.0
        })

        csv_rows.append({
            "patient_id": meta['patient_id'],
            "label_time": meta['label_time'],
            "target_task": args.task_name,
            "ground_truth": meta['ground_truth'],
            "probability_score": score,
            "has_valid_samples": num_valid > 0
        })

    # 5. Save Files
    with open(os.path.join(args.output_dir, "detailed_logs.json"), 'w') as f:
        json.dump(detailed_logs, f, indent=2)

    pd.DataFrame(csv_rows).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    run_stats = {
        "total_responses": global_total,
        "total_valid_predictions": global_valid,
        "total_invalid_predictions": global_invalid,
        "invalid_format_percentage": global_invalid / global_total if global_total > 0 else 0
    }
    with open(os.path.join(args.output_dir, "run_stats.json"), 'w') as f:
        json.dump(run_stats, f, indent=2)

if __name__ == "__main__":
    main()
