import json
import os
import glob
import re

BASE_DIR = "/home/demirel/clinical-reasoning/ehrshot-benchmark/ehrshot/sft_clean/data_gpt-5-mini/sft_rt_unsupervised"
SUBFOLDERS = ["train_all", "val_small"]

def format_conversation(entry):
    patient_id = entry.get("patient_id")
    label_time = entry.get("label_time")
    label_value = entry.get("label_value")
    input_text = entry.get("input_text", "").strip()
    task_instruction = entry.get("task_instruction", "")
    final_prediction = entry.get("final_prediction", "")
    
    reasoning_raw = entry.get("forward_reasoning", "")
    
    # Process reasoning to separate reasoning from Final Answer
    # Expected format ends with "Final Answer: [Positive/Negative]<STOP>" or similar
    reasoning_content = reasoning_raw
    
    # Regex to find Final Answer at the end
    match = re.search(r'(?s)(.*?)(Final Answer:.*$)', reasoning_raw)
    if match:
        reasoning_content = match.group(1).strip()
    
    # Construct System Message
    system_content = (
        f"You are a medical expert. You will be provided with a patient's Electronic Healthcare Record (EHR).\n"
        f"Your task is to predict: {task_instruction}\n"
        f"Think step-by-step to analyze the patient's condition and risk factors.\n"
        f"Format your response as follows:\n"
        f"<think>\n"
        f"[Detailed clinical reasoning]\n"
        f"</think>\n\n"
        f"Final Answer: [Positive/Negative]"
    )

    # Construct User Message
    user_content = input_text

    # Construct Assistant Message
    assistant_content = (
        f"<think>\n"
        f"{reasoning_content}\n"
        f"</think>\n\n"
        f"Final Answer: {final_prediction}"
    )

    new_conversations = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": assistant_content
        }
    ]
    
    new_entry = {
        "conversations": new_conversations,
        "patient_id": patient_id,
        "label_time": label_time,
        "label_value": label_value,
        "task_instruction": task_instruction
    }
    return new_entry

def process_folder(folder_path):
    print(f"Processing folder: {folder_path}")
    # Find all *_forward_reasoning.json files
    pattern = os.path.join(folder_path, "*_forward_reasoning.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"  No files found matching {pattern}")
        return

    for fr_file in files:
        filename = os.path.basename(fr_file)
        if "_forward_reasoning.json" in filename:
            output_filename = filename.replace("_forward_reasoning.json", "_sft_dataset.json")
            output_path = os.path.join(folder_path, output_filename)
            
            print(f"  Reading {filename} -> Writing {output_filename}")
            
            with open(fr_file, 'r') as f:
                data = json.load(f)
            
            new_data = []
            for entry in data:
                new_entry = format_conversation(entry)
                new_data.append(new_entry)
            
            with open(output_path, 'w') as f:
                json.dump(new_data, f, indent=4)
        else:
            print(f"Skipping file with unexpected name pattern: {filename}")

def main():
    for sub in SUBFOLDERS:
        folder_path = os.path.join(BASE_DIR, sub)
        if os.path.exists(folder_path):
            process_folder(folder_path)
        else:
            print(f"Subfolder not found: {folder_path}")

if __name__ == "__main__":
    main()
